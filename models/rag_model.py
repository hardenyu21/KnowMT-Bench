"""RAG model wrapper for KnowMT-Bench"""

import gc
import os
import pickle
import torch
import faiss
import numpy as np
import torch.nn.functional as F
from typing import List, Union, Optional, Dict, Any
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from llama_index.core.text_splitter import SentenceSplitter
import logging

from .hf_model import HFModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGModel:

    def __init__(
        self,
        base_model_name: str,
        corpus: List[str],
        corpus_name: str,
        embed_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        rerank_model_name: str = "Qwen/Qwen3-Reranker-0.6B",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        max_length: int = 8192,
        **kwargs
    ):
        self.base_model_name = base_model_name
        self.corpus = corpus
        self.corpus_name = corpus_name
        self.device = device
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_length = max_length

        logger.info(f"Initializing base model: {base_model_name}")
        self.base_model = HFModel(base_model_name, device=device, **kwargs)

        logger.info(f"Loading embedding model: {embed_model_name}")
        self.embed_tokenizer = AutoTokenizer.from_pretrained(embed_model_name, padding_side='left')
        self.embed_model = AutoModel.from_pretrained(embed_model_name).to(device).eval()

        logger.info(f"Loading rerank model: {rerank_model_name}")
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(rerank_model_name, padding_side='left')
        self.rerank_model = AutoModelForCausalLM.from_pretrained(rerank_model_name).to(device).eval()

        self.token_false_id = self.rerank_tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.rerank_tokenizer.convert_tokens_to_ids("yes")

        self.rerank_prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        self.rerank_suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.rerank_prefix_tokens = self.rerank_tokenizer.encode(self.rerank_prefix, add_special_tokens=False)
        self.rerank_suffix_tokens = self.rerank_tokenizer.encode(self.rerank_suffix, add_special_tokens=False)

        self.splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        self.chunks, self.embeddings = self._prepare_corpus_embedding()

        logger.info("RAG model initialized successfully!")

    def _last_token_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            seq_len = attention_mask.sum(dim=1) - 1
            return last_hidden_states[torch.arange(last_hidden_states.size(0)), seq_len]

    def _get_query_prompt(self, query: str) -> str:
        return f'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:{query}'

    def _get_embedding(self, texts: Union[str, List[str]], is_query: bool = False) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if is_query:
            texts = [self._get_query_prompt(t) for t in texts]

        batch = self.embed_tokenizer(
            texts, padding=True, truncation=True,
            max_length=8192, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.embed_model(**batch)
            emb = self._last_token_pool(outputs.last_hidden_state, batch['attention_mask'])
            emb = F.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy()

    def _batch_get_embedding(self, texts: List[str], is_query=False, batch_size=8) -> np.ndarray:
        results = []
        if batch_size == 1:
            for i in list(range(0, len(texts), batch_size)):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self._get_embedding(batch_texts, is_query=is_query)
                results.append(batch_embeddings)
        else:
            for i in tqdm(list(range(0, len(texts), batch_size))):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self._get_embedding(batch_texts, is_query=is_query)
                results.append(batch_embeddings)
        return np.vstack(results)

    def _prepare_corpus_embedding(self):
        save_dir = os.path.join("results", "intermediate")
        os.makedirs(save_dir, exist_ok=True)
        chunk_path = os.path.join(save_dir, f"{self.corpus_name}_chunks.pkl")
        emb_path = os.path.join(save_dir, f"{self.corpus_name}_embeddings.npy")

        if os.path.exists(chunk_path) and os.path.exists(emb_path):
            with open(chunk_path, "rb") as f:
                chunks = pickle.load(f)
            embeddings = np.load(emb_path)
            logger.info(f"Loaded {len(chunks)} chunks from cache")
            return chunks, embeddings

        logger.info("Building embeddings from scratch...")
        chunks = []
        for doc in self.corpus:
            chunks.extend(self.splitter.split_text(doc))
        chunks = list(set(chunks))

        logger.info(f"Split into {len(chunks)} chunks, starting encoding...")
        embeddings = self._batch_get_embedding(chunks, is_query=False, batch_size=8)

        with open(chunk_path, "wb") as f:
            pickle.dump(chunks, f)
        np.save(emb_path, embeddings)
        logger.info(f"Embeddings saved to cache: {save_dir}/")
        return chunks, embeddings

    def _format_rerank_instruction(self, instruction: str, query: str, doc: str) -> str:
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

    def _process_rerank_inputs(self, pairs: List[str]):
        inputs = self.rerank_tokenizer(
            pairs,
            padding=False,
            truncation='longest_first',
            return_attention_mask=False,
            max_length=self.max_length - len(self.rerank_prefix_tokens) - len(self.rerank_suffix_tokens)
        )
        for i in range(len(inputs["input_ids"])):
            inputs["input_ids"][i] = self.rerank_prefix_tokens + inputs["input_ids"][i] + self.rerank_suffix_tokens
        inputs = self.rerank_tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        for k in inputs:
            inputs[k] = inputs[k].to(self.device)
        return inputs

    @torch.no_grad()
    def _compute_rerank_scores(
        self,
        queries: List[str],
        documents: List[str],
        instruction: str = "Given a web search query, retrieve relevant passages that answer the query",
        batch_size: int = 1
    ):
        assert len(queries) == len(documents), "Length mismatch between queries and documents"
        scores = []

        for i in range(0, len(queries), batch_size):
            q_batch = queries[i:i + batch_size]
            d_batch = documents[i:i + batch_size]

            pairs = [self._format_rerank_instruction(instruction, q, d) for q, d in zip(q_batch, d_batch)]
            inputs = self._process_rerank_inputs(pairs)

            logits = self.rerank_model(**inputs).logits[:, -1, :]
            yes = logits[:, self.token_true_id]
            no = logits[:, self.token_false_id]
            batch_scores = torch.stack([no, yes], dim=1)
            probs = torch.nn.functional.log_softmax(batch_scores, dim=1)
            scores.extend(probs[:, 1].exp().tolist())

        return scores

    def _rerank(self, query: str, chunks: List[str], top_k: int = 5) -> List[str]:
        scores = self._compute_rerank_scores([query] * len(chunks), chunks)
        reranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return [text for text, _ in reranked[:top_k]]

    def _search_chunks(self, query: str, top_k: int = 5) -> List[str]:
        query_vec = self._batch_get_embedding([query], is_query=True, batch_size=1)
        torch.cuda.empty_cache()
        gc.collect()

        index = faiss.IndexFlatL2(self.embeddings.shape[1])
        index.add(self.embeddings)
        _, I = index.search(query_vec, top_k * 3)
        retrieved = [self.chunks[i] for i in I[0]]
        reranked = self._rerank(query, retrieved, top_k=top_k)
        return reranked

    def _build_rag_prompt(self, retrieved_chunks: List[str], query: str) -> str:
        context = "\n".join([
            f"{i+1}.\n\"\"\"\n{chunk.strip()}\n\"\"\"" for i, chunk in enumerate(retrieved_chunks)
        ])

        prompt = (
            f"The following information has been retrieved and may be helpful:\n\n"
            f"{context}\n\n"
            f"Please answer the following question based only on the information above:\n{query}"
        )
        return prompt

    def generate_response(
        self,
        prompt: str,
        rag_mode: str = "base",
        generation_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        if rag_mode == "base":
            retrieved_chunks = self._search_chunks(prompt)
            rag_prompt = self._build_rag_prompt(retrieved_chunks, prompt)
        else:
            rag_prompt = prompt

        return self.base_model.generate_response(rag_prompt, generation_config, **kwargs)

    def generate_multi_turn(
        self,
        conversation: List[str],
        rag_mode: str = "base",
        generation_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[str]:
        responses = []
        conversation_history = []

        for idx, user_msg in enumerate(conversation):
            if rag_mode == "base":
                if idx == len(conversation) - 1:
                    retrieved_chunks = self._search_chunks(user_msg)
                    user_msg = self._build_rag_prompt(retrieved_chunks, user_msg)
            elif rag_mode == "rounds":
                retrieved_chunks = self._search_chunks(user_msg)
                user_msg = self._build_rag_prompt(retrieved_chunks, user_msg)
            elif rag_mode == "all":
                past_context = "\n\n".join(conversation_history) if conversation_history else ""
                past_context = past_context + "\n\n" + user_msg
                retrieved_chunks = self._search_chunks(past_context)
                user_msg = self._build_rag_prompt(retrieved_chunks, user_msg)
            elif rag_mode == "last":
                if idx == len(conversation) - 1:
                    past_context = "\n\n".join(conversation_history) if conversation_history else ""
                    past_context = past_context + "\n\n" + user_msg
                    retrieved_chunks = self._search_chunks(past_context)
                    user_msg = self._build_rag_prompt(retrieved_chunks, user_msg)
                else:
                    retrieved_chunks = self._search_chunks(user_msg)
                    user_msg = self._build_rag_prompt(retrieved_chunks, user_msg)

            conversation_history.append(user_msg)

            messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": m}
                       for i, m in enumerate(conversation_history)]

            response = self.base_model._generate_from_formatted_prompt(
                messages, generation_config, **kwargs
            )

            responses.append(response)
            conversation_history.append(response)

        return responses

    def cleanup(self):
        try:
            if hasattr(self, 'base_model'):
                self.base_model.cleanup()
            if hasattr(self, 'embed_model'):
                del self.embed_model
            if hasattr(self, 'rerank_model'):
                del self.rerank_model
            if hasattr(self, 'embed_tokenizer'):
                del self.embed_tokenizer
            if hasattr(self, 'rerank_tokenizer'):
                del self.rerank_tokenizer
            torch.cuda.empty_cache()
            gc.collect()
            logger.info(f"Cleaned up RAG model resources")
        except Exception as e:
            logger.warning(f"Error during RAG cleanup: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        base_info = self.base_model.get_model_info()
        return {
            **base_info,
            "model_type": "RAG",
            "corpus_name": self.corpus_name,
            "corpus_size": len(self.corpus),
            "chunk_count": len(self.chunks),
            "embed_model": "Qwen/Qwen3-Embedding-0.6B",
            "rerank_model": "Qwen/Qwen3-Reranker-0.6B"
        }

    def __repr__(self):
        return f"RAGModel(base_model='{self.base_model_name}', corpus='{self.corpus_name}')"

    def __del__(self):
        try:
            self.cleanup()
        except:
            pass