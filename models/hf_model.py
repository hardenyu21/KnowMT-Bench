"""
Hugging Face model wrapper for KnowMT-Bench
"""

import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional, Dict, Any
import warnings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HFModel:
    """Hugging Face model wrapper for loading and running inference"""
    SUPPORTED_MODELS = {
        "qwen2.5-7b": {
            "model_name": "Qwen/Qwen2.5-7B-Instruct"
        },
        "qwen2.5-14b": {
            "model_name": "Qwen/Qwen2.5-14B-Instruct"
        },
        "qwen2.5-32b": {
            "model_name": "Qwen/Qwen2.5-32B-Instruct"
        },
        "qwen2.5-72b": {
            "model_name": "Qwen/Qwen2.5-72B-Instruct"
        },
        "llama3-8b": {
            "model_name": "meta-llama/Meta-Llama-3-8B-Instruct"
        },
        "llama3-70b": {
            "model_name": "meta-llama/Meta-Llama-3-70B-Instruct"
        },
        "llama3.1-8b": {
            "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct"
        },
        "llama3.1-70b": {
            "model_name": "meta-llama/Meta-Llama-3.1-70B-Instruct"
        },
        "llama3.3-70b": {
            "model_name": "meta-llama/Llama-3.3-70B-Instruct"
        },
        "qwq-32b": {
            "model_name": "Qwen/QwQ-32B"
        },
        "huatuogpt-o1-7b": {
            "model_name": "FreedomIntelligence/HuatuoGPT-o1-7B",
            "special_handling": "medical_cot"
        },
        "fin-r1": {
            "model_name": "SUFE-AIFLM-Lab/Fin-R1",
            "special_handling": "financial_cot"
        }
    }
    
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        use_flash_attention: bool = False,
        **kwargs
    ):
        """
        Initialize the HF model
        
        Args:
            model_name: Key from SUPPORTED_MODELS dict (e.g., "qwen2.5-7b", "llama3-8b", "llama3.3-70b", "qwq-32b")
            device: Device to load model on (default: "auto")
            use_flash_attention: Whether to use Flash Attention 2 for efficiency
            **kwargs: Additional arguments for model loading
        """
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model name: {model_name}. "
                           f"Supported models: {list(self.SUPPORTED_MODELS.keys())}")
        
        self.model_name = model_name
        self.config = self.SUPPORTED_MODELS[model_name]
        self.device = device
        self.use_flash_attention = use_flash_attention
        
        logger.info(f"Initializing {model_name} model...")
        self._load_model_and_tokenizer(**kwargs)
        
    def _load_model_and_tokenizer(self, **kwargs):
        """Load the model and tokenizer"""
        model_name = self.config["model_name"]
        
        logger.info(f"Loading tokenizer for {model_name}...")
        tokenizer_kwargs = {}
        model_kwargs = {
            "dtype": torch.bfloat16,
            "device_map": "auto",
            **kwargs
        }
        
        if "qwen" in model_name.lower() or "qwq" in model_name.lower():
            tokenizer_kwargs["trust_remote_code"] = True
            model_kwargs["trust_remote_code"] = True
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            **tokenizer_kwargs,
            **kwargs
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if self.use_flash_attention:
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Using Flash Attention 2 for improved efficiency")
            except Exception as e:
                logger.warning(f"Flash Attention 2 not available, falling back to default: {e}")
                self.use_flash_attention = False
        
        logger.info(f"Loading model {model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
            
        logger.info(f"Model {self.model_name} loaded successfully!")
    
    def generate_response(
        self,
        prompt: str,
        generation_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        Generate a single response to a prompt

        Args:
            prompt: Input prompt
            generation_config: Dict with generation parameters:
                - do_sample: Whether to use sampling (default: False)
                - max_new_tokens: Maximum number of new tokens (default: 1024)
                - temperature: Sampling temperature
                - top_p: Top-p sampling parameter
                - etc.
            **kwargs: Additional generation parameters

        Returns:
            Generated response string
        """
        messages = [{"role": "user", "content": prompt}]
        return self._generate_from_formatted_prompt(messages, generation_config, **kwargs)

    
    def generate_multi_turn(
        self,
        conversation: List[str],
        generation_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[str]:
        """
        Generate responses for multi-turn conversation
        
        Args:
            conversation: List of user messages
            generation_config: Dict with generation parameters (same as generate_response)
            **kwargs: Additional generation parameters
            
        Returns:
            List of model responses
        """
        responses = []
        conversation_history = []
        
        for user_message in conversation:
            conversation_history.append(user_message)
            
            messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": m} 
                       for i, m in enumerate(conversation_history)]
            
            response = self._generate_from_formatted_prompt(
                messages, generation_config, **kwargs
            )
            
            responses.append(response)
            conversation_history.append(response)
        
        return responses
    

    def clean_response(self, response: str, model_type: str = "") -> str:
        """
        Clean the response based on model type to extract final answer
        Handles edge cases like truncation and empty content after markers

        Args:
            response: Raw model response
            model_type: Type of model (cot, huatuo, etc.) or empty to use current model

        Returns:
            Cleaned response text with robust fallback handling
        """
        # Use current model name if model_type not provided
        if not model_type:
            model_type = self.model_name

        original_response = response.strip()

        if 'huatuogpto1_7b_cots' in model_type.lower():
            marker = '## Final Response'
            if marker in response:
                # (i) If a clear final answer is present, only that answer is retained
                final_answer = response.split(marker)[1].strip()

                if final_answer:  # Non-empty final answer exists
                    logger.info(f"HuatuoGPT: Extracted final answer ({len(final_answer)} chars) after '{marker}'")
                    return final_answer
                else:
                    # Empty after marker - treat as truncated
                    logger.warning(f"HuatuoGPT: No content after '{marker}' - treating entire response as answer (truncation case)")
                    return original_response
            else:
                # (ii) No explicit answer marker - treat entire reasoning as answer
                logger.warning(f"HuatuoGPT: No '{marker}' marker found - treating entire reasoning output as answer")
                return original_response

        elif 'cot' in model_type.lower():
            marker = '</think>'
            if marker in response:
                # (i) If a clear final answer is present, only that answer is retained
                last_marker_pos = response.rfind(marker)
                final_answer = response[last_marker_pos + len(marker):].strip()

                if final_answer:  # Non-empty final answer exists
                    logger.info(f"CoT: Extracted final answer ({len(final_answer)} chars) after '{marker}'")
                    return final_answer
                else:
                    # Empty after marker - treat as truncated
                    logger.warning(f"CoT: No content after '{marker}' - treating entire response as answer (truncation case)")
                    return original_response
            else:
                # (ii) No explicit answer marker - treat entire reasoning as answer
                logger.warning(f"CoT: No '{marker}' marker found - treating entire reasoning output as answer")
                return original_response

        else:
            return original_response
    
    def _generate_from_formatted_prompt(
        self,
        messages: List[Dict[str, str]],
        generation_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """Generate from chat messages"""
        if "qwen" in self.config["model_name"].lower() or "qwq" in self.config["model_name"].lower():
            
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        else:
            
            input_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(self.model.device)
            inputs = {"input_ids": input_ids}
        
        default_config = {
            "do_sample": False,
            "max_new_tokens": 1024,
        }

        # Handle special model configurations
        if self.config.get("special_handling") == "medical_cot":
            # Add system prompt for medical reasoning models
            if len(messages) == 1 and messages[0]["role"] == "user":
                system_prompt = {
                    "role": "system",
                    "content": "You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"
                }
                messages = [system_prompt] + messages
        elif self.config.get("special_handling") == "financial_cot":
            # Add system prompt for financial reasoning models
            if len(messages) == 1 and messages[0]["role"] == "user":
                system_prompt = {
                    "role": "system",
                    "content": "You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"
                }
                messages = [system_prompt] + messages
        
        gen_kwargs = {
            "pad_token_id": self.tokenizer.eos_token_id,
            **default_config,
            **kwargs
        }
        
        if generation_config:
            gen_kwargs.update(generation_config)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        response = self.clean_response(response)
        
        return response
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "model_path": self.config["model_name"],
            "device": str(self.model.device) if hasattr(self.model, 'device') else "auto",
            "dtype": "torch.bfloat16",
            "flash_attention": self.use_flash_attention,
            "device_map": "auto"
        }
    
    @classmethod
    def list_supported_models(cls) -> List[str]:
        """List all supported model keys"""
        return list(cls.SUPPORTED_MODELS.keys())
    
    def cleanup(self):
        """
        Clean up GPU memory by deleting model and clearing cache
        Based on patterns from the notebook for memory management
        """
        try:
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'tokenizer'):
                del self.tokenizer
            torch.cuda.empty_cache()
            gc.collect()
            logger.info(f"Cleaned up resources for {self.model_name}")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")


    def __repr__(self):
        return f"HFModel(model_name='{self.model_name}', device='{self.device}')"

    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.cleanup()
        except:
            pass
