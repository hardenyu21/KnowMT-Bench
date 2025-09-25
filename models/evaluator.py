"""
Response Evaluator for KnowMT-Bench
Based on the implementation from benchmark-code/evaluate_LLMs_all.ipynb
Evaluates entailment and contradiction relationships between claims and ground truth
"""

import logging
import numpy as np
import tiktoken
from typing import List, Dict, Any, Optional
from collections import Counter
from .hf_model import HFModel

# Import prompt templates
try:
    from utils.prompt import IS_ENTAIL, IS_CONTRADICT
except ImportError:
    # Fallback if utils.prompt is not available
    IS_ENTAIL = None
    IS_CONTRADICT = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResponseEvaluator:
    """
    Evaluates entailment and contradiction relationships for factuality assessment
    Based on the evaluation logic from the notebook
    """

    def __init__(
        self,
        eval_model_name: str = "qwen2.5-14b",
        device: str = "auto",
        **kwargs
    ):
        """
        Initialize the evaluator

        Args:
            eval_model_name: Model name for evaluation (from HFModel)
            device: Device to use
            **kwargs: Additional arguments for the model
        """
        self.eval_model_name = eval_model_name
        self.device = device

        logger.info(f"Initializing evaluator with model: {eval_model_name}")
        self.model = HFModel(eval_model_name, device=device, **kwargs)

        # Initialize token counter for efficiency metrics
        self.tokenizer_counter = tiktoken.encoding_for_model("gpt-4o")

        # Load templates from utils.prompt
        self._load_templates()

        logger.info("ResponseEvaluator initialized successfully!")

    def _load_templates(self):
        """Load evaluation prompt templates from utils.prompt"""
        if IS_ENTAIL is not None and IS_CONTRADICT is not None:
            self.entails_template = IS_ENTAIL
            self.contradict_template = IS_CONTRADICT
            logger.info("Loaded evaluation templates from utils.prompt")
        else:
            # Use default templates if utils.prompt is not available
            logger.warning("utils.prompt templates not available, using defaults")
            self.entails_template = self._default_entails_template()
            self.contradict_template = self._default_contradict_template()

    def _default_entails_template(self) -> str:
        """Minimal fallback entailment template"""
        return """Question: {question}
Premise: {llm_answer}
Hypothesis: {answer}

Does the premise entail the hypothesis? Answer "The answer is True" or "The answer is False".

Answer:"""

    def _default_contradict_template(self) -> str:
        """Minimal fallback contradiction template"""
        return """Question: {question}
Premise: {llm_answer}
Hypothesis: {answer}

Does the premise contradict the hypothesis? Answer "The answer is True" or "The answer is False".

Answer:"""

    def _ask(self, prompt: str, generation_config: Optional[Dict[str, Any]] = None) -> str:
        """Ask model with deterministic generation"""
        return self.model.generate_response(
            prompt=prompt,
            generation_config=generation_config or {"max_new_tokens": 1024, "do_sample": False}
        )

    def _ask_sample(self, prompt: str, generation_config: Optional[Dict[str, Any]] = None) -> str:
        """Ask model with sampling for uncertainty handling"""
        return self.model.generate_response(
            prompt=prompt,
            generation_config=generation_config or {
                "max_new_tokens": 1024,
                "do_sample": True,
                "temperature": 0.9,
                "top_p": 0.9999
            }
        )

    def _most_frequent(self, results: List[int]) -> int:
        """Get most frequent result with fallback"""
        if len(results) == 0:
            return 0

        results.sort()
        count = Counter(results)
        max_count = max(count.values())
        frequent_elements = [key for key, value in count.items() if value == max_count]

        if len(frequent_elements) >= 1:
            return frequent_elements[0]
        return 1

    def _parse_binary_result(self, response: str) -> Optional[int]:
        """Parse binary true/false response"""
        response_lower = response.lower().replace('*', '')

        # Check for false indicators
        if ('answer is false' in response_lower or
            'answer as false' in response_lower or
            'answer: false' in response_lower):
            return 0
        # Check for true indicators
        elif ('answer is true' in response_lower or
              'answer as true' in response_lower or
              'answer: true' in response_lower):
            return 1
        else:
            return None

    def is_entails(
        self,
        question: str,
        llm_answer: str,
        reference_answer: str,
        max_retries: int = 5
    ) -> int:
        """
        Determine if LLM answer entails the reference answer

        Args:
            question: Original question
            llm_answer: LLM's answer
            reference_answer: Ground truth reference
            max_retries: Maximum sampling retries for uncertain cases

        Returns:
            1 if entails, 0 if not
        """
        prompt = self.entails_template.format(
            question=question,
            llm_answer=llm_answer,
            answer=reference_answer
        )

        # First try deterministic
        response = self._ask(prompt)
        result = self._parse_binary_result(response)

        if result is not None:
            return result

        # If uncertain, try sampling
        entail_results = []
        for _ in range(max_retries):
            response = self._ask_sample(prompt)
            result = self._parse_binary_result(response)

            if result is not None:
                entail_results.append(result)

            if len(entail_results) == 3:  # Early stopping
                break

        return self._most_frequent(entail_results)

    def is_contradict(
        self,
        question: str,
        llm_answer: str,
        reference_answer: str,
        max_retries: int = 5
    ) -> int:
        """
        Determine if LLM answer contradicts the reference answer

        Args:
            question: Original question
            llm_answer: LLM's answer
            reference_answer: Ground truth reference
            max_retries: Maximum sampling retries for uncertain cases

        Returns:
            1 if contradicts, 0 if not
        """
        prompt = self.contradict_template.format(
            question=question,
            llm_answer=llm_answer,
            answer=reference_answer
        )

        # First try deterministic
        response = self._ask(prompt)
        result = self._parse_binary_result(response)

        if result is not None:
            return result

        # If uncertain, try sampling
        contradict_results = []
        for _ in range(max_retries):
            response = self._ask_sample(prompt)
            result = self._parse_binary_result(response)

            if result is not None:
                contradict_results.append(result)

            if len(contradict_results) == 3:  # Early stopping
                break

        return self._most_frequent(contradict_results)

    def evaluate_sample(
        self,
        question: str,
        response: str,
        ground_truth: str,
        must_have_facts: List[str],
        statements: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate a single sample comprehensively

        Args:
            question: Original question
            response: Model response
            ground_truth: Ground truth answer
            must_have_facts: List of facts that must be present
            statements: Decomposed statements from response

        Returns:
            Dictionary with all evaluation metrics
        """
        # Token count for efficiency metrics
        token_count = len(self.tokenizer_counter.encode(response))

        # Fact entailment (recall): Do facts appear in the response?
        fact_entails = [
            self.is_entails(question, response, fact)
            for fact in must_have_facts
        ]

        # Fact contradiction: Do facts contradict the response?
        fact_contradict = [
            self.is_contradict(question, response, fact)
            for fact in must_have_facts
        ]

        # Statement entailment (precision): Are statements supported by ground truth?
        statement_entails = [
            self.is_entails(question, ground_truth, stmt)
            for stmt in statements
        ]

        # Statement contradiction: Do statements contradict ground truth?
        statement_contradict = [
            self.is_contradict(question, ground_truth, stmt)
            for stmt in statements
        ]

        # Calculate metrics
        recall_fact = np.array(fact_entails).mean()
        precision_fact = np.array(statement_entails).mean()

        # F1 for factuality
        if (recall_fact + precision_fact) == 0:
            f1_fact = 0
        else:
            f1_fact = 2 * recall_fact * precision_fact / (recall_fact + precision_fact)

        # Hallucination metrics
        recall_hall = np.array(statement_contradict).mean()
        precision_hall = np.array(fact_contradict).mean()

        if (recall_hall + precision_hall) == 0:
            f1_hall = 0
        else:
            f1_hall = 2 * recall_hall * precision_hall / (recall_hall + precision_hall)

        # Efficiency metrics
        fact_entails_sum = np.array(fact_entails).sum()
        fact_contradict_sum = np.array(fact_contradict).sum()

        fact_eff = token_count / fact_entails_sum if fact_entails_sum > 0 else float('inf')
        hall_eff = token_count / fact_contradict_sum if fact_contradict_sum > 0 else float('inf')
        recall_eff = token_count / (fact_entails_sum / len(must_have_facts)) if fact_entails_sum > 0 else float('inf')

        return {
            'token_count': token_count,
            'fact_entails': fact_entails,
            'fact_contradict': fact_contradict,
            'statement_entails': statement_entails,
            'statement_contradict': statement_contradict,
            'num_facts': len(must_have_facts),
            'num_statements': len(statements),
            'recall_fact': recall_fact,
            'precision_fact': precision_fact,
            'F1_fact': f1_fact,
            'recall_hall': recall_hall,
            'precision_hall': precision_hall,
            'F1_hall': f1_hall,
            'fact_eff': fact_eff,
            'hall_eff': hall_eff,
            'recall_eff': recall_eff
        }

    def evaluate_batch(
        self,
        samples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Evaluate a batch of samples

        Args:
            samples: List of sample dictionaries containing:
                - question: str
                - response: str
                - ground_truth: str
                - must_have_facts: List[str]
                - statements: List[str]

        Returns:
            List of evaluation results
        """
        results = []

        for i, sample in enumerate(samples):
            logger.info(f"Evaluating sample {i+1}/{len(samples)}")

            result = self.evaluate_sample(
                question=sample['question'],
                response=sample['response'],
                ground_truth=sample['ground_truth'],
                must_have_facts=sample['must_have_facts'],
                statements=sample['statements']
            )

            result['idx'] = i
            results.append(result)

        return results

    @staticmethod
    def compute_mean_with_nan_max(sample_metrics: List[Dict[str, Any]], key: str, default_if_all_inf: float = 0) -> float:
        """
        Compute mean while handling infinite values

        Args:
            sample_metrics: List of metric dictionaries
            key: Key to compute mean for
            default_if_all_inf: Default value if all are infinite

        Returns:
            Mean value with inf handling
        """
        values = np.array([item[key] for item in sample_metrics])
        valid_values = values[np.isfinite(values)]

        if len(valid_values) == 0:
            max_val = default_if_all_inf
        else:
            max_val = valid_values.max()

        values = np.where(np.isfinite(values), values, max_val)
        return values.mean()

    def aggregate_metrics(self, sample_metrics: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Aggregate metrics across all samples

        Args:
            sample_metrics: List of per-sample metrics

        Returns:
            Aggregated metrics dictionary
        """
        if not sample_metrics:
            return {}

        # Simple averages
        recall_fact = np.array([m['recall_fact'] for m in sample_metrics]).mean()
        precision_fact = np.array([m['precision_fact'] for m in sample_metrics]).mean()
        f1_fact = np.array([m['F1_fact'] for m in sample_metrics]).mean()

        recall_hall = np.array([m['recall_hall'] for m in sample_metrics]).mean()
        precision_hall = np.array([m['precision_hall'] for m in sample_metrics]).mean()
        f1_hall = np.array([m['F1_hall'] for m in sample_metrics]).mean()

        # Efficiency metrics with inf handling
        fact_eff = self.compute_mean_with_nan_max(sample_metrics, 'fact_eff')
        hall_eff = self.compute_mean_with_nan_max(sample_metrics, 'hall_eff')
        recall_eff = self.compute_mean_with_nan_max(sample_metrics, 'recall_eff')

        return {
            'recall_fact': recall_fact,
            'precision_fact': precision_fact,
            'F1_fact': f1_fact,
            'recall_hall': recall_hall,
            'precision_hall': precision_hall,
            'F1_hall': f1_hall,
            'fact_eff': fact_eff,
            'hall_eff': hall_eff,
            'recall_eff': recall_eff
        }

    def cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'model'):
                self.model.cleanup()
            logger.info("ResponseEvaluator cleaned up successfully")
        except Exception as e:
            logger.warning(f"Error during evaluator cleanup: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        base_info = self.model.get_model_info()
        return {
            **base_info,
            "component": "ResponseEvaluator",
            "task": "entailment_contradiction_evaluation"
        }

    def __repr__(self):
        return f"ResponseEvaluator(model='{self.eval_model_name}')"

    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.cleanup()
        except:
            pass