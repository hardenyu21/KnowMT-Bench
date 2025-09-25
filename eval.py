#!/usr/bin/env python3

import json
import os
import gc
import torch
import argparse
import logging
import math
from datetime import datetime
from typing import Dict, Any, List, Optional
from tqdm import tqdm

from models.hf_model import HFModel
from models.decomposer import ResponseDecomposer
from models.evaluator import ResponseEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def convert_inf_to_str(obj):
    if isinstance(obj, float):
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        elif math.isnan(obj):
            return "nan"
    elif isinstance(obj, dict):
        return {k: convert_inf_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_inf_to_str(item) for item in obj]
    return obj


def safe_json_dump(obj, file_path: str, indent: int = 2):
    converted_obj = convert_inf_to_str(obj)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(converted_obj, f, indent=indent, ensure_ascii=False)


class StagedKnowMTEvaluator:

    def __init__(
        self,
        model_name: str,
        decompose_model_name: str = "qwen2.5-32b",
        eval_model_name: str = "qwen2.5-14b",
        device: str = "auto",
        output_path: str = "results/",
        **kwargs
    ):
        self.model_name = model_name
        self.decompose_model_name = decompose_model_name
        self.eval_model_name = eval_model_name
        self.device = device
        self.output_path = output_path
        self.model_kwargs = kwargs

        self.target_model = None
        self.decomposer = None
        self.evaluator = None

        self.unified_results = {
            "metadata": {
                "model_name": model_name,
                "decompose_model": decompose_model_name,
                "eval_model": eval_model_name,
                "device": device,
                "timestamp_start": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                "aggregated_metrics": {}
            },
            "samples": [],
            "status": "initialized"
        }

        os.makedirs(output_path, exist_ok=True)
        logger.info("StagedKnowMTEvaluator initialized - models will be loaded per stage")

    def load_data(self, data_path: str) -> List[Dict[str, Any]]:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data["samples"]

    def cleanup_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def get_unified_results_path(self, suffix: str = "") -> str:
        filename = f"{self.model_name}_evaluation{suffix}.json"
        return os.path.join(self.output_path, filename)

    def save_unified_results(self, suffix: str = ""):
        filepath = self.get_unified_results_path(suffix)
        safe_json_dump(self.unified_results, filepath)
        logger.info(f"Unified results saved to: {filepath}")

    def load_unified_results(self, suffix: str = "") -> Optional[Dict[str, Any]]:
        filepath = self.get_unified_results_path(suffix)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                self.unified_results = json.load(f)
                return self.unified_results
        return None

    def update_sample_result(self, sample_idx: int, update_data: Dict[str, Any], suffix: str = ""):
        while len(self.unified_results["samples"]) <= sample_idx:
            self.unified_results["samples"].append({})
        self.unified_results["samples"][sample_idx].update(update_data)
        self.save_unified_results(suffix)

    def stage1_generate_responses(
        self,
        samples: List[Dict[str, Any]],
        mode: str = "multi",
        suffix: str = ""
    ) -> Dict[str, Any]:
        logger.info("=== Stage 1: Generating Responses ===")

        self.unified_results["status"] = "stage1_generating"
        self.unified_results["metadata"]["mode"] = mode
        self.unified_results["metadata"]["total_samples"] = len(samples)
        self.save_unified_results(suffix)

        logger.info(f"Loading target model: {self.model_name}")
        self.target_model = HFModel(self.model_name, device=self.device, **self.model_kwargs)

        for i, sample in enumerate(tqdm(samples, desc="Generating responses")):
            try:
                sample_result = {
                    "sample_id": sample["sample_id"],
                    "domain": sample["domain"],
                    "ground_truth": sample["answer"],
                    "must_have_facts": sample["must_have"]
                }

                if mode == "single":
                    question = sample["single-turn question"]
                    response = self.target_model.generate_response(question)
                    sample_result.update({
                        "question": question,
                        "response": response
                    })
                elif mode == "multi":
                    questions = sample["multi-turn questions"]
                    responses = self.target_model.generate_multi_turn(questions)
                    sample_result.update({
                        "questions": questions,
                        "responses": responses,
                        "final_question": questions[-1],
                        "final_response": responses[-1]
                    })

                self.update_sample_result(i, sample_result, suffix)

                if (i + 1) % 10 == 0:
                    logger.info(f"Generated responses for {i + 1}/{len(samples)} samples")

            except Exception as e:
                logger.error(f"Error generating response for sample {sample.get('sample_id', i)}: {e}")
                error_result = {
                    "sample_id": sample.get("sample_id", f"sample_{i}"),
                    "error": str(e),
                    "stage1_status": "failed"
                }
                self.update_sample_result(i, error_result, suffix)
                continue

        self.target_model.cleanup()
        self.target_model = None
        self.cleanup_memory()
        logger.info("Target model cleaned up")

        self.unified_results["status"] = "stage1_completed"
        self.unified_results["metadata"]["stage1_timestamp"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        self.save_unified_results(suffix)

        return self.unified_results

    def stage2_decompose_responses(
        self,
        suffix: str = ""
    ) -> Dict[str, Any]:
        logger.info("=== Stage 2: Decomposing Responses ===")

        self.unified_results["status"] = "stage2_decomposing"
        self.save_unified_results(suffix)

        logger.info(f"Loading decomposer: {self.decompose_model_name}")
        self.decomposer = ResponseDecomposer(self.decompose_model_name, device=self.device, **self.model_kwargs)

        mode = self.unified_results["metadata"]["mode"]
        samples = self.unified_results["samples"]

        for i, sample_result in enumerate(tqdm(samples, desc="Decomposing responses")):
            try:
                if "error" in sample_result or "stage1_status" in sample_result:
                    continue

                if mode == "single":
                    question = sample_result["question"]
                    response = sample_result["response"]
                elif mode == "multi":
                    question = sample_result["final_question"]
                    response = sample_result["final_response"]

                statements = self.decomposer.decompose_response(question, response, self.model_name)
                update_data = {"statements": statements}
                self.update_sample_result(i, update_data, suffix)

                if (i + 1) % 10 == 0:
                    logger.info(f"Decomposed responses for {i + 1}/{len(samples)} samples")

            except Exception as e:
                logger.error(f"Error decomposing response for sample {sample_result.get('sample_id', i)}: {e}")
                error_update = {
                    "statements": [],
                    "stage2_error": str(e),
                    "stage2_status": "failed"
                }
                self.update_sample_result(i, error_update, suffix)
                continue

        self.decomposer.cleanup()
        self.decomposer = None
        self.cleanup_memory()
        logger.info("Decomposer cleaned up")

        self.unified_results["status"] = "stage2_completed"
        self.unified_results["metadata"]["stage2_timestamp"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        self.save_unified_results(suffix)

        return self.unified_results

    def stage3_evaluate_responses(
        self,
        suffix: str = ""
    ) -> Dict[str, Any]:
        logger.info("=== Stage 3: Evaluating Responses ===")

        self.unified_results["status"] = "stage3_evaluating"
        self.save_unified_results(suffix)

        logger.info(f"Loading evaluator: {self.eval_model_name}")
        self.evaluator = ResponseEvaluator(self.eval_model_name, device=self.device, **self.model_kwargs)

        mode = self.unified_results["metadata"]["mode"]
        samples = self.unified_results["samples"]

        for i, sample_result in enumerate(tqdm(samples, desc="Evaluating responses")):
            try:
                if ("error" in sample_result or
                    "stage1_status" in sample_result or
                    "stage2_status" in sample_result):
                    continue

                if mode == "single":
                    question = sample_result["question"]
                    response = sample_result["response"]
                elif mode == "multi":
                    question = sample_result["final_question"]
                    response = sample_result["final_response"]

                evaluation = self.evaluator.evaluate_sample(
                    question=question,
                    response=response,
                    ground_truth=sample_result["ground_truth"],
                    must_have_facts=sample_result["must_have_facts"],
                    statements=sample_result.get("statements", [])
                )

                update_data = {"evaluation": evaluation}
                self.update_sample_result(i, update_data, suffix)

                if (i + 1) % 10 == 0:
                    logger.info(f"Evaluated responses for {i + 1}/{len(samples)} samples")

            except Exception as e:
                logger.error(f"Error evaluating response for sample {sample_result.get('sample_id', i)}: {e}")
                error_update = {
                    "evaluation": {},
                    "stage3_error": str(e),
                    "stage3_status": "failed"
                }
                self.update_sample_result(i, error_update, suffix)
                continue

        sample_metrics = [s.get("evaluation", {}) for s in samples if s.get("evaluation")]
        aggregated_metrics = self.evaluator.aggregate_metrics(sample_metrics) if sample_metrics else {}

        self.evaluator.cleanup()
        self.evaluator = None
        self.cleanup_memory()
        logger.info("Evaluator cleaned up")

        self.unified_results["status"] = "completed"
        self.unified_results["metadata"]["aggregated_metrics"] = aggregated_metrics
        self.unified_results["metadata"]["stage3_timestamp"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        self.unified_results["metadata"]["timestamp_end"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        self.save_unified_results(suffix)

        return self.unified_results

    def evaluate_dataset_staged(
        self,
        data_path: str,
        mode: str = "multi",
        limit: int = None,
        start_idx: int = 0,
        suffix: str = "",
        resume_from_stage: int = 1
    ) -> Dict[str, Any]:
        if resume_from_stage > 1:
            existing_results = self.load_unified_results(suffix)
            if existing_results:
                logger.info(f"Loaded existing results, resuming from stage {resume_from_stage}")
            else:
                raise ValueError(f"Cannot resume from stage {resume_from_stage}: no existing results found")

        if resume_from_stage == 1:
            samples = self.load_data(data_path)

            if limit:
                samples = samples[start_idx:start_idx + limit]
            else:
                samples = samples[start_idx:]

            logger.info(f"Evaluating {len(samples)} samples in {mode}-turn mode")

            self.stage1_generate_responses(samples, mode, suffix)

        if resume_from_stage <= 2:
            self.stage2_decompose_responses(suffix)

        if resume_from_stage <= 3:
            final_results = self.stage3_evaluate_responses(suffix)
        else:
            final_results = self.unified_results

        return final_results

    def print_summary(self, final_results: Dict[str, Any]):
        metadata = final_results.get("metadata", {})
        metrics = metadata.get("aggregated_metrics", {})
        completed_samples = len([s for s in final_results.get("samples", []) if s.get("evaluation")])

        logger.info("=== Evaluation Summary ===")
        logger.info(f"Model: {metadata.get('model_name', 'Unknown')}")
        logger.info(f"Mode: {metadata.get('mode', 'Unknown')}-turn")
        logger.info(f"Total Samples: {metadata.get('total_samples', 0)}")
        logger.info(f"Completed Samples: {completed_samples}")
        logger.info(f"Status: {final_results.get('status', 'Unknown')}")
        logger.info(f"Factuality F1: {metrics.get('F1_fact', 0):.4f}")
        logger.info(f"Hallucination F1: {metrics.get('F1_hall', 0):.4f}")
        logger.info(f"Recall Fact: {metrics.get('recall_fact', 0):.4f}")
        logger.info(f"Precision Fact: {metrics.get('precision_fact', 0):.4f}")

        logger.info(f"Results file: {self.get_unified_results_path()}")

    def cleanup(self):
        logger.info("Final cleanup...")
        try:
            if self.target_model:
                self.target_model.cleanup()
                self.target_model = None
            if self.decomposer:
                self.decomposer.cleanup()
                self.decomposer = None
            if self.evaluator:
                self.evaluator.cleanup()
                self.evaluator = None
            self.cleanup_memory()
            logger.info("Final cleanup completed")
        except Exception as e:
            logger.warning(f"Error during final cleanup: {e}")


def main():
    parser = argparse.ArgumentParser(description="KnowMT-Bench Staged Evaluation")

    parser.add_argument("--model", required=True, type=str,
                       help="Model name from HFModel.SUPPORTED_MODELS")
    parser.add_argument("--data_path", required=True, type=str,
                       help="Path to KnowMT_QA.json")
    parser.add_argument("--output_path", required=True, type=str,
                       help="Output directory for results")

    parser.add_argument("--mode", default="multi", choices=["single", "multi"],
                       help="Evaluation mode: single or multi-turn")
    parser.add_argument("--decompose_model", default="qwen2.5-32b", type=str,
                       help="Model for response decomposition")
    parser.add_argument("--eval_model", default="qwen2.5-14b", type=str,
                       help="Model for evaluation")
    parser.add_argument("--device", default="auto", type=str,
                       help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--limit", type=int,
                       help="Maximum number of samples to evaluate")
    parser.add_argument("--start_idx", default=0, type=int,
                       help="Starting index for evaluation")
    parser.add_argument("--suffix", default="", type=str,
                       help="Suffix for output filename")
    parser.add_argument("--resume_from_stage", default=1, type=int, choices=[1, 2, 3],
                       help="Resume from stage (1=generation, 2=decomposition, 3=evaluation)")

    args = parser.parse_args()

    if args.model not in HFModel.list_supported_models():
        logger.error(f"Unsupported model: {args.model}")
        logger.error(f"Supported models: {HFModel.list_supported_models()}")
        return

    evaluator = StagedKnowMTEvaluator(
        model_name=args.model,
        decompose_model_name=args.decompose_model,
        eval_model_name=args.eval_model,
        device=args.device,
        output_path=args.output_path
    )

    try:
        results = evaluator.evaluate_dataset_staged(
            data_path=args.data_path,
            mode=args.mode,
            limit=args.limit,
            start_idx=args.start_idx,
            suffix=args.suffix,
            resume_from_stage=args.resume_from_stage
        )

        evaluator.print_summary(results)

    finally:
        evaluator.cleanup()


if __name__ == "__main__":
    main()