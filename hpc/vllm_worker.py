"""
Ray actor wrapping a vLLM engine for distributed LLM inference.
Each actor uses one node with 4 GPUs via tensor parallelism.
"""

import json
import logging
import socket
from typing import Dict, List, Any, Optional

import ray
from vllm import LLM, SamplingParams

from src.llm_prompts import (
    SYSTEM_PROMPT,
    DOMAIN_CLASSIFICATION_PROMPT,
    ALGORITHMS_PROMPT,
    PARADIGMS_AND_PATTERNS_PROMPT,
)

logger = logging.getLogger(__name__)

# Prompt types used to map results back to samples.
PROMPT_TYPE_DOMAINS = "domains"
PROMPT_TYPE_ALGORITHMS = "algorithms"
PROMPT_TYPE_PARADIGMS_PATTERNS = "paradigms_and_patterns"


@ray.remote(num_gpus=4)
class VLLMPipelineActor:
    """
    Ray actor that owns a vLLM engine on 4 GPUs.
    Processes batches of code samples with three extraction prompts.

    Uses distributed_executor_backend="mp" so vLLM manages tensor parallelism
    internally via multiprocessing.
    """

    def __init__(
        self,
        model_name: str,
        tp_size: int = 4,
        max_model_len: int = 16384,
        gpu_memory_utilization: float = 0.85,
        max_num_seqs: int = 64,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.8,
        local_model_dir: Optional[str] = None,
        prompt_overhead_tokens: int = 500,
    ):
        self.model_name = model_name

        import os

        # Prefer a node-local directory to reduce shared filesystem I/O contention.
        if local_model_dir and os.path.isdir(local_model_dir):
            model_path = local_model_dir
            logger.info(f"Using node-local model from: {model_path}")
        else:
            # Fallback: resolve the model from the Hugging Face cache.
            from huggingface_hub import snapshot_download

            try:
                model_path = snapshot_download(
                    repo_id=model_name,
                    local_files_only=True,
                    cache_dir=os.environ.get("HF_HOME"),
                )
                logger.info(f"Using cached model from: {model_path}")
            except Exception as e:
                logger.warning(f"Could not resolve cached model path: {e}")
                logger.warning(f"Falling back to model name: {model_name}")
                model_path = model_name

        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tp_size,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            gpu_memory_utilization=gpu_memory_utilization,
            distributed_executor_backend="mp",
            trust_remote_code=True,
            enforce_eager=False,
        )

        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        self.prompt_overhead_tokens = prompt_overhead_tokens
        self.tokenizer = self.llm.get_tokenizer()

        self._hostname = socket.gethostname()
        logger.info(
            f"VLLMPipelineActor initialized on {self._hostname} "
            f"with model={model_name}, TP={tp_size}"
        )

    def health_check(self) -> str:
        return f"alive on {self._hostname}"

    def process_samples(self, samples: List[Dict]) -> List[Dict]:
        """
        Process a batch of code samples. For each sample, generates 3 prompts
        (domain classification, algorithms, paradigms+patterns), runs them all
        through vLLM in a single batched generate() call, then maps results back.

        Args:
            samples: List of dicts with keys: content, blob_id, language, size,
                     repository, file_path, etc. (as produced by download_stack_edu_dataset)

        Returns:
            List of result dicts matching main.py process_code_sample() output format:
            {sample_id, language, domains, functions, algorithms, paradigms,
             design_patterns, file_node_id}
        """
        prompts = []
        prompt_map = []  # (sample_index, prompt_type)

        # Truncate code to fit within max_model_len.
        # Reserve tokens for: prompt template overhead + max output (max_tokens).
        max_code_tokens = self.llm.llm_engine.model_config.max_model_len - self.sampling_params.max_tokens - self.prompt_overhead_tokens

        for i, sample in enumerate(samples):
            code = sample.get("content", "")
            if not code.strip():
                continue

            token_ids = self.tokenizer.encode(code, add_special_tokens=False)
            if len(token_ids) > max_code_tokens:
                token_ids = token_ids[:max_code_tokens]
                code = self.tokenizer.decode(token_ids, skip_special_tokens=True) + "\n# ... [truncated]"

            prompts.append(
                DOMAIN_CLASSIFICATION_PROMPT.format(code=code)
            )
            prompt_map.append((i, PROMPT_TYPE_DOMAINS))

            prompts.append(ALGORITHMS_PROMPT.format(code=code))
            prompt_map.append((i, PROMPT_TYPE_ALGORITHMS))

            prompts.append(
                PARADIGMS_AND_PATTERNS_PROMPT.format(code=code)
            )
            prompt_map.append((i, PROMPT_TYPE_PARADIGMS_PATTERNS))

        if not prompts:
            return []

        # Single batched vLLM call for all prompts
        outputs = self.llm.generate(prompts, self.sampling_params)

        # Initialize per-sample accumulators
        parsed = {i: {} for i in range(len(samples))}

        for (sample_idx, prompt_type), output in zip(prompt_map, outputs):
            text = output.outputs[0].text
            data = self._parse_json(text)

            if prompt_type == PROMPT_TYPE_DOMAINS:
                domains = data.get("domains", [])
                if not isinstance(domains, list):
                    domains = ["unknown"]
                parsed[sample_idx]["domains"] = domains if domains else ["unknown"]

            elif prompt_type == PROMPT_TYPE_ALGORITHMS:
                parsed[sample_idx]["algorithms"] = data.get("algorithms", [])

            elif prompt_type == PROMPT_TYPE_PARADIGMS_PATTERNS:
                parsed[sample_idx]["paradigms"] = data.get("paradigms", [])
                parsed[sample_idx]["design_patterns"] = data.get(
                    "design_patterns", []
                )

        # Build result dicts in the same format as main.py:237-246
        results = []
        for i, sample in enumerate(samples):
            code = sample.get("content", "")
            if not code.strip():
                continue

            sample_id = sample.get("blob_id", "unknown")
            language = sample.get("language", "unknown")
            file_node_id = f"{sample_id}_{language}"

            p = parsed.get(i, {})
            results.append(
                {
                    "sample_id": sample_id,
                    "language": language,
                    "domains": p.get("domains", ["unknown"]),
                    "functions": [],
                    "algorithms": p.get("algorithms", []),
                    "paradigms": p.get("paradigms", []),
                    "design_patterns": p.get("design_patterns", []),
                    "file_node_id": file_node_id,
                }
            )

        return results

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """
        Robust JSON parsing matching llm_extractor.py:162-186.
        First tries direct parse, then tries extracting JSON substring.
        """
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass

        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(text[start:end])
        except (json.JSONDecodeError, TypeError):
            pass

        return {}
