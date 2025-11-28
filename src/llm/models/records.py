from typing import List

from llm.models import ModelRecordAny
from llm.models.engine_params import EngineParamsLlamacpp
from llm.models.model_record import ModelRecordLlamaCpp, SamplingParams, ModelRecordLMStudio


__all__ = ["RECORDS"]


RECORDS: List[ModelRecordAny] = [
    # https://huggingface.co/unsloth/gpt-oss-20b-GGUF/tree/main
    ModelRecordLlamaCpp(
        model="unsloth/gpt-oss-20b-GGUF",
        resolve_name="gpt-oss-20b",
        tokenizer="openai/gpt-oss-20b",
        sampling_params=SamplingParams(
            max_tokens=8096,
            temperature=1.0,
        ),
        model_file="gpt-oss-20b-F16.gguf",
        engine_params=EngineParamsLlamacpp(
            ctx_size=64_000,
            args=[
                "--threads", "16",
                "--n-gpu-layers", "99",
                "--reasoning-format", "deepseek",
                "-fa", "on", # flash attention kernels
                "-cb", # continuous batching
                "--jinja",
            ]
        ),
    ),
    ModelRecordLMStudio(
        model="unsloth/gpt-oss-20b",
        resolve_name="gpt-oss-20b",
        tokenizer="openai/gpt-oss-20b",
        sampling_params=SamplingParams(
            max_tokens=8096,
        ),
        ctx_size=32_000,
    ),
]
