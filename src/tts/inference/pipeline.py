import torch

from core.globals import MODELS_DIR
from hf.download import download_repo_paths
from kokoro import KPipeline
from models.definitions import ModelTTSAny


def init_pipeline(model: ModelTTSAny) -> KPipeline:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    d_paths, err = download_repo_paths(
        model.record.model,
        MODELS_DIR / model.record.model.replace("/", "_"),
        model.record.files,
    )
    if err is not None:
        raise RuntimeError(f"Failed to download model paths: {err}")

    pipeline = KPipeline(lang_code='a', device=device)

    for voice in model.record.voices:
        voice_pt = d_paths["voices"] / f"{voice}.pt"
        if not voice_pt.exists():
            raise RuntimeError(f"Failed to find voice file: {voice_pt}")
        voice_tensor = torch.load(str(voice_pt), weights_only=True).to(device)
        pipeline.voices[voice] = voice_tensor

    return pipeline
