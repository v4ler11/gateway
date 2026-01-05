import re


def validate_huggingface_path(path: str) -> str:
    path_parts = path.split("/")
    if len(path_parts) != 2:
        raise ValueError(f"HF path should have exactly two elements separated by '/', got: {path}")
    return path


def snake_to_kebab(name: str) -> str:
    return re.sub(r'_+', '-', name.strip('_'))
