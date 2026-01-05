import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Dict

from huggingface_hub import snapshot_download
from core.logger import warn, info


def download_repo_paths(
        repo_id: str,
        dest_dir: Path,
        paths: List[str]
) -> Tuple[Dict[str, Path], Optional[str]]:
    if len(paths) != len(set(paths)):
        raise ValueError(f"Paths list contains duplicates: {paths}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    final_paths: Dict[str, Path] = {}

    try:
        paths_to_download = []
        for path in paths:
            target_path = dest_dir / path
            if target_path.exists():
                final_paths[path] = target_path
            else:
                paths_to_download.append(path)

        if not paths_to_download:
            return final_paths, None

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)

            for target in paths_to_download:
                info(f"Downloading {target} from {repo_id}")

                snapshot_download(
                    repo_id=repo_id,
                    local_dir=temp_dir_path,
                    allow_patterns=[f"{target}/**", target],
                    local_dir_use_symlinks=False
                )

                source_path = temp_dir_path / target
                dest_path = dest_dir / target

                if not source_path.exists():
                    return {}, f"Could not find {target} in repository {repo_id}"

                if dest_path.exists():
                    warn(f"Path {target} from {repo_id} was downloaded, but it already exists in {dest_dir}")
                else:
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(source_path), str(dest_path))

                final_paths[target] = dest_path

            return final_paths, None

    except Exception as e:
        return {}, str(e)
