import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Dict

from huggingface_hub import hf_hub_download
from core.logger import warn, info


def download_repo_files(
        repo_id: str,
        dest_dir: Path,
        files: List[str]
) -> Tuple[Dict[str, Path], Optional[str]]:
    if len(files) != len(set(files)):
        raise ValueError(f"Files list contains duplicates: {files}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    final_paths: Dict[str, Path] = {}

    try:
        files_to_download = []
        for filename in files:
            target_path = dest_dir / filename
            if target_path.is_file():
                final_paths[filename] = target_path
            else:
                files_to_download.append(filename)

        if not files_to_download:
            return final_paths, None

        with tempfile.TemporaryDirectory() as temp_dir:
            for filename in files_to_download:
                info(f"Downloading {filename} from {repo_id}")
                temp_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=temp_dir
                )

                target_path = dest_dir / filename

                if target_path.is_file():
                    warn(f"File {filename} from {repo_id} was downloaded, but it already exists in {dest_dir}")
                else:
                    shutil.move(temp_path, target_path)

                final_paths[filename] = target_path

            return final_paths, None

    except Exception as e:
        return {}, str(e)
