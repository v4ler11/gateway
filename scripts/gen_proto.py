import subprocess
import sys

from pathlib import Path


def generate():
    root = Path(__file__).parent.parent
    proto_dir = root / "proto"
    out_dir = root / "src" / "generated"

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating protos from {proto_dir} into {out_dir}...")

    proto_files = list(proto_dir.glob("*.proto"))

    if not proto_files:
        print(f"No .proto files found in {proto_dir}")
        sys.exit(0)

    proto_files_str = [str(p) for p in proto_files]

    cmd = [
        sys.executable, "-m", "grpc_tools.protoc",
        f"-I{proto_dir}",
        f"--python_betterproto_out={out_dir}",
        *proto_files_str
    ]

    result = subprocess.run(cmd, cwd=root)

    if result.returncode == 0:
        print("Generate Proto: Success!")
    else:
        print("Generate Proto: Failed")
        sys.exit(result.returncode)


if __name__ == "__main__":
    generate()
