#!/usr/bin/env -S uv run

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "grpcio-tools",
#     "betterproto[compiler]>=2.0.0b6",
# ]
# ///

import subprocess
import sys
from pathlib import Path


def generate():
    root = Path(__file__).parent.parent
    proto_dir = root / "proto"
    out_dir = root / "src" / "generated"

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating protos from {proto_dir} into {out_dir}...")

    cmd = [
        sys.executable, "-m", "grpc_tools.protoc",
        f"-I{proto_dir}",
        f"--python_betterproto_out={out_dir}",
        f"{proto_dir}/*.proto"
    ]

    full_cmd = " ".join(cmd).replace(f"{proto_dir}/*.proto", f"{proto_dir}/*.proto")

    result = subprocess.run(full_cmd, shell=True, cwd=root)

    if result.returncode == 0:
        print("Generate Porto: Success!")
    else:
        print("Generate Porto: Failed")
        sys.exit(result.returncode)


if __name__ == "__main__":
    generate()