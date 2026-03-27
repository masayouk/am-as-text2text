import os
import shutil
import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel

from am_text2text.schemas.common import GpuConfig


def run_training_worker(module: str, request: BaseModel, request_path: Path, *, gpu: GpuConfig) -> None:
    request_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_text(request.model_dump_json(indent=2), encoding="utf-8")

    env = os.environ.copy()
    visible_devices = gpu.visible_devices_env()
    if visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = visible_devices

    if gpu.num_gpus == 1:
        cmd = [sys.executable, "-m", module, "--request", str(request_path)]
    else:
        deepspeed_command = shutil.which("deepspeed")
        if deepspeed_command is None:
            raise RuntimeError("deepspeed command is not available in PATH")
        cmd = [
            deepspeed_command,
            f"--num_gpus={gpu.num_gpus}",
            "--module",
            module,
            "--request",
            str(request_path),
        ]
    subprocess.run(cmd, check=True, env=env)
