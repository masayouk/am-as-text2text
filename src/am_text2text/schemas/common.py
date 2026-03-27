from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


SplitName = Literal["train", "dev", "test"]


class GpuConfig(BaseModel):
    num_gpus: int = 1
    visible_devices: Optional[list[int]] = None

    @model_validator(mode="after")
    def validate_spec(self) -> "GpuConfig":
        if self.num_gpus <= 0:
            raise ValueError("gpu.num_gpus must be positive")
        if self.visible_devices and len(self.visible_devices) < self.num_gpus:
            raise ValueError("gpu.visible_devices is shorter than gpu.num_gpus")
        return self

    def visible_devices_env(self) -> Optional[str]:
        if not self.visible_devices:
            return None
        return ",".join(str(device) for device in self.visible_devices)


class DeepSpeedConfig(BaseModel):
    enabled: bool = False
    config_path: Optional[Path] = None

    @model_validator(mode="after")
    def validate_spec(self) -> "DeepSpeedConfig":
        if self.enabled and self.config_path is None:
            raise ValueError("deepspeed.config_path is required when deepspeed.enabled is true")
        return self


class RunConfig(BaseModel):
    name: str
    seed: int = 42


class OutputConfig(BaseModel):
    root_dir: Path = Field(default_factory=lambda: Path("outputs"))
