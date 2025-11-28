from typing import List, Optional

from pydantic import BaseModel


class EngineParamsBase(BaseModel):
    args: Optional[List[str]] = None

    @property
    def context_size(self) -> int:
        raise NotImplementedError("Context size must be implemented for EngineParamsBase")


class EngineParamsLlamacpp(EngineParamsBase):
    ctx_size: int

    @property
    def context_size(self) -> int:
        return self.ctx_size

    def model_dump_to_args(self) -> List[str]:
        args =  [
            "-c", str(self.context_size),
        ]
        if self.args:
            args.extend(self.args)
        return args
