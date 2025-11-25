from typing import List, Any, Dict

from pydantic import BaseModel

from models.utils import snake_to_kebab


class EngineParamsBase(BaseModel):
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        data = super().model_dump(**kwargs)
        converted = {snake_to_kebab(key): value for key, value in data.items()}

        if kwargs.get('exclude_none'):
            return {k: v for k, v in converted.items() if v is not None}
        return converted

    def model_dump_to_args(self) -> List[str]:
        params = self.model_dump(exclude_none=True)
        args = []
        for k, v in params.items():
            if k in ['enable-prefix-caching', 'enable-chunked-prefill']:
                if str(v).lower() == 'true':
                    args.append(f"--{k}")
            else:
                args.extend([f"--{k}", str(v)])
        return args

    @property
    def context_size(self) -> int:
        raise NotImplementedError("Context size must be implemented for EngineParamsBase")


class EngineParamsLlamacpp(EngineParamsBase):
    ctx_size: int
    n_gpu_layers: int = 999 # GPU only

    @property
    def context_size(self) -> int:
        return self.ctx_size
