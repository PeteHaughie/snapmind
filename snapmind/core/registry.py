# ─── SECTION: Registry ──────────────────────────────────
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


# ANCHOR: RegistryError
class RegistryError(Exception):
    pass


# ENDANCHOR: RegistryError


# ANCHOR: Registry
class Registry:
    def __init__(self, name: str, expected_type: type):
        self._name = name
        self._expected_type = expected_type
        self._registry: dict[str, type] = {}

    @property
    def name(self) -> str:
        return self._name

    def register(self, key: str, cls: type | None = None, *, override: bool = False) -> Callable:
        def _register(cls: type) -> type:
            if not issubclass(cls, self._expected_type):
                raise TypeError(
                    f"Cannot register {cls.__name__} with key '{key}': must subclass {self._expected_type.__name__}"
                )
            if key in self._registry and not override:
                raise RegistryError(f"already registered: '{key}' in '{self._name}' registry")
            self._registry[key] = cls
            return cls

        if cls is not None:
            return _register(cls)
        return _register

    def create(self, key: str, **kwargs) -> Any:
        if key not in self._registry:
            raise RegistryError(f"unknown key '{key}' in '{self._name}' registry")
        return self._registry[key](**kwargs)

    def list(self) -> list[str]:
        return list(self._registry.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._registry


# ANCHOR: GlobalRegistrySingletons
ATTENTION = Registry("attention", expected_type=object)
NORM = Registry("norm", expected_type=object)
PE = Registry("pe", expected_type=object)
ACTIVATION = Registry("activation", expected_type=object)
KV_CACHE = Registry("kv_cache", expected_type=object)
TOKENIZER = Registry("tokenizer", expected_type=object)
SAMPLER = Registry("sampler", expected_type=object)
LOADER = Registry("loader", expected_type=object)
MODEL = Registry("model", expected_type=object)
# ENDANCHOR: GlobalRegistrySingletons

__all__ = [
    "Registry",
    "RegistryError",
    "ATTENTION",
    "NORM",
    "PE",
    "ACTIVATION",
    "KV_CACHE",
    "TOKENIZER",
    "SAMPLER",
    "LOADER",
    "MODEL",
]
# ─── ENDSECTION: Registry ───────────────────────────────
