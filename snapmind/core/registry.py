# ─── SECTION: Registry ──────────────────────────────────
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


# ANCHOR: RegistryError
class RegistryError(Exception):
    """Raised on registry conflicts (duplicate key) or lookups (unknown key)."""


# ENDANCHOR: RegistryError


# ANCHOR: Registry
class Registry:
    """Generic plugin registry with decorator-based registration and string-key dispatch.

    Usage:
        @REGISTRY.register("my_key")
        class MyPlugin(PluginABC):
            ...

        instance = REGISTRY.create("my_key", **kwargs)
    """

    def __init__(self, name: str, expected_type: type):
        self._name = name
        self._expected_type = expected_type
        self._registry: dict[str, Any] = {}

    @property
    def name(self) -> str:
        """Human-readable name for error messages (e.g. ``"sampler"``)."""
        return self._name

    def register(self, key: str, cls: Any = None, *, override: bool = False) -> Callable:
        """Register a class (or instance) under *key*, optionally as a decorator.

        Args:
            key: String key for dispatch.
            cls: Class or instance to register. If ``None``, return a decorator.
            override: If ``True``, silently replace an existing registration.

        Raises:
            TypeError: If *cls* is a class but does not subclass *expected_type*.
            RegistryError: If *key* already registered and *override* is ``False``.
        """
        def _register(cls: Any) -> Any:
            if isinstance(cls, type) and not issubclass(cls, self._expected_type):
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
        """Instantiate the class registered under *key*, passing ``**kwargs``.

        Raises:
            RegistryError: If *key* is not registered.
        """
        if key not in self._registry:
            raise RegistryError(f"unknown key '{key}' in '{self._name}' registry")
        return self._registry[key](**kwargs)

    def get(self, key: str) -> Any:
        """Return the stored object for *key* without instantiating it.

        Use this for registries that store non-class objects (e.g. dataclass instances).
        """
        if key not in self._registry:
            raise RegistryError(f"unknown key '{key}' in '{self._name}' registry")
        return self._registry[key]

    def list(self) -> list[str]:
        """Return a copy of all registered keys."""
        return list(self._registry.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._registry


# ANCHOR: GlobalRegistrySingletons
ATTENTION: Registry = Registry("attention", expected_type=object)
"""Registry for :class:`~snapmind.layers.attention.base.AttentionABC` subclasses."""
NORM: Registry = Registry("norm", expected_type=object)
"""Registry for :class:`~snapmind.layers.normalization.base.NormABC` subclasses."""
PE: Registry = Registry("pe", expected_type=object)
"""Registry for :class:`~snapmind.layers.positional.base.PositionalEncodingABC` subclasses."""
ACTIVATION: Registry = Registry("activation", expected_type=object)
"""Registry for :class:`~snapmind.layers.activation.base.ActivationABC` subclasses."""
KV_CACHE: Registry = Registry("kv_cache", expected_type=object)
"""Registry for :class:`~snapmind.kv_cache.base.KVCacheABC` subclasses."""
TOKENIZER: Registry = Registry("tokenizer", expected_type=object)
"""Registry for :class:`~snapmind.tokenizer.base.TokenizerABC` subclasses."""
SAMPLER: Registry = Registry("sampler", expected_type=object)
"""Registry for :class:`~snapmind.sampling.base.SamplerABC` subclasses."""
LOADER: Registry = Registry("loader", expected_type=object)
"""Registry for :class:`~snapmind.loaders.base.WeightLoaderABC` subclasses."""
MODEL: Registry = Registry("model", expected_type=object)
"""Registry for :class:`~snapmind.models.base.BaseModelABC` subclasses."""
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
