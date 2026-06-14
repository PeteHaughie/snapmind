# ─── SECTION: Weight Loader ABC ─────────────────────────
import abc


# ANCHOR: WeightLoaderABC
class WeightLoaderABC(abc.ABC):
    @abc.abstractmethod
    def load(self, path: str, model, config):
        ...
# ENDANCHOR: WeightLoaderABC
# ─── ENDSECTION: Weight Loader ABC ──────────────────────
