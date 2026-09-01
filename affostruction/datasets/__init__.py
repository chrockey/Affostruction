import importlib

__attributes = {
    "ReconstructionDataset": "reconstruction",
    "AffordanceDataset": "affordance",
}

__all__ = list(__attributes.keys())


def __getattr__(name):
    if name not in globals():
        if name not in __attributes:
            raise AttributeError(f"module {__name__} has no attribute {name}")
        module = importlib.import_module(f".{__attributes[name]}", __name__)
        globals()[name] = getattr(module, name)
    return globals()[name]
