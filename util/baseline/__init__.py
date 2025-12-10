import importlib
import pkgutil

# global registry of filter functions
BASELINE_REGISTRY = {}

def register_baseline(func):
    """Decorator to register filter functions by name."""
    BASELINE_REGISTRY[func.__name__] = func
    return func


# automatically import all submodules under this package
for module_info in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{module_info.name}")
