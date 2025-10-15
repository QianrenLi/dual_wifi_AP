# net_util/__init__.py

import importlib
import pkgutil

# global registry of policies
POLICY_REGISTRY = {}
POLICY_CFG_REGISTRY = {}

BUFFER_REGISTRY = {}

def register_policy(cls):
    """Decorator to register policy classes by name."""
    POLICY_REGISTRY[cls.__name__] = cls
    return cls

def register_policy_cfg(cls):
    """Decorator to register policy classes by name."""
    POLICY_CFG_REGISTRY[cls.__name__.split('_')[0]] = cls
    return cls

def register_buffer(cls):
    BUFFER_REGISTRY[cls.__name__] = cls
    return cls


# automatically import all submodules in net_util (ppo, sac, …)
for module_info in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{module_info.name}")
