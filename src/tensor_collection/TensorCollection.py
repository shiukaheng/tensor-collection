from dataclasses import dataclass, field
from typing import Any, Callable, Dict, NamedTuple
import torch
import torch.nn as nn
from typing_extensions import dataclass_transform
from beartype import beartype
from jaxtyping import jaxtyped

class Transform(NamedTuple):
    set: Callable[[Any], Any] | None = None
    get: Callable[[Any], Any] | None = None

@dataclass_transform(eq_default=False, kw_only_default=True) # Use dataclass_transform to provide dataclass like behaviour
class TensorCollection(nn.Module): # Inherit from nn.Module to have all the nn.Module functionality
    property_transforms: Dict[str, Transform] = field(default_factory=dict, init=False) # Special to modify getting and setting behaviour of properties

    # We initialize the class with __new__ to ensure nn.Module is initialized properly
    def __new__(cls, *_, **__):
        obj = super().__new__(cls)
        nn.Module.__init__(obj)
        return obj

    # We use __init_subclass__ to apply the dataclass decorator and jaxtyped with beartype to the class
    def __init_subclass__(cls) -> None:
        dataclass(eq=False, kw_only=True)(cls)
        jaxtyped(typechecker=beartype)(cls)
        return super().__init_subclass__()

    def __setattr__(self, name, value):
        original_value = value # Lets keep the original value
        if name in self.property_transforms: # If the name is in the property transforms
            transform = self.property_transforms[name] # We transform it
            if transform.set: # If we have an encode function, we use it
                value = transform.set(value)
                # If original value is a instance of Parameter, and the output is a tensor with grad, we need to convert it back to a Parameter
                if isinstance(original_value, nn.Parameter) and isinstance(value, torch.Tensor) and value.requires_grad:
                    value = nn.Parameter(value)
        if isinstance(value, torch.Tensor) and not isinstance(value, nn.Parameter): # If it's a tensor but not a parameter, register as buffer
            self._buffers[name] = value
            return
        super().__setattr__(name, value)

    def __getattr__(self, name):
        val = super().__getattr__(name)
        # First check if the name is in the property transforms
        if name in self.property_transforms:
            # Get the value from the super class
            # If we have a decode function, we use it
            if self.property_transforms[name].get:
                val = self.property_transforms[name].get(val)
        return val

    def __post_init__(self):
        # Re-trigger __setattr__ on all public attributes
        for name in self.__dict__:
            if not name.startswith('_'):
                setattr(self, name, getattr(self, name))

        # Merge description dicts from all base classes
        merged_property_transforms = {}
        for base in self.__class__.__mro__:
            if hasattr(base, 'property_transforms') and isinstance(getattr(base, 'property_transforms'), dict):
                merged_property_transforms.update(getattr(base, 'property_transforms'))
        self.property_transforms = merged_property_transforms

    def __str__(self):
        return nn.Module.__repr__(self)