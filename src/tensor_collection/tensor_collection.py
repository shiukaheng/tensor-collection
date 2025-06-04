from __future__ import annotations

"""
TensorCollection: a type-safe, declarative, grouped tensor structure
that behaves like a PyTorch nn.Module and supports batched stacking.

Features:
- Automatically wraps tensors with gradients as Parameters.
- Registers buffers for tensors without gradients.
- Type-checked with jaxtyping and beartype.
- Supports batching via `batch_tensor_collections`.

Author: You
Date: 2025-06-04
"""

from dataclasses import dataclass, field, fields as dataclass_fields
from typing import Any, Dict, List, TypeVar, Type

import torch
import torch.nn as nn
from beartype import beartype
from jaxtyping import jaxtyped
from typing_extensions import dataclass_transform

T = TypeVar("T", bound="TensorCollection")


@dataclass_transform(eq_default=False, kw_only_default=True)
class TensorCollection(nn.Module):
    """
    A lightweight, declarative container for grouping tensors/parameters
    while retaining full `nn.Module` semantics.
    """

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        nn.Module.__init__(obj)
        return obj

    def __init_subclass__(cls) -> None:
        dataclass(eq=False, kw_only=True)(cls)
        jaxtyped(typechecker=beartype)(cls)
        super().__init_subclass__()

    def __setattr__(self, name: str, value: Any):
        if isinstance(value, torch.Tensor) and not isinstance(value, nn.Parameter):
            if value.requires_grad and value.grad_fn is None:
                super().__setattr__(name, nn.Parameter(value))
            else:
                self.register_buffer(name, value, persistent=False)
            return
        super().__setattr__(name, value)

    def __str__(self):
        return nn.Module.__repr__(self)


# ----------------------------------------------------------------------------
# Batching utility
# ----------------------------------------------------------------------------

def batch_tensor_collections(instances: List[T]) -> T:
    if len(instances) == 0:
        raise ValueError("Cannot batch an empty list of TensorCollections.")

    cls: Type[T] = type(instances[0])
    if not all(isinstance(x, cls) for x in instances):
        raise TypeError("All instances must be of the same TensorCollection subclass.")

    field_names = [f.name for f in dataclass_fields(cls) if f.init]
    batched_data = {}

    for name in field_names:
        tensors = [getattr(inst, name) for inst in instances]

        if not all(isinstance(t, torch.Tensor) for t in tensors):
            raise TypeError(f"Field '{name}' contains non-tensor types.")

        shape = tensors[0].shape
        dtype = tensors[0].dtype
        device = tensors[0].device

        for t in tensors:
            if t.shape != shape:
                raise ValueError(f"Shape mismatch in field '{name}': {t.shape} != {shape}")
            if t.dtype != dtype:
                raise ValueError(f"Dtype mismatch in field '{name}': {t.dtype} != {dtype}")
            if t.device != device:
                raise ValueError(f"Device mismatch in field '{name}': {t.device} != {device}")

        batched_data[name] = torch.stack(tensors, dim=0)

    # Avoid triggering __setattr__ logic again
    batched = cls.__new__(cls)
    nn.Module.__init__(batched)
    for k, v in batched_data.items():
        object.__setattr__(batched, k, v)

    return batched