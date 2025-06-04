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