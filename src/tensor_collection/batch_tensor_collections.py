# tensor_batching.py  ── place in the same package as tensor_collection.py
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Dict, List, Type, TypeVar, overload

import torch
import torch.nn as nn

from tensor_collection import TensorCollection

__all__ = [
    "TENSOR_STACK_AGGREGATOR",
    "LIST_AGGREGATOR",
    "DEFAULT_TYPE_AGGREGATORS",
    "batch_tensor_collections",
]

T = TypeVar("T", bound=TensorCollection)

# --------------------------------------------------------------------------- #
# Built-in aggregators
# --------------------------------------------------------------------------- #

def stack_tensors(values: List[torch.Tensor]) -> torch.Tensor:
    """
    Stack a list of tensors (or Parameters) along a new leading dimension.

    Raises
    ------
    ValueError
        * shape mismatch
        * dtype mismatch
        * device mismatch
    """
    ref = values[0]
    if any(v.shape != ref.shape for v in values):
        raise ValueError("Shape mismatch while stacking tensors.")
    if any(v.dtype != ref.dtype for v in values):
        raise ValueError("Dtype mismatch while stacking tensors.")
    if any(v.device != ref.device for v in values):
        raise ValueError("Device mismatch while stacking tensors.")

    # `torch.stack` keeps the computational graph: gradients flow back to
    # all original leaves, which is exactly what the tests expect.
    return torch.stack(values, dim=0)


def stack_arbitrary(values: List[Any]) -> List[Any]:
    """Fallback aggregator: simply collect items into a list."""
    return list(values)

def undefined_aggregator(values: List[Any]) -> Any:
    """
    Placeholder aggregator that raises an error if called.
    Useful for debugging or as a default in type_aggregators.
    """
    raise NotImplementedError("Unhandled property encountered during batching, "
                              "please define an appropriate aggregator for this type or field.")

DEFAULT_TYPE_AGGREGATORS: Dict[Type[Any], Callable[[List[Any]], Any]] = {
    torch.Tensor: stack_tensors,  # also matches nn.Parameter
}

# --------------------------------------------------------------------------- #
# Core batching routine
# --------------------------------------------------------------------------- #

def batch_tensor_collections(
    collections: List[T],
    *,
    type_aggregators: Dict[Type[Any], Callable[[List[Any]], Any]] = DEFAULT_TYPE_AGGREGATORS,
    default_type_aggregator: Callable[[List[Any]], Any] = undefined_aggregator,
    field_aggregators: Dict[str, Callable[[List[Any]], Any]] = {},
) -> T:
    """
    Batch together a list of TensorCollection instances, with fine-grained control
    over how each field is aggregated via type-based or field-specific functions.

    Precedence for aggregator resolution:
    1. field_aggregators[field_name]
    2. type_aggregators[type(value)] (via isinstance)
    3. default_type_aggregator

    Parameters
    ----------
    collections : List[T]
        A non-empty list of TensorCollection instances of the same subclass.

    type_aggregators : dict
        Mapping from type -> function(List[Any]) -> Any, used to batch fields
        by type.

    default_type_aggregator : Callable
        Fallback function used when a field's type is not matched in the above.

    field_aggregators : dict
        Mapping from field name -> function(List[Any]) -> Any, which overrides
        all type-based aggregators for that field.

    Returns
    -------
    T : TensorCollection
        A single batched instance of the same subclass.
    """
    if not collections:
        raise ValueError("Expected a non-empty list of TensorCollections.")

    cls = type(collections[0])
    if any(type(c) is not cls for c in collections):
        raise TypeError("All items must be instances of the same TensorCollection subclass.")

    # Compose type aggregator map
    agg_map: Dict[Type[Any], Callable[[List[Any]], Any]] = {
        **DEFAULT_TYPE_AGGREGATORS,
        **type_aggregators,
    }

    # Resolve aggregator per field
    def _get_aggregator(name: str, v: Any) -> Callable[[List[Any]], Any]:
        if name in field_aggregators:
            return field_aggregators[name]
        for typ, fn in agg_map.items():
            if isinstance(v, typ):
                return fn
        if isinstance(v, TensorCollection):
            return lambda xs: batch_tensor_collections(
                xs,
                type_aggregators=agg_map,
                default_type_aggregator=default_type_aggregator,
                field_aggregators=field_aggregators,
            )
        return default_type_aggregator

    # Aggregate all fields
    aggregated_fields = {}
    for field in cls.__dataclass_fields__.values():  # type: ignore[attr-defined]
        name = field.name
        values = [getattr(c, name) for c in collections]
        aggregator = _get_aggregator(name, values[0])
        aggregated_fields[name] = aggregator(values)

    return cls(**aggregated_fields)