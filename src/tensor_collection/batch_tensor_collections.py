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

def _stack_tensors(values: List[torch.Tensor]) -> torch.Tensor:
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


def _list_values(values: List[Any]) -> List[Any]:
    """Fallback aggregator: simply collect items into a list."""
    return list(values)


# Public aliases ------------------------------------------------------------- #
TENSOR_STACK_AGGREGATOR: Callable[[List[torch.Tensor]], torch.Tensor] = _stack_tensors
LIST_AGGREGATOR: Callable[[List[Any]], List[Any]] = _list_values

DEFAULT_TYPE_AGGREGATORS: Dict[Type[Any], Callable[[List[Any]], Any]] = {
    torch.Tensor: TENSOR_STACK_AGGREGATOR,  # also matches nn.Parameter
}

# --------------------------------------------------------------------------- #
# Core batching routine
# --------------------------------------------------------------------------- #

def batch_tensor_collections(
    collections: List[T],
    *,
    type_aggregators: Dict[Type[Any], Callable[[List[Any]], Any]] = DEFAULT_TYPE_AGGREGATORS,
    default_type_aggregator: Callable[[List[Any]], Any] = LIST_AGGREGATOR,
) -> T:
    """
    Batch together a list of ``TensorCollection`` instances.

    Parameters
    ----------
    collections
        Instances of the **same** ``TensorCollection`` subclass to be batched.
    type_aggregators
        Mapping from *type* → *callable* that knows how to reduce a list of that
        type into a single batched object.  If omitted, uses
        ``DEFAULT_TYPE_AGGREGATORS``.
    default_type_aggregator
        Fallback reducer for unseen types.  If ``None`` (default) and a field’s
        value type is not found in *either* mapping, a ``TypeError`` is raised.

    Returns
    -------
    T
        A *new* instance of the same subclass with each field aggregated across
        the leading batch dimension.

    Notes
    -----
    * Leaf parameters remain intact, so gradients flow back to the originals.
    * Nested ``TensorCollection`` fields are batched **recursively**.
    """
    if not collections:
        raise ValueError("Expected a non-empty list of TensorCollections.")

    cls = type(collections[0])
    if any(type(c) is not cls for c in collections):
        raise TypeError("All items must be instances of the same TensorCollection subclass.")

    # Merge user-defined mapping over the defaults, keeping user overrides.
    agg_map: Dict[Type[Any], Callable[[List[Any]], Any]] = {
        **DEFAULT_TYPE_AGGREGATORS,
        **type_aggregators,  # no need for `or {}` since default is non-None
    }
    # Local helper ----------------------------------------------------------- #
    def _get_aggregator(v: Any) -> Callable[[List[Any]], Any]:
        for typ, fn in agg_map.items():
            if isinstance(v, typ):
                return fn
        if isinstance(v, TensorCollection):  # recurse by default
            return lambda xs: batch_tensor_collections(
                xs,
                type_aggregators=agg_map,
                default_type_aggregator=default_type_aggregator,
            )
        if default_type_aggregator is not None:
            return default_type_aggregator
        raise TypeError(
            f"No aggregator registered for type {type(v).__name__!r} "
            "and no default_type_aggregator provided."
        )

    # Aggregate each dataclass field ---------------------------------------- #
    aggregated_fields = {}
    for field in cls.__dataclass_fields__.values():  # type: ignore[attr-defined]
        name = field.name
        values = [getattr(c, name) for c in collections]
        aggregator = _get_aggregator(values[0])
        aggregated_fields[name] = aggregator(values)

    # Construct the batched TensorCollection -------------------------------- #
    return cls(**aggregated_fields)
