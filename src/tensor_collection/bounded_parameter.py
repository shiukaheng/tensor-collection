import math
import torch

from pytorch_dataclasses.TensorCollection import Transform

def bounded_parameter(min: float = -math.inf, max: float = math.inf, epsilon=0):
    if min == max:
        raise ValueError("Domain must have a non-zero range")
    if min > max:
        raise ValueError("Domain must be increasing")
    
    if min == -math.inf and max == math.inf:
        return Transform(lambda x: x, lambda z: z)

    if min == -math.inf:
        return Transform(
            lambda x: torch.log(max - torch.clamp(x, max=max - epsilon)),
            lambda z: max - torch.exp(z),
        )

    if max == math.inf:
        return Transform(
            lambda x: torch.log(torch.clamp(x, min=min + epsilon) - min),
            lambda z: torch.exp(z) + min,
        )

    # Finite domain
    return Transform(
        lambda x: torch.log((torch.clamp(x, min=min + epsilon, max=max - epsilon) - min) / (max - torch.clamp(x, min=min + epsilon, max=max - epsilon))),
        lambda z: (torch.sigmoid(z) * (max - min)) + min,
    )