import pytest
import torch
from jaxtyping import Float

from tensor_collection import TensorCollection, batch_tensor_collections


# -----------------------------------------------------------------------------
# Helper classes
# -----------------------------------------------------------------------------

class MLPParams(TensorCollection):
    """Simple two‑layer MLP parameter container.

    Shape annotation uses *batch so the same class works before and after batching.
    """

    weight1: Float[torch.Tensor, "*batch hidden input"]
    bias1: Float[torch.Tensor, "*batch hidden"]
    weight2: Float[torch.Tensor, "*batch output hidden"]
    bias2: Float[torch.Tensor, "*batch output"]


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture(scope="function")
def example_instances():
    """Return a pair of unbatched MLPParams for testing."""
    hidden, input_dim, output = 8, 4, 2

    a = MLPParams(
        weight1=torch.randn(hidden, input_dim, requires_grad=True),
        bias1=torch.zeros(hidden),
        weight2=torch.randn(output, hidden, requires_grad=True),
        bias2=torch.zeros(output),
    )
    b = MLPParams(
        weight1=torch.randn(hidden, input_dim, requires_grad=True),
        bias1=torch.zeros(hidden),
        weight2=torch.randn(output, hidden, requires_grad=True),
        bias2=torch.zeros(output),
    )
    return a, b


# -----------------------------------------------------------------------------
# Registration tests
# -----------------------------------------------------------------------------

def test_parameter_and_buffer_registration(example_instances):
    a, _ = example_instances

    # Grad‑enabled tensors become Parameters
    assert isinstance(a.weight1, torch.nn.Parameter)
    assert isinstance(a.weight2, torch.nn.Parameter)

    # Non‑grad tensors become buffers (persistent=False)
    assert isinstance(a.bias1, torch.Tensor) and not a.bias1.requires_grad
    assert isinstance(a.bias2, torch.Tensor) and not a.bias2.requires_grad


# -----------------------------------------------------------------------------
# Batching tests
# -----------------------------------------------------------------------------

def test_batch_shapes(example_instances):
    a, b = example_instances
    batched = batch_tensor_collections([a, b])

    assert batched.weight1.shape == (2, 8, 4)
    assert batched.bias2.shape == (2, 2)


def test_batch_shape_mismatch_raises(example_instances):
    a, b = example_instances
    # introduce shape mismatch
    b.weight1 = torch.randn(7, 4, requires_grad=True)
    with pytest.raises(ValueError):
        _ = batch_tensor_collections([a, b])


def test_batch_dtype_mismatch_raises(example_instances):
    a, b = example_instances

    # Create a third instance with a different dtype (float64)
    c = MLPParams(
        weight1=torch.randn(8, 4, dtype=torch.float64, requires_grad=True),
        bias1=torch.zeros(8, dtype=torch.float64),
        weight2=torch.randn(2, 8, dtype=torch.float64, requires_grad=True),
        bias2=torch.zeros(2, dtype=torch.float64),
    )

    with pytest.raises(ValueError):
        _ = batch_tensor_collections([a, c])
        _ = batch_tensor_collections([a, b])


# -----------------------------------------------------------------------------
# Autograd tests
# -----------------------------------------------------------------------------

def _forward(params: MLPParams, x: torch.Tensor) -> torch.Tensor:
    """Forward that works for both unbatched and batched `params`.

    * If `params.weight1` is 2‑D ``[hidden, input]`` (unbatched), use plain
      matmul.
    * If it is 3‑D ``[B, hidden, input]`` (batched), use batched matmul (bmm).
    """

    if params.weight1.dim() == 2:
        # Unbatched path ------------------------------------------------------
        # x: (input,) or (1, input)
        if x.dim() == 1:
            x_ = x
        else:
            x_ = x.squeeze(0)
        h = params.weight1 @ x_ + params.bias1          # (hidden)
        h = torch.relu(h)
        out = params.weight2 @ h + params.bias2           # (output)
        return out.sum()

    # Batched path -----------------------------------------------------------
    # params.weight1: (B, hidden, input)
    # x            : (B, input)
    x_batched = x.unsqueeze(-1)                           # (B, input, 1)
    h = torch.bmm(params.weight1, x_batched).squeeze(-1)  # (B, hidden)
    h = h + params.bias1                                  # (B, hidden)
    h = torch.relu(h)
    out = (
        torch.bmm(params.weight2, h.unsqueeze(-1)).squeeze(-1) + params.bias2
    )                                                     # (B, output)
    return out.sum()


def test_gradient_flow_to_original_leaves(example_instances):
    a, b = example_instances
    batched = batch_tensor_collections([a, b])

    x = torch.randn(2, 4)
    loss = _forward(batched, x)
    loss.backward()

    # Gradients should flow back to original leaf parameters
    assert a.weight1.grad is not None and torch.any(a.weight1.grad != 0)
    assert b.weight1.grad is not None and torch.any(b.weight1.grad != 0)

    # Biases are buffers, grads should remain None
    assert a.bias1.grad is None and b.bias1.grad is None


def test_gradients_match_manual_split(example_instances):
    """Compare batched gradients against manual unbatched computation."""
    a, b = example_instances
    batched = batch_tensor_collections([a, b])

    # Forward with batched params
    x = torch.randn(2, 4)
    loss_batch = _forward(batched, x)
    loss_batch.backward()

    grad_a_batched = a.weight1.grad.clone()
    grad_b_batched = b.weight1.grad.clone()

    # Manually compute losses independently (reset grads first)
    for p in [*a.parameters(), *b.parameters()]:
        p.grad = None

    loss_a = _forward(a, x[0:1])
    loss_b = _forward(b, x[1:2])
    (loss_a + loss_b).backward()

    # Gradients should be identical
    assert torch.allclose(a.weight1.grad, grad_a_batched, atol=1e-6)
    assert torch.allclose(b.weight1.grad, grad_b_batched, atol=1e-6)


# -----------------------------------------------------------------------------
# Device tests (CPU vs CUDA) – optional
# -----------------------------------------------------------------------------

def test_batching_on_cuda(example_instances):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    a, b = example_instances
    a = a.to("cuda")
    b = b.to("cuda")

    batched = batch_tensor_collections([a, b])
    assert batched.weight1.is_cuda

    x = torch.randn(2, 4, device="cuda")
    loss = _forward(batched, x)
    loss.backward()
    assert a.weight1.grad is not None
