import torch

from kora import MultiplicativeKoRA, SharedBilinearKoRA


def test_zero_initial_output_and_gradient_flow():
    torch.manual_seed(0)
    adapter = MultiplicativeKoRA(4, 3, rank=2, interaction_rank=1)
    x = torch.randn(8, 4)
    y = adapter(x)
    assert torch.equal(y, torch.zeros_like(y))
    loss = (y - torch.randn_like(y)).square().mean()
    loss.backward()
    assert adapter.lora_B.grad is not None
    assert adapter.interaction_U.grad is not None


def test_scalar_product_is_representable():
    adapter = MultiplicativeKoRA(2, 1, rank=1, interaction_rank=1)
    with torch.no_grad():
        adapter.lora_A.zero_(); adapter.lora_B.zero_()
        adapter.interaction_P[:] = torch.tensor([[1.0, 0.0]])
        adapter.interaction_Q[:] = torch.tensor([[0.0, 1.0]])
        adapter.interaction_U[:] = torch.tensor([[1.0]])
    x = torch.tensor([[2.0, 3.0], [-1.0, 4.0]])
    assert torch.allclose(adapter(x).flatten(), torch.tensor([6.0, -4.0]))


def test_shared_bilinear_interacts_in_latent_space():
    adapter = SharedBilinearKoRA(3, 1, rank=2, interaction_rank=1)
    with torch.no_grad():
        adapter.lora_A[:] = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
        adapter.lora_B.zero_()
        adapter.interaction_P[:] = torch.tensor([[1., 0.]])
        adapter.interaction_Q[:] = torch.tensor([[0., 1.]])
        adapter.interaction_U[:] = torch.tensor([[1.]])
    x = torch.tensor([[2., 3., 9.], [-1., 4., 8.]])
    assert torch.allclose(adapter(x).flatten(), torch.tensor([6., -4.]))
    assert adapter.trainable_parameters == 2*3 + 1*2 + 1*2 + 1*2 + 1
