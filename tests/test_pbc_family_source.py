import torch

from src.models.ldm_family_medoid_module import _family_lookup
from src.models.vae_pbc_module import wrapped_fractional_delta


def test_wrapped_fractional_delta_uses_minimum_image():
    predicted = torch.tensor([[0.99, 0.02, 0.50]])
    target = torch.tensor([[0.01, 0.98, 0.50]])
    delta = wrapped_fractional_delta(predicted, target)
    expected = torch.tensor([[-0.02, 0.04, 0.00]])
    torch.testing.assert_close(delta, expected, atol=1.0e-6, rtol=0.0)


def test_periodic_table_family_map_is_complete():
    lookup = _family_lookup()
    assert torch.all(lookup[1:119] >= 0)
    assert set(lookup[1:119].tolist()) == set(range(10))
