import pytest
import torch

from models.heads import CoordinateHead


def test_coordinate_head_output_shapes():
    head = CoordinateHead(in_channels=128)

    x = torch.randn(2, 128, 16, 16, 16)
    out = head(x)

    assert set(out.keys()) == {"coord_wt", "coord_et"}
    assert out["coord_wt"].shape == (2, 3)
    assert out["coord_et"].shape == (2, 3)


def test_coordinate_head_output_range_with_tanh():
    head = CoordinateHead(in_channels=128, use_tanh=True)

    x = torch.randn(2, 128, 16, 16, 16)
    out = head(x)

    assert torch.all(out["coord_wt"] >= -1.0)
    assert torch.all(out["coord_wt"] <= 1.0)
    assert torch.all(out["coord_et"] >= -1.0)
    assert torch.all(out["coord_et"] <= 1.0)


def test_coordinate_head_without_tanh_runs():
    head = CoordinateHead(in_channels=128, use_tanh=False)

    x = torch.randn(2, 128, 16, 16, 16)
    out = head(x)

    assert out["coord_wt"].shape == (2, 3)
    assert out["coord_et"].shape == (2, 3)


def test_coordinate_head_invalid_input_dim_raises():
    head = CoordinateHead(in_channels=128)

    x = torch.randn(2, 128, 16, 16)

    with pytest.raises(ValueError, match="Expected input shape"):
        head(x)


def test_coordinate_head_invalid_channel_count_raises():
    head = CoordinateHead(in_channels=128)

    x = torch.randn(2, 64, 16, 16, 16)

    with pytest.raises(ValueError, match="Expected 128 input channels"):
        head(x)


def test_coordinate_head_invalid_in_channels_raises():
    with pytest.raises(ValueError):
        CoordinateHead(in_channels=0)


def test_coordinate_head_invalid_hidden_dim_raises():
    with pytest.raises(ValueError):
        CoordinateHead(in_channels=128, hidden_dim=0)


def test_coordinate_head_backward_pass():
    head = CoordinateHead(in_channels=128)

    x = torch.randn(2, 128, 16, 16, 16, requires_grad=True)
    out = head(x)

    loss = out["coord_wt"].mean() + out["coord_et"].mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()