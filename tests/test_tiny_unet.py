import pytest
import torch

from models.tiny_unet import ConvBlock3D, TinyUNet3D


def test_conv_block_3d_output_shape():
    block = ConvBlock3D(in_channels=4, out_channels=8)

    x = torch.randn(2, 4, 16, 16, 16)
    y = block(x)

    assert y.shape == (2, 8, 16, 16, 16)


def test_tiny_unet_output_shapes_with_coord_head():
    model = TinyUNet3D(
        in_channels=4,
        out_channels=4,
        base_channels=4,
        use_coord_head=True,
    )

    x = torch.randn(2, 4, 32, 32, 32)
    out = model(x)

    assert set(out.keys()) == {"seg_logits", "coord_wt", "coord_et"}
    assert out["seg_logits"].shape == (2, 4, 32, 32, 32)
    assert out["coord_wt"].shape == (2, 3)
    assert out["coord_et"].shape == (2, 3)


def test_tiny_unet_output_shapes_without_coord_head():
    model = TinyUNet3D(
        in_channels=4,
        out_channels=4,
        base_channels=4,
        use_coord_head=False,
    )

    x = torch.randn(2, 4, 32, 32, 32)
    out = model(x)

    assert out["seg_logits"].shape == (2, 4, 32, 32, 32)
    assert out["coord_wt"].shape == (2, 3)
    assert out["coord_et"].shape == (2, 3)

    assert torch.equal(out["coord_wt"], torch.zeros_like(out["coord_wt"]))
    assert torch.equal(out["coord_et"], torch.zeros_like(out["coord_et"]))


def test_tiny_unet_output_range_with_coord_head():
    model = TinyUNet3D(
        in_channels=4,
        out_channels=4,
        base_channels=4,
        use_coord_head=True,
    )

    x = torch.randn(2, 4, 32, 32, 32)
    out = model(x)

    assert torch.all(out["coord_wt"] >= -1.0)
    assert torch.all(out["coord_wt"] <= 1.0)
    assert torch.all(out["coord_et"] >= -1.0)
    assert torch.all(out["coord_et"] <= 1.0)


def test_tiny_unet_backward_pass():
    model = TinyUNet3D(
        in_channels=4,
        out_channels=4,
        base_channels=4,
        use_coord_head=True,
    )

    x = torch.randn(2, 4, 32, 32, 32, requires_grad=True)
    out = model(x)

    loss = (
        out["seg_logits"].mean()
        + out["coord_wt"].mean()
        + out["coord_et"].mean()
    )

    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_tiny_unet_odd_spatial_size_runs():
    model = TinyUNet3D(
        in_channels=4,
        out_channels=4,
        base_channels=4,
        use_coord_head=True,
    )

    x = torch.randn(1, 4, 31, 33, 35)
    out = model(x)

    assert out["seg_logits"].shape == (1, 4, 31, 33, 35)
    assert out["coord_wt"].shape == (1, 3)
    assert out["coord_et"].shape == (1, 3)


def test_tiny_unet_invalid_input_dim_raises():
    model = TinyUNet3D()

    x = torch.randn(1, 4, 32, 32)

    with pytest.raises(ValueError, match="Expected input shape"):
        model(x)


def test_tiny_unet_invalid_channel_count_raises():
    model = TinyUNet3D(in_channels=4)

    x = torch.randn(1, 3, 32, 32, 32)

    with pytest.raises(ValueError, match="Expected 4 input channels"):
        model(x)


def test_tiny_unet_invalid_constructor_args_raise():
    with pytest.raises(ValueError):
        TinyUNet3D(in_channels=0)

    with pytest.raises(ValueError):
        TinyUNet3D(out_channels=0)

    with pytest.raises(ValueError):
        TinyUNet3D(base_channels=0)