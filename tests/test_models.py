"""Unit tests for MLP, SimpleCNN, ResNet-18, differential LRs, and BatchNorm freezing."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.models import (
    SimpleCNN,
    SimpleMLP,
    build_model,
    build_optimizer,
    build_resnet18,
    freeze_bn,
)


def test_mlp_output_shapes_with_and_without_dropout() -> None:
    x = torch.randn(4, 3, 64, 64)
    mlp_no_drop = SimpleMLP(num_classes=7, image_size=64, dropout=0.0)
    mlp_drop = SimpleMLP(num_classes=7, image_size=64, dropout=0.3)

    assert not any(isinstance(m, nn.Dropout) for m in mlp_no_drop.modules())
    assert any(isinstance(m, nn.Dropout) and m.p == 0.3 for m in mlp_drop.modules())

    assert mlp_no_drop(x).shape == (4, 7)
    assert mlp_drop(x).shape == (4, 7)


def test_cnn_variants_batchnorm_and_adaptive_pooling_shapes() -> None:
    x64 = torch.randn(4, 3, 64, 64)

    cnn_no_bn = SimpleCNN(num_classes=7, use_batchnorm=False, use_adaptive_pool=False, image_size=64)
    cnn_bn = SimpleCNN(num_classes=7, use_batchnorm=True, use_adaptive_pool=False, image_size=64)
    cnn_gap = SimpleCNN(num_classes=7, use_batchnorm=True, use_adaptive_pool=True, first_kernel_size=5)

    assert not any(isinstance(m, nn.BatchNorm2d) for m in cnn_no_bn.modules())
    assert sum(isinstance(m, nn.BatchNorm2d) for m in cnn_bn.modules()) == 4
    assert any(isinstance(m, nn.AdaptiveAvgPool2d) for m in cnn_gap.modules())

    assert cnn_no_bn(x64).shape == (4, 7)
    assert cnn_bn(x64).shape == (4, 7)
    assert cnn_gap(torch.randn(2, 3, 100, 130)).shape == (2, 7)
    assert cnn_gap(torch.randn(2, 3, 48, 48)).shape == (2, 7)

    with pytest.raises(ValueError):
        SimpleCNN(channels=(16, 32, 64, 128, 256))


def test_freeze_bn_excludes_bn_from_optimizer_and_preserves_stats() -> None:
    model = build_resnet18(num_classes=7, pretrained=False, freeze_bn_stats=True)

    # All BN layers must be in eval mode and have requires_grad=False
    bn_modules = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    assert len(bn_modules) > 0
    for m in bn_modules:
        assert not m.training
        for p in m.parameters():
            assert not p.requires_grad

    # Differential optimizer must cleanly exclude frozen BN parameters
    optimizer = build_optimizer(model, backbone_lr=1e-4, head_lr=1e-3, weight_decay=1e-4)
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(1e-3)

    opt_param_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    for m in bn_modules:
        for p in m.parameters():
            assert id(p) not in opt_param_ids

    # Snapshot BN buffers & affine weights before a training step
    bn0 = bn_modules[0]
    before_mean = bn0.running_mean.clone()
    before_var = bn0.running_var.clone()
    before_weight = bn0.weight.clone()
    before_bias = bn0.bias.clone()
    fc_before = model.fc.weight.clone()

    # Simulate epoch train() + freeze_bn(model) guard
    model.train()
    freeze_bn(model)
    assert not bn0.training

    x = torch.randn(4, 3, 224, 224)
    y = torch.tensor([0, 2, 4, 6], dtype=torch.long)
    optimizer.zero_grad()
    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()
    optimizer.step()

    assert torch.equal(bn0.running_mean, before_mean)
    assert torch.equal(bn0.running_var, before_var)
    assert torch.equal(bn0.weight, before_weight)
    assert torch.equal(bn0.bias, before_bias)
    assert not torch.equal(model.fc.weight, fc_before)


def test_build_model_factory_validation() -> None:
    assert isinstance(build_model("mlp", num_classes=7), SimpleMLP)
    assert isinstance(build_model("cnn", num_classes=7), SimpleCNN)
    with pytest.raises(ValueError):
        build_model("unknown_arch", num_classes=7)

