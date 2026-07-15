"""
Module: test_agilab_models_CNN
Stage: Library
Author: AngusHsu
Date: 2026-07-15
Description: 編寫agilab_models_CNN的測試檔
"""

import torch

from agilab.models import CNN


def test_cnn_forward():
    model = CNN()
    input_tensor = torch.randn(
        1, 1, 28, 28
    )  # 模擬一個批次的輸入，形狀為 (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    assert output_tensor.shape == (
        1,
        10,
    ), f"Output shape should be (1, 10), but got {output_tensor.shape}"


def test_softmax_output():
    model = CNN()
    input_tensor = torch.randn(1, 1, 28, 28)
    output_tensor = model(input_tensor)
    softmax_output = torch.exp(output_tensor)
    sum_probs = torch.sum(softmax_output, dim=1)
    assert torch.allclose(sum_probs, torch.tensor([1.0])), (
        f"Softmax output should sum to 1, but got {sum_probs}"
    )
