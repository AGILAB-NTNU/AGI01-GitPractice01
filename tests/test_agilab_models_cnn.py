"""
Module: test_agilab_models_cnn
Stage: Library
Author: AngusHsu
Date: 2026-07-16
Description: 編寫agilab.models的CNN測試腳本
"""

import torch
from agilab.models import CNN


def test_cnn_initialization():
    model = CNN()
    assert isinstance(model, CNN), "CNN model initialization failed."


def test_cnn_forward_pass():
    model = CNN()
    input_tensor = torch.randn(1, 1, 28, 28)  # 假設輸入為28x28的單通道圖像
    output = model(input_tensor)
    assert output.shape == (
        1,
        10,
    ), f"CNN forward pass output shape mismatch. Expected (1, 10), got {output.shape}."


def test_prepare_images():
    model = CNN()
    # 產生(5,28,28)的uint8張量，模擬5張28x28的圖像
    input_tensor = torch.randint(0, 256, (5, 28, 28), dtype=torch.uint8)
    output_tensor = model.prepare_images(input_tensor)
    assert (
        output_tensor.type() == "torch.FloatTensor"
    ), f"prepare_images output type mismatch. Expected torch.floatTensor, \
        got {output_tensor.type()}."


def test_range_of_cnn_output():
    model = CNN()
    input_tensor = torch.randn(1, 1, 28, 28)
    output = model(input_tensor)
    assert torch.all(
        output <= 0
    ), rf"CNN output values should be less than or equal to 0 \
        due to log_softmax. Got {output.max()}"


def test_softmax_output_sum():
    model = CNN()
    input_tensor = torch.randn(1, 1, 28, 28)
    output = model(input_tensor)
    softmax_output = torch.exp(output)  # 將log_softmax輸出轉換回softmax
    sum_output = softmax_output.sum(dim=1)
    assert torch.allclose(
        sum_output, torch.tensor([1.0])
    ), f"Softmax output should sum to 1. Got {sum_output.item()}"
