import torch

from trainer.common import calc_errors


class TestCalcErrors:
    def test_top1_error(self):
        dim_batch = 32
        dim_class = 100

        output = torch.zeros(dim_class)
        output[0] = 1.0
        output = output[None, :].repeat(dim_batch, 1)
        output[0, 1] = 2.0

        target = torch.zeros(dim_batch, dtype=torch.long)

        errors = calc_errors(output, target)
        assert errors[0] == torch.Tensor([100.0 * (1.0 / dim_batch)])

    def test_top1_and_top2_error(self):
        dim_batch = 32
        dim_class = 100

        output = torch.zeros(dim_class)
        output[0] = 1.0
        output[1] = 2.0
        output = output[None, :].repeat(dim_batch, 1)

        target = torch.zeros(dim_batch, dtype=torch.long)

        errors = calc_errors(output, target, (1, 2))
        assert errors[0] == torch.Tensor([100.0])
        assert errors[1] == torch.Tensor([0.0])
