from torch.autograd import Function


class ReverseLayerF(Function):
    """
    Gradient reversal layer (https://arxiv.org/abs/1505.07818)
    """

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg()

        return output, None
