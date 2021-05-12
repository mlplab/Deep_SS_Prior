# coding: UTF-8


from layers import ReLU, Leaky, Swish, Mish
import torch
from torchsummary import summary


class TestLearnInverseMask(torch.nn.Module):

    def __init__(self, mask_ch: int, *args, feature_num: int=64, block_num: int=3, **kwargs):
        super().__init__()
        activations = {'relu': ReLU, 'leaky': Leaky, 'swish': Swish, 'mish': Mish}
        activation = kwargs.get('activation', 'relu').lower()
        self.main = torch.nn.ModuleList([torch.nn.Conv2d(mask_ch, mask_ch, 3, 1, 1) for _ in range(block_num)])
        self.activation = torch.nn.ModuleList([activations[activation]() for _ in range(block_num)])

    def forward(self, x):

        for i, (layer, activation) in enumerate(zip(self.main, self.activation)):
            x = activation(layer(x))

        return x


if __name__ == '__main__':

    model = TestLearnInverseMask(31).to('cuda')
    summary(model, (31, 48, 48))
