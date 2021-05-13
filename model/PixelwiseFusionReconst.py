# coding: UTF-8


from .layers import ReLU, Leaky, Swish, Mish
from .layers import FeatureFusion
import torch
from torchsummary import summary


class FusionReconstHSI(torch.nn.Module):

    def __init__(self, input_ch: int, output_ch: int, *args, feature_num: int=64,
                 block_num: int=9, **kwargs):
        super().__init__()

        activations = {'relu': ReLU, 'leaky': Leaky, 'swish': Swish, 'mish': Mish}
        activation = str(kwargs.get('activation', 'relu')).lower()
        feature_block = kwargs.get('feature_block', 3)

        self.input_conv = torch.nn.Conv2d(input_ch, feature_num, 3, 1, 1)
        self.features_conv = torch.nn.ModuleList([FeatureFusion(feature_num, feature_num, feature_block=feature_block, activation=activation)
                                                  for _ in range(block_num)])
        self.output_conv = torch.nn.Conv2d(feature_num, output_ch, 3, 1, 1)

    def forward(self, x):

        x = self.input_conv(x)
        for layer in self.features_conv:
            x_in = x
            x = layer(x_in)
        x = self.output_conv(x)

        return x


if __name__ == '__main__':

    model = FusionReconstHSI(1, 31, feature_num=64, feature_block=4).to('cuda')
    summary(model, (1, 48, 48))
