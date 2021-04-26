# coding: utf-8


import torch
from torchsummary import summary
from layers import Mish


class UNet(torch.nn.Module):

    def __init__(self, input_ch: int, output_ch: int, *args, **kwargs):
        super(UNet, self).__init__()
        self.activation = kwargs.get('activation', 'relu').lower()
        self.activations = {'relu': torch.nn.ReLU, 'leaky': torch.nn.LeakyReLU,
                            'swish': torch.nn.SiLU, 'mish': Mish}
        features = [32, 64, 128, 256, 512]
        enc_features = [input_ch] + features
        enc_layer = []
        pool_layer = []
        for i in range(1, len(enc_features)):
            layer = []
            layer.append(self.init_convlayer(enc_features[i - 1], enc_features[i]))
            layer.append(self.init_convlayer(enc_features[i], enc_features[i]))
            # layer.append(torch.nn.MaxPool2d(2, 2))
            enc_layer.append(torch.nn.Sequential(*layer))
            pool_layer.append(torch.nn.MaxPool2d(2, 2))

        self.enc_layers = torch.nn.ModuleList(enc_layer)
        self.enc_pool = torch.nn.ModuleList(pool_layer)
        # Bottleneck
        self.bn = torch.nn.ModuleList([torch.nn.Conv2d(512, 1024, 3, 1, 1),
                                       torch.nn.Conv2d(1024, 1024, 3, 1, 1)])
        self.bn_activations = torch.nn.ModuleList([torch.nn.ReLU(),
                                                   torch.nn.ReLU()])

        # First Decoder
        dec_features = [1024] + features[::-1] + [output_ch]
        dec_layer = []
        transpose_layer = []
        for i in range(1, len(dec_features) - 1):
            transpose_layer.append(torch.nn.ConvTranspose2d(dec_features[i - 1], dec_features[i], 2, 2))
            dec_layer.append(self.init_convlayer(dec_features[i] + dec_features[i + 1], dec_features[i]))
        self.dec_layers = torch.nn.ModuleList(dec_layer)
        print(self.dec_layers)
        self.transpose_layers = torch.nn.ModuleList(transpose_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_features = []
        for layers, pool in zip(self.enc_layers, self.enc_pool):
            x = layers(x)
            enc_features.append(x)
            x = pool(x)
        for layers, activation in zip(self.bn, self.bn_activations):
            x = activation(layers(x))
        x = self.transpose_layers[0](x)
        x = torch.cat([x, enc_features[-1]], dim=1)
        x = self.dec_layers[0]
        # for i, (layers, transpose) in enumerate(zip(self.dec_layers, self.transpose_layers)):
        #     x = transpose(x)
        #     x = torch.cat([x, enc_features[-i]], dim=1)
        #     print(x.shape)
        return x

    def init_convlayer(self, input_ch: int, output_ch: int) -> torch.nn.Module:
        return torch.nn.Sequential(torch.nn.Conv2d(input_ch, output_ch, 3, 1, 1),
                                   self.activations[self.activation]())


model = UNet(1, 3)
summary(model, (1, 64, 64))
