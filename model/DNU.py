# coding: utf-8


import torch


class DNU_Block(torch.nn.Module):

    def __init__(self, input_ch: int, output_ch: int, *args, feature_num: int=64, **kwargs):
        super(DNU_Block, self).__init__()

        deta = kwargs.get('deta', .04)
        eta = kwargs.get('eta', .8)
        wz1 = kwargs.get('wz1', .8)
        # DNU Parameters
        self.DNU_Params = torch.nn.ParameterDict({'deta': torch.nn.Parameter(torch.tensor([deta])),
                                                  'eta': torch.nn.Parameter(torch.tensor([eta])),
                                                  'wz1': torch.nn.Parameter(torch.tensor([wz1]))})

        # local module
        self.resx1 = torch.nn.Sequential(torch.nn.ReLU(),
                                         torch.nn.Conv2d(input_ch, feature_num, 3, 1, 1))
        self.resx2 = torch.nn.Conv2d(feature_num, output_ch, 3, 1, 1)

        # Non Local module
        self.nlConv = torch.nn.Conv2d(input_ch, output_ch, 1, 1, 0)

    def forward(self, xt: torch.Tensor, x0: torch.Tensor, Cu: torch.Tensor) -> torch.Tensor:

        bt, Ct, Ht, Wt = xt.shape
        b0, C0, H0, W0 = x0.shape

        resx1 = self.resx1(xt)
        resx2 = self.resx2(resx1)
        z1 = resx2 + xt

        x_g = self.nlConv(xt)

        x_phi_reshape = xt.view(-1, Ct, Ht * Wt)
        x_phi_permute = x_phi_reshape.permute(0, 2, 1)
        x_g_reshape = x_g.view(-1, Ct, Ht * Wt)
        x_mul1 = torch.matmul(x_phi_permute, x_g_reshape)
        x_mul2 = torch.matmul(x_phi_reshape, x_mul1)
        x_mul2_softmax = x_mul2 * (1 / (Ht + Ct - 1) * Wt)
        z2_tmp = x_mul2_softmax.view(-1, Ct, Ht, Wt)
        z2 = torch.nn.functional.relu(z2_tmp)
        z = self.DNU_Params['wz1'] * z1 + (1 - self.DNU_Params['wz1']) * z2

        yt = x * Cu
        yt1 = yt.sum(dim=1, keepdims=True)
        yt2 = yt1.tile(1, Ct, 1, 1)
        xt2 = yt2 * Cu
        x_output = (1 - self.DNU_Params['deta'] * self.DNU_Params['eta']) * xt - \
                    self.DNU_Params['deta'] * xt2 + self.DNU_Params['deta'] * x0 + \
                    self.DNU_Params['deta'] * self.DNU_Params['eta'] * z

        return x_output


class DNU(torch.nn.Module):

    def __init__(self, input_ch: int, output_ch: int, *args, feature_num: int=64,
                 layer_num=9, **kwargs):
        super(DNU, self).__init__()

        self.output_ch = output_ch

        deta = kwargs.get('deta', .04)
        eta = kwargs.get('eta', .8)
        wz1 = kwargs.get('wz1', .8)
        self.recon_block = torch.nn.ModuleList([DNU_Block(output_ch, output_ch,
                                                          feature_num=feature_num)
                                                for _ in range(layer_num)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        y, Cu = x
        y = y.tile((1, self.output_ch, 1, 1))
        x0 = y * Cu
        xt = x0
        for layer in self.recon_block:
            xt = layer(xt, x0, Cu)

        return xt
