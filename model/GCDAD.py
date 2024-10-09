import torch
import torch.nn as nn
from einops import rearrange
from model.RevIN import RevIN
from tkinter import _flatten


class GCDAD(nn.Module):
    def __init__(self, win_size, d_model=256,  patch_size=[3, 5, 7], channel=55,space_num = [3], dropout=0.05, output_attention=True):
        super(GCDAD, self).__init__()
        self.output_attention = output_attention
        self.patch_size = patch_size
        self.space_num = space_num
        self.channel = channel
        self.win_size = win_size
        self.gru_enc1 = nn.GRU(1, d_model)
        self.gru_enc2 = nn.GRU(1, d_model)
        self.gru_enc3 = nn.ModuleList(
            nn.GRU(1, d_model) for patch_index, patchsize in enumerate(self.patch_size))
        self.gru_enc4 = nn.ModuleList(
            nn.GRU(space_num[patch_index], d_model) for patch_index, patchsize in enumerate(self.patch_size))
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, M = x.shape  # Batch win_size channel   128 100 51
        series_patch_mean = []
        prior_patch_mean = []
        series_patch_mean_1 = []
        prior_patch_mean_1 = []
        revin_layer = RevIN(num_features=M)

        # Instance Normalization Operation
        x = revin_layer(x, 'norm')

        #time
        for patch_index, patchsize in enumerate(self.patch_size):
            x_patch_size, x_patch_num = x, x  # 128,100,51
            x_patch_size = rearrange(x_patch_size, 'b l m -> b m l')  # Batch channel win_size  128 51 100
            x_patch_num = rearrange(x_patch_num, 'b l m -> b m l')  # Batch channel win_size    128 51 100

            processed_results = []
            x_patch_size = rearrange(x_patch_size, 'b m (n p) -> (b m) n p', p=patchsize)  # 6528 20 5  内 128 51 20 5
            x_patch_num = rearrange(x_patch_num, 'b m (p n) -> (b m) p n', p=patchsize)  # 6528 5 20    128 51 5 20
            for i in range(x_patch_size.size(1)):
                part = x_patch_size[:, i:i + 1, :]  # 128 51 1 5
                part = part.squeeze(1).unsqueeze(-1)  # batch fea size128 51 5   6528 1 5
                part, _ = self.gru_enc1(part.permute(1, 0, 2))
                processed_results.append(part.permute(1, 0, 2))

            x_patch_size = torch.cat(processed_results, axis=1)
            x_patch_size = x_patch_size.reshape(B, M, L, self.d_model)
            x_patch_size = torch.mean(x_patch_size, dim=1)
            x_patch_size = self.dropout(torch.softmax(x_patch_size, dim=-1))

            processed_results = []
            for i in range(x_patch_num.size(1)):
                part = x_patch_num[:, i:i + 1, :]  # 128 51 1 5
                part = part.squeeze(1).unsqueeze(-1)  # batch fea size128 51 5   6528 1 5
                part, _ = self.gru_enc2(part.permute(1, 0, 2))
                processed_results.append(part.permute(1, 0, 2))

            # batch num fea 128 100 128
            x_patch_num = torch.cat(processed_results, axis=1)  # .permute( 0, 2, 1)
            x_patch_num = x_patch_num.reshape(B, M, L, self.d_model)
            x_patch_num = torch.mean(x_patch_num, dim=1)
            x_patch_num = self.dropout(torch.softmax(x_patch_num, dim=-1))

            series_patch_mean.append(x_patch_size), prior_patch_mean.append(x_patch_num)

            num = self.space_num[patch_index]
            result = []
            front = num // 2
            back = num - front
            boundary = L - back
            for i in range(self.win_size):
                if (i < front):
                    temp = x[:, 0, :].unsqueeze(1).repeat(1, front - i, 1)
                    temp1 = torch.cat((temp, x[:, 0:i, :]), dim=1)
                    temp1 = torch.cat((temp1, x[:, i:i + back, :]), dim=1)
                    result.append(temp1)
                elif (i > boundary):
                    temp = x[:, L - 1, :].unsqueeze(1).repeat(1, back + i - L, 1)
                    temp1 = torch.cat((x[:, i - front:self.win_size, :], temp), dim=1)
                    result.append(temp1)
                else:
                    temp = x[:, i - front:i + back, :].reshape(B, -1, M)
                    result.append(temp)

            x_patch_num = torch.cat(result, axis=0).reshape(L, B, num, M).permute(1, 0, 3, 2).reshape(-1, M, num)
            x_patch_size = x.unsqueeze(-1).reshape(-1, M, 1)
            x_patch_num, _ = self.gru_enc4[patch_index](x_patch_num.permute(1, 0, 2))
            x_patch_num = x_patch_num.permute(1, 0, 2).reshape(B, L, M, self.d_model).permute(0, 2, 1, 3)
            x_patch_num = self.dropout(torch.softmax(x_patch_num, dim=1))

            x_patch_size, _ = self.gru_enc3[patch_index](x_patch_size.permute(1, 0, 2))
            x_patch_size = x_patch_size.permute(1, 0, 2).reshape(B, L, M, self.d_model).permute(0, 2, 1,
                                                                                                3)  # batch fea win d
            x_patch_size = self.dropout(torch.softmax(x_patch_size, dim=1))

            series_patch_mean_1.append(x_patch_size), prior_patch_mean_1.append(x_patch_num)

        series_patch_mean = list(_flatten(series_patch_mean))  # 3
        prior_patch_mean = list(_flatten(prior_patch_mean))  # 3
        prior_patch_mean_1 = list(_flatten(prior_patch_mean_1))  # 3
        series_patch_mean_1 = list(_flatten(series_patch_mean_1))  # 3

        if self.output_attention:
            return series_patch_mean, prior_patch_mean,series_patch_mean_1, prior_patch_mean_1
        else:
            return None


