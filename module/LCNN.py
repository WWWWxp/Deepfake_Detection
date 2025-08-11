
import torch
import torch.nn as nn
import torch.nn.functional as F

class MFM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * 2, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv(x)
        out1, out2 = torch.chunk(x, 2, dim=1)
        return torch.max(out1, out2)


class LCNN(nn.Module):
    def __init__(self, global_pool=True):
        super().__init__()
        self.global_pool = global_pool

        self.features = nn.Sequential(
            MFM(1, 64, kernel_size=5, stride=1, padding=2),     # [B, 64, 100, T]
            nn.MaxPool2d(kernel_size=2, stride=2),              # [B, 64, 50, T/2]

            MFM(64, 96, kernel_size=3, stride=1, padding=1),    # [B, 96, 50, T/2]
            nn.MaxPool2d(kernel_size=2, stride=2),              # [B, 96, 25, T/4]

            MFM(96, 128, kernel_size=3, stride=1, padding=1),   # [B, 128, 25, T/4]
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),    # [B, 128, 25, T/8]

            MFM(128, 128, kernel_size=3, stride=1, padding=1),  # [B, 128, 25, T/8]
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),    # [B, 128, 25, T/16]

            # 加深网络
            MFM(128, 64, kernel_size=3, stride=1, padding=1),   # [B, 64, 25, T/16]
            MFM(64, 32, kernel_size=3, stride=1, padding=1),    # [B, 32, 25, T/16]
        )

        self.pool = nn.AdaptiveMaxPool2d((16, 8))               # [B, 32, 16, 8]

        self.fc29 = nn.Linear(32 * 16 * 8, 128)
        self.mfm_fc = MFM(1, 64, kernel_size=1, stride=1, padding=0)  # 模拟全连接 MFM
        self.bn_fc = nn.BatchNorm1d(64)
        self.fc32 = nn.Linear(64, 2)

    def forward(self, mel, return_hidden=False):
        assert mel.ndim == 3 and mel.size(1) == 100
        x = mel.unsqueeze(1)                     # [B, 1, 100, T]
        x = self.features(x)                     # [B, 32, H, W]
        x = self.pool(x)                         # [B, 32, 16, 8]
        x = x.view(x.size(0), -1)                # [B, 4096]

        x = self.fc29(x).unsqueeze(1)            # [B, 1, 128]
        x = self.mfm_fc(x).squeeze(1)            # [B, 64]
        emb = self.bn_fc(x)                      # [B, 64]
        logits = self.fc32(emb)                  # [B, 2]

        return (emb, logits) if return_hidden else logits



class LCNN_ori(nn.Module):
    def __init__(self, num_class=2):
        super().__init__()
        self.dropout1 = nn.Dropout(0.2)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(5, 5),
                                    padding=(2, 2), stride=(1, 1))
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1),
                               padding=(0, 0), stride=(1, 1))
        self.batchnorm6 = nn.BatchNorm2d(32)
        self.conv7 = nn.Conv2d(in_channels=32, out_channels=96, kernel_size=(3, 3),
                               padding=(1, 1), stride=(1, 1))
        self.maxpool9 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.batchnorm10 = nn.BatchNorm2d(48)

        self.conv11 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=(1, 1),
                               padding=(0, 0), stride=(1, 1))
        self.batchnorm13 = nn.BatchNorm2d(48)
        self.conv14 = nn.Conv2d(in_channels=48, out_channels=128, kernel_size=(3, 3),
                               padding=(1, 1), stride=(1, 1))
        self.maxpool16 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv17 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1),
                               padding=(0, 0), stride=(1, 1))
        self.batchnorm19 = nn.BatchNorm2d(64)
        self.conv20 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3),
                               padding=(1, 1), stride=(1, 1))
        self.batchnorm22 = nn.BatchNorm2d(32)
        self.conv23 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1),
                               padding=(0, 0), stride=(1, 1))
        self.batchnorm25 = nn.BatchNorm2d(32)
        self.conv26 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3),
                               padding=(1, 1), stride=(1, 1))
        self.maxpool28 = nn.AdaptiveMaxPool2d((16, 8))
        # self.fc29 = nn.Linear(32*20*3, 160) #5#nn.Linear(12*32*4*16,2*64)
        self.fc29 = nn.Linear(32*16*8, 128)
        self.batchnorm31 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.7)
        
        self.fc32 = nn.Linear(64, num_class)


    def mfm2(self, x):
        out1, out2 = torch.chunk(x, 2, 1)
        return torch.max(out1, out2)

    def mfm3(self, x):
        n, c, y, z = x.shape
        out1, out2, out3 = torch.chunk(x, 3, 1)
        res1 = torch.max(torch.max(out1, out2), out3)
        tmp1 = out1.flatten()
        tmp1 = tmp1.reshape(len(tmp1), -1)
        tmp2 = out2.flatten()
        tmp2 = tmp2.reshape(len(tmp2), -1)
        tmp3 = out3.flatten()
        tmp3 = tmp3.reshape(len(tmp3), -1)
        res2 = torch.cat((tmp1, tmp2, tmp3), 1)
        res2 = torch.median(res2, 1)[0]
        res2 = res2.reshape(n,-1, y,z)
        return torch.cat((res1,res2), 1)
    def forward(self, x, return_hidden: bool = False):
        assert x.ndim == 3, f"Expected 3D input, got {x.ndim}D tensor"

        if x.size(2) != 100:
            # 自动交换维度，变成 [..., 100]
            x = x.permute(0, 2, 1)  # 从 [B, 100, T] -> [B, T, 100]
            assert x.size(2) == 100, f"Failed permute: last dim now {x.size(2)}"
        x = self.conv1(x.unsqueeze(1))
        x = self.mfm2(x)
        x = self.maxpool3(x)

        x = self.conv4(x)
        x = self.mfm2(x)
        x = self.batchnorm6(x)
        x = self.conv7(x)
        x = self.mfm2(x)

        x = self.maxpool9(x)
        x = self.batchnorm10(x)

        x = self.conv11(x)
        x = self.mfm2(x)
        x = self.batchnorm13(x)
        x = self.conv14(x)
        x = self.mfm2(x)

        x = self.maxpool16(x)

        x = self.conv17(x)
        x = self.mfm2(x)
        x = self.batchnorm19(x)
        x = self.conv20(x)
        x = self.mfm2(x)
        x = self.batchnorm22(x)
        x = self.conv23(x)
        x = self.mfm2(x)
        x = self.batchnorm25(x)
        x = self.conv26(x)
        x = self.mfm2(x)

        x = self.maxpool28(x)
        x = x.view(-1, 32 * 16 * 8)

        x = self.mfm2(self.fc29(x))
        x = self.batchnorm31(x)
        logits = self.fc32(x)

        if return_hidden:
            return x, logits
        return logits
