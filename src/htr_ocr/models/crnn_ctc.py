import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding=pad)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class CNN12Backbone(nn.Module):
    def __init__(self, in_ch: int = 1):
        super().__init__()

        layers: list[nn.Module] = []

        ch = in_ch
        for _ in range(2):
            layers.append(ConvBlock(ch, 32))
            ch = 32
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        for _ in range(4):
            layers.append(ConvBlock(ch, 64))
            ch = 64
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        for _ in range(6):
            layers.append(ConvBlock(ch, 128))
            ch = 128

        self.net = nn.Sequential(*layers)
        self.out_channels = 128
        self.pool_factor_w = 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CRNNCTC(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_ch: int = 1,
        rnn_hidden: int = 256,
        rnn_layers: int = 2,
        fc_hidden: int = 256,
    ):
        super().__init__()
        self.backbone = CNN12Backbone(in_ch=in_ch)

        self.rnn = nn.LSTM(
            input_size=self.backbone.out_channels,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            dropout=0,
            bidirectional=True,
        )

        self.fc1 = nn.Linear(2 * rnn_hidden, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, num_classes)

    @property
    def time_downsample_factor(self) -> int:
        return self.backbone.pool_factor_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, 1, H, W]
        f = self.backbone(x)  # [B, C, H', W']

        # [B, C, W']
        f = torch.max(f, dim=2).values

        # [W', B, C]
        f = f.permute(2, 0, 1).contiguous()

        # [W', B, 2H]
        y, _ = self.rnn(f)

        y = F.relu(self.fc1(y))
        logits = self.fc2(y)  # [T, B, C]

        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs
