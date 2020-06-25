import torch
from torch import nn
import torch.nn.functional as F


class MetricGenerator(nn.Module):
    def __init__(self, sg_dim=257, rnn_hidden_dim=100, linear_hidden_dim=300):
        super().__init__()
        self.rec_1 = nn.LSTM(sg_dim, rnn_hidden_dim, bidirectional=True, batch_first=True)
        self.rec_2 = nn.LSTM(2 * rnn_hidden_dim, rnn_hidden_dim, bidirectional=True, batch_first=True)
        self.linear_layers = nn.Sequential(
            nn.Linear(2 * rnn_hidden_dim, linear_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(linear_hidden_dim, sg_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x, _ = self.rec_1(x)
        x, _ = self.rec_2(x)
        x = self.linear_layers(x)
        return x


class MetricDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(2, 10, 5),
            nn.LeakyReLU(),
            nn.Conv2d(10, 15, 4, dilation=2),
            nn.LeakyReLU(),
            nn.Conv2d(15, 25, 5, dilation=2),
            nn.LeakyReLU(),
            nn.Conv2d(25, 25, 11)
        )
        self.linears = nn.Sequential(
            nn.Linear(25, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, noisy, reference):
        x = torch.cat((noisy, reference), dim=1)
        x = self.convs(x)
        x = F.avg_pool2d(x, x.size()[-2:]).reshape((-1, 25))
        x = self.linears(x)
        return x
