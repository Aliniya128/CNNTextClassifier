import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Classifier(nn.Module):
    def __init__(self, n_classes, args):
        super(Classifier, self).__init__()
        self.emb = nn.Embedding(20300, 1024)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 1024), stride=1),
            nn.MaxPool2d(kernel_size=(16, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

        )

        self.classifier = nn.Sequential(
            nn.Linear(32, n_classes),
        )

    def forward(self, x, length):
        x = self.emb(x)
        pd = pack_padded_sequence(x, lengths=length, batch_first=True, enforce_sorted=False)
        data = pad_packed_sequence(pd, batch_first=True, total_length=18)[0].unsqueeze(1)
        x = self.conv(data).squeeze()
        x = self.classifier(x)
        return x
