import torch
import torch.nn as nn

class CNNLSTM(nn.Module):
    def __init__(self, input_size, cnn_out_channels, lstm_hidden_size, num_classes):
        super(CNNLSTM, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, cnn_out_channels, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,2))
        )

        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x):
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.mean(dim=2)
        cnn_out = cnn_out.permute(0, 2, 1)
        lstm_out, _ = self.lstm(cnn_out)
        out = self.fc(lstm_out[:, -1, :])
        return out
