import torch
from torch.utils.data import DataLoader
from model import CNNLSTM
from dataset import SeismicDataset
import config

data_files = ["data/example_test1.npy", "data/example_test2.npy"]
labels = [0, 1]

dataset = SeismicDataset(data_files, labels)
dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

model = CNNLSTM(input_size=128, cnn_out_channels=16, lstm_hidden_size=32, num_classes=2)
model.load_state_dict(torch.load("models/cnn_lstm_model.pth"))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for batch_data, batch_labels in dataloader:
        outputs = model(batch_data.unsqueeze(1))
        _, predicted = torch.max(outputs.data, 1)
        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()

print(f"Accuracy on test data: {100 * correct / total:.2f}%")
