import torch
from torch.utils.data import DataLoader
from model import CNNLSTM
from dataset import SeismicDataset
import config

data_files = ["data/example1.npy", "data/example2.npy"]
labels = [0, 1]

dataset = SeismicDataset(data_files, labels)
dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

model = CNNLSTM(input_size=128, cnn_out_channels=16, lstm_hidden_size=32, num_classes=2)
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(config.NUM_EPOCHS):
    for batch_data, batch_labels in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_data.unsqueeze(1))
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}, Loss: {loss.item():.4f}")
