import gzip
import struct

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


def read_images(filepath):
	with gzip.open(filepath, "rb") as f:
		magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
		buf = f.read(num * rows * cols)
		data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
		data = data.reshape(num, rows * cols) / 255.0
	return torch.from_numpy(data)


def read_labels(filepath):
	with gzip.open(filepath, "rb") as f:
		magic, num = struct.unpack(">II", f.read(8))
		buf = f.read(num)
		data = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
	return torch.from_numpy(data)


class MnistDataset(Dataset):
	def __init__(self, img_path, label_path):
		self.imgs = read_images(img_path)
		self.labels = read_labels(label_path)

	def __len__(self):
		return len(self.imgs)

	def __getitem__(self, idx):
		return self.imgs[idx], self.labels[idx]


class MnistModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(784, 128)
		self.fc2 = nn.Linear(128, 64)
		self.fc3 = nn.Linear(64, 10)

	def forward(self, x):
		x = torch.relu(self.fc1(x))
		x = torch.relu(self.fc2(x))
		return self.fc3(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_set = MnistDataset("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz")
test_set = MnistDataset("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64)

model = MnistModel().to(device)
loss_func = nn.CrossEntropyLoss()
opt = optim.SGD(model.parameters(), lr=0.01)

epoch_num = 5
for epoch in range(epoch_num):
	model.train()
	total_loss = 0
	for img, label in train_loader:
		img, label = img.to(device), label.to(device)
		opt.zero_grad()
		pred = model(img)
		loss = loss_func(pred, label)
		loss.backward()
		opt.step()
		total_loss += loss.item()
	avg_loss = total_loss / len(train_loader)

	model.eval()
	correct = 0
	with torch.no_grad():
		for img, label in test_loader:
			img, label = img.to(device), label.to(device)
			pred = model(img)
			pred_class = torch.argmax(pred, dim=1)
			correct += (pred_class == label).sum().item()
	acc = correct / len(test_set)
	print(f"Epoch {epoch + 1} | loss:{avg_loss:.4f} | 测试准确率:{acc:.4f}")
