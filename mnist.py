import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        return torch.nn.functional.log_softmax(x, dim=1)


def get_data_loaders(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    dataset = MNIST(root="./data", train=is_train, transform=to_tensor, download=True)
    return DataLoader(dataset, batch_size=128, shuffle=is_train)


def evaluate(test_data, net):
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for x, y in test_data:
            outputs = net.forward(x.view(-1, 28 * 28))
            predictions = torch.argmax(outputs, dim=1)
            n_correct += (predictions == y).sum().item()
            n_total += y.numel()
    return n_correct / n_total if n_total else 0.0


def main():
    train_data = get_data_loaders(True)
    test_data = get_data_loaders(False)
    net = Net()

    print("initial accuracy:", evaluate(test_data, net))
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(2):
        for x, y in train_data:
            net.zero_grad()
            outputs = net.forward(x.view(-1, 28 * 28))
            loss = torch.nn.functional.nll_loss(outputs, y)
            loss.backward()
            optimizer.step()
        accuracy = evaluate(test_data, net)
        print(f"epoch: {epoch} accuracy: {accuracy:.4f}")

    for n, (x, _) in enumerate(test_data):
        if n >= 4:
            break
        sample = x[0]
        prediction = torch.argmax(net.forward(sample.view(-1, 28 * 28)), dim=1)
        plt.figure(n)
        plt.imshow(sample.view(28, 28), cmap="gray")
        plt.title(f"prediction: {int(prediction[0])}")
        plt.show()


if __name__ == "__main__":
    main()