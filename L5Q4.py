import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# Configurable CNN
class Net(nn.Module):
    def __init__(self, c1=16, c2=32, fc=128, gap=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, c1, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1) if gap else nn.Identity()
        )
        self.fc = nn.Sequential(
            nn.Linear(c2 * (1 if gap else 7 * 7), fc),
            nn.ReLU(),
            nn.Linear(fc, 10)
        ) if not gap else nn.Linear(c2, 10)

    def forward(self, x):
        x = self.conv(x).flatten(1)
        return self.fc(x)


# Data setup
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))])
train = DataLoader(torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=64,
                   shuffle=True)
test = DataLoader(torchvision.datasets.MNIST('./data', train=False, transform=transform), batch_size=64)

# Experiment configs
configs = [
    {'name': 'Baseline', 'args': {}},
    {'name': 'SmallFC', 'args': {'fc': 64}},
    {'name': 'GAP', 'args': {'gap': True}},
    {'name': 'Tiny', 'args': {'c1': 8, 'c2': 16, 'fc': 64}}
]

# Training/eval loop
results = []
for cfg in configs:
    model = Net(**cfg['args']).to('cuda')
    opt = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    # Train
    for _ in range(5):
        for x, y in train:
            opt.zero_grad()
            loss = criterion(model(x.to('cuda')), y.to('cuda'))
            loss.backward()
            opt.step()

    # Eval
    correct = 0
    with torch.no_grad():
        for x, y in test:
            correct += (model(x.to('cuda')).argmax(1) == y.to('cuda')).sum().item()

    params = sum(p.numel() for p in model.parameters())
    results.append((cfg['name'], params, 100 * correct / len(test.dataset)))

# Plot results
baseline = next(r[1] for r in results if r[0] == 'Baseline')
plt.figure(figsize=(8, 5))
for name, params, acc in results:
    plt.scatter(100 * (baseline - params) / baseline, acc, label=name, s=100)
plt.legend(), plt.grid(), plt.xlabel('Parameter Reduction (%)'), plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs Parameter Reduction'), plt.show()