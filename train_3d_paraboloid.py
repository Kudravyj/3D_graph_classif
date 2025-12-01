import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import TensorDataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

N = 20000

X = torch.rand(N, 3) * 2 - 1 

labels = (X[:, 2] > (X[:, 0]**2 + X[:, 1]**2)).long()


train_X = X[:16000].to(device)
train_y = labels[:16000].to(device)

test_X = X[16000:].to(device)
test_y = labels[16000:].to(device)
dataset = TensorDataset(train_X, train_y)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.model(x)

net = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

epochs = 40
for epoch in range(epochs):
    for batch_X, batch_y in loader:
        optimizer.zero_grad()
        out = net(batch_X)
        loss = criterion(out, batch_y)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss = {loss.item():.4f}")

with torch.no_grad():
    preds = torch.argmax(net(test_X), dim=1)
    acc = (preds == test_y).float().mean()
    print("\nAccuracy:", acc.item())

pts = test_X.cpu()
preds_cpu = preds.cpu()

colors = ["red" if p == 0 else "blue" for p in preds_cpu]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors, s=4)

res = 50
xs = torch.linspace(-1, 1, res)
ys = torch.linspace(-1, 1, res)
Xg, Yg = torch.meshgrid(xs, ys)
Zg = Xg**2 + Yg**2

ax.plot_surface(Xg, Yg, Zg, alpha=0.5, color="black")

ax.set_title("3D Paraboloid Classification")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()
