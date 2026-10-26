import os
from radiomana import HighwayDataModule

import torch
from torch.utils.data import DataLoader

import torch.nn as nn
import matplotlib.pyplot as plt

from model.cnn import CNN

os.environ["DSET_FIOT_HIGHWAY2"] = os.path.expanduser("~/aerospace_corp/fiot_highway2_f32")
print(os.environ["DSET_FIOT_HIGHWAY2"])

torch.set_num_threads(4)

dm = HighwayDataModule(batch_size=32, num_workers=0, pin_memory=False)
dm.setup()    # root_dir default is the env var NAME, so leave it alone

train_loader = dm.train_dataloader()
val_loader   = dm.val_dataloader()
test_loader  = dm.test_dataloader()

print(len(dm.data_train), len(dm.data_val), len(dm.data_test))
x, y = next(iter(train_loader))
print(x.shape, x.dtype, y[:8])

train_loader = DataLoader(
    dm.data_train,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CNN(num_classes=9).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

labels = torch.tensor([
    int(dm.data_train[i][1])
    for i in range(len(dm.data_train))
])

class_counts = torch.bincount(labels, minlength=9).float()

class_weights = 1.0 / torch.sqrt(class_counts)
class_weights = class_weights / class_weights.mean()

train_loss_fn = nn.CrossEntropyLoss(
    weight=class_weights.to(device)
)

val_loss_fn = nn.CrossEntropyLoss()


def prep(x, y):
    x = torch.as_tensor(x).float()
    x = (x + 29) / 12
    if x.ndim == 3:                      # (B, H, W) -> (B, 1, H, W)
        x = x.unsqueeze(1)
    return x.to(device), torch.as_tensor(y).long().to(device)

train_losses, val_losses, val_accs = [], [], []
best_val_loss = float("inf")

for epoch in range(20):
    # training
    model.train()

    running, n = 0.0, 0

    # count how many samples of each class are seen this epoch
    epoch_class_counts = torch.zeros(9, dtype=torch.long)

    for x, y in train_loader:
        # count labels before moving to GPU/CPU device if you want
        y_cpu = torch.as_tensor(y).long()
        epoch_class_counts += torch.bincount(
            y_cpu,
            minlength=9
        )

        x, y = prep(x, y)

        opt.zero_grad()
        logits = model(x)
        loss = train_loss_fn(logits, y)

        loss.backward()
        opt.step()

        running += loss.item() * y.size(0)
        n += y.size(0)

    train_losses.append(running / n)

    # validation
    model.eval()
    running, n, correct = 0.0, 0, 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = prep(x, y)

            logits = model(x)

            running += val_loss_fn(logits, y).item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += y.size(0)

    val_losses.append(running / n)
    val_accs.append(correct / n)

    print(
        f"epoch {epoch+1:2d} | "
        f"train loss {train_losses[-1]:.4f} | "
        f"val loss {val_losses[-1]:.4f} | "
        f"val acc {val_accs[-1]:.3f}"
    )

    print("samples seen this epoch:")
    print(epoch_class_counts)

# plot
plt.figure(figsize=(7, 4))
plt.plot(train_losses, label="train loss")
plt.plot(val_losses, label="val loss")
plt.xlabel("epoch")
plt.ylabel("cross-entropy loss")
plt.legend()
plt.show()
