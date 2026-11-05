import time
import torch

import torch.nn as nn
from radiomana.radiomana import HighwayDataModule

from model import CNN

import os

os.environ["DSET_FIOT_HIGHWAY2"] = os.path.expanduser("~/aerospace_corp/fiot_highway2_f32")
print(os.environ["DSET_FIOT_HIGHWAY2"])

torch.set_num_threads(4)

dm = HighwayDataModule(batch_size=32, num_workers=4, pin_memory=False)
dm.setup()    # root_dir default is the env var NAME, so leave it alone

train_loader = dm.train_dataloader()

timing_model = CNN(num_classes=9).to("cpu")
timing_opt = torch.optim.Adam(timing_model.parameters(), lr=1e-3)

timing_model.train()

device = "cuda" if torch.cuda.is_available() else "cpu"

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

def prep(x, y):
    x = torch.as_tensor(x).float()
    x = (x + 29) / 12
    if x.ndim == 3:                      # (B, H, W) -> (B, 1, H, W)
        x = x.unsqueeze(1)
    return x.to(device), torch.as_tensor(y).long().to(device)

t0 = time.time()
n_batches = 0

for x, y in train_loader:
    x, y = prep(x, y)
    timing_opt.zero_grad()
    loss = train_loss_fn(timing_model(x), y)
    loss.backward()
    timing_opt.step()
    n_batches += 1
    if n_batches == 5:
        break
per_batch = (time.time() - t0) / n_batches
print(f"{per_batch:.2f} s/batch, ~{per_batch * len(train_loader) / 60:.1f} min/epoch")
