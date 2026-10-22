import os
from radiomana import HighwayDataModule

import torch
from torch.utils.data import WeightedRandomSampler
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


