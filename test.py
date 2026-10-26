from radiomana import HighwayDataModule
from model.cnn import CNN

import torch

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report

from sklearn.metrics import balanced_accuracy_score

def prep(x, y):
    x = torch.as_tensor(x).float()
    x = (x + 29) / 12
    if x.ndim == 3:                      # (B, H, W) -> (B, 1, H, W)
        x = x.unsqueeze(1)
    return x.to(device), torch.as_tensor(y).long().to(device)


def evaluate(loader):
    preds, labels = [], []

    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x, y = prep(x, y)

            logits = model(x)
            pred = logits.argmax(1)

            preds.append(pred.cpu())
            labels.append(y.cpu())

    preds = torch.cat(preds)
    labels = torch.cat(labels)

    print("acc:", (preds == labels).float().mean().item())
    print("label counts:", torch.bincount(labels, minlength=9).tolist())
    print("pred counts: ", torch.bincount(preds, minlength=9).tolist())

    return preds, labels


dm = HighwayDataModule(batch_size=32, num_workers=0, pin_memory=False)
dm.setup()

test_loader  = dm.test_dataloader()

print(len(dm.data_test))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load (in a new session)
model = CNN(num_classes=9).to(device)
model.load_state_dict(torch.load("best.pth"))
model.eval()  # switch to inference mode if you're not training further

preds, labels = evaluate(test_loader)

cm = confusion_matrix(labels.numpy(), preds.numpy())
cm_norm = cm / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(8, 6))
plt.imshow(cm_norm)
plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.colorbar()
plt.show()

print(classification_report(
    labels.numpy(),
    preds.numpy(),
    digits=3
))

balanced_acc = balanced_accuracy_score(
    labels.numpy(),
    preds.numpy()
)

print("balanced accuracy:", balanced_acc)