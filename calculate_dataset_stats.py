import json
import math
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from radiomana.radiomana import HighwayDataModule

os.environ["DSET_FIOT_HIGHWAY2"] = os.path.expanduser(
    "~/aerospace_corp/fiot_highway2_f32"
)

OUTPUT_PATH = Path(__file__).with_name("dataset_stats.json")


def main():
    dm = HighwayDataModule(
        batch_size=256,
        num_workers=4,
        pin_memory=False,
    )
    dm.setup()

    train_loader = DataLoader(
        dm.data_train,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
    )

    sum_x = 0.0
    sum_x2 = 0.0
    num_values = 0

    with torch.no_grad():
        for i, (x, _) in enumerate(train_loader):
            x = x.float()

            sum_x += x.sum().item()
            sum_x2 += torch.square(x).sum().item()
            num_values += x.numel()

            if i % 100 == 0:
                print(f"Processed {i}/{len(train_loader)} batches")

    if num_values == 0:
        raise ValueError("Cannot calculate statistics for an empty training dataset.")

    mean = sum_x / num_values
    variance = sum_x2 / num_values - mean**2
    std = math.sqrt(max(variance, 0.0))

    stats = {
        "mean": mean,
        "std": std,
    }

    OUTPUT_PATH.write_text(
        json.dumps(stats, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Mean:", mean)
    print("Standard deviation:", std)
    print("Saved statistics to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
