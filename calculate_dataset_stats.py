"""Calculate raw training-set statistics and save them to dataset_stats.json."""

import json
import math
import os
from pathlib import Path

from torch.utils.data import DataLoader

from radiomana import HighwayDataModule

OUTPUT_PATH = Path(__file__).with_name("dataset_stats.json")


def main() -> None:
    os.environ["DSET_FIOT_HIGHWAY2"] = str(
        Path.home() / "aerospace_corp" / "fiot_highway2_f32"
    )

    dm = HighwayDataModule(batch_size=32, num_workers=0, pin_memory=False)
    dm.setup()

    train_loader = DataLoader(
        dm.data_train,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    sum_x = 0.0
    sum_x2 = 0.0
    num_values = 0

    for x, _ in train_loader:
        x = x.float()
        sum_x += x.sum().item()
        sum_x2 += (x ** 2).sum().item()
        num_values += x.numel()

    if num_values == 0:
        raise ValueError("Cannot calculate statistics for an empty training dataset.")

    mean = sum_x / num_values
    variance = (sum_x2 / num_values) - mean**2
    std = math.sqrt(max(variance, 0.0))

    stats = {"mean": mean, "std": std}
    OUTPUT_PATH.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    print("Mean:", mean)
    print("Standard deviation:", std)
    print("Saved statistics to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
