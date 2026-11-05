# CNN training

Run training as usual:

```sh
python train.py --epochs 20
```

Each invocation creates a new directory under `runs/` beside `train.py`, for
example `runs/2026-11-04_23-15-00_UTC_a1b2c3d4/`. The timestamp is UTC and the
automatically generated suffix keeps simultaneous jobs separate. The full output
path is printed when training starts. Each directory contains `best.pth`,
`cnn_epoch_1.pth` (and subsequent epoch checkpoints), and `loss_plot.png` after
training finishes.

To use a different parent directory, such as school scratch storage:

```sh
python train.py --epochs 20 --output-dir /path/to/scratch/runs
```

Evaluate a particular run by passing its checkpoint:

```sh
python test.py --checkpoint runs/<run-directory>/best.pth
```

Without `--checkpoint`, `test.py` still loads `best.pth` from the working directory.
