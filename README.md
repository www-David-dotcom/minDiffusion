# minDiffusion

<!-- #region -->
<p align="center">
<img  src="contents/_ddpm_sample_19.png">
</p>

Goal of this educational repository is to provide a self-contained, minimalistic implementation of diffusion models using Pytorch.

Many implementations of diffusion models can be a bit overwhelming. Here, `superminddpm` : under 200 lines of code, fully self contained implementation of DDPM with Pytorch is a good starting point for anyone who wants to get started with Denoising Diffusion Models, without having to spend time on the details.

## Environment

This project uses `uv` for the Python environment. PyTorch CUDA 12.1 wheels are available for Python 3.10-3.12, so use Python 3.12 on this cluster:

```bash
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
uv sync
```

On this cluster, the NVIDIA driver reports CUDA driver API 12.2, so the uv config uses PyTorch CUDA 12.1 wheels (`cu121`). Those wheels are compatible with a 12.2-capable driver and avoid installing a newer PyTorch CUDA build that requires a newer NVIDIA driver.

If you still see a driver mismatch, recreate the environment after updating the lockfile:

```bash
uv lock
uv sync --reinstall
```

If uv already made a Python 3.13 `.venv`, remove and recreate it with Python 3.12:

```bash
rm -rf .venv
uv venv --python 3.12
uv sync
```

If you move to a cluster with a different CUDA/driver stack, update the PyTorch index in `pyproject.toml` accordingly.

## GPU selection

Training scripts accept `--gpu N` for multi-GPU systems. For example, `--gpu 0` maps to PyTorch device `cuda:0`. If a scheduler sets `CUDA_VISIBLE_DEVICES`, this is relative to the visible GPUs, so `--gpu 0` means the first visible GPU.

You can also pass `--device` directly, such as `--device cpu` or `--device cuda:1`; `--device` takes precedence over `--gpu`.

Simply:

```
$ uv run python superminddpm.py --gpu 0
```

Above script is self-contained. (Of course, you need to have pytorch and torchvision installed. Latest version should suffice. We do not use any cutting edge features.)

If you want to use the bit more refactored code, that runs CIFAR10 dataset:

```
$ uv run python train_cifar10.py --gpu 0
```

For CelebA, set `CELEBA_PATH` in `.env` or pass `--celeba-path`:

```
$ uv run python train_celeba.py --gpu 1 --celeba-path /path/to/celeba
```

<!-- #region -->
<p align="center">
<img  src="contents/_ddpm_sample_cifar43.png">
</p>

Above result took about 2 hours of training on single 3090 GPU. Top 8 images are generated, bottom 8 are ground truth.

Here is another example, trained on 100 epochs (about 1.5 hours)

<p align="center">
<img  src="contents/_ddpm_sample_cifar100.png">
</p>

Currently has:

- [x] Tiny implementation of DDPM
- [x] MNIST, CIFAR dataset.
- [x] Simple unet structure. + Simple Time embeddings.
- [x] CelebA dataset.
- [x] DPM-Solver-2 and DPM-Solver-3 samplers.

## DPM-Solver MNIST comparison

`mindiffusion/dpm_solver.py` implements DPM-Solver-1, DPM-Solver-2, and DPM-Solver-3 from the paper equations. DPM-Solver-1 is equivalent to deterministic DDIM, so it is used as the DDIM baseline in the MNIST comparison script.

Run a quick MNIST comparison:

```
uv run python compare_mnist_samplers.py --gpu 0 --epochs 5 --nfe-values 6,12,18
```

The script writes comparison grids to `contents/mnist_dpm_solver/` and a markdown report to `reports/mnist_dpm_solver_report.md`. For assignment-quality samples, train longer or load an existing `ddpm_mnist.pth` checkpoint and pass `--epochs 0`.

TODOS

- [x] DDIM
- [ ] Classifier Guidance
- [ ] Multimodality

# Updates!

- Using more parameter yields better result for MNIST.
- More comments in superminddpm.py

# Running MNIST comparison of DPM-Solver samplers
If you want a fresh training,
```bash
uv run python compare_mnist_samplers.py \
  --epochs 20 \
  --batch-size 128 \
  --n-feat 64 \
  --n-T 1000 \
  --nfe-values 6,12,18,24 \
  --checkpoint ddpm_mnist.pth \
  --output-dir contents/mnist_dpm_solver \
  --report reports/mnist_dpm_solver_report.md \
  --gpu 0
```
If you alreay have a checkpoint, you can skip training by setting `--epochs 0` and just load the checkpoint:
```bash
uv run python compare_mnist_samplers.py \
  --gpu 0 \
  --epochs 0 \
  --nfe-values 6,12,18,24 \
  --checkpoint ddpm_mnist.pth
```
