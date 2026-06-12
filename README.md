# minDiffusion

<!-- #region -->
<p align="center">
<img  src="contents/_ddpm_sample_19.png">
</p>

Goal of this educational repository is to provide a self-contained, minimalistic implementation of diffusion models using Pytorch.

Many implementations of diffusion models can be a bit overwhelming. Here, `superminddpm` : under 200 lines of code, fully self contained implementation of DDPM with Pytorch is a good starting point for anyone who wants to get started with Denoising Diffusion Models, without having to spend time on the details.

Simply:

```
$ python superminddpm.py
```

Above script is self-contained. (Of course, you need to have pytorch and torchvision installed. Latest version should suffice. We do not use any cutting edge features.)

If you want to use the bit more refactored code, that runs CIFAR10 dataset:

```
$ python train_cifar10.py
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
python compare_mnist_samplers.py --epochs 5 --nfe-values 6,12,18
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
python3 compare_mnist_samplers.py \
  --epochs 20 \
  --batch-size 128 \
  --n-feat 64 \
  --n-T 1000 \
  --nfe-values 6,12,18,24 \
  --checkpoint ddpm_mnist.pth \
  --output-dir contents/mnist_dpm_solver \
  --report reports/mnist_dpm_solver_report.md
```
If you alreay have a checkpoint, you can skip training by setting `--epochs 0` and just load the checkpoint:
```bash
python3 compare_mnist_samplers.py \
  --epochs 0 \
  --nfe-values 6,12,18,24 \
  --checkpoint ddpm_mnist.pth
```