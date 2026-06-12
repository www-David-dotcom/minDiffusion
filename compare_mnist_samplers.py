import argparse
import os
import time
from pathlib import Path
from typing import Iterable, List, Optional

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid, save_image

from mindiffusion.ddpm import DDPM
from mindiffusion.dpm_solver import DPMSolverSampler
from mindiffusion.unet import NaiveUnet


def parse_nfe_values(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",")]


def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_device(device: Optional[str], gpu: Optional[int]) -> str:
    if device is not None:
        return device
    if gpu is not None:
        return f"cuda:{gpu}"
    return choose_device()


def mnist_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


def train_mnist(
    ddpm: DDPM,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    n_workers: int,
    max_train_batches: int,
    data_dir: Path,
) -> None:
    dataset = MNIST(
        str(data_dir),
        train=True,
        download=True,
        transform=mnist_transform(),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
    )
    optimizer = torch.optim.Adam(ddpm.parameters(), lr=lr)

    for epoch in range(epochs):
        ddpm.train()
        loss_ema = None
        pbar = tqdm(dataloader, desc=f"epoch {epoch + 1}/{epochs}")
        for batch_idx, (x, _) in enumerate(pbar):
            optimizer.zero_grad()
            loss = ddpm(x.to(device))
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            loss_ema = loss_value if loss_ema is None else 0.9 * loss_ema + 0.1 * loss_value
            pbar.set_postfix(loss=f"{loss_ema:.4f}")

            if max_train_batches and batch_idx + 1 >= max_train_batches:
                break


def sample_grid(
    sampler: DPMSolverSampler,
    base_noise: torch.Tensor,
    nfe: int,
    nrow: int,
) -> torch.Tensor:
    time_steps_ddim = sampler.make_time_steps(nfe, device=base_noise.device)
    time_steps_dpm2 = sampler.make_time_steps(nfe // 2, device=base_noise.device)
    time_steps_dpm3 = sampler.make_time_steps(nfe // 3, device=base_noise.device)

    samples = [
        sampler.sample_from(base_noise.clone(), time_steps_ddim, order=1),
        sampler.sample_from(base_noise.clone(), time_steps_dpm2, order=2),
        sampler.sample_from(base_noise.clone(), time_steps_dpm3, order=3),
    ]
    return make_grid(
        torch.cat(samples, dim=0),
        nrow=nrow,
        normalize=True,
        value_range=(-1, 1),
    )


def write_report(
    report_path: Path,
    output_dir: Path,
    nfe_values: Iterable[int],
    checkpoint_path: Path,
    epochs: int,
    max_train_batches: int,
    n_T: int,
    n_feat: int,
    n_samples: int,
) -> None:
    training_scope = "all batches" if max_train_batches == 0 else f"{max_train_batches} batch(es)"
    lines = [
        "# MNIST DPM-Solver Sampler Report",
        "",
        "This report compares DDIM, DPM-Solver-2, and DPM-Solver-3 on the same MNIST checkpoint.",
        "DDIM is the first-order DPM-Solver update, so the DDIM row uses `order=1` with the same log-SNR time grid.",
        "",
        "Rows in each image are: DDIM / DPM-Solver-1, DPM-Solver-2, DPM-Solver-3.",
        f"The checkpoint used by this run was `{checkpoint_path}` after `{epochs}` training epoch(s) over {training_scope}.",
        f"Sampler/model settings for this run: `n_T={n_T}`, `n_feat={n_feat}`, `n_samples={n_samples}`.",
        "",
        "## Qualitative Samples",
        "",
    ]

    for nfe in nfe_values:
        image_path = output_dir / f"mnist_nfe_{nfe:03d}_comparison.png"
        image_link = os.path.relpath(image_path, report_path.parent)
        lines.extend(
            [
                f"### NFE = {nfe}",
                "",
                f"![MNIST sampler comparison at NFE {nfe}]({Path(image_link).as_posix()})",
                "",
            ]
        )

    lines.extend(
        [
            "## Notes",
            "",
            "- DPM-Solver-2 uses `nfe / 2` second-order steps.",
            "- DPM-Solver-3 uses `nfe / 3` third-order steps.",
            "- Use NFE values divisible by both 2 and 3 for exact apples-to-apples budgets.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--n-T", type=int, default=1000)
    parser.add_argument("--n-feat", type=int, default=64)
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--nfe-values", type=parse_nfe_values, default=parse_nfe_values("6,12,18"))
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", type=Path, default=Path("ddpm_mnist.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("contents/mnist_dpm_solver"))
    parser.add_argument("--report", type=Path, default=Path("reports/mnist_dpm_solver_report.md"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    device = resolve_device(args.device, args.gpu)

    torch.manual_seed(args.seed)
    ddpm = DDPM(
        eps_model=NaiveUnet(1, 1, n_feat=args.n_feat),
        betas=(1e-4, 0.02),
        n_T=args.n_T,
    ).to(device)

    if args.checkpoint.exists():
        ddpm.load_state_dict(torch.load(args.checkpoint, map_location=device))

    if args.epochs:
        start = time.time()
        train_mnist(
            ddpm,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            n_workers=args.num_workers,
            max_train_batches=args.max_train_batches,
            data_dir=args.data_dir,
        )
        print(f"training_time_sec={time.time() - start:.2f}")
        torch.save(ddpm.state_dict(), args.checkpoint)

    ddpm.eval()
    sampler = DPMSolverSampler(ddpm.eps_model, betas=(1e-4, 0.02), n_T=args.n_T).to(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for nfe in args.nfe_values:
            base_noise = torch.randn(args.n_samples, 1, 32, 32, device=device)
            grid = sample_grid(sampler, base_noise, nfe=nfe, nrow=args.n_samples)
            save_image(grid, args.output_dir / f"mnist_nfe_{nfe:03d}_comparison.png")

    write_report(
        report_path=args.report,
        output_dir=args.output_dir,
        nfe_values=args.nfe_values,
        checkpoint_path=args.checkpoint,
        epochs=args.epochs,
        max_train_batches=args.max_train_batches,
        n_T=args.n_T,
        n_feat=args.n_feat,
        n_samples=args.n_samples,
    )
    print(f"report={args.report}")


if __name__ == "__main__":
    main()
