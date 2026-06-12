import argparse
import os
from typing import Optional

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from mindiffusion.unet import NaiveUnet
from mindiffusion.ddpm import DDPM

from dotenv import load_dotenv

load_dotenv("./.env")
CELEBA_PATH = os.getenv("CELEBA_PATH")


def resolve_device(device: Optional[str], gpu: Optional[int], default: str) -> str:
    if device is not None:
        return device
    if gpu is not None:
        return f"cuda:{gpu}"
    return default


def train_celeba(
    n_epoch: int = 100,
    device: str = "cuda:1",
    load_pth: Optional[str] = None,
    celeba_path: Optional[str] = None,
) -> None:

    ddpm = DDPM(eps_model=NaiveUnet(3, 3, n_feat=128), betas=(1e-4, 0.02), n_T=1000)

    if load_pth is not None:
        ddpm.load_state_dict(torch.load(load_pth, map_location=device))

    ddpm.to(device)

    tf = transforms.Compose(  # resize to 512 x 512, convert to tensor, normalize
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    if celeba_path is None:
        raise ValueError("CelebA path is required. Set CELEBA_PATH in .env or pass --celeba-path.")

    dataset = ImageFolder(
        root=celeba_path,
        transform=tf,
    )

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=20)
    optim = torch.optim.Adam(ddpm.parameters(), lr=2e-5)

    for i in range(n_epoch):
        print(f"Epoch {i} : ")
        ddpm.train()

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, _ in pbar:
            optim.zero_grad()
            x = x.to(device)
            loss = ddpm(x)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        ddpm.eval()
        with torch.no_grad():
            xh = ddpm.sample(8, (3, 128, 128), device)
            xset = torch.cat([xh, x[:8]], dim=0)
            grid = make_grid(xset, normalize=True, value_range=(-1, 1), nrow=4)
            save_image(grid, f"./contents/ddpm_sample_celeba{i:03d}.png")

            # save model
            torch.save(ddpm.state_dict(), f"./ddpm_celeba.pth")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--load-pth", default=None)
    parser.add_argument("--celeba-path", default=CELEBA_PATH)
    args = parser.parse_args()

    train_celeba(
        n_epoch=args.epochs,
        device=resolve_device(args.device, args.gpu, "cuda:1"),
        load_pth=args.load_pth,
        celeba_path=args.celeba_path,
    )


if __name__ == "__main__":
    main()
