from typing import Iterable, Tuple

import torch
import torch.nn as nn

from .ddpm import ddpm_schedules


class DPMSolverSampler(nn.Module):
    def __init__(
        self,
        eps_model: nn.Module,
        betas: Tuple[float, float],
        n_T: int,
    ) -> None:
        super().__init__()
        self.eps_model = eps_model
        self.n_T = n_T

        schedules = ddpm_schedules(betas[0], betas[1], n_T)
        log_alpha = torch.log(schedules["sqrtab"])
        log_sigma = torch.log(schedules["sqrtmab"])

        self.register_buffer("log_alpha_t", log_alpha)
        self.register_buffer("log_sigma_t", log_sigma)
        self.register_buffer("lambda_t", log_alpha - log_sigma)

    def _interpolate(self, values: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(t, device=values.device, dtype=values.dtype)
        index = t * self.n_T
        lower = torch.floor(index).long().clamp(0, self.n_T)
        upper = (lower + 1).clamp(0, self.n_T)
        weight = index - lower.to(index.dtype)
        return values[lower] + weight * (values[upper] - values[lower])

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(self._interpolate(self.log_alpha_t, t))

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(self._interpolate(self.log_sigma_t, t))

    def marginal_lambda(self, t: torch.Tensor) -> torch.Tensor:
        return self._interpolate(self.lambda_t, t)

    def inverse_lambda(self, lamb: torch.Tensor) -> torch.Tensor:
        lamb = torch.as_tensor(lamb, device=self.lambda_t.device, dtype=self.lambda_t.dtype)
        lambdas = torch.flip(self.lambda_t, dims=(0,))
        times = torch.linspace(1.0, 0.0, self.n_T + 1, device=lamb.device, dtype=lamb.dtype)

        upper = torch.searchsorted(lambdas.contiguous(), lamb.contiguous()).clamp(1, self.n_T)
        lower = upper - 1
        lamb_lower = lambdas[lower]
        lamb_upper = lambdas[upper]
        weight = (lamb - lamb_lower) / (lamb_upper - lamb_lower)
        return times[lower] + weight * (times[upper] - times[lower])

    def make_time_steps(
        self,
        n_steps: int,
        device: torch.device,
        start_time: float = 1.0,
        end_time: float = None,
        skip_type: str = "logsnr",
    ) -> torch.Tensor:
        end_time = 1.0 / self.n_T if end_time is None else end_time
        dtype = self.log_alpha_t.dtype
        if skip_type == "time":
            return torch.linspace(start_time, end_time, n_steps + 1, device=device, dtype=dtype)

        start = torch.tensor(start_time, device=device, dtype=dtype)
        end = torch.tensor(end_time, device=device, dtype=dtype)
        lambdas = torch.linspace(
            self.marginal_lambda(start),
            self.marginal_lambda(end),
            n_steps + 1,
            device=device,
            dtype=dtype,
        )
        return self.inverse_lambda(lambdas)

    def model_fn(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_batch = torch.ones(x.shape[0], device=x.device, dtype=x.dtype) * t.to(x.device, x.dtype)
        return self.eps_model(x, t_batch)

    def dpm_solver_first_update(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        eps_s: torch.Tensor = None,
    ) -> torch.Tensor:
        eps_s = self.model_fn(x, s) if eps_s is None else eps_s
        h = self.marginal_lambda(t) - self.marginal_lambda(s)
        return self.alpha(t) / self.alpha(s) * x - self.sigma(t) * torch.expm1(h) * eps_s

    def dpm_solver_second_update(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        r1: float = 0.5,
    ) -> torch.Tensor:
        eps_s = self.model_fn(x, s)
        h = self.marginal_lambda(t) - self.marginal_lambda(s)
        s1 = self.inverse_lambda(self.marginal_lambda(s) + r1 * h)

        u = self.alpha(s1) / self.alpha(s) * x - self.sigma(s1) * torch.expm1(r1 * h) * eps_s
        eps_s1 = self.model_fn(u, s1)
        return (
            self.alpha(t) / self.alpha(s) * x
            - self.sigma(t) * torch.expm1(h) * eps_s
            - self.sigma(t) / (2 * r1) * torch.expm1(h) * (eps_s1 - eps_s)
        )

    def dpm_solver_third_update(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        r1: float = 1.0 / 3.0,
        r2: float = 2.0 / 3.0,
    ) -> torch.Tensor:
        eps_s = self.model_fn(x, s)
        lambda_s = self.marginal_lambda(s)
        h = self.marginal_lambda(t) - lambda_s
        s1 = self.inverse_lambda(lambda_s + r1 * h)
        s2 = self.inverse_lambda(lambda_s + r2 * h)

        u1 = self.alpha(s1) / self.alpha(s) * x - self.sigma(s1) * torch.expm1(r1 * h) * eps_s
        eps_s1 = self.model_fn(u1, s1)
        d1 = eps_s1 - eps_s

        u2 = (
            self.alpha(s2) / self.alpha(s) * x
            - self.sigma(s2) * torch.expm1(r2 * h) * eps_s
            - self.sigma(s2)
            * r2
            / r1
            * (torch.expm1(r2 * h) / (r2 * h) - 1)
            * d1
        )
        eps_s2 = self.model_fn(u2, s2)
        d2 = eps_s2 - eps_s

        return (
            self.alpha(t) / self.alpha(s) * x
            - self.sigma(t) * torch.expm1(h) * eps_s
            - self.sigma(t) / r2 * (torch.expm1(h) / h - 1) * d2
        )

    def sample_from(
        self,
        x: torch.Tensor,
        time_steps: torch.Tensor,
        order: int,
    ) -> torch.Tensor:
        x_t = x
        for i in range(1, len(time_steps)):
            s, t = time_steps[i - 1], time_steps[i]
            if order == 1:
                x_t = self.dpm_solver_first_update(x_t, s, t)
            elif order == 2:
                x_t = self.dpm_solver_second_update(x_t, s, t)
            else:
                x_t = self.dpm_solver_third_update(x_t, s, t)
        return x_t

    def sample_from_orders(
        self,
        x: torch.Tensor,
        time_steps: torch.Tensor,
        orders: Iterable[int],
    ) -> torch.Tensor:
        x_t = x
        for i, order in enumerate(orders, start=1):
            s, t = time_steps[i - 1], time_steps[i]
            if order == 1:
                x_t = self.dpm_solver_first_update(x_t, s, t)
            elif order == 2:
                x_t = self.dpm_solver_second_update(x_t, s, t)
            else:
                x_t = self.dpm_solver_third_update(x_t, s, t)
        return x_t

    def sample_with_nfe(
        self,
        n_sample: int,
        size,
        device,
        nfe: int,
        order: int,
        skip_type: str = "logsnr",
    ) -> torch.Tensor:
        x = torch.randn(n_sample, *size).to(device)
        n_steps = nfe // order
        time_steps = self.make_time_steps(n_steps, device=x.device, skip_type=skip_type)
        return self.sample_from(x, time_steps, order)

    def sample(
        self,
        n_sample: int,
        size,
        device,
        nfe: int,
        order: int = 3,
        skip_type: str = "logsnr",
    ) -> torch.Tensor:
        return self.sample_with_nfe(n_sample, size, device, nfe, order, skip_type)

    def sample_fast(
        self,
        x: torch.Tensor,
        nfe: int,
        skip_type: str = "logsnr",
    ) -> torch.Tensor:
        n_steps = nfe // 3 + 1
        time_steps = self.make_time_steps(n_steps, device=x.device, skip_type=skip_type)
        remainder = nfe % 3

        if remainder == 0:
            orders = [3] * (n_steps - 2) + [2, 1]
        elif remainder == 1:
            orders = [3] * (n_steps - 1) + [1]
        else:
            orders = [3] * (n_steps - 1) + [2]

        return self.sample_from_orders(x, time_steps, orders)

    def sample_fast_with_nfe(
        self,
        n_sample: int,
        size,
        device,
        nfe: int,
        skip_type: str = "logsnr",
    ) -> torch.Tensor:
        x = torch.randn(n_sample, *size).to(device)
        return self.sample_fast(x, nfe=nfe, skip_type=skip_type)
