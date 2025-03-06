import torch
import numpy as np

class Reconstruction:
    def __init__(self, unet, config):
        self.unet = unet
        self.config = config

    def _alpha(self, t):
        betas = torch.linspace(self.config.model.beta_start, self.config.model.beta_end, self.config.model.trajectory_steps, device=self.config.model.device)
        return (1 - torch.cat([torch.zeros(1, device=betas.device), betas])).cumprod(0)[t + 1].view(-1, 1, 1, 1)

    def _denoise_step(self, xt, t, next_t, w=0, y0=None):
        at, at_next = self._alpha(t.long()), self._alpha(next_t.long())
        et = self.unet(xt, t)
        yt = at.sqrt() * (y0 if y0 is not None else (xt - et * (1 - at).sqrt()) / at.sqrt()) + (1 - at).sqrt() * et
        et_hat = et - (1 - at).sqrt() * w * (yt - xt)
        x0_t = (xt - et_hat * (1 - at).sqrt()) / at.sqrt()
        c1 = self.config.model.eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
        return at_next.sqrt() * x0_t + c1 * torch.randn_like(xt) + ((1 - at_next) - c1 ** 2).sqrt() * et_hat

    def generate_y0(self, x, steps=None, skip=None, anomaly_strength=0):
        steps = steps or self.config.model.small_trajectory_steps
        skip = skip or self.config.model.small_skip
        seq = list(range(0, steps, skip))
        xt = self._alpha(torch.tensor([steps], device=x.device)).sqrt() * x + (1 - self._alpha(torch.tensor([steps], device=x.device))).sqrt() * torch.randn_like(x)
        for i, j in zip(reversed(seq), reversed([-1] + seq[:-1])):
            xt = self._denoise_step(xt, torch.ones(x.size(0), device=x.device) * i, torch.ones(x.size(0), device=x.device) * j)
        return xt

    def __call__(self, x, y0, w):
        y0_new = self.generate_y0(y0)
        xt = self._alpha(torch.tensor([self.config.model.test_trajectoy_steps], device=x.device)).sqrt() * x + (1 - self._alpha(torch.tensor([self.config.model.test_trajectoy_steps], device=x.device))).sqrt() * torch.randn_like(x)
        seq = list(range(0, self.config.model.test_trajectoy_steps, self.config.model.skip))
        return [xt := self._denoise_step(xt, torch.ones(x.size(0), device=x.device) * i, torch.ones(x.size(0), device=x.device) * j, w, y0_new) for i, j in zip(reversed(seq), reversed([-1] + seq[:-1]))], y0_new