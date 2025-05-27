import torch
import numpy as np
from net.NoiseNet import NoiseUnet
import matplotlib.pyplot as plt
from net.Discriminator import Discriminator

class NoiseLearner:
    def __init__(self, recon, degrad, discrm, L=10, sigma_low=0.01, sigma_high=1, device=None):
        self.recon = recon
        self.degrad = degrad
        self.discrm = discrm
        self.L = L
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.recon.to(self.device)
        self.degrad.to(self.device)
        self.discrm.to(self.device)

        # Generate a sequence of noise levels
        sigma = [sigma_high]
        for _ in range(L - 1):
            sigma.append(sigma[-1] * (sigma_low / sigma_high) ** (1 / (L - 1)))
        self.sigma = torch.tensor(sigma, device=self.device)

    def sample_from(self, X0, T=100, eps=2e-5):
        x_step = X0.to(self.device)
        x_hist = torch.zeros(self.L + 1, *x_step.shape, device=self.device)
        x_hist[0] = x_step

        with torch.no_grad():
            for i in range(self.L):
                alpha_i = eps * self.sigma[i] ** 2 / self.sigma[-1] ** 2
                for _ in range(T):
                    noise_level = self.sigma[i].expand(x_step.shape[0], 1)
                    noise_level = noise_level.view(-1, 1, 1, 1)
                    next_noise_level = self.sigma[i + 1].expand(x_step.shape[0], 1)
                    next_noise_level = next_noise_level.view(-1, 1, 1, 1)
                    x_step = self.degrad(self.recon(x_step, noise_level), next_noise_level)

                x_hist[i + 1] = x_step

        return x_step, x_hist

    def train(self, optimizer, epochs, dataloader, print_interval=1):
        print("Training Noise Learner...")
        self.recon.train()
        self.degrad.train()
        self.discrm.train()

        for epoch in range(epochs):
            total_loss = 0.0
            total_batches = 0
            c = 0
            for batch in dataloader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(self.device)

                batch_size = x.size(0)
                sigma_level_idx = torch.randint(0, self.L, (batch_size,), device=self.device)
                sigma_level = self.sigma[sigma_level_idx].unsqueeze(1)

                print(sigma_level[0])

                noise = torch.randn_like(x)
                # noise = self.NoiseNet(x, sigma_level)
                sigma_level = sigma_level.unsqueeze(1).unsqueeze(2)


                x_degradated = self.degrad(x, sigma_level, noise)

                optimizer.zero_grad()
                recon_pred = self.recon(x_degradated, sigma_level)



                loss = ((x - recon_pred) ** 2).mean() + torch.log(1 - self.discrm(x_degradated, x)).mean()

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_batches += 1

            if epoch % print_interval == 0:
                avg_loss = total_loss / total_batches
                print(f"Epoch {epoch + 1}/{epochs} - Avg Loss: {avg_loss:.6f}")