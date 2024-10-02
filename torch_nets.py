import torch
from torch import nn


class RunningMeanStd(nn.Module):
    def __init__(self, epsilon: float = 1e-4, shape=(), *args, **kwargs):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        super().__init__(*args, **kwargs)
        self.mean = nn.Parameter(
            torch.zeros(shape, dtype=torch.float), requires_grad=False
        )
        self.var = nn.Parameter(
            torch.ones(shape, dtype=torch.float), requires_grad=False
        )
        self.count = epsilon
        self.epsilon = nn.Parameter(torch.tensor(epsilon), requires_grad=False)

    def update(self, arr: torch.Tensor) -> None:
        batch_mean = torch.mean(arr, dim=0)
        batch_var = torch.var(arr, dim=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(
        self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_count: float
    ) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = (
            m_a
            + m_b
            + torch.square(delta)
            * self.count
            * batch_count
            / (self.count + batch_count)
        )
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean.data = new_mean
        self.var.data = new_var
        self.count = new_count

    def normalize(self, arr: torch.Tensor) -> torch.Tensor:
        return torch.clip(
            (arr - self.mean) / torch.sqrt(self.var + self.epsilon), -5, 5
        )

    def unnormalize(self, arr: torch.Tensor) -> torch.Tensor:
        return arr * torch.sqrt(self.var + self.epsilon) + self.mean


class VAE(nn.Module):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.args = args
        instance.kwargs = kwargs
        return instance

    def __init__(self, input_size: int, hidden_size: int, latent_size: int):
        super().__init__()
        self.input_rms = RunningMeanStd(shape=(input_size,))

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(input_size + hidden_size, latent_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(latent_size + hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(latent_size + hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(latent_size + hidden_size, input_size),
        )
        # Add mu and log_var layers for reparameterization
        self.mu = nn.Sequential(nn.LeakyReLU(), nn.Linear(latent_size, latent_size))
        self.log_var = nn.Sequential(
            nn.LeakyReLU(), nn.Linear(latent_size, latent_size)
        )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor):
        x = self.input_rms.normalize(x * 1)
        x = x.reshape(x.shape[0], -1)

        z = x * 1
        for j, op in enumerate(self.encoder):
            if isinstance(op, torch.nn.Linear):
                if j > 0:
                    z = torch.cat((x, z), dim=-1)
            z = op(z)
        z = z.reshape(z.shape[0], self.latent_size)

        return z

    def decode(self, z: torch.Tensor):
        z = z.reshape(z.shape[0], -1)

        x = z * 1
        for j, op in enumerate(self.decoder):
            if isinstance(op, torch.nn.Linear):
                if j > 0:
                    x = torch.cat((z, x), dim=-1)
            x = op(x)
        x = x.reshape(x.shape[0], self.input_size)
        return x

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        encoded = self.encode(x)
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        log_var = torch.clamp(log_var, -5, 5)
        z = self.reparameterize(mu, log_var)
        decoded = self.decode(z)
        return z, decoded, mu, log_var
