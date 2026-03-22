"""
Lab: Generative Models - VAE and Diffusion Model (Minimal)
===========================================================
Minimal implementations showing the core mechanics of:
1. Variational Autoencoder (VAE) with reparameterization trick
2. Denoising Diffusion Probabilistic Model (DDPM) training loop
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# =============================================================================
# VARIATIONAL AUTOENCODER
# =============================================================================
class VAE(nn.Module):
    """Minimal VAE for MNIST: Encoder → μ,σ → Reparameterize → Decoder."""

    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """The reparameterization trick: z = μ + σ * ε, ε ~ N(0,I)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    """ELBO = Reconstruction (BCE) + KL divergence."""
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction="sum")
    # KL(q(z|x) || p(z)) where p(z) = N(0,I)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def train_vae():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    model = VAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(5):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(data)
            loss = vae_loss(recon, data, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/5 | Avg Loss: {avg_loss:.4f}")

    return model


# =============================================================================
# MINIMAL DDPM TRAINING LOOP
# =============================================================================
class SimpleUNet(nn.Module):
    """Simplified U-Net for DDPM noise prediction (for demonstration)."""

    def __init__(self, channels=1, dim=64):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.down1 = nn.Sequential(nn.Conv2d(channels, dim, 3, padding=1), nn.GELU())
        self.down2 = nn.Sequential(nn.Conv2d(dim, dim * 2, 3, stride=2, padding=1), nn.GELU())
        self.mid = nn.Sequential(nn.Conv2d(dim * 2, dim * 2, 3, padding=1), nn.GELU())
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(dim * 2, dim, 4, stride=2, padding=1), nn.GELU()
        )
        self.out = nn.Conv2d(dim, channels, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_embed(t.float().unsqueeze(-1))
        h1 = self.down1(x)
        h2 = self.down2(h1)
        h2 = h2 + t_emb.unsqueeze(-1).unsqueeze(-1).expand_as(h2[:, : t_emb.size(1)])
        h = self.mid(h2)
        h = self.up1(h)
        return self.out(h)


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


def train_ddpm_step(model, x_0, timesteps=1000):
    """Single training step of DDPM: sample t, add noise, predict noise."""
    betas = linear_beta_schedule(timesteps).to(x_0.device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    batch_size = x_0.size(0)
    t = torch.randint(0, timesteps, (batch_size,), device=x_0.device)

    # Forward diffusion: q(x_t | x_0) = N(sqrt(α̅_t) x_0, (1-α̅_t) I)
    sqrt_alpha_cumprod = alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
    sqrt_one_minus = (1 - alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)

    noise = torch.randn_like(x_0)
    x_t = sqrt_alpha_cumprod * x_0 + sqrt_one_minus * noise

    # Predict the noise
    noise_pred = model(x_t, t / timesteps)
    loss = F.mse_loss(noise_pred, noise)
    return loss


if __name__ == "__main__":
    print("=" * 60)
    print("VAE TRAINING ON MNIST")
    print("=" * 60)
    model = train_vae()
    print("\nVAE Training complete!")

    print("\n" + "=" * 60)
    print("DDPM SINGLE TRAINING STEP DEMO")
    print("=" * 60)
    unet = SimpleUNet(channels=1, dim=32)
    dummy_images = torch.randn(4, 1, 28, 28)
    loss = train_ddpm_step(unet, dummy_images, timesteps=100)
    print(f"DDPM training step loss: {loss.item():.4f}")
