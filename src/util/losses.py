import torch


def segan_generator_loss(
    g_out: torch.Tensor, d_out: torch.Tensor, x_clean: torch.Tensor, l1_mag: int = 1
):
    return torch.mean((d_out - 1) ** 2) / 2 + l1_mag * torch.mean(
        torch.abs(g_out - x_clean)
    )


def segan_discriminator_loss(d_out_real: torch.Tensor, d_out_fake: torch.Tensor):
    return torch.mean((d_out_real - 1) ** 2) / 2 + torch.mean(d_out_fake**2) / 2
