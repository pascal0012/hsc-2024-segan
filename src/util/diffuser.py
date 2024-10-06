import torch


class Diffuser:
    """
    Class handling the diffusion process for Diffusion GANs.

    Args:
        t_min (int, optional): Minimum value for T. Defaults to 5.
        t_max (int, optional): Maximum value for T. Defaults to 500.
        sigma (float, optional): Scale for noise. Defaults to 0.05.
        C (int, optional): Step size for updating T. Defaults to 1.
        d_target (float, optional): Target value for r. Defaults to 0.6.
    """

    def __init__(
        self,
        device: torch.device,
        t_min: int = 5,
        t_max: int = 500,
        sigma: float = 0.24,
        C: int = 3,
        d_target: float = 0.8,
    ):
        self.device = device

        # Bounds and current value for T
        self.t_min = t_min
        self.t_max = t_max
        self.t_curr = t_min

        # Initialize list of t values to sample from
        self.update_t_list()

        # Initialize alpha values
        self.update_alphas()

        # Scale for noise
        self.sigma = sigma

        # Multiplier for new T
        self.C = C

        # Target value for r
        self.d_target = d_target

    def get_t_list(self) -> torch.Tensor:
        """
        Returns a list of t values to sample from with 32 zero
        elements and 32 elements uniformally sampled from 0 to T
        """
        return torch.cat([torch.zeros(32, dtype=torch.int32), self.t_list], dim=0).to(
            device=self.device
        )

    def update_t_list(self):
        """
        Updates the list of t values to sample from with 32 elements from 0 to T
        """
        self.t_list = torch.randint(0, self.t_curr, size=(32,))

    def sample_t_list(self, batch_size: int) -> torch.Tensor:
        """
        Returns batch_size t values sampled from the list of t values

        Args:
            batch_size (int): Number of t values to sample
        """

        t_list = self.get_t_list()
        indices = torch.randint(
            0, t_list.shape[0], size=(batch_size,), dtype=torch.int32
        ).to(device=self.device)
        return t_list[indices]

    def update_alphas(self):
        """
        Updates alpha_t values for current T
        """
        alphas = 1 - torch.linspace(1e-3, 0.02, self.t_curr, device=self.device)
        self.alphas = torch.cumprod(alphas, dim=0)

    def update_curr_t(self, d_out: torch.Tensor):
        """
        Calculates new T depending on last discriminator output and
        updates the list of t values and alpha_t accordingly

        Args:
            d_out (torch.Tensor): Last discriminator output
        """
        r = torch.mean(torch.sign(d_out - 0.5))

        self.t_curr = (self.t_curr + torch.sign(r - self.d_target) * self.C).int().item()

        if self.t_curr < self.t_min:
            self.t_curr = self.t_min

        if self.t_curr > self.t_max:
            self.t_curr = self.t_max

        self.update_t_list()
        self.update_alphas()

    def diffuse(self, d_in: torch.Tensor) -> torch.Tensor:
        """
        Diffuses input tensor with current parameters

        Args:
            d_in (torch.Tensor): Input tensor to diffuse
        """

        batch_size = d_in.shape[0]

        t_list = self.sample_t_list(batch_size)

        alpha_ts = self.alphas[t_list, None, None]

        epsilons = torch.normal(0, 1, size=d_in.shape, device=self.device)
        ys = (
            torch.sqrt(alpha_ts) * d_in
            + torch.sqrt(1 - alpha_ts) * self.sigma * epsilons
        )

        return ys
