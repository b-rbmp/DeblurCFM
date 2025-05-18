import torch
import numpy as np
from tqdm.auto import tqdm

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule defines the noise level added at each diffusion step.
    """
    if beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    else:
        raise NotImplementedError(f"unknown beta schedule: {beta_schedule}")
    return betas

def get_alpha_schedule(betas):
    """
    Compute alphas and cumulative alphas from betas.
    """
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    return alphas, alphas_cumprod

class DDIMSampler:
    def __init__(self, model, beta_schedule="linear", beta_start=0.0001, beta_end=0.02, num_diffusion_timesteps=1000, device='cuda'):
        """
        Initializes the DDIM Sampler.

        Args:
            model: The trained U-Net model.
            beta_schedule: Type of beta schedule (e.g., 'linear').
            beta_start: Starting value for beta.
            beta_end: Ending value for beta.
            num_diffusion_timesteps: Total number of diffusion steps (T).
            device: The device to run the model and sampling on.
        """
        self.model = model
        self.device = device
        self.num_diffusion_timesteps = num_diffusion_timesteps

        betas = get_beta_schedule(
            beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            num_diffusion_timesteps=num_diffusion_timesteps,
        )

        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=self.device), self.alphas_cumprod[:-1]],
            dim=0,
        )
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)


    def _get_variance(self, timestep, prev_timestep, eta=0.0):
        """
        Compute the variance required for the DDIM step.
        """
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod_prev[prev_timestep+1] if prev_timestep >= 0 else torch.tensor(1.0, device=self.device) # Accounts for t=0 case
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # sigma_t = eta * sqrt((1 - alpha_{t-1}) / (1 - alpha_t)) * sqrt(1 - alpha_t / alpha_{t-1})
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        variance = variance.clamp(min=1e-20) # Avoid division by zero
        
        # Adjust variance based on eta
        sigma_t = eta * torch.sqrt(variance)
        return sigma_t


    @torch.no_grad()
    def sample(self, batch_size, image_size, num_inference_steps=50, eta=0.0, class_labels=None, condition=None):
        # Ensure condition has the correct batch size if provided
        if condition is not None:
            if condition.shape[0] != batch_size:
                raise ValueError("Condition batch size must match batch_size")
            # move condition to the correct device
            condition = condition.to(self.device)
        # - 1. Setup timesteps -
        # Use linear spacing for the sampling timesteps τ 
        step_ratio = self.num_diffusion_timesteps // num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        timesteps_pt = torch.from_numpy(timesteps).to(self.device)

        # - 2. Initial noise -
        shape = (batch_size,) + image_size
        latents = torch.randn(shape, device=self.device)

        # - 3. DDIM sampling loop -
        for i, t in enumerate(timesteps_pt):
            prev_t_idx = i + 1
            prev_t = timesteps_pt[prev_t_idx] if prev_t_idx < len(timesteps_pt) else -1 # -1 indicates the final step to x_0

            # - a. Prepare model input and predict noise component  -
            t_input = t.repeat(batch_size)

            # Prepare the input 'x' for the UNetModel
            if condition is not None:
                # Concatenate noisy latents and condition
                latents_input = torch.cat([latents, condition], dim=1) # Shape becomes (B, 6, H, W)
            else:
                latents_input = latents

            # Call the model
            model_output = self.model(x=latents_input, t=t_input, y=class_labels)

            # - b. Get predicted x_0 -
            # pred_x0 = (xt - sqrt(1 - alpha_t) * epsilon) / sqrt(alpha_t)
            sqrt_alpha_t = self.sqrt_alphas_cumprod[t]
            sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t]
            pred_original_sample = (latents - sqrt_one_minus_alpha_t * model_output) / sqrt_alpha_t
            pred_original_sample = torch.clamp(pred_original_sample, -1.0, 1.0) # Clamp prediction

            # - c. Compute x_{t-1} components -
            # 1. Direction pointing to x_t
            pred_epsilon = model_output
            sqrt_alpha_prev = self.alphas_cumprod_prev[prev_t+1] if prev_t >= 0 else torch.tensor(1.0, device=self.device) # sqrt(alpha_{t-1})
            sqrt_alpha_prev = torch.sqrt(sqrt_alpha_prev)

            # 2. Calculate variance term sigma_t
            sigma_t = self._get_variance(t, prev_t, eta)
            sqrt_one_minus_alpha_prev_minus_sigma_sq = torch.sqrt(
                (1.0 - self.alphas_cumprod_prev[prev_t+1] if prev_t >= 0 else 0.0) - sigma_t**2
            )
            sqrt_one_minus_alpha_prev_minus_sigma_sq = sqrt_one_minus_alpha_prev_minus_sigma_sq.clamp(min=0.0) # Ensure non-negative


            # - d. Compute x_{t-1} -
            # x_{t-1} = sqrt(alpha_{t-1}) * pred_x0 + sqrt(1 - alpha_{t-1} - sigma_t^2) * epsilon + sigma_t * noise
            term1 = sqrt_alpha_prev * pred_original_sample
            term2 = sqrt_one_minus_alpha_prev_minus_sigma_sq * pred_epsilon

            # Add noise if eta > 0
            if eta > 0:
                noise = torch.randn_like(latents)
                term3 = sigma_t * noise
            else:
                term3 = 0.0

            latents = term1 + term2 + term3

        # - 4. Return generated image -
        # Clamp to [-1, 1]
        latents = torch.clamp(latents, -1.0, 1.0)
        return latents