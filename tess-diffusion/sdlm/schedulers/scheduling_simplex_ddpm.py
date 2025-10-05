"""DDPM scheduler for the simplex diffusion model."""

from diffusers import DDPMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMSchedulerOutput
from dataclasses import dataclass
from typing import Union, Tuple, Optional
import torch
import numpy as np
from diffusers.configuration_utils import register_to_config
from diffusers.utils import BaseOutput
import math
import pdb


@dataclass
class SimplexDDPMSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's step function output.
    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample (x_{t-1}) of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        projected_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, vocab_size)`):
            The projected logits sample (x_{0}) based on the model output from the current timestep.
    """

    prev_sample: torch.FloatTensor
    projected_logits: Optional[torch.FloatTensor] = None


def betas_for_alpha_bar(num_diffusion_timesteps, device, max_beta=0.999, improved_ddpm=False):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].
    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.
    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """

    def default_alpha_bar(time_step):
        return math.cos((time_step + 1e-4) / (1 + 1e-4) * math.pi / 2) ** 2

    if improved_ddpm:
        # Implements eqn. 17 in https://arxiv.org/pdf/2102.09672.pdf.
        alpha_bar = lambda x: (default_alpha_bar(x) / default_alpha_bar(0.0))
        alphas_cumprod = []
    else:
        alpha_bar = default_alpha_bar
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        alpha_bar_t1 = alpha_bar(t1)
        betas.append(min(1 - alpha_bar(t2) / alpha_bar_t1, max_beta))
        if improved_ddpm:
            alphas_cumprod.append(alpha_bar_t1)
    # TODO(rabeeh): maybe this cause memory issue.
    betas = torch.tensor(betas, dtype=torch.float32, device=device)
    if improved_ddpm:
        return betas, torch.tensor(alphas_cumprod, dtype=torch.torch.float32, device=device)
    return betas


class SimplexDDPMScheduler(DDPMScheduler):
    @register_to_config
    def __init__(
        self,
        device,
        simplex_value: float,
        num_train_timesteps: int = 1000,
        num_inference_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[np.ndarray] = None,
        variance_type: str = "fixed_small",
        clip_sample: bool = False,        
    ):
        if trained_betas is not None:
            self.betas = torch.from_numpy(trained_betas)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32, device=device)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = (
                torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32, device=device)
                ** 2
            )
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(num_train_timesteps, device=device)
        elif beta_schedule == "squaredcos_improved_ddpm":
            self.betas, self.alphas_cumprod = betas_for_alpha_bar(num_train_timesteps, device=device, improved_ddpm=True)
        elif beta_schedule == "sigmoid":
            # GeoDiff sigmoid schedule
            betas = torch.linspace(-6, 6, num_train_timesteps, device=device)
            self.betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        if beta_schedule == "squaredcos_improved_ddpm":
            self.alphas = None
        else:
            self.alphas = 1.0 - self.betas
            self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.one = torch.tensor(1.0, device=device)

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # setable values
        self.num_inference_steps = None
        # TODO(rabeeh): if memory issue, we can not add this to GPU and convert them iteratively.
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy()).to(device=device)

        self.variance_type = variance_type

    def step(
        self,
        projected_logits: torch.FloatTensor,
        timestep: int,
        noise: torch.FloatTensor,
        generator=None,
    ) -> Union[DDPMSchedulerOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).
        Args:
            projected_logits (`torch.FloatTensor`): projected logits from the diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            noise (`torch.FloatTensor`): a random noise with simplex_value standard deviation.
            generator: random number generator.
        Returns:
            [`~schedulers.scheduling_utils.DDPMSchedulerOutput`] resulted values.
        """
        t = timestep

        # 1. compute alphas, betas
        alpha_prod_t_prev = self.alphas_cumprod[t - 1] if t > 0 else self.one

        # 3. Clip "predicted x_0"
        if self.config.clip_sample:
            projected_logits = torch.clamp(projected_logits, -1, 1)

        # See algorithm 2 in Figure 3 in https://arxiv.org/pdf/2210.17432.pdf.
        predicted_logits_coeff = alpha_prod_t_prev ** (0.5)
        noise_coeff = (1 - alpha_prod_t_prev) ** (0.5)
        pred_prev_sample = predicted_logits_coeff * projected_logits + noise_coeff * noise

        return SimplexDDPMSchedulerOutput(prev_sample=pred_prev_sample, projected_logits=projected_logits)

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        # self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        # timesteps = timesteps.to(original_samples.device)

        alphas_cumprod_timesteps = self.alphas_cumprod[timesteps].view(-1, 1, 1)
        sqrt_alpha_prod = alphas_cumprod_timesteps**0.5
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod_timesteps) ** 0.5
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

class ModifiedSimplexDDPMScheduler(DDPMScheduler):
    @register_to_config
    def __init__(
        self,
        device,
        simplex_value: float,
        num_train_timesteps: int = 1000,
        num_inference_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[np.ndarray] = None,
        variance_type: str = "fixed_small",
        clip_sample: bool = False,
        token_schedule: str = "linear_l2r",
        token_alpha: float = 1.0,
        token_beta: float = 0.0,
        ddim: bool = False,
        eta: float = 0.0,
        schedule_warm_up_step: int = 0,
        monotonic: bool = False,
        ignore_prompts: bool = False,
    ):
        if trained_betas is not None:
            self.betas = torch.from_numpy(trained_betas)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32, device=device)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = (
                torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32, device=device)
                ** 2
            )
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(num_train_timesteps, device=device)
        elif beta_schedule == "squaredcos_improved_ddpm":
            self.betas, self.alphas_cumprod = betas_for_alpha_bar(num_train_timesteps, device=device, improved_ddpm=True)
        elif beta_schedule == "sigmoid":
            # GeoDiff sigmoid schedule
            betas = torch.linspace(-6, 6, num_train_timesteps, device=device)
            self.betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        if beta_schedule == "squaredcos_improved_ddpm":
            self.alphas = None
        else:
            self.alphas = 1.0 - self.betas
            self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.one = torch.tensor(1.0, device=device)

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # setable values
        self.num_inference_steps = None
        # TODO(rabeeh): if memory issue, we can not add this to GPU and convert them iteratively.
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy()).to(device=device)

        self.variance_type = variance_type
        self.token_schedule = token_schedule
        self.token_alpha = token_alpha
        self.token_beta = token_beta
        self.ddim = ddim
        self.eta = eta
        self.token_grad = None
        self.schedule_warm_up_step = schedule_warm_up_step
        self.monotonic = monotonic
        self.previous_token_timesteps = None
        self.ignore_prompts = ignore_prompts
   

    def reset_token_timesteps(self):
        self.previous_token_timesteps = None

    def step(
        self,
        projected_logits: torch.FloatTensor,
        timestep: int,
        noise: torch.FloatTensor,
        generator=None,
        token_grad=None,
        schedule_warm_up_step=None,
        prompt_mask: Optional[torch.BoolTensor] = None,
    ) -> Union[DDPMSchedulerOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).
        Args:
            projected_logits (`torch.FloatTensor`): projected logits from the diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            noise (`torch.FloatTensor`): a random noise with simplex_value standard deviation.
            generator: random number generator.
            token_grad (`torch.FloatTensor`): gradient values per token, used for adaptive scheduling.
        Returns:
            [`~schedulers.scheduling_utils.DDPMSchedulerOutput`] resulted values.
        """
        t = timestep
        batch_size, seq_len = projected_logits.shape[:2]
        # print(f"logit shape: {projected_logits.shape}")

        # Get token-specific timesteps for the batch
        self.token_grad = token_grad
        self.current_prompt_mask = prompt_mask if self.ignore_prompts else None
        if self.ignore_prompts:
            print("ignoring prompt") 
        
        token_timesteps = self.get_token_timestep(t, seq_len, batch_size=batch_size)

        # # Apply prompt masking if enabled
        # if self.ignore_prompts and prompt_mask is not None:
        #     assert prompt_mask.shape == (batch_size, seq_len), "Invalid prompt_mask shape"
        #     token_timesteps = token_timesteps.clone()  # ensure we don't overwrite anything
        #     token_timesteps[prompt_mask] = self.num_train_timesteps - 1  # no noise for prompt
        

        # print(self.monotonic)
        # Apply monotonic constraint
        if self.monotonic:
            if self.previous_token_timesteps is None:
                self.previous_token_timesteps = token_timesteps.clone()
            else:
                # Compute the mask of affected tokens
                # erase when decoding
                # affected_mask = token_timesteps > self.previous_token_timesteps  # shape: (batch_size, seq_len)
                # num_affected = affected_mask.sum().item()
                # total_tokens = token_timesteps.numel()
                # ratio_affected = num_affected / total_tokens

                # print(f"[Monotonic Constraint] {num_affected} / {total_tokens} tokens affected ({ratio_affected:.2%})")

                # Apply monotonic constraint
                token_timesteps = torch.minimum(token_timesteps, self.previous_token_timesteps)
                self.previous_token_timesteps = token_timesteps.clone()
        # print(f"token_timesteps shape: {token_timesteps.shape}")
        #print(token_timesteps.shape)    

        # if t%10 == 0:
        #     print(f"token_timesteps: {token_timesteps}")    

        if t > 0:
            # Flatten token_timesteps for proper indexing
            #flat_token_timesteps = token_timesteps.view(-1)  # Shape: (batch_size * seq_len,)
            #print(flat_token_timesteps.shape)
            alphas_prod_t_prev = self.alphas_cumprod[token_timesteps - 1]  # Shape: (batch_size * seq_len,)
            alphas_prod_t_prev = alphas_prod_t_prev.view(batch_size, seq_len, 1)  # Reshape back
        else:
            # Handle the case where t == 0
            alphas_prod_t_prev = torch.ones((batch_size, seq_len, 1), device=projected_logits.device)

        # Clip "predicted x_0" if needed
        if self.config.clip_sample:
            projected_logits = torch.clamp(projected_logits, -1, 1)

        if self.ddim:
            # Use DDIM deterministic or semi-deterministic sampling            
            alphas_prod_t = self.alphas_cumprod[token_timesteps]
            alphas_prod_t = alphas_prod_t.view(batch_size, seq_len, 1)
            ddim_coeff = (1 - alphas_prod_t / (alphas_prod_t_prev + 1e-9)) ** 0.5
            predicted_logits_coeff = (alphas_prod_t_prev + 1e-6) ** 0.5
            ddim_noise = self.eta * ddim_coeff * torch.randn_like(projected_logits)

            projected_logits = torch.clamp(projected_logits, min=-1e9, max=1e9)
            alphas_prod_t_prev = torch.clamp(alphas_prod_t_prev, min=1e-9, max=1.0)
            alphas_prod_t = torch.clamp(alphas_prod_t, min=1e-9, max=1.0)

            # Compute x_t-1 using DDIM formula
            pred_prev_sample = (
                predicted_logits_coeff * projected_logits + ddim_coeff * noise + ddim_noise
            )
            # print(f"projected_logits: {projected_logits}")
            # print(f"pred_prev_sample: {pred_prev_sample}")

        else:
            # Use DDPM stochastic sampling
            predicted_logits_coeff = alphas_prod_t_prev ** 0.5
            noise_coeff = (1 - alphas_prod_t_prev) ** 0.5
            pred_prev_sample = (
                predicted_logits_coeff * projected_logits + noise_coeff * noise
            )

        return SimplexDDPMSchedulerOutput(
            prev_sample=pred_prev_sample, projected_logits=projected_logits
        )

    def get_token_timestep(self, timestep: int, seq_len: int, batch_size: int = 1) -> torch.IntTensor:
        max_steps = self.num_train_timesteps
        prompt_mask = getattr(self, "current_prompt_mask", None)

        ## timestep is constant when larger then warm up step
        if timestep > (self.num_train_timesteps - self.schedule_warm_up_step):
            return torch.full((batch_size, seq_len), timestep, dtype=torch.long, device=self.device)

        if self.token_schedule == "constant":
            return torch.full((batch_size, seq_len), timestep, dtype=torch.long, device=self.device)

        if self.token_schedule == "scaled":
            token_steps = self.token_alpha * timestep + self.token_beta
            token_steps = torch.clamp(token_steps.long(), 1, max_steps - 1)
            return token_steps.unsqueeze(0).expand(batch_size, seq_len)

        if self.token_schedule == "linear":
            token_steps = torch.zeros(batch_size, seq_len, dtype=torch.float32, device=self.device)
            for x in range(seq_len):
                token_steps[:, x] = timestep - (seq_len - x - 1) * (max_steps - timestep) * self.token_alpha + self.token_beta
            token_steps = torch.clamp(token_steps.long(), 1, max_steps - 1)
            return token_steps.unsqueeze(0).expand(batch_size, seq_len)
        
        if self.token_schedule == "linear_l2r":
            if self.ignore_prompts and prompt_mask is not None:
                token_steps = torch.full((batch_size, seq_len), self.num_train_timesteps - 1, dtype=torch.long, device=self.device)

                num_gen_tokens = (~prompt_mask).sum(dim=1).max().item()
                l2r_steps = torch.linspace(timestep, 0, steps=num_gen_tokens, dtype=torch.float32, device=self.device)
                l2r_steps = self.token_alpha * l2r_steps + self.token_beta
                l2r_steps = torch.clamp(l2r_steps.long(), 1, max_steps - 1)

                for b in range(batch_size):
                    gen_indices = (~prompt_mask[b]).nonzero(as_tuple=False).squeeze(1)
                    fill_vals = l2r_steps[-len(gen_indices):]  # right-align if shorter than max
                    token_steps[b, gen_indices] = fill_vals
            else:
                token_steps = torch.linspace(timestep, 0, seq_len, dtype=torch.float32, device=self.device)
                token_steps = self.token_alpha * token_steps + self.token_beta
                token_steps = torch.clamp(token_steps.long(), 1, max_steps - 1)
                token_steps = token_steps.unsqueeze(0).expand(batch_size, seq_len)

            return token_steps

        if self.token_schedule == "linear_r2l":
            if self.ignore_prompts and prompt_mask is not None:
                token_steps = torch.full((batch_size, seq_len), self.num_train_timesteps - 1, dtype=torch.long, device=self.device)

                num_gen_tokens = (~prompt_mask).sum(dim=1).max().item()
                r2l_steps = torch.linspace(0, timestep, steps=num_gen_tokens, dtype=torch.float32, device=self.device)
                r2l_steps = self.token_alpha * r2l_steps + self.token_beta
                r2l_steps = torch.clamp(r2l_steps.long(), 1, max_steps - 1)

                for b in range(batch_size):
                    gen_indices = (~prompt_mask[b]).nonzero(as_tuple=False).squeeze(1)
                    fill_vals = r2l_steps[-len(gen_indices):]  # right-align if shorter than max
                    token_steps[b, gen_indices] = fill_vals
            else:
                token_steps = torch.linspace(0, timestep, seq_len, dtype=torch.float32, device=self.device)
                token_steps = self.token_alpha * token_steps + self.token_beta
                token_steps = torch.clamp(token_steps.long(), 1, max_steps - 1)
                token_steps = token_steps.unsqueeze(0).expand(batch_size, seq_len)

            return token_steps

        if self.token_schedule == "linear_with_warmup":
            token_steps = torch.zeros(batch_size, seq_len, dtype=torch.float32, device=self.device)
            for x in range(seq_len):
                token_steps[:, x] = timestep - (seq_len - x - 1) * (max_steps - timestep) * self.token_alpha + self.token_beta
            token_steps = torch.clamp(token_steps.long(), 1, timestep)
            return token_steps

        if self.token_schedule == "wave_r2l":
            token_steps = torch.linspace(max_steps - timestep, timestep, seq_len, dtype=torch.float32, device=self.device)
            token_steps = self.token_alpha * token_steps + self.token_beta
            token_steps = torch.clamp(token_steps.long(), 1, max_steps - 1)
            return token_steps.unsqueeze(0).expand(batch_size, seq_len)

        if self.token_schedule == "wave_l2r":
            token_steps = torch.linspace(timestep, max_steps - timestep, seq_len, dtype=torch.float32, device=self.device)
            token_steps = self.token_alpha * token_steps + self.token_beta
            token_steps = torch.clamp(token_steps.long(), 1, max_steps - 1)
            return token_steps.unsqueeze(0).expand(batch_size, seq_len)

        if self.token_schedule == "adaptive_grad":
            if self.token_grad is None:
                token_steps = torch.full((batch_size, seq_len), timestep, dtype=torch.long, device=self.device)
            else:
                grad = self.token_grad.clone()
                if self.ignore_prompts and prompt_mask is not None:
                    grad = grad.masked_fill(prompt_mask, float("-inf"))
                normalize_grad = (grad - grad.amin(dim=1, keepdim=True)) / (
                    grad.amax(dim=1, keepdim=True) - grad.amin(dim=1, keepdim=True) + 1e-8
                )
                adaptive_timestep = (1 - normalize_grad) * timestep
                smooth_factor = self.token_alpha
                final_timestep = smooth_factor * timestep + (1 - smooth_factor) * adaptive_timestep
                token_steps = torch.clamp(final_timestep.long(), 1, max_steps - 1)

            if self.ignore_prompts and prompt_mask is not None:
                token_steps = token_steps.clone()
                token_steps[prompt_mask] = self.num_train_timesteps - 1
            return token_steps

        if self.token_schedule == "adaptive_grad_max":
            if self.token_grad is None:
                return torch.full((batch_size, seq_len), timestep, dtype=torch.long, device=self.device)
            else:
                # Assume self.token_grad has shape (batch_size, seq_len)
                normalize_grad = (self.token_grad - self.token_grad.min(dim=1, keepdim=True)[0]) / \
                                (self.token_grad.max(dim=1, keepdim=True)[0] - self.token_grad.min(dim=1, keepdim=True)[0] + 1e-8)

                adaptive_timestep = (1 - normalize_grad) * self.num_train_timesteps
                smooth_factor = self.token_alpha
                final_timestep = smooth_factor * self.num_train_timesteps + (1 - smooth_factor) * adaptive_timestep

                final_timestep = torch.clamp(final_timestep.long(), 1, max_steps - 1)

                return final_timestep.long()
        
        if self.token_schedule == "topk_grad":
            if self.token_grad is None:
                return torch.full((batch_size, seq_len), timestep, dtype=torch.long, device=self.device)
            else:
                k = int(seq_len * self.token_beta)
                topk = torch.topk(self.token_grad, k=k, dim=1).indices
                mask = torch.full_like(self.token_grad, fill_value=timestep)
                batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, k)
                mask[batch_indices, topk] = 1
                return mask.long()

        if self.token_schedule == "topp_grad":
            if self.token_grad is None:
                return torch.full((batch_size, seq_len), timestep, dtype=torch.long, device=self.device)
            else:
                mask = torch.full_like(self.token_grad, fill_value=timestep)
                for b in range(batch_size):
                    grad_norms = self.token_grad[b]
                    sorted_vals, sorted_idx = torch.sort(grad_norms, descending=True)
                    cum_weights = torch.cumsum(sorted_vals, dim=0) / (sorted_vals.sum() + 1e-8)
                    cutoff = (cum_weights <= self.token_beta).sum().item()
                    top_indices = sorted_idx[:cutoff]
                    mask[b, top_indices] = 1
                return mask.long()
        
        if self.token_schedule == "softmax_grad":
            if self.token_grad is None:
                return torch.full((batch_size, seq_len), timestep, dtype=torch.long, device=self.device)
            else:
                weight = torch.softmax(self.token_grad / self.token_beta, dim=1)
                step_weight = 1 * weight + timestep * (1 - weight)
                return step_weight.clamp(min=1, max=self.num_train_timesteps - 1).long()
        
        if self.token_schedule == "threshold_grad":
            if self.token_grad is None:
                return torch.full((batch_size, seq_len), timestep, dtype=torch.long, device=self.device)
            else:
                mean = self.token_grad.mean(dim=1, keepdim=True)
                std = self.token_grad.std(dim=1, keepdim=True)
                threshold = mean + self.token_beta * std
                mask = torch.where(self.token_grad >= threshold, torch.ones_like(self.token_grad), torch.full_like(self.token_grad, timestep))
                return mask.long()

        if self.token_schedule == "cumgrad_ratio":
            if self.token_grad is None:
                return torch.full((batch_size, seq_len), timestep, dtype=torch.long, device=self.device)
            else:
                mask = torch.full_like(self.token_grad, fill_value=timestep)
                for b in range(batch_size):
                    grad_vals = self.token_grad[b]
                    sorted_vals, sorted_idx = torch.sort(grad_vals, descending=True)
                    cum_ratio = torch.cumsum(sorted_vals, dim=0) / (sorted_vals.sum() + 1e-8)
                    cutoff = (cum_ratio <= self.token_beta).sum().item()
                    top_indices = sorted_idx[:cutoff]
                    mask[b, top_indices] = 1
                return mask.long()
        
        if self.token_schedule == "grad_pos":
            if self.token_grad is None:
                return torch.full((batch_size, seq_len), timestep, dtype=torch.long, device=self.device)
            else:
                pos = torch.arange(seq_len, dtype=torch.float32, device=self.device) / (seq_len - 1)
                pos = 1 + self.token_alpha * pos  # broadcastable: shape (seq_len,)
                importance = self.token_grad * pos.unsqueeze(0)
                k = int(seq_len * self.token_beta)
                topk = torch.topk(importance, k=k, dim=1).indices
                mask = torch.full_like(self.token_grad, fill_value=timestep)
                batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, k)
                mask[batch_indices, topk] = 1
                return mask.long()

        if self.token_schedule == "adaptive_grad_reverse":
            if self.token_grad is None:
                return torch.full((batch_size, seq_len), timestep, dtype=torch.long, device=self.device)
            else:                
                grad_min = self.token_grad.min(dim=1, keepdim=True)[0]
                grad_max = self.token_grad.max(dim=1, keepdim=True)[0]                
               
                inverse_normalize_grad = (grad_max - self.token_grad) / (grad_max - grad_min + 1e-8)                
               
                adaptive_timestep = inverse_normalize_grad * timestep                
                
                smooth_factor = self.token_alpha
                final_timestep = smooth_factor * timestep + (1 - smooth_factor) * adaptive_timestep
                
                final_timestep = torch.clamp(final_timestep.long(), 1, max_steps - 1)

                return final_timestep.long()

        if self.token_schedule == "random":
            return torch.randint(0, max_steps - 1, (batch_size, seq_len), dtype=torch.long, device=self.device)

        # Add similar logic for other schedules as needed, ensuring the `batch_size` dimension is accounted for.
        raise ValueError(f"Unknown token schedule: {self.token_schedule}")



    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        # self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        # timesteps = timesteps.to(original_samples.device)

        alphas_cumprod_timesteps = self.alphas_cumprod[timesteps].view(-1, 1, 1)
        sqrt_alpha_prod = alphas_cumprod_timesteps**0.5
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod_timesteps) ** 0.5
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
