"""Denoising with edit mask"""
import torch
import torch.nn as nn
import k_diffusion as K


def masked_denoiser(
        denoiser_model: nn.Module,
        edit_mask: torch.Tensor,
        sigmas: torch.Tensor,
        z_t: torch.Tensor,
        t: int,
        extra_args: dict,
    ) -> torch.Tensor:
    """
    Denoises a noisy latent using en edit mask.
    :denoiser_model: The model used by the sampler for the denoising task. (nn.Module)
    :param edit_mask: Binary edit mask to be applied during iterative denoising. (torch.Tensor)
    :param sigmas: Vector of noise levels, in descending order. (torch.Tensor)
    :param extra_args: Extra arguments for the sampler, such as conditioning and unconditioning.
    (dict)
    :return: Latent denoised with the edit mask and (un)conditioning. (torch.Tensor)
    """
    z = torch.randn_like(edit_mask)
    # print(t)
    # print(sigmas[t])

    z = K.sampling.sample_euler_ancestral(denoiser_model, z, sigmas[:t+2], extra_args=extra_args)
    z = z * edit_mask + z_t * (1 - edit_mask)
    z = K.sampling.sample_euler_ancestral(denoiser_model, z, sigmas[t+1:], extra_args=extra_args)
    return z
