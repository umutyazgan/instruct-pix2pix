"""Relevance map implementation"""
import torch
import torch.nn as nn
import k_diffusion as K
import numpy as np


class RelevanceMap:
    """Class implementing relevance map generation and editing with masks"""
    def __init__(
            self,
            denoiser_model: nn.Module,
            device: torch.device,
            z_0: torch.Tensor,
            sigmas: torch.Tensor,
            t: int,
            extra_args: dict,
            threshold: float = 0.5
        ):
        """
        Initializes the relevance map
        :param denoiser_model: Model used to denoise the latent from step t. (nn.Module)
        :param device: Device to store the edit mask. (torch.device)
        :param z_t: Encoded input image. (torch.Tensor)
        :param sigmas: Vector of noise levels, in descending order. (torch.Tensor)
        :param t: Noise level of the latent. (int)
        :param extra_args: Contains (un)conditioning and text/image scales. This has to contain
        "cond", "uncond", "text_cfg_scale" and "image_cfg_scale" fields. (dict)
        :param threshold: Threshold for the edit mask. (float)
        """
        self.denoiser_model = denoiser_model
        self.device = device
        self.t = t
        self.sigmas = sigmas
        self.noise = torch.randn_like(z_0)
        self.z_t = forward_process(z_0=z_0, noise=self.noise, sigma_t=self.sigmas[t])

        self.extra_args = extra_args

        cond_no_text = dict(self.extra_args["cond"])
        # Assuming extra_args["uncond"]["c_crossattn"] == [null_token], meaning conditioned on
        # empty string.
        cond_no_text["c_crossattn"] = self.extra_args["uncond"]["c_crossattn"]
        cond_no_text["c_concat"] = [z_0]

        self.extra_args_no_text = {
            "cond": cond_no_text,
            "uncond": self.extra_args["uncond"],
            "text_cfg_scale": self.extra_args["text_cfg_scale"],
            "image_cfg_scale": self.extra_args["image_cfg_scale"],
        }

        self.relevance_map = self.generate_relevance_map()
        self.edit_mask = self.generate_edit_mask(threshold=threshold)

    def generate_relevance_map(self) -> torch.Tensor:
        """
        Generates the relevance map. Denoises the noisy latent with and without text conditioning,
        calculates the substracted noise and returns the absolute difference between two noise
        tensors.
        :return: Relevance map. (torch.Tensor)
        """
        denoised_latent = unet_single_pass(
            model=self.denoiser_model,
            z_t=self.z_t,
            sigma_t=self.sigmas[self.t],
            extra_args=self.extra_args
        )
        predicted_noise = self.z_t - denoised_latent
        denoised_latent_no_text = unet_single_pass(
            model=self.denoiser_model,
            z_t=self.z_t,
            sigma_t=self.sigmas[self.t],
            extra_args=self.extra_args_no_text
        )
        predicted_noise_no_text = self.z_t - denoised_latent_no_text
        return torch.abs(predicted_noise - predicted_noise_no_text)

    def generate_edit_mask(self, threshold: float = 0.5) -> torch.Tensor:
        """
        Generates a latent binary edit mask from the relevance map.
        :param threshold: Threshold for setting values to 1 or 0. (float)
        :returns: A 1/0 tensor, acting as a latent edit mask. (torch.Tensor)
        """
        # IQR
        q1 = np.percentile(self.relevance_map.cpu(), 25)
        q3 = np.percentile(self.relevance_map.cpu(), 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        # Clamp and normalize
        clamped_rm = np.clip(self.relevance_map.cpu(), lower_bound, upper_bound)
        normalized_rm = (clamped_rm - torch.min(clamped_rm)) / (torch.max(clamped_rm) - torch.min(clamped_rm))
        # Apply threshold
        edit_mask = normalized_rm.clone().detach()
        edit_mask[normalized_rm >= threshold] = 1.
        edit_mask[normalized_rm < threshold] = 0.
        return edit_mask.to(self.device)


def forward_process(z_0: torch.Tensor, noise: torch.Tensor, sigma_t: float) -> torch.Tensor:
    """
    Applies noise the the encoded image until step t.
    :param z_0: Encoded input image. (torch.Tensor)
    :param noise: Noise to apply times sigma_t. Usually Gaussian noise. (torch.Tensor)
    :param sigma_t: Noise level for step t. (float)
    :return: t step noised latent. (torch.Tensor)
    """
    noisy_latent = z_0 + noise * K.utils.append_dims(sigma_t, z_0.ndim)
    return noisy_latent

@torch.no_grad()
def unet_single_pass(
    model: nn.Module, z_t: torch.Tensor, sigma_t: torch.Tensor, extra_args: dict = None
) -> torch.Tensor:
    """
    Does a single pass of the deniosing UNet and returns the denoised latent.
    :param model: Denoising UNet model. (nn.Module)
    :param z_t: Encoded noisy latent. (torch.Tensor)
    :param sigma_t: Noise levels of the z_t. (int)
    :param extra_args: Conditioning and unconditioning used for the UNet model. (dict)
    :return: Denoised latent. (torch.Tensor)
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = z_t.new_ones([z_t.shape[0]])
    denoised = model(z_t, sigma_t * s_in, **extra_args)  # mask unaware prediction
    return denoised
