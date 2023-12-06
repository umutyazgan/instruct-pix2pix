from __future__ import annotations

import math
import random
import sys
from argparse import ArgumentParser

import einops
import k_diffusion as K
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import autocast

sys.path.append("./stable_diffusion")

from stable_diffusion.ldm.util import instantiate_from_config


class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model  # UNet

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        # print(f"cfg_z: {cfg_z.shape}, cfg_sigma: {cfg_sigma.shape}, c_concat: {cfg_cond['c_concat'][0].shape}, c_crossattn: {cfg_cond['c_crossattn'][0].shape}")
        out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)


@torch.no_grad()
def unet_single_pass(
    model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, mask=None, z_0=None, sigma_level=None
):
    """
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    denoised = model(x, sigmas[sigma_level] * s_in, **extra_args)  # mask unaware prediction

    return denoised

# model = CompVisDenoiser
def predict_noise(model, z, sigma, cond):
    t = model.sigma_to_t(sigma) 
    # TODO try using the unet model instead of this
    eps = model.get_eps(z, t, cond)
    return eps

def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu") # TODO may need to remove decoder related weights
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    if vae_ckpt is not None:
        print(f"Loading VAE from {vae_ckpt}")
        vae_sd = torch.load(vae_ckpt, map_location="cpu")["state_dict"] # TODO may need to remove decoder related weights
        sd = {
            k: vae_sd[k[len("first_stage_model.") :]] if k.startswith("first_stage_model.") else v
            for k, v in sd.items()
        }
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model


def main():
    parser = ArgumentParser()
    parser.add_argument("--resolution", default=512, type=int)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--config", default="configs/generate.yaml", type=str)
    parser.add_argument("--ckpt", default="checkpoints/instruct-pix2pix-00-22000.ckpt", type=str)
    parser.add_argument("--vae-ckpt", default=None, type=str)
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--noisy", required=False, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--edit", default="", required=False, type=str)
    parser.add_argument("--cfg-text", default=7.5, type=float)
    parser.add_argument("--cfg-image", default=1.5, type=float)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt, args.vae_ckpt)
    model.eval().cuda()
    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = model.get_learned_conditioning([""])

    seed = random.randint(0, 100000) if args.seed is None else args.seed

    # Original input image
    input_image = Image.open(args.input).convert("RGB")
    width, height = input_image.size
    factor = args.resolution / max(width, height)
    factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
    width = int((width * factor) // 64) * 64
    height = int((height * factor) // 64) * 64
    input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)

    # # Noisy image
    # noisy_image = Image.open(args.noisy).convert("RGB")
    # width, height = noisy_image.size
    # factor = args.resolution / max(width, height)
    # factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
    # width = int((width * factor) // 64) * 64
    # height = int((height * factor) // 64) * 64
    # noisy_image = ImageOps.fit(noisy_image, (width, height), method=Image.Resampling.LANCZOS)

    with torch.no_grad(), autocast("cuda"), model.ema_scope(): # .ema_scope() makes it so that the model is run with the EMA params
        cond = {}
        cond["c_crossattn"] = [model.get_learned_conditioning([args.edit])]
        input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
        input_image = rearrange(input_image, "h w c -> 1 c h w").to(model.device)


        encoded_input_image = model.encode_first_stage(input_image).mode()
        cond["c_concat"] = [encoded_input_image]
        # NOTE This is a DiagonalGaussianDistribution().mode(), which actually returns the mean...
        # If we set z = cond["c_concat"] and skip from here to decode_first_stage(z), we get the same image but a bit disorted.
        uncond = {}
        uncond["c_crossattn"] = [null_token]
        uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]
        # TODO Figure out what below line does
        # model_wrap = K.external.CompVisDenoiser(model); model = ddpm_edit.LatentDiffusion
        # CompVisDenoiser is a type of DiscreteEpsDDPMDenoiser, which is a type of DiscreteSchedule
        # DiscreteSchedule: A mapping between continuous noise levels (sigmas) and a list of discrete noise levels
        # .get_sigmas returns those sigmas. So I assume this has something to do with the noise schedule.
        sigmas = model_wrap.get_sigmas(args.steps)
        # TODO: Find out what do sigmas look like

        cond_no_text = {}
        cond_no_text["c_crossattn"] = [model.get_learned_conditioning([""])]
        cond_no_text["c_concat"] = [encoded_input_image]

        extra_args = {
            "cond": cond,
            "uncond": uncond,
            "text_cfg_scale": args.cfg_text,
            "image_cfg_scale": args.cfg_image,
        }

        extra_args_no_text = {
            "cond": cond_no_text,
            "uncond": uncond,
            "text_cfg_scale": args.cfg_text,
            "image_cfg_scale": args.cfg_image,
        }

        torch.manual_seed(0)

        # TODO Figure out how to get the actual noised image at step t, rather than giving an arbitrary amount of noise
        sigma_level = 10 # NOTE small number means more noise
        noise = torch.randn_like(encoded_input_image)
        noisy_latent = encoded_input_image + noise * K.utils.append_dims(sigmas[sigma_level], encoded_input_image.ndim)
        # NOTE maybe we need to multiply with sigma[0] to sigma[20] or sigma[20] to sigma[100], instad of just multiplying with simga[20]

        # NOTE why multiply with sigmas[0]?
        z = noisy_latent# * sigmas[0]

        # eps = predict_noise(model_wrap, z, torch.Tensor([sigmas[sigma_level]]).to(device="cuda"), cond)
        # eps_no_text = predict_noise(model_wrap, z, torch.Tensor([sigmas[sigma_level]]).to(device="cuda"), cond_no_text)
        print(sigmas[sigma_level:].shape)
        # denoised_latent = K.sampling.sample_euler_ancestral(model_wrap_cfg, noisy_latent, sigmas[sigma_level:], extra_args=extra_args)
        denoised_latent = unet_single_pass(model_wrap_cfg, noisy_latent, sigmas, extra_args=extra_args, sigma_level=sigma_level)
        predicted_noise = noisy_latent - denoised_latent
        # denoised_latent_no_text = K.sampling.sample_euler_ancestral(model_wrap_cfg, noisy_latent, sigmas[sigma_level:], extra_args=extra_args_no_text)
        denoised_latent_no_text = unet_single_pass(model_wrap_cfg, noisy_latent, sigmas, extra_args=extra_args_no_text, sigma_level=sigma_level)
        predicted_noise_no_text = noisy_latent - denoised_latent_no_text

        relevance_map = torch.abs(predicted_noise - predicted_noise_no_text)

        x = model.decode_first_stage(relevance_map)

        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
        x = 255.0 * rearrange(x, "1 c h w -> h w c")
        x = Image.fromarray(x.type(torch.uint8).cpu().numpy())

        x.save("relevance_maps/normalized_single_pass/relevance_map.png")

        q1 = np.percentile(relevance_map.cpu(), 25)
        q3 = np.percentile(relevance_map.cpu(), 75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        clamped_relevance_map = np.clip(relevance_map.cpu(), lower_bound, upper_bound)

        normalized_relevance_map = (clamped_relevance_map - torch.min(clamped_relevance_map)) / (torch.max(clamped_relevance_map) - torch.min(clamped_relevance_map))
        edit_mask = normalized_relevance_map
        edit_mask[edit_mask > 0.5] = 1
        edit_mask[edit_mask <= 0.5] = 0
        edit_mask = edit_mask.to("cuda")

        # Denoising stage
        z = noise
        z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args, mask=edit_mask, z_0=encoded_input_image)

        x = model.decode_first_stage(z)

        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
        x = 255.0 * rearrange(x, "1 c h w -> h w c")
        x = Image.fromarray(x.type(torch.uint8).cpu().numpy())

    x.save(args.output)


if __name__ == "__main__":
    main()
