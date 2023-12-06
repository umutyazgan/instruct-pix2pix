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
        self.inner_model = model

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


# model = CompVisDenoiser
def predict_noise(model, z, sigma, cond):
    t = model.sigma_to_t(sigma) 
    eps = model.get_eps(z, t, cond) # FIXME Out of memory
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
        # noisy_image = 2 * torch.tensor(np.array(noisy_image)).float() / 255 - 1
        # noisy_image = rearrange(noisy_image, "h w c -> 1 c h w").to(model.device)
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

        extra_args = {
            "cond": cond,
            "uncond": uncond,
            "text_cfg_scale": args.cfg_text,
            "image_cfg_scale": args.cfg_image,
        }

        torch.manual_seed(seed)
        # # TODO replace this random noise with a noised input image, also remove the decoding part below
        # encoded_noisy_image = torch.load(args.noisy)

        # decoded_noisy_image = model.decode_first_stage(encoded_noisy_image)
        # decoded_noisy_image = torch.clamp((decoded_noisy_image + 1.0) / 2.0, min=0.0, max=1.0)
        # decoded_noisy_image = 255.0 * rearrange(decoded_noisy_image, "1 c h w -> h w c")
        # decoded_noisy_image = Image.fromarray(decoded_noisy_image.type(torch.uint8).cpu().numpy())
        # decoded_noisy_image.save("../examples/decoded_noisy_image_2.png")

        # 0.8 noise + 0.2 image
        encoded_noisy_image = torch.randn_like(encoded_input_image) * 0.8 + encoded_input_image * 0.2

        z = encoded_noisy_image * sigmas[0]

        eps = predict_noise(model_wrap, z, sigmas, cond)

        # decoded_random_noise = model.decode_first_stage(torch.randn_like(cond["c_concat"][0]))
        # decoded_random_noise = torch.clamp((decoded_random_noise + 1.0) / 2.0, min=0.0, max=1.0)
        # decoded_random_noise = 255.0 * rearrange(decoded_random_noise, "1 c h w -> h w c")
        # decoded_random_noise = Image.fromarray(decoded_random_noise.type(torch.uint8).cpu().numpy())
        # decoded_random_noise.save("../examples/decoded_random_noise.png")

        # z = encoded_noisy_image * sigmas[0]
        # model_wrap_cfg = CFGDenoiser, CFG : Classifier Free Guidance
        # I think here z is just random noise and the line below creates a denoised image from the noise.
        # So this is the "reverse process". It's somehow conditioned on the input image, and unconditioned on empty string and empty image?
        # print(model_wrap_cfg)
        # print(model_wrap)
        # print(model)
        z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)
        x = model.decode_first_stage(z)
        # x = z  # No encode/decode
        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
        x = 255.0 * rearrange(x, "1 c h w -> h w c")
        edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())
    edited_image.save(args.output)


if __name__ == "__main__":
    main()
