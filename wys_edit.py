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
        out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)


def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    if vae_ckpt is not None:
        print(f"Loading VAE from {vae_ckpt}")
        vae_sd = torch.load(vae_ckpt, map_location="cpu")["state_dict"]
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

def remove_alpha_background(img: Image):
    # Check if the image has an alpha channel (RGBA)
    if img.mode == 'RGBA':
        # Get the image data
        data = img.getdata()
        # Create a new image with the same size and mode
        new_img = Image.new('RGB', img.size)
        # Update the RGB values where alpha is 0
        updated_data = [(r, g, b) if a != 0 else (255, 255, 255) for r, g, b, a in data]
        # Put the updated data into the new image
        new_img.putdata(updated_data)
    else:
        print("The image doesn't have an alpha channel.")
    return new_img


def main():
    parser = ArgumentParser()
    parser.add_argument("--resolution", default=512, type=int)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--config", default="configs/generate.yaml", type=str)
    parser.add_argument("--ckpt", default="checkpoints/instruct-pix2pix-00-22000.ckpt", type=str)
    parser.add_argument("--vae-ckpt", default=None, type=str)
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--edit", required=True, type=str)
    parser.add_argument("--cfg-text", default=7.5, type=float)
    parser.add_argument("--cfg-image", default=1.5, type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--threshold", default=0.0, type=float)
    args = parser.parse_args()
    print(args.cfg_image)

    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt, args.vae_ckpt)
    model.eval().cuda()
    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = model.get_learned_conditioning([""])
    # print(1. - model.betas)
    # print(len(model.betas))
    # print(model.alphas_cumprod)
    # print(len(model.alphas_cumprod))
    # print(model.sqrt_alphas_cumprod)
    # print(model.sqrt_one_minus_alphas_cumprod)

    seed = random.randint(0, 100000) if args.seed is None else args.seed
    print(seed)
    input_image = Image.open(args.input)#.convert("RGB")
    input_image = remove_alpha_background(input_image)
    width, height = input_image.size
    factor = args.resolution / max(width, height)
    factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
    width = int((width * factor) // 64) * 64
    height = int((height * factor) // 64) * 64
    input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)

    if args.edit == "":
        input_image.save(args.output)
        return

    with torch.no_grad(), autocast("cuda"), model.ema_scope():
        cond = {}
        cond["c_crossattn"] = [model.get_learned_conditioning([args.edit])]
        input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
        input_image = rearrange(input_image, "h w c -> 1 c h w").to(model.device)
        cond["c_concat"] = [model.encode_first_stage(input_image).mode()]

        uncond = {}
        uncond["c_crossattn"] = [null_token]
        uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

        sigmas = model_wrap.get_sigmas(args.steps)

        extra_args = {
            "cond": cond,
            "uncond": uncond,
            "text_cfg_scale": args.cfg_text,
            "image_cfg_scale": args.cfg_image,
        }

        extra_args_no_text = {
            "cond": {
                "c_crossattn": [model.get_learned_conditioning([""])],
                "c_concat": [model.encode_first_stage(input_image).mode()]
            },
            "uncond": {
                "c_crossattn": [null_token],
                "c_concat": [torch.zeros_like(cond["c_concat"][0])]
            },
            "text_cfg_scale": args.cfg_text,
            "image_cfg_scale": args.cfg_image,
        }
        torch.manual_seed(seed)

        # Forward process
        # t = 20
        # noise_t = torch.randn_like(cond["c_concat"][0]) * sigmas[t]
        # z_0 = model.encode_first_stage(input_image).mode()
        # z_t = z_0 + noise_t
        z_0 = model.encode_first_stage(input_image).mode()
        noise = torch.randn_like(z_0)
        t_rel = round(0.8 * args.steps)
        t_rel_fw = round(0.8*len(model.sqrt_alphas_cumprod))
        z_t_rel = model.sqrt_alphas_cumprod[t_rel_fw] * z_0 + model.sqrt_one_minus_alphas_cumprod[t_rel_fw] * noise
        # Relevance map
        s_in = z_t_rel.new_ones([z_t_rel.shape[0]])
        denoised = model_wrap_cfg(z_t_rel, sigmas[t_rel] * s_in, **extra_args)
        noise_pred = z_t_rel - denoised
        denoised_no_text = model_wrap_cfg(z_t_rel, sigmas[t_rel] * s_in, **extra_args_no_text)
        noise_pred_no_text = z_t_rel - denoised_no_text
        relevance_map = torch.abs(noise_pred - noise_pred_no_text)
        #IQR
        q1 = np.percentile(relevance_map.cpu(), 25)
        q3 = np.percentile(relevance_map.cpu(), 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        # Clamp and normalize
        clamped_rm = np.clip(relevance_map.cpu(), lower_bound, upper_bound)
        normalized_rm = (clamped_rm - torch.min(clamped_rm)) / (torch.max(clamped_rm) - torch.min(clamped_rm))
        # Edit mask
        threshold = args.threshold
        mask = (normalized_rm >= threshold).float()
        edit_mask = torch.zeros_like(normalized_rm).masked_fill(mask.bool(), 1.0).to("cuda")

        # Denoising stage
        z = torch.randn_like(z_0) * sigmas[0]
        z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas[:args.steps-t_rel+1], extra_args=extra_args)
        z = z * edit_mask + z_t_rel * (1-edit_mask)
        z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas[args.steps-t_rel:], extra_args=extra_args)
    
        x = model.decode_first_stage(z)
        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
        x = 255.0 * rearrange(x, "1 c h w -> h w c")
        edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())
    edited_image.save(args.output)


if __name__ == "__main__":
    main()
