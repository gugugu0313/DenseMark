"""
Test with:
    python -m watermark_anything.augmentation.repair
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionInpaintPipeline,
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel,
    StableDiffusionXLInpaintPipeline,
    RePaintPipeline,
    RePaintScheduler,
    StableDiffusionInstructPix2PixPipeline,
    EulerAncestralDiscreteScheduler,
    DDIMScheduler
)

# ==========================================
# 1. Stable Diffusion v2 Inpainter
# ==========================================
class SDInpainter:
    def __init__(self, device="cuda"):
        self.device = device
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "sd2-community/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16,
        ).to(device)
        print(f"Initialized SDInpainter on {device}")

    def process(self, image_tensor, mask_tensor, prompt=""):
        b, c, h, w = image_tensor.shape
        # numpy for PIL
        image_batch = image_tensor.permute(0, 2, 3, 1).detach().cpu().numpy()
        mask_batch = mask_tensor.permute(0, 2, 3, 1).detach().cpu().numpy()
        forw_list = []

        for j in range(b):
            img_pil = Image.fromarray((image_batch[j] * 255).astype(np.uint8))
            mask_pil = Image.fromarray((mask_batch[j, :, :, 0] * 255).astype(np.uint8)).convert("L")
            
            # inference at 512 (default sweet spot)
            output = self.pipe(
                prompt=prompt, 
                image=img_pil.resize((512, 512)), 
                mask_image=mask_pil.resize((512, 512))
            ).images[0]
            
            img_inpaint = np.array(output.resize((w, h))) / 255.
            # blend inpaint into original
            fused = image_batch[j] * (1 - mask_batch[j]) + img_inpaint * mask_batch[j]
            forw_list.append(torch.from_numpy(fused).permute(2, 0, 1))

        return torch.stack(forw_list, dim=0).float().to(self.device)

    def __call__(self, image_tensor, mask_tensor=None, prompt=""):
        # default: all-zero mask if missing
        if mask_tensor is None:
            b, _, h, w = image_tensor.shape
            mask_tensor = torch.zeros(b, 1, h, w).to(self.device)
        result = self.process(image_tensor, mask_tensor, prompt)
        return result, mask_tensor.to(self.device)


# ==========================================
# 2. ControlNet Inpainter (SD 1.5)
# ==========================================
class ControlNetInpainter:
    def __init__(self, device="cuda"):
        self.device = device
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float32
        ).to(device)
        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            controlnet=controlnet, 
            torch_dtype=torch.float32
        ).to(device)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.vae.to(dtype=torch.float32)

    def process(self, image_tensor, mask_tensor, prompt=""):
        b, c, h, w = image_tensor.shape
        image_batch = image_tensor.permute(0, 2, 3, 1).detach().cpu().numpy()
        mask_batch = mask_tensor.permute(0, 2, 3, 1).detach().cpu().numpy()
        generator = torch.Generator(device=self.device).manual_seed(1)
        forw_list = []

        for j in range(b):
            img_pil = Image.fromarray((image_batch[j] * 255).astype(np.uint8))
            mask_np = mask_batch[j, :, :, 0]
            mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8)).convert("L")

            # ControlNet conditioning image
            # control_img_np = image_batch[j].copy()
            # control_img_np[mask_np > 0.5] = 0.0
            control_tensor = self.make_inpaint_condition(img_pil, mask_pil).to(torch.float32)

            output = self.pipe(
                prompt=prompt,
                image=img_pil,
                mask_image=mask_pil,
                control_image=control_tensor,
                num_inference_steps=20,
                generator=generator,
            ).images[0]

            img_inpaint = np.array(output.resize((w, h))) / 255.
            fused = image_batch[j] * (1 - mask_batch[j]) + img_inpaint * mask_batch[j]
            forw_list.append(torch.from_numpy(fused).permute(2, 0, 1))

        return torch.stack(forw_list, dim=0).float().to(self.device)

    def __call__(self, image_tensor, mask_tensor=None, prompt=""):
        if mask_tensor is None:
            b, _, h, w = image_tensor.shape
            mask_tensor = torch.zeros(b, 1, h, w).to(self.device)
        result = self.process(image_tensor, mask_tensor, prompt)
        return result, mask_tensor.to(self.device)

    @staticmethod
    def make_inpaint_condition(image, image_mask):
        image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

        assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
        image[image_mask > 0.5] = 0.0  # set as masked pixel
        image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return image

# ==========================================
# 3. SDXL Inpainter
# ==========================================
class SDXLInpainter:
    def __init__(self, device="cuda"):
        self.device = device
        self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=torch.float16, 
            variant="fp16", 
            use_safetensors=True
        ).to(device)

    def process(self, image_tensor, mask_tensor, prompt=""):
        b, c, h, w = image_tensor.shape
        image_batch = image_tensor.permute(0, 2, 3, 1).detach().cpu().numpy()
        mask_batch = mask_tensor.permute(0, 2, 3, 1).detach().cpu().numpy()
        forw_list = []

        for j in range(b):
            img_pil = Image.fromarray((image_batch[j] * 255).astype(np.uint8))
            # SDXL inpaint expects RGB mask
            mask_pil = Image.fromarray((mask_batch[j, :, :, 0] * 255).astype(np.uint8)).convert("RGB")

            output = self.pipe(
                prompt=prompt, 
                image=img_pil, 
                mask_image=mask_pil, 
                num_inference_steps=50, 
                strength=0.80,
                target_size=(512, 512)
            ).images[0]

            img_inpaint = np.array(output.resize((w, h))) / 255.
            fused = image_batch[j] * (1 - mask_batch[j]) + img_inpaint * mask_batch[j]
            forw_list.append(torch.from_numpy(fused).permute(2, 0, 1))

        return torch.stack(forw_list, dim=0).float().to(self.device)

    def __call__(self, image_tensor, mask_tensor=None, prompt=""):
        if mask_tensor is None:
            b, _, h, w = image_tensor.shape
            mask_tensor = torch.zeros(b, 1, h, w).to(self.device)
        result = self.process(image_tensor, mask_tensor, prompt)
        return result, mask_tensor.to(self.device)


# ==========================================
# 4. RePainter
# ==========================================
class RePainter:
    def __init__(self, device="cuda"):
        self.device = device
        self.scheduler = RePaintScheduler.from_pretrained("google/ddpm-ema-celebahq-256")
        self.pipe = RePaintPipeline.from_pretrained(
            "google/ddpm-ema-celebahq-256", 
            scheduler=self.scheduler
        ).to(device)

    def process(self, image_tensor, mask_tensor=None, prompt=None):
        # default zeros mask = nothing to inpaint
        if mask_tensor is None:
            b, c, h, w = image_tensor.shape
            mask_tensor = torch.zeros(b, 1, h, w).to(self.device)

        # RePaint: no text prompt
        b, c, h, w = image_tensor.shape
        image_batch = image_tensor.permute(0, 2, 3, 1).detach().cpu().numpy()
        mask_batch = mask_tensor.permute(0, 2, 3, 1).detach().cpu().numpy()
        generator = torch.Generator(device=self.device).manual_seed(0)
        forw_list = []

        for j in range(b):
            img_pil = Image.fromarray((image_batch[j] * 255).astype(np.uint8)).resize((256, 256))
            # RePaint: 0 = inpaint region, 1 = keep — invert user mask
            mask_inv_np = 1.0 - mask_batch[j, :, :, 0]
            mask_pil = Image.fromarray((mask_inv_np * 255).astype(np.uint8)).convert("L").resize((256, 256))

            output = self.pipe(
                image=img_pil,
                mask_image=mask_pil,
                num_inference_steps=150,
                eta=0.0,
                jump_length=10,
                jump_n_sample=10,
                generator=generator,
            ).images[0]

            img_inpaint = np.array(output.resize((w, h))) / 255.
            fused = image_batch[j] * (1 - mask_batch[j]) + img_inpaint * mask_batch[j]
            forw_list.append(torch.from_numpy(fused).permute(2, 0, 1))

        return torch.stack(forw_list, dim=0).float().to(self.device)

    def __call__(self, image_tensor, mask_tensor=None, prompt=None):
        if mask_tensor is None:
            b, _, h, w = image_tensor.shape
            mask_tensor = torch.zeros(b, 1, h, w).to(self.device)
        result = self.process(image_tensor, mask_tensor, prompt)
        return result, mask_tensor.to(self.device)

# ==========================================
# 5. InstructP2P
# ==========================================
class InstructP2P:
    def __init__(self, device="cuda"):
        self.device = device
        # Instruct-Pix2Pix pipeline
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix",
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to(device)
        # Euler ancestral scheduler (recommended in upstream examples)
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)

    def process(self, image_tensor, prompt=""):
        """
        image_tensor: torch.Tensor [B, C, H, W], RGB in [0,1]
        prompt: text instruction
        """
        b, c, h, w = image_tensor.shape


        # batch images
        image_batch = image_tensor.permute(0, 2, 3, 1).detach().cpu().numpy()
        generator = torch.Generator(device=self.device).manual_seed(0)
        output_list = []

        for j in range(b):
            # tensor -> PIL
            img_pil = Image.fromarray((image_batch[j] * 255).astype(np.uint8)).convert("RGB").resize((512, 512))
            
            # no mask: model decides edit region
            mask_pil = None

            output = self.pipe(
                prompt=prompt,
                image=img_pil,
                num_inference_steps=50,  # tunable
                # generator=generator,
                image_guidance_scale = 1
            ).images[0]
            
            # resize back
            img_out = np.array(img_pil.resize((w, h))) / 255.0
            output_list.append(torch.from_numpy(img_out).permute(2, 0, 1))

        return torch.stack(output_list, dim=0).float().to(self.device)

    def __call__(self, image_tensor, mask_tensor=None, prompt=""):

        result = self.process(image_tensor, prompt)
        return result, mask_tensor
    
    
# ==========================================
# Example
# ==========================================
if __name__ == "__main__":
    # dummy batch: 1 x 3 x 512 x 512
    img = torch.randn(1, 3, 512, 512).clamp(0, 1).cuda()
    mask = torch.zeros(1, 1, 512, 512).cuda()
    mask[:, :, 100:300, 100:300] = 1.0  # rectangular mask
    
    # swap class to try other pipelines
    # worker = SDInpainter() 
    # worker = ControlNetInpainter()
    # worker = SDXLInpainter()
    # worker = RePainter()
    worker = InstructP2P()
    
    with torch.no_grad():
        result_img, result_mask = worker(img, mask, prompt="a beautiful landscape")
    
    print(f"out: {result_img.shape}, mask: {result_mask.shape}")  # e.g. [1,3,512,512], [1,1,512,512]