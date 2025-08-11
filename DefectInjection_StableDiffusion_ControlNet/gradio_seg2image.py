from share import *
import config
import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
from PIL import Image

from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


def process(input_image, defect_mask, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, strength, scale, seed, eta):
    with torch.no_grad():
        # Resize input and mask to desired resolution
        img = cv2.resize(input_image, (image_resolution, image_resolution))
        mask = cv2.resize(defect_mask, (image_resolution, image_resolution), interpolation=cv2.INTER_NEAREST)

        H, W, C = img.shape

        # Normalize and prepare image
        image_tensor = torch.from_numpy(img).float().cuda() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
        image_tensor = 2. * image_tensor - 1.  # [-1, 1]

        # Normalize and prepare mask
        mask_tensor = torch.from_numpy(mask).float().cuda() / 255.0
        if len(mask_tensor.shape) == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        else:
            mask_tensor = mask_tensor.permute(2, 0, 1)
        mask_tensor = mask_tensor.unsqueeze(0)

        # Repeat for batch size
        image_tensor = image_tensor.repeat(num_samples, 1, 1, 1)
        mask_tensor = mask_tensor.repeat(num_samples, 1, 1, 1)

        # Conditioning
        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {
            "c_concat": [mask_tensor],
            "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]
        }
        un_cond = {
            "c_concat": [mask_tensor],
            "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]
        }

        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength] * 13

        init_latent = model.get_first_stage_encoding(model.encode_first_stage(image_tensor))

        samples, _ = ddim_sampler.sample(
            steps=ddim_steps,
            batch_size=num_samples,
            shape=shape,
            conditioning=cond,
            x_T=init_latent,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond,
            eta=eta
        )

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5)
        x_samples = x_samples.cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [Image.fromarray(x_samples[i]) for i in range(num_samples)]
        return results


# Load inpainting model
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./models/control_v11p_sd15_inpaint.pth', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)

# Gradio UI
block = gr.Blocks().queue()
with block:
    gr.Markdown("## 🩺 Defect Injection with ControlNet Inpainting")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="numpy", label="Normal Image")
            defect_mask = gr.Image(type="numpy", label="Defect Mask (white = inject here)")
            prompt = gr.Textbox(label="Defect Prompt", placeholder="e.g., a scratch, crack, or burn mark")
            run_button = gr.Button(value="Inject Defect")

            with gr.Accordion("Advanced Settings", open=False):
                a_prompt = gr.Textbox(label="Added Prompt", value="best quality, extremely detailed")
                n_prompt = gr.Textbox(label="Negative Prompt", value="blurry, low quality, bad anatomy")
                num_samples = gr.Slider(label="Images", minimum=1, maximum=4, value=1, step=1)
                image_resolution = gr.Slider(label="Resolution", minimum=256, maximum=768, value=512, step=64)
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=30, step=1)
                strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.1)
                scale = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=30.0, value=9.0, step=0.5)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=999999, step=1, value=-1)
                eta = gr.Number(label="eta (DDIM)", value=0.0)

        with gr.Column():
            gallery = gr.Gallery(label="Injected Results", columns=2, height="auto")

    inputs = [input_image, defect_mask, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, strength, scale, seed, eta]
    run_button.click(fn=process, inputs=inputs, outputs=gallery)

block.launch(server_name='0.0.0.0')
