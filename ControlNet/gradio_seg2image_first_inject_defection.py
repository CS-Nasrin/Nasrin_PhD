from share import *
import config
import cv2

import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

def process(input_image, defect_mask, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    with torch.no_grad():
        # Force resolution to nearest multiple of 64
        #image_resolution = 384
        #detect_resolution = 384

        # ✅ Use the values passed from the UI directly (don't override!)
        # Resize both input and defect mask to detect_resolution
        img = cv2.resize(input_image, (detect_resolution, detect_resolution))
        mask = cv2.resize(defect_mask, (detect_resolution, detect_resolution), interpolation=cv2.INTER_NEAREST)

        # Then resize to image_resolution for actual processing
        img = cv2.resize(img, (image_resolution, image_resolution))
        mask = cv2.resize(mask, (image_resolution, image_resolution), interpolation=cv2.INTER_NEAREST)

        H, W, C = img.shape

        # Normalize mask to [0, 1] and format for input
        control = torch.from_numpy(mask.copy()).float().cuda() / 255.0
        if len(control.shape) == 2:
            control = control[None, :, :].repeat(3, 1, 1)
        else:
            control = control.permute(2, 0, 1)

        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b c h w -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {
            "c_concat": [control],
            "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]
        }
        un_cond = {
            "c_concat": None if guess_mode else [control],
            "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]
        }
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)

        samples, intermediates = ddim_sampler.sample(
            ddim_steps, num_samples, shape, cond, verbose=False, eta=eta,
            unconditional_guidance_scale=scale, unconditional_conditioning=un_cond
        )

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5)
        x_samples = x_samples.cpu().numpy().clip(0, 255).astype(np.uint8)

        from PIL import Image
        
        #results = [Image.fromarray(x_samples[i]) for i in range(num_samples)]
        results = [Image.fromarray(img) for img in x_samples]
        #results = [x_samples[i] for i in range(num_samples)]
               # Save for debugging
        import os
        os.makedirs("debug_outputs", exist_ok=True)
        for i, img in enumerate(results):
            path = f"debug_outputs/output_{i}.png"
            img.save(path)  # PIL-based saving
            print("Saved:", path)

        print("DONE: returning images")
        #print("RESULT SHAPE:", [r.shape for r in results])

        print("RESULT SIZE:", [r.size for r in results])  # PIL.Image.size is (width, height)
        print("DTYPE:", [r.mode for r in results])  
        print("Returning to Gradio:", type(results), len(results), type(results[0]))
    return results
    #return [(img, None) for img in results]


# Load model
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./models/control_sd15_seg.pth', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)

# Gradio UI
block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Defect Injection using ControlNet + Segmentation Mask")
    with gr.Row():
        with gr.Column():
            #input_image = gr.Image(source='upload', type="numpy", label="Normal Image")
            #defect_mask = gr.Image(source='upload', type="numpy", label="Defect Mask")
            input_image = gr.Image(type="numpy", label="Normal Image")
            defect_mask = gr.Image(type="numpy", label="Defect Mask")
            prompt = gr.Textbox(label="Prompt")
            run_button = gr.Button(value="Run")
            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                detect_resolution = gr.Slider(label="Segmentation Resolution", minimum=128, maximum=1024, value=512, step=1)
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                eta = gr.Number(label="eta (DDIM)", value=0.0)
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
                n_prompt = gr.Textbox(label="Negative Prompt", value='lowres, bad anatomy, blurry, worst quality')
        with gr.Column():
            #result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
            #result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery", columns=2, height='auto')
            result_gallery = gr.Gallery(label='Output', columns=2, height='auto')
            


    ips = [input_image, defect_mask, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta]
    #run_button.click(fn=process, inputs=ips, outputs=[result_gallery])
    run_button.click(fn=process, inputs=ips, outputs=result_gallery)


block.launch(server_name='0.0.0.0')
