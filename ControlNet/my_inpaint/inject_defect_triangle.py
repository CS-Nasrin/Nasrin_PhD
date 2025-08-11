import torch
from PIL import Image
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers import AutoPipelineForInpainting
from diffusers import StableDiffusionInpaintPipeline
from diffusers import StableDiffusionPipeline
from diffusers.utils import load_image
import os

# ========= USER CONFIGURATION =========
lora_path = "/home/tavakoln/ControlNet/models/lora/lora_triangle_defect.safetensors"
normal_image_path = "/home/tavakoln/ControlNet/my_inpaint/normal_triangle.jpg"
mask_image_path   = "/home/tavakoln/ControlNet/my_inpaint/mask_triangle.png"
control_image_path = "/home/tavakoln/ControlNet/my_inpaint/mask_wood.png"
output_path = "/home/tavakoln/ControlNet/my_inpaint/output/injected_scratch.png"


#prompt = "scratched, scratched, scratched metal texture" # scratch_4 with compatible stable-diffusion-2-1

prompt = "Add oxidized rust patches on the metal surface."
#prompt= "A dog sitting on a metal"

import torch
generator = torch.manual_seed(42)
# =======================================


# pipe =StableDiffusionInpaintPipeline.from_pretrained(
#     #"stabilityai/stable-diffusion-2-inpainting", 
#     "stabilityai/stable-diffusion-2-1", 
#     torch_dtype=torch.float32
# ).to("cuda")


pipe = AutoPipelineForInpainting.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    torch_dtype=torch.float32
).to("cuda")



pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# Step 3: Load LoRA weights
#pipe.load_lora_weights(lora_path)

#pipe.fuse_lora()

# Optional: for better performance
pipe.enable_model_cpu_offload()
pipe.enable_xformers_memory_efficient_attention()

# Step 4: Load input images and resize to 512x512
image = load_image(normal_image_path).resize((512, 512))
mask = load_image(mask_image_path).resize((512, 512))
#control = load_image(control_image_path).resize((512, 512))


result = pipe(
    prompt=prompt,
    #negative_prompt=negative_prompt,
    image=image,
    mask_image=mask,
    num_inference_steps=30,
    guidance_scale=8,
    generator=generator,
    #cross_attention_kwargs={"scale": 10.0}
)


# Step 6: Save output
os.makedirs(os.path.dirname(output_path), exist_ok=True)
result.images[0].save(output_path)

print(f"✅ Defective image saved at: {output_path}")
