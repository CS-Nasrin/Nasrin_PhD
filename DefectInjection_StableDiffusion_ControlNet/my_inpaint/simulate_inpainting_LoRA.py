import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
from diffusers.utils import load_image
import os

# ========= USER CONFIGURATION =========
lora_path = "/home/tavakoln/ControlNet/models/lora/lora_triangle_defect.safetensors"
masked_image_path = "/home/tavakoln/ControlNet/my_inpaint/new_masked_triangle.png"  # <— manually masked!
output_path = "/home/tavakoln/ControlNet/my_inpaint/output/injected_scratch.png"
prompt = "a scratched metal surface with damage in the top left center"

# Set seed for reproducibility
generator = torch.manual_seed(42)

# =======================================

# Step 1: Load base SD pipeline (must match LoRA base)
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float32
).to("cuda")

# Step 2: Load LoRA weights
pipe.load_lora_weights(lora_path)
pipe.fuse_lora()

# Optional: for performance
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.enable_xformers_memory_efficient_attention()

# Step 3: Load masked image (simulate inpainting)
image = Image.open(masked_image_path).resize((512, 512)).convert("RGB")


# Step 4: Run inference (img2img-style)
result = pipe(
    prompt=prompt,
    image=image,  # Treated like img2img (not inpaint!)
    strength=0.75,
    num_inference_steps=30,
    guidance_scale=8,
    generator=generator,
)

# Step 5: Save output
os.makedirs(os.path.dirname(output_path), exist_ok=True)
result.images[0].save(output_path)

print(f"✅ Defective image saved at: {output_path}")
