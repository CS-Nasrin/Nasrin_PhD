import torch
from PIL import Image
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers import StableDiffusionInpaintPipeline
from diffusers import StableDiffusionPipeline
from diffusers.utils import load_image
import os

# ========= USER CONFIGURATION =========
lora_path = "/home/tavakoln/ControlNet/models/lora/last.safetensors"
normal_image_path = "/home/tavakoln/ControlNet/my_inpaint/normal_wood.png"
mask_image_path   = "/home/tavakoln/ControlNet/my_inpaint/mask_wood.png"
control_image_path = "/home/tavakoln/ControlNet/my_inpaint/mask_wood.png"
output_path = "/home/tavakoln/ControlNet/my_inpaint/output/injected_scratch.png"
#prompt = "a deep scratch on wooden surface"
#prompt= "high quality photo of a scratched wooden surface, deep damaged groove, broken texture, 4k, realistic"
# prompt = "high resolution image of wood with a large deep scratch, damaged texture, broken surface, realistic, macro shot, gouge, crack, splinter"
# negative_prompt = "clean, undamaged, smooth, polished"

# prompt = (
#     "extreme surface damage on wood, deep irregular scratch, cracked surface, rough texture, broken wood fibers, "
#     "splintered and torn wood, heavy abrasion, photorealistic, close-up, 8k"
# )

# negative_prompt = (
#     "smooth, clean, polished, undamaged, perfect, soft texture, shiny surface, cartoonish, blurry"
# )

prompt = "add a dog"
negative_prompt = "no dog, empty, clean, plain surface"


import torch
generator = torch.manual_seed(42)
# =======================================

# Step 1: Load ControlNet model (you can change to canny, seg, etc.)
# controlnet = ControlNetModel.from_pretrained(
#     #"lllyasviel/sd-controlnet-depth",
#     "lllyasviel/control_v11p_sd21_depth",
#     #"/home/tavakoln/ControlNet/models/control_sd15_seg",
#     torch_dtype=torch.float16
# )


pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",  # or "stabilityai/stable-diffusion-2-1" if your LoRA is trained on that
    torch_dtype=torch.float32
).to("cuda")


# # Step 2: Load base inpainting model with ControlNet
# pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
#     #"runwayml/stable-diffusion-inpainting",
#     "stabilityai/stable-diffusion-2-1",
#     #"runwayml/stable-diffusion-inpainting",
#     #controlnet=controlnet,
#     torch_dtype=torch.float16
# ).to("cuda")


# pipe = StableDiffusionPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-2-1",
#     torch_dtype=torch.float32
# ).to("cuda")

# Test without LoRA
# image = pipe(prompt="a wooden texture", num_inference_steps=30).images[0]
# image.save("test_without_lora.png")



# image = load_image(normal_image_path).resize((512, 512))
# mask = load_image(mask_image_path).resize((512, 512))

# result = pipe(
#     prompt="a wooden texture",
#     image=image,
#     mask_image=mask,
#     num_inference_steps=30,
#     guidance_scale=7.5
# )

# result.images[0].save("test_without_lora.png")





pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# Step 3: Load LoRA weights
from diffusers.loaders import AttnProcsLayers
#pipe.load_attn_procs(lora_path)
#pipe.load_lora_weights(lora_path)
#pipe.load_lora_weights("/home/tavakoln/ControlNet/models/lora/converted_diffusers_lora_sd21")

# Optional: for better performance
pipe.enable_model_cpu_offload()
pipe.enable_xformers_memory_efficient_attention()

# Step 4: Load input images and resize to 512x512
image = load_image(normal_image_path).resize((512, 512))
mask = load_image(mask_image_path).resize((512, 512))
#control = load_image(control_image_path).resize((512, 512))


result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=image,
    mask_image=mask,
    num_inference_steps=30,
    guidance_scale=15,
    generator=generator
)

# Step 5: Generate image with injected defect
# result = pipe(
#     prompt=prompt,
#     image=image,
#     mask_image=mask,
#     #control_image=control,
#     num_inference_steps=30,
#     guidance_scale=7.5,
# )
#result = pipe(prompt=prompt, num_inference_steps=30, generator=generator)



# Step 6: Save output
os.makedirs(os.path.dirname(output_path), exist_ok=True)
result.images[0].save(output_path)

print(f"✅ Defective image saved at: {output_path}")
