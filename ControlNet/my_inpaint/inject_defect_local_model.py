import torch
from PIL import Image
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
from diffusers.loaders import AttnProcsLayers
import os

# ========= USER CONFIGURATION =========
lora_path = "/home/tavakoln/ControlNet/models/lora/last.safetensors"
normal_image_path = "/home/tavakoln/ControlNet/my_inpaint/normal_wood.png"
mask_image_path   = "/home/tavakoln/ControlNet/my_inpaint/mask_wood.png"
#control_image_path = "/home/tavakoln/ControlNet/my_inpaint/mask_wood.png"
output_path = "/home/tavakoln/ControlNet/my_inpaint/output/injected_scratch.png"
prompt = "a deep scratch on wooden surface"
# =======================================

# # ✅ Step 1: Load ControlNet model from local folder
# controlnet = ControlNetModel.from_pretrained(
#     "/home/tavakoln/ControlNet/models/controlnet-depth",  # download this from Hugging Face
#     torch_dtype=torch.float16
# )

# ✅ Step 2: Load base inpainting pipeline from local folder
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "/home/tavakoln/ControlNet/models/sd-inpaint",  # download this from Hugging Face
    #controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# ✅ Step 3: Load LoRA weights
#pipe.load_attn_procs(lora_path)
#pipe.load_lora_weights(lora_path)
pipe.load_lora_weights(
    lora_path,
    low_cpu_mem_usage=False,
    ignore_mismatched_sizes=True,
)


# ✅ Step 4: Enable performance options
pipe.enable_model_cpu_offload()
pipe.enable_xformers_memory_efficient_attention()

# ✅ Step 5: Load input images
image = load_image(normal_image_path).resize((512, 512))
mask = load_image(mask_image_path).resize((512, 512))
#control = load_image(control_image_path).resize((512, 512))

# ✅ Step 6: Generate image with injected defect
result = pipe(
    prompt=prompt,
    image=image,
    mask_image=mask,
    #control_image=control,
    num_inference_steps=30,
    guidance_scale=7.5,
)

# ✅ Step 7: Save output
os.makedirs(os.path.dirname(output_path), exist_ok=True)
result.images[0].save(output_path)

print(f"✅ Defective image saved at: {output_path}")
