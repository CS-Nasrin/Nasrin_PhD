
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
from diffusers.utils import make_image_grid

# === Load the inpainting pipeline (float32 for safety) ===
pipeline = StableDiffusionInpaintPipeline.from_pretrained(
    #"runwayml/stable-diffusion-inpainting"
    "stabilityai/stable-diffusion-2-inpainting"
).to("cuda")

# Disable safety checker
pipeline.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

# === Load the original triangle image ===
image = Image.open("triangle_normal.png").convert("RGB")

# === Load the improved mask ===
mask = Image.open("mask_2.png").convert("L")  # 368x368




# === Resize both to 512x512 (required by model) ===
init_image = image.resize((512, 512))
mask_image = mask.resize((512, 512))

# Save inputs for debugging
init_image.save("debug_aligned_image.png")
mask_image.save("debug_aligned_mask.png")

# === Define prompt and generator ===
prompt ="fill in the holes with similar metal texture"



generator = torch.Generator("cuda").manual_seed(42)

# === Inpaint ===
output = pipeline(
    prompt=prompt,
    image=init_image,
    mask_image=mask_image,
    generator=generator,
    num_inference_steps=20,
    strength=1.0  # Fully regenerate masked regions
).images[0]

# === Clean output image ===
np_image = np.array(output)
print("NaNs in output?", np.isnan(np_image).any())
np_image = np.nan_to_num(np_image, nan=0.0)
np_image = np.clip(np_image, 0, 255).astype("uint8")
clean_image = Image.fromarray(np_image)

# === Save final results ===
clean_image.save("triangle_output_clean.png")
print("✅ Saved: triangle_output_clean.png")

# === Optional: Create a side-by-side grid ===
grid = make_image_grid([init_image, mask_image.convert("RGB"), clean_image], rows=1, cols=3)
grid.save("triangle_comparison_grid.png")
print("✅ Saved: triangle_comparison_grid.png")



from PIL import ImageDraw, ImageFont

# Load the saved grid image
grid_with_prompt = Image.open("triangle_comparison_grid.png")

# Prepare drawing
draw = ImageDraw.Draw(grid_with_prompt)

# Try loading a system font that supports larger sizes
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
except:
    font = ImageFont.load_default()
    print("⚠️ Font not found, using default")

# Draw text on the image
text = f"Prompt: {prompt}"
draw.text((10, 10), text, fill="purple", font=font)

# Save the final labeled grid
grid_with_prompt.save("triangle_comparison_grid_labeled.png")
print("✅ Saved: triangle_comparison_grid_labeled.png (with prompt text)")


