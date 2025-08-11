from PIL import Image
import numpy as np

# Load images
normal = Image.open("/home/tavakoln/ControlNet/my_inpaint/normal_triangle.jpg").resize((512, 512)).convert("RGB")
mask = Image.open("/home/tavakoln/ControlNet/my_inpaint/mask_triangle.png").resize((512, 512)).convert("L")

# Convert to numpy
normal_np = np.array(normal)
mask_np = np.array(mask)

# White out the masked region
normal_np[mask_np > 128] = [255, 255, 255]  # or [127, 127, 127] for gray

# Save
Image.fromarray(normal_np).save("/home/tavakoln/ControlNet/my_inpaint/new_masked_triangle.png")
