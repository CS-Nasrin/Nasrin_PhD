from PIL import Image
import numpy as np
import cv2

# === Input paths ===
normal_image_path = "triangle_normal.jpg"
defect_mask_path = "triangle_mask.png"  # white area = desired defect region

# === Load and resize images ===
img = Image.open(normal_image_path).convert("RGB").resize((512, 512))
defect_mask = Image.open(defect_mask_path).convert("L").resize((512, 512))

# === Convert to numpy arrays ===
img_np = np.array(img)
gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
defect_np = np.array(defect_mask)

# === Segment the triangle metal part ===
# Adaptive thresholding to separate triangle from background
_, part_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

# Optional: Morphological cleaning to remove small speckles
kernel = np.ones((3, 3), np.uint8)
part_mask_clean = cv2.morphologyEx(part_mask, cv2.MORPH_OPEN, kernel)

# === Create the semantic segmentation mask ===
semantic_mask = np.zeros((512, 512, 3), dtype=np.uint8)  # default black

# Set triangle part to gray
semantic_mask[part_mask_clean > 127] = [128, 128, 128]

# Set defect area to red
semantic_mask[defect_np > 127] = [255, 0, 0]

# === Save the semantic mask ===
out_img = Image.fromarray(semantic_mask)
out_img.save("semantic_mask_clean.png")
print("✅ Saved clean semantic mask as 'semantic_mask_clean.png'")