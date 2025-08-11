import os

# === Set these paths to your real folders ===
image_dir = r"C:\Users\tavakoln\OneDrive - University of Windsor\Desktop\Triangle\test\defective"
mask_dir = r"C:\Users\tavakoln\OneDrive - University of Windsor\Desktop\Triangle\ground_truth\defective"

# === Get sorted image filenames ===
image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

# === Check count matches ===
if len(image_files) != len(mask_files):
    print("⚠️ Warning: Number of images and masks do not match!")
    print(f"Images: {len(image_files)}, Masks: {len(mask_files)}")
    print("Make sure to double check the folder contents before renaming.")
    exit()

# === Rename masks to match image names + '_mask.png' ===
for image_file, mask_file in zip(image_files, mask_files):
    image_name = os.path.splitext(image_file)[0]  # e.g., '002'
    new_mask_name = image_name + '_mask.png'
    src_mask_path = os.path.join(mask_dir, mask_file)
    dst_mask_path = os.path.join(mask_dir, new_mask_name)
    os.rename(src_mask_path, dst_mask_path)
    print(f"✅ Renamed: {mask_file} -> {new_mask_name}")

print("🎉 Done renaming all masks!")
