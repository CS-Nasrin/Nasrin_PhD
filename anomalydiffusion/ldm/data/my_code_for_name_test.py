# from PIL import Image
# import os

# # Change this to your actual paths:
# folders = [
#     "C:/Users/tavakoln/OneDrive - University of Windsor/Desktop/Triangle_Dataset/Triangle/train/good",
#     "C:/Users/tavakoln/OneDrive - University of Windsor/Desktop/anomalydiffusion/generated_mask/triangle/scratch"
# ]

# for folder in folders:
#     for filename in os.listdir(folder):
#         if filename.lower().endswith(".jpg"):
#             path = os.path.join(folder, filename)
#             img = Image.open(path)
#             new_path = os.path.join(folder, os.path.splitext(filename)[0] + ".png")
#             img.save(new_path)
#             os.remove(path)
#             print(f"Converted {filename} → {os.path.basename(new_path)}")


import os

img_dir = r'C:\Users\tavakoln\OneDrive - University of Windsor\Desktop\Triangle_Dataset\Triangle\train\good'
mask_dir = r'C:\Users\tavakoln\OneDrive - University of Windsor\Desktop\anomalydiffusion\generated_mask\triangle\scratch'

print("🔍 Checking good images:")
for file in os.listdir(img_dir):
    if not file.lower().endswith('.png') or not file.split('.')[0].isdigit():
        print("❗️ Bad or hidden file:", file)

print("\n🔍 Checking generated masks:")
for file in os.listdir(mask_dir):
    if not file.lower().endswith('.png') or not file.split('.')[0].isdigit():
        print("❗️ Bad or hidden file:", file)