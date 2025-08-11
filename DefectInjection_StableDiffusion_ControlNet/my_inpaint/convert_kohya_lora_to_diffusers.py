import os
import torch
from safetensors.torch import load_file
from diffusers import UNet2DConditionModel, StableDiffusionPipeline
from transformers import CLIPTextModel

def convert_lora(kohya_path, output_path, base_model):
    print("🔄 Loading base pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        safety_checker=None,
    )

    unet = pipe.unet
    text_encoder = pipe.text_encoder

    print("📦 Loading Kohya LoRA weights...")
    lora_sd = load_file(kohya_path)

    visited = []

    print("🔁 Applying LoRA...")
    for key in lora_sd:
        if "lora_down" in key:
            base_key = key.replace(".lora_down.weight", "")
            up_key = base_key + ".lora_up.weight"
            alpha_key = base_key + ".alpha"
            if base_key in visited:
                continue
            visited.append(base_key)

            layer = get_target_module(unet, text_encoder, base_key)
            if layer is None:
                print(f"❌ Skipping: {base_key} (not found)")
                continue

            weight_down = lora_sd[key]
            weight_up = lora_sd[up_key]
            alpha = lora_sd.get(alpha_key, torch.tensor(weight_up.shape[0]))

            # Normalize
            scale = alpha / weight_up.shape[0]

            if weight_up.ndim == 4:
                # Conv2D
                result = torch.nn.functional.conv2d(weight_down.unsqueeze(0), weight_up.unsqueeze(0)).squeeze(0)
            else:
                # Linear
                result = torch.mm(weight_up, weight_down)

            if hasattr(layer, "weight"):
                layer.weight.data += scale * result.to(layer.weight.data.device).to(layer.weight.data.dtype)
            else:
                print(f"⚠️ No weight param in {base_key}, skipping.")

    print("✅ Saving converted pipeline...")
    pipe.save_pretrained(output_path)
    print(f"🎉 LoRA converted and saved to: {output_path}")

def get_target_module(unet, text_encoder, key):
    try:
        if key.startswith("unet."):
            target = unet
            key = key.replace("unet.", "")
        elif key.startswith("text_encoder."):
            target = text_encoder
            key = key.replace("text_encoder.", "")
        else:
            return None

        parts = key.split(".")
        for p in parts:
            if p.isdigit():
                target = target[int(p)]
            else:
                target = getattr(target, p)
        return target
    except Exception as e:
        return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--kohya_path", type=str, required=True, help="Path to Kohya LoRA .safetensors file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save converted model")
    parser.add_argument("--base_model", type=str, required=True, help="Base model (e.g., runwayml/stable-diffusion-inpainting)")
    args = parser.parse_args()

    convert_lora(args.kohya_path, args.output_path, args.base_model)
