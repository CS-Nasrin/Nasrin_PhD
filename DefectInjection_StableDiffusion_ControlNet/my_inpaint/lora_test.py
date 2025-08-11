from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float32
).to("cuda")

pipe.load_lora_weights("/home/tavakoln/ControlNet/models/lora/lora_triangle_defect.safetensors")

image = pipe(
    prompt="a metallic triangular part with six holes",
    num_inference_steps=30,
    guidance_scale=8,
    cross_attention_kwargs={"scale": 1.0}
).images[0]

image.save("lora_test_output.png")
