from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",  # known good default
    torch_dtype=torch.float32,
    safety_checker=None 
).to("cuda")

prompt = "a high quality wooden table surface with visible grain and texture"
generator = torch.manual_seed(42)

image = pipe(prompt=prompt, num_inference_steps=30, generator=generator).images[0]
image.save("test_v15_output.png")
