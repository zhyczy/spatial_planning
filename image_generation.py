import torch
from diffusers import FluxPipeline
from diffusers.utils import load_image

# 1. 既然你有 8 张卡，我们可以指定特定的 GPU
device = "cuda:0" 

# 2. 加载模型（在 48GB 显存下，你可以尝试加载原始 BF16 或 FP8）
model_path = "./flux_dev_model" # 你刚才下载的路径
pipe = FluxPipeline.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16 # A6000 支持 BF16
).to(device)

# 3. 准备 4 张参考图
ref_images = [load_image(f"ref_{i}.jpg") for i in range(1, 5)]

# 4. 执行多图参考生成
# FLUX.2 原生支持 image_reference 参数，这在空间推理研究中非常稳
prompt = "A robot based on the design in img1, holding the object from img2, placed in the room of img3, using the lighting style of img4."

image = pipe(
    prompt=prompt,
    image_reference=ref_images,      # 关键：直接传入 4 张图
    reference_strength=0.85,         # 对参考图的遵循强度
    num_inference_steps=35,
    guidance_scale=3.5
).images[0]

image.save("spatial_rai_flux2.png")