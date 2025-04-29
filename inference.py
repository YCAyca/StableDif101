import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers import ControlNetModel
from PIL import Image
import numpy as np
import cv2

import os

from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import torch


def generate_images(args):
    outdir = f"inference/{args.input_type}/"

    # Dynamically add enabled features
    enabled_features = []
    if args.finetuned:
        enabled_features.append("finetuned")
    if args.use_controlnet:
        enabled_features.append("controlnet")
        enabled_features.append(f"controlnet_scale{args.controlnet_scale}")
    if args.direct_injection:
        enabled_features.append("direct_injection")
    if args.im2im:
        enabled_features.append("im2im")
    if args.prompting:
        if args.negative_prompting:
            enabled_features.append("detailed_and_negative_prompting")
        else:
            enabled_features.append("detailed_prompting")
    elif args.negative_prompting:
        enabled_features.append("negative_prompting")
    
    enabled_features.append(f"guidance_scale{args.guidance_scale}")

    # Combine into final sub_dir
    if len(enabled_features):
        outdir += "_".join(enabled_features)
    else:
        outdir += "default"

    os.makedirs("inference", exist_ok=True)
    os.makedirs(outdir, exist_ok=True)

    num_images_per_prompt = 1 # max number of images without having CUDA memory error

    # Model paths
    base_model_id = "runwayml/stable-diffusion-v1-5"  # Base model

    # Load the base Stable Diffusion model
    if args.im2im:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16).to("cuda")
    else:
        pipe = StableDiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16).to("cuda")
    
    if args.use_controlnet:
        # Load ControlNet model (choose the correct one based on your needs)
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16).to("cuda")
        # Attach ControlNet to your existing pipeline
        pipe.controlnet = controlnet  # ✅ Directly integrate ControlNet into your fine-tuned model

    # Load LoRA adapters
    if args.finetuned:
        lora_weights_path = "./flag_lora_trained"  # Path to your LoRA fine-tuned weights
        pipe.unet.load_attn_procs(lora_weights_path)

    # Set model to evaluation mode
    pipe.unet.eval()

    if args.prompting: #use more detailed prompts
        # prompts = [
        # "A United States flag with high quality for its stars and lines in the pattern",
        # "A Turkish flag with high quality for its moon and star in the pattern",
        # "A France flag with blue, white, red lines generated in correct order (from left to right)",
        # "A Greece flag with correct pattern of blue and white lines",
        # "A Spain flag, red and yellow, with a high quality and well designed logo in the middle",
        # "A Germany flag with black, red, red yellow lines generated in correct order (from top to bottom)"
        # ]
        prompts = [
        "A United States flag with correct design",
        "A Turkish flag with correct design",
        "A France flag with correct design",
        "A Greek flag with correct pattern",
        "A Spain flag with correct pattern",
        "A Germany flag with correct shape"
        ]
    else: 
        prompts = [
        "A United States flag",
        "A Turkish flag",
        "A France flag",
        "A Greece flag",
        "A Spain flag",
        "A Germany flag" 
    ]

    if args.input_type == "flags":
        pass 
    elif args.input_type == "flags_with_env":
        env = ["placed in front of a building", "in the mountains", "inside a stadium", "in a rural village", " in front of a building", "next to a monument"]
        prompts = [prompt + " " + env[i] for i,prompt in enumerate(prompts)]
    else:
        AssertionError("Input type not defined")



    if args.negative_prompting:
        negative_prompt = "blurry, distorted, low quality, unrealistic, unrealistic flag pattern, wrong flag"
        # Match the length of the negative prompts to the number of input prompts
        negative_prompts = [negative_prompt] * len(prompts)
    else:
        negative_prompts = [""] * len(prompts)
    # Load an input image for ControlNet (as conditioning input)
    if args.use_controlnet or args.im2im:
        if args.input_type == "flags_with_env":
            image_folder = "controlnet_prompts/flags_env"  # Folder containing input images
        elif args.input_type == "flags":
            image_folder = "controlnet_prompts/flags"
        else:
            AssertionError("Input type not defined") 

        #image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith((".jpg", ".png"))]
        image_paths = sorted(
        [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith((".jpg", ".jpeg",".png"))]
        )
        prompts = sorted(prompts) # align controlnet condition images with correct prompts. i.e a turkish flag conditional input image shouldnt be match with an american flag
        # **2️⃣ Load Input Images**
        input_images = [Image.open(img).convert("RGB") for img in image_paths]

        if args.use_controlnet :
            control_images = []
            for i,img in enumerate(input_images):
                img_np = np.array(img)
                edges = cv2.Canny(img_np, 100, 200)  # Detect edges
                pil_edges = Image.fromarray(edges)
                pil_edges.save(f"controlnet_prompts/edges/{i}.jpg")
                control_images.append(Image.fromarray(edges))  # Convert back to PIL


    # Generate images with ControlNet conditioning
    print("Image generation starts: ", prompts)
    if args.use_controlnet:
        images = pipe(prompt=prompts, controlnet_conditioning_scale=args.controlnet_scale, negative_prompt=negative_prompts, image=control_images, guidance_scale=args.guidance_scale , num_images_per_prompt=num_images_per_prompt).images
    elif args.im2im:
        # prompt = "A child sketch of this flag"
        # prompts = [prompt] * len(input_images)
        images = pipe(prompt=prompts, image=input_images, strength=0.7, guidance_scale=7.5, num_images_per_prompt=num_images_per_prompt).images
    else:
        images = pipe(prompt=prompts,  negative_prompt=negative_prompts, guidance_scale=args.guidance_scale , num_images_per_prompt=num_images_per_prompt).images


    # Save each image
    for i, img,in enumerate(images):
        k = int(i / num_images_per_prompt) 
        prompt = prompts[k]
        img.save(f"{outdir}/{prompt}_{i}.jpg")

if __name__ == "__main__":
    from types import SimpleNamespace
    args = SimpleNamespace()

    args.finetuned = False 
    args.use_controlnet = False 
    args.controlnet_scale = 1.0
    args.guidance_scale = 7.5
    args.direct_injection = False 
    args.im2im = True
    args.input_type = "flags_with_env"
    args.prompting = True
    args.negative_prompting = True


    grid_search = False 

    if grid_search:
        controlnet_scales = [1.0, 1.5, 2.0]
        guidance_scales = [7.5, 9, 12]
        for guidance in guidance_scales:
            args.guidance_scale = guidance
            generate_images(args)
        
        args.use_controlnet = True
        for control in controlnet_scales:
            args.controlnet_scale = control
            for guidance in guidance_scales:
                args.guidance_scale = guidance
                generate_images(args)
    else:
        generate_images(args)

