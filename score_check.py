import argparse
import os
from PIL import Image
import torch
import numpy as np
from torchvision import transforms

# Your imports for metrics
from torchmetrics.image.fid import FrechetInceptionDistance
from transformers import CLIPProcessor, CLIPModel  # if using CLIP score
from diffusers import StableDiffusionPipeline

# ---------- Argument Parser ----------
parser = argparse.ArgumentParser(description="Compute image generation evaluation score.")
parser.add_argument("--real_images", type=str, default="score_samples/real_images_categorized/Spanish building", help="Path to folder containing real images, necessary for fid")
parser.add_argument("--prompts", type=str, nargs='+')
parser.add_argument("--num_images", type=str, default=10,  help="How many images to generate per prompt")
parser.add_argument("--output_path", type=str,default="score_check",  help="Save generated images and scores")


args = parser.parse_args()


os.makedirs(args.output_path, exist_ok=True)

# ---------- Load Real Images ----------
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(folder, filename)
            img = Image.open(path).convert("RGB")
            images.append(img)
    return images

# --- Calculate CLIP and FID scores -----

def calculate_clip_score(images, prompts, num_images):
    # Load CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    
    # Preprocess images
    total_scores = []
    for p, prompt in enumerate(prompts):
        processed_images = []
        for i in range(num_images):
            img = images[num_images*p+i]
            # Convert numpy array to PIL Image
            pil_img = Image.fromarray((img * 255).astype('uint8').transpose(1, 2, 0))
            # Preprocess image using CLIP's processor
            processed_img = processor(images=pil_img, return_tensors="pt")["pixel_values"]
            processed_images.append(processed_img)
        
        # Stack processed images
        image_input = torch.cat(processed_images)
        
        # Prepare text input
        text_input = processor(text=[prompt] * len(processed_images), return_tensors="pt", padding=True)
        
        # Calculate CLIP scores
        with torch.no_grad():
            image_features = model.get_image_features(image_input)
            text_features = model.get_text_features(**text_input)
            clip_scores = torch.nn.functional.cosine_similarity(image_features, text_features)
        
        # Log individual scores
        for i, score in enumerate(clip_scores):
            print(f"Image {i} from {prompt}: CLIP Score = {score.item()}")
            file.write(f"Image {i} from {prompt} : CLIP Score = {score.item()}\n")
        
        prompt_score = round(float(clip_scores.mean()), 4)
        print(f"Prompt score for {prompt} : CLIP Score = {prompt_score}\n")
        total_scores.append(prompt_score)
        file.write(f"Prompt score for {prompt} : CLIP Score = {prompt_score}\n")

    return round(float(np.mean(total_scores)), 4)  # Return average CLIP score

def compute_fid(real_images, generated_images):
    fid = FrechetInceptionDistance(feature=2048).to("cuda")
    generated_images = (generated_images * 255).clamp(0, 255).byte()
    real_images = (real_images * 255).clamp(0, 255).byte()
    fid.update(real_images, real=True)
    fid.update(generated_images, real=False)
    score = fid.compute().item()
    return score
# ---------- Preprocessing ----------
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5], [0.5])
])

# ---- Load Stable Diffusion ----

# Model paths
base_model_id = "runwayml/stable-diffusion-v1-5"  # Base model
# Load the base Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16).to("cuda")
# Generate images
generated_images = []
for prompt in args.prompts:
    for i in range(args.num_images):
        generated = pipe(prompt=prompt, guidance_scale=7.5, num_images_per_prompt=1).images[0]
        generated.save(f"{args.output_path}/{prompt}_{i}.jpg")
        generated_images.append(transform(generated))

generated_images_tensor = torch.stack(generated_images).to("cuda")
        
# ---------- Compute Metric FID ----------
real_images = load_images_from_folder(args.real_images)
real_images_tensor = torch.stack([transform(img) for img in real_images]).to("cuda")
# real_images_tensor = (real_images_tensor * 255).clamp(0, 255).byte().to("cuda")

# Dummy fake images for demo (replace with real generated ones)
fid_score = compute_fid(real_images_tensor, generated_images_tensor)
print(f"FID Score: {fid_score}")
file = open(args.output_path+"/scores.log", "w+") 
file.write(f"FID Score: {fid_score}\n")


# ---------- Compute Metric CLIP Score ----------

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

sd_clip_score = calculate_clip_score(np.array(generated_images), args.prompts, args.num_images)
print(f"CLIP score: {sd_clip_score}")
file.write(f"CLIP Score: {sd_clip_score}\n")
file.close()
