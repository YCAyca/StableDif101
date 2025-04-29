# Google Colab Adapted LoRA Fine-Tuning with Grid Search

import os
import torch
import itertools
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler
from diffusers.utils import is_xformers_available
from transformers import TrainingArguments
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from diffusers.models.attention_processor import LoRAAttnProcessor, LoRAAttnProcessor2_0
import numpy as np 
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms.functional import to_pil_image

# Load dataset
image_dir = "flag_images_with_env/"
image_paths = []
image_names = []

batch_size = 8

for folder in os.listdir(image_dir):
    folder_path = os.path.join(image_dir, folder)
    if os.path.isdir(folder_path):
        for image_name in os.listdir(folder_path):
            image_paths.append(os.path.join(folder_path, image_name))
            image_names.append(image_name)

# Create dataset
from datasets import Dataset, Image
from PIL import Image as PIL_Image

dataset = Dataset.from_dict({
    "image": image_paths,
    "file_name": image_names
}).cast_column("image", Image(decode=True))

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5], [0.5])
])

transform_fid = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5], [0.5])
])

def preprocess_and_add_text(example):
    img = example["image"].convert("RGB")
    text_prompt = example["file_name"].split("_")[0]
    return {"image": transform(img), "text": text_prompt}


def compute_fid(real_images, fake_images):
    fid = FrechetInceptionDistance(feature=2048).to("cuda")

   # real_images = torch.stack(real_images).to("cuda")
   # fake_images = torch.stack(fake_images).to("cuda")
    # Convert to uint8 before computing FID
    fake_images = (fake_images * 255).clamp(0, 255).byte()
    real_images = (real_images * 255).clamp(0, 255).byte()

    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)

    
    return fid.compute().item()

dataset = dataset.map(preprocess_and_add_text)
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

grid_search = False
experiment_id = 0
best_fid_score = float("inf")
best_model_dir = "./best_model"
log_file = "experiment_results.log"

prompts_fid = [
        "A United States flag in front of a building",
        "A Turkey flag in the mountains",
        "A France flag inside a stadium",
        "A Greece flag in a rural village",
        "A Spain flag in front of a building",
        "A Germany flag next to a monument" 
    ]               

real_image_paths = [
    "flag_images_with_env/United States/United States flag in front of a building_108.jpg",
    "flag_images_with_env/Turkey/Turkey flag in the mountains_70.jpg",
    "flag_images_with_env/France/France flag inside a stadium_202.jpg",
    "flag_images_with_env/Greece/Greece flag in a rural village_255.jpg",
    "flag_images_with_env/Spain/Spain flag in front of a building_103.jpg",
    "flag_images_with_env/Germany/Germany flag next to a monument_480.jpg"
]
# Compute FID for the pretrained model
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id).to("cuda")

os.makedirs("experiments/pretrained", exist_ok=True)

generated_images = []
for i, prompt_fid in enumerate(prompts_fid):
    generated = pipe(prompt_fid).images[0]
    generated.save(f"experiments/pretrained/{prompt_fid}_{i}.jpg")
    generated_images.append(transform_fid(generated))

generated_images = torch.stack(generated_images).to("cuda")

fid = FrechetInceptionDistance(feature=2048).to("cuda")

real_images = []
for image_path in real_image_paths:
    image = PIL_Image.open(image_path)
    real_images.append(transform_fid(image))

if isinstance(real_images, list):
    real_images = torch.stack([torch.tensor(np.array(img), dtype=torch.float16, device="cuda") for img in real_images])
else:
    real_images = real_images.to(dtype=torch.float16, device="cuda")


if real_images.dim() == 3 and len(real_images) == 1:
    real_images = real_images.unsqueeze(0)

real_images = real_images.permute(3, 0, 1, 2) if real_images.shape[1] != 3 else real_images
            
# Convert list of PIL images to PyTorch tensors
best_fid_score = compute_fid(real_images, generated_images)
print("Pretrained fid score:", best_fid_score)

with open("best_fid.log", "w+") as f:
    f.write(f"Pretrained Model FID: {best_fid_score}\n")

os.makedirs("experiments/pretrained", exist_ok=True)

for i,image in enumerate(generated_images):
    pil_image = to_pil_image(image.cpu().detach().clamp(0, 1))  # Convert to PIL image
   
# Define hyperparameter search space
 
if grid_search:
    lr_rates = [5e-7, 1e-6, 5e-5]
    lora_ranks = [4,8,16,32]
    lora_alphas = [8,16,32]
    dropouts = [0.1, 0.2, 0.3, 0.4, 0.5]
    timesteps = [50,100,150,200]
    max_norms = [0.5,1.0]
else:
    lr_rates = [5e-5]
    lora_ranks = [4]
    lora_alphas = [8]
    dropouts = [0.5]
    timesteps = [50]
    max_norms = [1.0]


for lr_rate, lora_rank, lora_alpha, dropout, num_timesteps, max_norm in itertools.product(
    lr_rates, lora_ranks, lora_alphas, dropouts, timesteps, max_norms
):
    
    # Clear GPU memory before each experiment
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.ipc_collect()

    experiment_id += 1
    nan_break = False
    exp_name = f"lr{lr_rate}_rank{lora_rank}_alpha{lora_alpha}_dropout{dropout}_steps{num_timesteps}_max_norm{max_norm}"
    exp_dir = f"./experiments/{exp_name}"
    if os.path.isdir(exp_dir):
        if len(os.listdir(exp_dir)) > 0:
            print("This experiment was already run, passing to the next one")
            continue
    os.makedirs(exp_dir, exist_ok=True)
    
    print(f"Starting experiment: {exp_name}")

    # Load model
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to("cuda")

    # Apply LoRA
    pipe.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to("cuda")
    for name, attn in pipe.unet.attn_processors.items():
        if isinstance(attn, torch.nn.Module):
            pipe.unet.attn_processors[name] = LoRAAttnProcessor2_0(r=lora_rank, alpha=lora_alpha, dropout=dropout)

    if is_xformers_available():
        pipe.unet.disable_xformers_memory_efficient_attention()
    
   # pipe.unet.enable_gradient_checkpointing()
    scheduler = DDPMScheduler(num_train_timesteps=num_timesteps)
    
    # Convert to float16
    pipe.unet.to(dtype=torch.float16)
    pipe.vae.to(dtype=torch.float16)
    pipe.text_encoder.to(dtype=torch.float16)
    
    optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=lr_rate, fused=True)
    pipe.unet.train()
    
    training_args = TrainingArguments(
        output_dir=exp_dir,
        num_train_epochs=5,
        logging_steps=1,
    )
    
    losses = []

    for epoch in range(training_args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            images = batch["image"]
            prompt = batch["text"]
            
            print("prompt", prompt)
            if isinstance(images, list):
                images = torch.stack([torch.tensor(np.array(img), dtype=torch.float16, device="cuda") for img in images])
            else:
                images = images.to(dtype=torch.float16, device="cuda")
            
            if images.dim() == 3 and len(batch) == 1:
                images = images.unsqueeze(0)
            
            images = images.permute(3, 0, 1, 2) if images.shape[1] != 3 else images
            
            with torch.no_grad():
                latents = pipe.vae.encode(images).latent_dist.sample().to(dtype=torch.float16, device="cuda")
                latents = latents * 0.18215
            
            with torch.cuda.amp.autocast(dtype=torch.float16):
                noise = torch.randn_like(latents, dtype=torch.float16, device="cuda")
                timesteps = torch.randint(1, scheduler.config.num_train_timesteps, (latents.shape[0],), device="cuda").long()
                noisy_latents = scheduler.add_noise(latents, noise, timesteps).to(dtype=torch.float16, device="cuda")
                
                optimizer.zero_grad()
                text_inputs = pipe.tokenizer(prompt, padding="max_length", truncation=True, max_length=77, return_tensors="pt").input_ids.to("cuda")
                encoder_hidden_states = pipe.text_encoder(text_inputs)[0].to(dtype=torch.float16, device="cuda")
                
                noise_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
                loss = torch.nn.functional.mse_loss(noise_pred, noise).to(dtype=torch.float16)
            
            if torch.isnan(loss):
                print(f"[NaN detected] Stopping experiment {exp_name}")
                nan_break = True
                break
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(), max_norm=max_norm)
            optimizer.step()
            torch.cuda.empty_cache()
            
            losses.append(loss.item())
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")
            

           
        # Save loss plot
        plt.plot(losses)
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.savefig(f"{exp_dir}/loss_plot.png")
        plt.close()

        if nan_break:
           break 

        generated_images = []
        # Generate images and compute FID        
        for i,prompt_fid in enumerate(prompts_fid):
            os.makedirs(exp_dir, exist_ok=True)
            os.makedirs(os.path.join(exp_dir,f"epoch{epoch}"), exist_ok=True)
            generated = pipe(prompt_fid).images[0]
            save_dir = f"{exp_dir}/epoch{epoch}/{prompt_fid}_epoch{epoch}.jpg"
            generated.save(save_dir)
            generated_images.append(transform_fid(generated))

        generated_images = torch.stack(generated_images).to("cuda")
        fid_score = compute_fid(real_images, generated_images)
        with open(log_file, "a") as log:
            log.write(f"{exp_name}, Epoch {epoch}, FID: {fid_score}\n")
            
        if fid_score < best_fid_score:
            best_fid_score = fid_score
            pipe.save_pretrained(best_model_dir)
            print("New best fid!!:", best_fid_score)
            print("Experiment: ", exp_name, "epoch ", epoch)
            with open("best_fid.log", "a+") as f:
                f.write(f"Best FID: {best_fid_score}, Experiment: {exp_name}, Epoch: {epoch}\n")

    print(f"Experiment {exp_name} completed.")
    del pipe 
