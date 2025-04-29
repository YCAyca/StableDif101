from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms


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

    # Convert to uint8 before computing FID
    fake_images = (fake_images * 255).clamp(0, 255).byte()
    real_images = (real_images * 255).clamp(0, 255).byte()

    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)

    
    return fid.compute().item()