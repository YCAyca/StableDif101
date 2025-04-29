from bing_image_downloader import downloader
import os 
import shutil 

# Define flags and environments
flags = [
    "United States", "France", "Germany", "Spain", "Greece", "Turkey", "Canada", "United Kingdom", "Italy", "Brazil",
    "Argentina", "Mexico", "Japan", "China", "India", "Russia", "Australia", "South Africa", "South Korea", "Netherlands",
    "Sweden", "Norway", "Denmark", "Finland", "Switzerland", "Belgium", "Portugal", "Poland", "Ukraine", "Thailand"
]

environments = [
    "in a city", "in the mountains", "in front of a building", "on a beach", "inside a stadium",
    "in a rural village", "in a forest", "on a bridge", "in a market square", "next to a monument"
]

# Download images for each flag in different environments
data_dir = 'flag_images'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

for flag in flags:
   for env in environments:
    query = f"{flag} flag {env}"
    downloader.download(query, limit=1, output_dir=data_dir, adult_filter_off=True, force_replace=False, timeout=60)

# Rename and move images into a single folder for each flag
for flag in flags:
    flag_folder = os.path.join(data_dir, flag.replace(" ", "_"))
    #if not os.path.exists(flag_folder):
     #   os.makedirs(flag_folder)

    if not os.path.exists("labels/"+flag):
        os.makedirs("labels/"+flag)
    
    i = 0
    for env in environments:
        env_folder = os.path.join(data_dir, f"{flag} flag {env}")
        print(env_folder)
        if os.path.exists(env_folder):
            for img_name in os.listdir(env_folder):
                src_path = os.path.join(env_folder, img_name)
                dest_path = os.path.join(flag_folder, f"{flag} flag {env}_{i}.jpg")
                shutil.move(src_path, dest_path)
                i += 1
            os.rmdir(env_folder)