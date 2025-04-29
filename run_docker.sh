docker run --rm  -it --gpus '"device=3"'  \
    -v /home/VICOMTECH/yaktas/StableDif101:/app/StableDif101\
    lora-finetune:2.0 bash
