#!./venv/bin/python
import os
import gc

# -- System / utils --

import argparse
import coloredlogs, logging
import utils

# -- AI / ML --

import torch
import gradio as gr

from diffusers import StableDiffusionPipeline, KDPM2DiscreteScheduler
from RealESRGAN import RealESRGAN
from compel import Compel
from gradio.components import Textbox, Number, Slider

# Setup logging
HOST = os.environ.get('HOST', '0.0.0.0')
PORT = int(os.environ.get('PORT', '8443'))
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()

logger = logging.getLogger(__name__)
coloredlogs.install(level=LOG_LEVEL)

# Ensure the /models directory exists
if not os.path.exists("models"):
    os.mkdir("models")

models = utils.load_models("models", logger)

# Parse arguments
parser = argparse.ArgumentParser(description='Gradio Interface for Stable Diffusion AI art generation')
parser.add_argument('--model', type=str, default=None, help='Name of the model to load from /models directory')
parser.add_argument('--nsfw', action='store_true', help='Enable NSFW mode')
args = parser.parse_args()

MODEL_NAME = None
MODEL_PATH = None

if args.model is not None:
    ext = args.model.split(".")[-1]
    MODEL_NAME = args.model.replace(f".{ext}", "")
    MODEL_PATH = os.path.join("models", args.model + ".safetensors")

LORAS = utils.load_models("models/lora", logger=logger)

COMPONENTS = [
    Textbox(lines=3, label="Prompt"),
    Textbox(lines=5, label="Negative prompt"),

    Number(value=512, label="Width"),
    Number(value=512, label="Height"),

    Number(value=20, label="Denoising steps"),
    Slider(minimum=1.0, maximum=10.0, step=0.1, value=6.0, label="Creativity")
]

# Load model
if args.model is None:
    logger.error("Please specify a model to load with --model")
    exit(1)

logger.info(f"Loading model {MODEL_NAME}...") 

PIPELINE = StableDiffusionPipeline.from_ckpt(
    MODEL_PATH,
    torch_dtype=torch.float16,
    use_safetensors=True,
    prediction_type="epsilon",
    load_safety_checker=False if args.nsfw else True
)

PIPELINE = PIPELINE.to("cuda")
PIPELINE.enable_xformers_memory_efficient_attention()
PIPELINE.scheduler = KDPM2DiscreteScheduler.from_config(PIPELINE.scheduler.config)

COMPEL = Compel(tokenizer=PIPELINE.tokenizer, text_encoder=PIPELINE.text_encoder)

def infer(prompt: str, negative: str, width: int, height: int, denoising_steps: int, creativity: float):
    global PIPELINE

    logger.info("Starting inference with options: ")

    tags = prompt.split(",")
    tags = [tag.strip() for tag in tags]
    
    params = None

    for tag in tags:
        if tag.startswith("<") and tag.endswith(">"):
            tag = tag.removeprefix("<").removesuffix(">")
            parts = tag.split(":")

            if len(parts) == 2:
                lora_name = parts[0]
                lora_file = lora_name + ".safetensors"
                lora_path = os.path.join("models", "lora", lora_file)

                params = { "scale": float(parts[1]) }

                if lora_file in LORAS:
                    logger.info(f"- LoRA: {lora_name}")
                    PIPELINE = utils.load_lora_weights(PIPELINE, lora_path)
                
    logger.info(f"- Prompt: {prompt}")
    logger.info(f"- Negative prompt: {negative}")
    logger.info(f"- Width: {width}")
    logger.info(f"- Height: {height}")
    logger.info(f"- Denoising steps: {denoising_steps}")
    logger.info(f"- Creativity: {creativity}")

    conditioning = COMPEL.build_conditioning_tensor(prompt)
    negative_conditioning = COMPEL.build_conditioning_tensor(negative)

    with torch.inference_mode():
        render = PIPELINE(
            prompt_embeds=conditioning,
            negative_prompt_embeds=negative_conditioning,
            width=int(width),
            height=int(height),
            num_inference_steps=int(denoising_steps),
            guidance_scale=((20 - creativity * 2)),
            cross_attention_kwargs=params,
        )

        conditioning = None
        negative_conditioning = None
        gc.collect()

        image = render.images[0]

        model = RealESRGAN('cuda', scale=2)
        model.load_weights('weights/RealESRGAN_x2.pth', download=True)

        upscaled = model.predict(image)
        upscaled.save("render.png")

        return "render.png"

def main():
    iface = gr.Interface(
        fn=infer,
        inputs=COMPONENTS,
        outputs="image",
        title=f"Stable Diffusion - {MODEL_NAME}",
        examples=[["A photo of a cat"]],
        allow_flagging='never',
        analytics_enabled=False,
    )
    iface.launch(server_name=HOST, server_port=PORT)

if __name__ == "__main__":
    main()
