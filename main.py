#!./venv/bin/python
import os

# -- System / utils --

import argparse
import coloredlogs, logging
import utils

from typing import Optional

# -- AI / ML --

import torch
import gradio as gr

from diffusers import StableDiffusionPipeline, KDPM2DiscreteScheduler
from gradio.components import Textbox, Number, Slider

# Setup logging
HOST = os.environ.get('HOST', '0.0.0.0')
PORT = os.environ.get('PORT', '8080')
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
MODEL_FILE = None
MODEL_PATH = None

if args.model is not None:
    ext = args.model.split(".")[-1]
    MODEL_NAME = args.model.replace(f".{ext}", "")
    MODEL_FILE = args.model + ".safetensors"
    MODEL_PATH = os.path.join("models", MODEL_FILE)

PIPELINE: Optional[StableDiffusionPipeline] = None
COMPONENTS = [
    Textbox(lines=3, label="Prompt"),
    Textbox(lines=5, label="Negative prompt"),

    Number(value=512, label="Width"),
    Number(value=512, label="Height"),

    Number(value=20, label="Denoising steps"),
    Slider(minimum=1.0, maximum=10.0, step=0.1, value=6.0, label="Creativity")
]

def main():
    def infer(prompt: str, negative: str, width: int, height: int, denoising_steps: int, creativity: float):
        logger.info("Starting inference with options: ")
        logger.info(f"- Prompt: {prompt}")
        logger.info(f"- Negative prompt: {negative}")
        logger.info(f"- Width: {width}")
        logger.info(f"- Height: {height}")
        logger.info(f"- Denoising steps: {denoising_steps}")
        logger.info(f"- Creativity: {creativity}")

        render = PIPELINE(
            prompt=prompt,
            negative_prompt=negative,
            width=int(width),
            height=int(height),
            num_inference_steps=int(denoising_steps),
            guidance_scale=((20 - creativity * 2)),
        )

        image = render.images[0]
        image.save("render.png")
        return "render.png"

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
    ).to("cuda")

    PIPELINE.scheduler = KDPM2DiscreteScheduler.from_config(PIPELINE.scheduler.config)

    iface = gr.Interface(
        fn=infer,
        inputs=COMPONENTS,
        outputs="image",
        title=f"Stable Diffusion - {MODEL_NAME}",
        examples=[["A photo of a cat"]],
        allow_flagging='never',
        analytics_enabled=False,
    )
    iface.launch(server_port=HOST, server_name=PORT)

if __name__ == "__main__":
    main()
