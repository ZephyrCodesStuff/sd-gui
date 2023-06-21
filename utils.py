import os
import logging
from typing import List, Optional


def load_models(path: str, logger: Optional[logging.Logger]) -> List[str]:
    models = []

    files = os.listdir(path)
    for model in files:
        ext = model.split(".")[-1]

        filetypes = ["pt", "safetensors"]
        if ext in filetypes:
            models.append(model)

    if logger is not None:
        if len(models) == 0:
            logger.error("No models found in /models directory. Please add a model checkpoint to /models directory")
            exit(1)
        else:
            logger.info(f"Found {len(models)} model(s) in /models directory")
            for model in models:
                logger.info(f"- {model}")
    
    return models
