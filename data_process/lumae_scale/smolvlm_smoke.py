"""One-case local vision smoke test; never writes annotation decisions."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from .models import ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("query")
    args = parser.parse_args()
    torch.set_num_threads(4)
    model_path = ROOT / "local_data/cache/smolvlm_256m"
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    processor.chat_template = processor.tokenizer.chat_template
    processor.image_processor.size = {"longest_edge": 1024}
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.float32).eval()
    loaded = time.perf_counter()
    question = ("Describe the visible people, animals, objects, and actions in this image."
                if args.query == "__describe__" else
                "This image has timestamped before, inside, and after panels from a video. "
                f"Does this exact event occur in the inside panels: {args.query} "
                "Describe what is actually visible and answer YES, NO, or UNCERTAIN.")
    prompt = f"<|im_start|>User:<image>{question}<end_of_utterance>\nAssistant:"
    with Image.open(args.image) as image:
        inputs = processor(text=prompt, images=[image.convert("RGB")],
                           return_tensors="pt")
    with torch.no_grad():
        generated = model.generate(**inputs, do_sample=False, max_new_tokens=100)
    answer = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:],
                                    skip_special_tokens=True)[0]
    print(f"model_load_seconds={loaded-started:.2f}")
    print(f"inference_seconds={time.perf_counter()-loaded:.2f}")
    print(answer)


if __name__ == "__main__":
    main()
