#!/usr/bin/env python3
"""Chatterbox-TTS Gradio App with checkpoint loading"""

import sys
import re
from typing import List
import os
import random
import numpy as np
import torch
import gradio as gr
import logging
from pathlib import Path
from huggingface_hub import hf_hub_download
from chatterbox import ChatterboxTTS
from safetensors.torch import load_file

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Running on device: {DEVICE}")

# Model repository and checkpoint file
MODEL_REPO = "havok2/Kartoffelbox-v0.1_0.65h2"
T3_CHECKPOINT_FILE = "merged_model/t3_cfg.safetensors"

# Global model initialization
MODEL = None


def get_or_load_model():
    global MODEL
    if MODEL is None:
        print("Model not loaded, initializing...")
        try:
            MODEL = ChatterboxTTS.from_pretrained(DEVICE)
            checkpoint_path = hf_hub_download(repo_id=MODEL_REPO, filename=T3_CHECKPOINT_FILE,
                                              token="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")
            t3_state = load_file(checkpoint_path, device="cpu")
            MODEL.t3.load_state_dict(t3_state)

            if hasattr(MODEL, 'to') and str(MODEL.device) != DEVICE:
                MODEL.to(DEVICE)
            print(f"Model loaded successfully. Internal device: {getattr(MODEL, 'device', 'N/A')}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    return MODEL


def set_seed(seed: int):
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def split_text_into_chunks(text: str, max_chars: int = 250) -> list:
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(sentence) > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            parts = re.split(r'(?<=,)\s+', sentence)
            for part in parts:
                if len(part) > max_chars:
                    words = part.split()
                    word_chunk = ""
                    for word in words:
                        if len(word_chunk + " " + word) <= max_chars:
                            word_chunk += " " + word if word_chunk else word
                        else:
                            if word_chunk:
                                chunks.append(word_chunk.strip())
                            word_chunk = word
                    if word_chunk:
                        chunks.append(word_chunk.strip())
                else:
                    if len(current_chunk + " " + part) <= max_chars:
                        current_chunk += " " + part if current_chunk else part
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = part
        else:
            if len(current_chunk + " " + sentence) <= max_chars:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return [chunk for chunk in chunks if chunk.strip()]


def generate_tts_audio(
        text_input: str,
        audio_prompt_path_input: str,
        exaggeration_input: float,
        temperature_input: float,
        seed_num_input: int,
        cfgw_input: float
) -> tuple[int, np.ndarray]:
    try:
        current_model = get_or_load_model()
        if current_model is None:
            raise RuntimeError("TTS model is not loaded.")

        if seed_num_input != 0:
            set_seed(int(seed_num_input))

        chunk_size = 400  # Hardcoded here
        text_chunks = split_text_into_chunks(text_input, chunk_size)
        logger.info(f"Processing {len(text_chunks)} text chunk(s)")

        generated_wavs = []
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)

        for i, chunk in enumerate(text_chunks):
            logger.info(f"Generating chunk {i + 1}/{len(text_chunks)}: '{chunk[:50]}...'")
            wav = current_model.generate(
                chunk,
                audio_prompt_path=audio_prompt_path_input,
                exaggeration=exaggeration_input,
                temperature=temperature_input,
                cfg_weight=cfgw_input,
            )
            generated_wavs.append(wav)

            if len(text_chunks) > 1:
                chunk_path = output_dir / f"chunk_{i + 1}_{random.randint(1000, 9999)}.wav"
                import torchaudio
                torchaudio.save(str(chunk_path), wav, current_model.sr)
                logger.info(f"Chunk {i + 1} saved to: {chunk_path}")

        if len(generated_wavs) > 1:
            silence_samples = int(0.3 * current_model.sr)
            first_wav = generated_wavs[0]
            target_device = first_wav.device
            target_dtype = first_wav.dtype
            silence = torch.zeros(1, silence_samples, dtype=target_dtype).to(target_device)
            final_wav = generated_wavs[0]
            for wav_chunk in generated_wavs[1:]:
                final_wav = torch.cat([final_wav, silence, wav_chunk], dim=1)
        else:
            final_wav = generated_wavs[0]

        logger.info("Audio generation complete.")
        output_path = output_dir / f"generated_full_{random.randint(1000, 9999)}.wav"
        import torchaudio
        torchaudio.save(str(output_path), final_wav, current_model.sr)
        logger.info(f"Final audio saved to: {output_path}")

        return (current_model.sr, final_wav.squeeze(0).cpu().numpy())
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise gr.Error(f"Generation failed: {str(e)}")


# Create Gradio interface
with gr.Blocks(
        title="Chatterbox-TTS",
        theme=gr.themes.Soft(),
        css=""" .gradio-container { max-width: 1200px; margin: auto; } """
) as demo:
    gr.HTML("""
    <div style="text-align: center; padding: 20px;">
        <h1>Chatterbox-TTS Demo</h1>
        <p style="font-size: 18px; color: #666;">
            Generate high-quality speech from text with reference audio styling
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                value="Hello! This is a test of the Chatterbox-TTS voice cloning system.",
                label="Text to synthesize",
                max_lines=10,
                lines=5
            )

            ref_wav = gr.Audio(
                type="filepath",
                label="Reference Audio File (Optional)",
                sources=["upload", "microphone"]
            )

            with gr.Row():
                exaggeration = gr.Slider(
                    0.25, 2, step=0.05,
                    label="Exaggeration",
                    value=0.5
                )
                cfg_weight = gr.Slider(
                    0.2, 1, step=0.05,
                    label="CFG/Pace",
                    value=0.5
                )

            with gr.Accordion("Advanced Options", open=False):
                seed_num = gr.Number(
                    value=0,
                    label="Random seed (0 for random)",
                    precision=0
                )
                temp = gr.Slider(
                    0.05, 5, step=0.05,
                    label="Temperature",
                    value=0.8
                )

            run_btn = gr.Button("Generate Speech", variant="primary", size="lg")

        with gr.Column():
            audio_output = gr.Audio(label="Generated Speech")

    run_btn.click(
        fn=generate_tts_audio,
        inputs=[
            text,
            ref_wav,
            exaggeration,
            temp,
            seed_num,
            cfg_weight,
        ],
        outputs=[audio_output],
        show_progress=True
    )

    gr.Examples(
        examples=[
            ["Hello! This is a test of voice cloning technology."],
            ["The quick brown fox jumps over the lazy dog."],
        ],
        inputs=[text],
        label="Example Texts"
    )


def main():
    try:
        logger.info("Loading model at startup...")
        get_or_load_model()
        logger.info("Startup model loading complete!")
        demo.launch(server_name="0.0.0.0", server_port=7860)
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
        print(f"Application may not function properly. Error: {e}")
        demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
