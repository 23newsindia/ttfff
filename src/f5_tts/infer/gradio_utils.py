import json
import gradio as gr
from typing import Tuple, Optional

from f5_tts.infer.json_parser import validate_tts_json
from f5_tts.infer.tts_generator import generate_timed_audio

def process_tts_input(
    text_input: str,
    ref_audio: str,
    ref_text: str,
    model,
    vocoder,
    show_info=print
) -> Optional[Tuple[int, torch.Tensor]]:
    """Process TTS input and handle both regular text and JSON"""
    
    # Try parsing as JSON first
    json_data = validate_tts_json(text_input)
    
    if json_data:
        show_info("Processing timestamped JSON input...")
        return generate_timed_audio(
            json_data,
            ref_audio,
            ref_text,
            model,
            vocoder,
            show_info=show_info
        )
    else:
        # Fall back to regular TTS processing
        show_info("Processing regular text input...")
        from f5_tts.infer.utils_infer import infer
        return infer(
            ref_audio,
            ref_text,
            text_input,
            model,
            vocoder,
            show_info=show_info
        )