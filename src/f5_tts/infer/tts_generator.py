import torch
import torchaudio
from typing import Dict, List, Tuple, Optional

from f5_tts.infer.utils_infer import (
    infer_process,
    preprocess_ref_audio_text,
)

def generate_timed_audio(
    json_data: List[Dict],
    ref_audio: str,
    ref_text: str,
    model,
    vocoder,
    sample_rate: int = 24000,
    show_info=print
) -> Tuple[int, torch.Tensor]:
    """Generate audio from JSON timestamp data"""
    
    # Process reference audio
    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio, ref_text, show_info=show_info)
    
    # Generate audio segments
    audio_segments = []
    current_sample = 0
    
    for segment in json_data:
        # Calculate start position in samples
        start_sample = int(segment["start"] * sample_rate)
        
        # Add silence if needed
        if start_sample > current_sample:
            silence_samples = start_sample - current_sample
            silence = torch.zeros(silence_samples)
            audio_segments.append(silence)
        
        # Generate audio for text segment
        show_info(f"Generating audio for: {segment['text']}")
        audio, sr, _ = infer_process(
            ref_audio,
            ref_text,
            segment["text"],
            model,
            vocoder,
            show_info=show_info
        )
        
        audio_segments.append(torch.from_numpy(audio))
        current_sample = start_sample + len(audio)
    
    # Combine all segments
    final_audio = torch.cat(audio_segments)
    
    return sample_rate, final_audio