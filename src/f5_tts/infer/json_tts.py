import json
import numpy as np
import torch
import torchaudio
from pathlib import Path
from typing import List, Dict

from f5_tts.infer.utils_infer import (
    load_model, 
    load_vocoder,
    preprocess_ref_audio_text,
    infer_process
)
from f5_tts.model import DiT

class TimestampedTTSGenerator:
    def __init__(
        self,
        model_type="F5-TTS",
        sample_rate=24000,
        ref_audio_paths: Dict[int, str] = None,
        ref_texts: Dict[int, str] = None,
    ):
        self.sample_rate = sample_rate
        self.model = load_model(
            model_cls=DiT,
            model_cfg=dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
            ckpt_path=None,  # Will use default F5-TTS model
            mel_spec_type="vocos"
        )
        self.vocoder = load_vocoder()
        
        # Store reference audio/text for each speaker
        self.ref_audio_paths = ref_audio_paths or {}
        self.ref_texts = ref_texts or {}
        
        # Pre-process reference audio/text pairs
        self.ref_pairs = {}
        for speaker_id in self.ref_audio_paths:
            ref_audio, ref_text = preprocess_ref_audio_text(
                self.ref_audio_paths[speaker_id],
                self.ref_texts.get(speaker_id, "")
            )
            self.ref_pairs[speaker_id] = (ref_audio, ref_text)

    def generate_from_json(self, json_data: List[Dict], output_path: str):
        """Generate audio from timestamped JSON data"""
        
        # Convert timestamps to samples
        segments = []
        for segment in json_data:
            start_sample = int(segment["start"] * self.sample_rate)
            text = segment["text"].strip()
            speaker = segment["speaker"]
            segments.append({
                "start_sample": start_sample,
                "text": text,
                "speaker": speaker
            })

        # Sort segments by start time
        segments.sort(key=lambda x: x["start_sample"])
        
        # Generate audio for each segment
        audio_segments = []
        current_sample = 0
        
        for i, segment in enumerate(segments):
            # Calculate silence padding before segment
            silence_samples = segment["start_sample"] - current_sample
            if silence_samples > 0:
                silence = torch.zeros(silence_samples)
                audio_segments.append(silence)
                
            # Get reference audio/text for speaker
            if segment["speaker"] not in self.ref_pairs:
                print(f"Warning: No reference audio for speaker {segment['speaker']}, using default")
                ref_audio, ref_text = list(self.ref_pairs.values())[0]
            else:
                ref_audio, ref_text = self.ref_pairs[segment["speaker"]]
                
            # Generate audio for segment text
            audio, sr, _ = infer_process(
                ref_audio,
                ref_text, 
                segment["text"],
                self.model,
                self.vocoder
            )
            
            audio_segments.append(torch.from_numpy(audio))
            current_sample = segment["start_sample"] + len(audio)

        # Combine all segments
        final_audio = torch.cat(audio_segments)
        
        # Save audio file
        torchaudio.save(output_path, final_audio.unsqueeze(0), self.sample_rate)
        
        return final_audio

def main():
    # Example usage
    ref_audio_paths = {
        1: "src/f5_tts/infer/examples/basic/basic_ref_en.wav"
    }
    ref_texts = {
        1: "Some reference text for the speaker"
    }
    
    generator = TimestampedTTSGenerator(
        ref_audio_paths=ref_audio_paths,
        ref_texts=ref_texts
    )
    
    # Example JSON data
    json_data = [
        {
            "start": 0.609,
            "text": "going for image to image make sure text to image is disabled and we are going to image to image so in which i have enable a",
            "speaker": 1
        },
        {
            "start": 10.894,
            "text": "so just i have created another group for loras it's just a lora you i have using three d render style excel model",
            "speaker": 1
        },
        {
            "start": 20.18,
            "text": "so you can download this model you can download this model from here i have given all the links in a description you can go through that",
            "speaker": 1
        }
    ]
    
    output_path = "output_timestamped.wav"
    generator.generate_from_json(json_data, output_path)

if __name__ == "__main__":
    main()