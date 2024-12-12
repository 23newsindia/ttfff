import re
from typing import List, Tuple
import numpy as np

def parse_timestamps(text: str) -> List[Tuple[float, str]]:
    """Parse text with timestamps into segments.
   
    Example input: "[0:50]Hello[1:30]World"
    Returns: [(0.833, "Hello"), (1.5, "World")]
    """
    # Pattern matches [minutes:seconds] or [seconds]
    pattern = r'\[(\d+:?\d*)\](.*?)(?=\[\d+:?\d*\]|$)'
    matches = re.findall(pattern, text)
   
    segments = []
    for timestamp, content in matches:
        # Convert timestamp to seconds
        if ':' in timestamp:
            minutes, seconds = map(float, timestamp.split(':'))
            time_in_seconds = minutes * 60 + seconds
        else:
            time_in_seconds = float(timestamp)
       
        if content.strip():
            segments.append((time_in_seconds, content.strip()))
           
    return segments

def validate_timestamps(text: str) -> bool:
    """Validate if timestamps are in correct format and order."""
    segments = parse_timestamps(text)
    if not segments:
        return False
       
    # Check if timestamps are in ascending order
    times = [t for t, _ in segments]
    return all(times[i] <= times[i+1] for i in range(len(times)-1))

def create_silence_padding(duration: float, sample_rate: int) -> np.ndarray:
    """Create silence padding of specified duration."""
    return np.zeros(int(duration * sample_rate))

def format_timestamp_info(segments: List[Tuple[float, str]]) -> str:
    """Format timestamp information for display."""
    timestamp_details = ["### Generated Audio Segments:"]
    for time, text in segments:
        timestamp_details.append(f"- [{time:.2f}s] {text}")
    return "\n".join(timestamp_details)