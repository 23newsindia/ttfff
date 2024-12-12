import json
from typing import List, Dict, Union, Optional

def validate_tts_json(text: str) -> Optional[List[Dict]]:
    """Validate and parse JSON input for TTS"""
    try:
        # Try parsing the JSON
        data = json.loads(text)
        
        # Ensure it's a list
        if not isinstance(data, list):
            return None
            
        # Validate each segment
        for segment in data:
            if not isinstance(segment, dict):
                return None
                
            # Check required fields
            required_fields = {"start", "text", "speaker"}
            if not all(field in segment for field in required_fields):
                return None
                
            # Validate types
            if not isinstance(segment["start"], (int, float)):
                return None
            if not isinstance(segment["text"], str):
                return None
            if not isinstance(segment["speaker"], int):
                return None
                
        # Sort by start time
        data.sort(key=lambda x: x["start"])
        return data
        
    except json.JSONDecodeError:
        return None