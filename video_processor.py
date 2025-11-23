import os
import json
import base64
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

import cv2
from scenedetect import detect, ContentDetector, split_video_ffmpeg
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SCREENSHOTS_DIR = "screenshots"

def ensure_screenshots_dir():
    """Create screenshots directory if it doesn't exist."""
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

def detect_scenes(video_path: str, threshold: float = 27.0) -> List[Dict]:
    """
    Detect scene changes in a video.
    
    Args:
        video_path: Path to video file
        threshold: Scene detection sensitivity (lower = more sensitive)
    
    Returns:
        List of scenes with start/end timestamps
    """
    print(f"\n🎬 Detecting scenes in {video_path}...")
    
    try:
        scene_list = detect(video_path, ContentDetector(threshold=threshold))
    except Exception as e:
        print(f"❌ Error detecting scenes: {e}")
        print("💡 Tip: Make sure the video file is valid and not corrupted")
        raise
    
    if not scene_list:
        print("⚠️  No scenes detected - treating entire video as one scene")
        # Fallback: create a single scene for entire video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        return [{
            "id": 1,
            "start": 0.0,
            "end": duration,
            "duration": duration,
            "timestamp": f"0:00-{format_timestamp(duration)}"
        }]
    
    scenes = []
    for i, scene in enumerate(scene_list):
        start_time = scene[0].get_seconds()
        end_time = scene[1].get_seconds()
        
        scenes.append({
            "id": i + 1,
            "start": start_time,
            "end": end_time,
            "duration": end_time - start_time,
            "timestamp": f"{format_timestamp(start_time)}-{format_timestamp(end_time)}"
        })
    
    print(f"✅ Found {len(scenes)} scenes")
    return scenes

def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}:{secs:02d}"

def extract_frame(video_path: str, timestamp: float, output_path: str):
    """
    Extract a single frame from video at given timestamp.
    
    Args:
        video_path: Path to video file
        timestamp: Time in seconds to extract frame
        output_path: Where to save the screenshot
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(timestamp * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    
    if ret:
        cv2.imwrite(output_path, frame)
    
    cap.release()

def extract_screenshots(video_path: str, scenes: List[Dict]) -> List[Dict]:
    """
    Extract a screenshot for each scene.
    
    Args:
        video_path: Path to video file
        scenes: List of scene dictionaries
    
    Returns:
        Updated scenes list with screenshot paths
    """
    print("\n📸 Extracting screenshots...")
    ensure_screenshots_dir()
    
    for scene in scenes:
        # Extract frame from middle of scene for best representation
        mid_timestamp = (scene["start"] + scene["end"]) / 2
        screenshot_filename = f"scene_{scene['id']:03d}.jpg"
        screenshot_path = os.path.join(SCREENSHOTS_DIR, screenshot_filename)
        
        extract_frame(video_path, mid_timestamp, screenshot_path)
        scene["screenshot"] = screenshot_path
        
        print(f"  Scene {scene['id']}: {screenshot_path}")
    
    print("✅ Screenshots extracted")
    return scenes

def encode_image(image_path: str) -> str:
    """Encode image to base64 for OpenAI API."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def describe_scene(image_path: str) -> str:
    """
    Use GPT-4 Vision to describe what's happening in a scene.
    
    Args:
        image_path: Path to screenshot
    
    Returns:
        Description of the scene
    """
    try:
        base64_image = encode_image(image_path)
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this video frame in 1-2 sentences. Focus on: What is the person doing? What's the setting? What's the mood/energy? Keep it concise and factual."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=150
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️  Error describing scene: {e}")
        return "Unable to analyze scene"

def add_scene_descriptions(scenes: List[Dict]) -> List[Dict]:
    """
    Add visual descriptions to each scene using GPT-4 Vision.
    
    Args:
        scenes: List of scenes with screenshots
    
    Returns:
        Updated scenes with descriptions
    """
    print("\n🔍 Analyzing scenes with GPT-4 Vision...")
    
    for scene in scenes:
        if "screenshot" in scene:
            description = describe_scene(scene["screenshot"])
            scene["description"] = description
            print(f"  Scene {scene['id']}: {description[:80]}...")
    
    print("✅ Scene descriptions complete")
    return scenes

def transcribe_audio(video_path: str) -> Dict:
    """
    Transcribe audio from video using Whisper API.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Transcription with timestamps
    """
    print("\n🎙️  Transcribing audio with Whisper...")
    
    try:
        # Check file size (Whisper has 25MB limit)
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        if file_size_mb > 25:
            print(f"⚠️  Warning: File size ({file_size_mb:.1f}MB) exceeds Whisper API limit (25MB)")
            print("💡 Tip: Consider compressing the video or extracting audio separately")
        
        with open(video_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json"
            )
        
        print("✅ Transcription complete")
        return transcript
    except Exception as e:
        print(f"❌ Error transcribing audio: {e}")
        print("💡 Tip: If file is too large, try extracting audio as mp3 first")
        raise

def align_transcript_to_scenes(scenes: List[Dict], transcript: Dict) -> List[Dict]:
    """
    Match transcribed segments to their corresponding scenes.
    
    Args:
        scenes: List of scenes with timestamps
        transcript: Whisper transcript with segment timestamps
    
    Returns:
        Scenes with aligned dialogue
    """
    print("\n🔗 Aligning transcript to scenes...")
    
    # Initialize dialogue for each scene
    for scene in scenes:
        scene["dialogue"] = []
    
    # Get segments with timestamps from transcript
    if hasattr(transcript, 'segments') and transcript.segments:
        segments = transcript.segments
        
        for segment in segments:
            # Access as attributes, not dict keys
            text = getattr(segment, 'text', '').strip()
            start_time = getattr(segment, 'start', 0)
            
            # Find which scene this segment belongs to
            for scene in scenes:
                if scene["start"] <= start_time < scene["end"]:
                    scene["dialogue"].append(text)
                    break
    elif hasattr(transcript, 'text'):
        # Fallback: if no segments, split full text across scenes evenly
        full_text = transcript.text
        words = full_text.split()
        words_per_scene = max(1, len(words) // len(scenes))
        
        for i, scene in enumerate(scenes):
            start_idx = i * words_per_scene
            end_idx = start_idx + words_per_scene if i < len(scenes) - 1 else len(words)
            scene["dialogue"].append(" ".join(words[start_idx:end_idx]))
    
    # Join dialogue for each scene
    for scene in scenes:
        scene["dialogue"] = " ".join(scene["dialogue"]).strip()
        if scene["dialogue"]:
            print(f"  Scene {scene['id']}: {scene['dialogue'][:60]}...")
    
    print("✅ Transcript aligned")
    return scenes

def process_video(video_path: str, output_json: str = "video_analysis.json") -> List[Dict]:
    """
    Full video analysis pipeline.
    
    Args:
        video_path: Path to input video
        output_json: Where to save structured output
    
    Returns:
        List of analyzed scenes
    """
    # Validate video file
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Check file extension
    valid_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.wmv']
    file_ext = os.path.splitext(video_path)[1].lower()
    if file_ext not in valid_extensions:
        print(f"⚠️  Warning: '{file_ext}' may not be supported. Recommended: mp4, mov, avi")
    
    print(f"\n{'='*60}")
    print(f"🎥 PROCESSING VIDEO: {video_path}")
    print(f"{'='*60}")
    
    try:
        # Step 1: Detect scenes
        scenes = detect_scenes(video_path)
        
        # Step 2: Extract screenshots
        scenes = extract_screenshots(video_path, scenes)
        
        # Step 3: Describe scenes
        scenes = add_scene_descriptions(scenes)
        
        # Step 4: Transcribe audio
        transcript = transcribe_audio(video_path)
        
        # Step 5: Align transcript to scenes
        scenes = align_transcript_to_scenes(scenes, transcript)
        
        # Save structured output
        output = {
            "video": video_path,
            "total_scenes": len(scenes),
            "scenes": scenes
        }
        
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Analysis complete! Saved to {output_json}")
        print(f"{'='*60}\n")
        
        return scenes
        
    except Exception as e:
        print(f"\n❌ Video processing failed: {e}")
        print("\n💡 Troubleshooting:")
        print("  1. Ensure video file is valid and not corrupted")
        print("  2. Check your OpenAI API key in .env")
        print("  3. Verify you have sufficient API credits")
        print("  4. Try with a smaller/shorter video first")
        raise

def scenes_to_script(scenes: List[Dict], output_file: str = "script.txt"):
    """
    Convert scene analysis to a simple script format for repurposing.
    
    Args:
        scenes: List of analyzed scenes
        output_file: Where to save the script
    """
    script_lines = []
    
    for scene in scenes:
        # Add scene header
        script_lines.append(f"[SCENE {scene['id']} | {scene['timestamp']} | {scene.get('screenshot', 'N/A')}]")
        
        # Add visual description
        if scene.get("description"):
            script_lines.append(f"Visual: {scene['description']}")
        
        # Add dialogue
        if scene.get("dialogue"):
            script_lines.append(f"Dialogue: {scene['dialogue']}")
        
        script_lines.append("")  # Blank line between scenes
    
    script_content = "\n".join(script_lines)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(script_content)
    
    print(f"📝 Script saved to {output_file}")
    return script_content

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python video_processor.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    # Process video
    scenes = process_video(video_path)
    
    # Convert to script format
    scenes_to_script(scenes)
    
    print("\n✨ Done! Next step: Run main.py to personalize the script")

