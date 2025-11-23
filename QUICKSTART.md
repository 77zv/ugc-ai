# Quick Start Guide

## 🚀 Get Running in 5 Minutes

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up OpenAI API Key
Create a `.env` file:
```
OPENAI_API_KEY=sk-your-key-here
```

### 3. Create Your Personality File
Edit `personality.txt` with your:
- Background and story
- Current projects
- Achievements and metrics
- Voice and tone preferences
- What you want people to remember

Example:
```
My name's [Your Name]. I'm building [Product] — [one-line description].

I came up with the idea when [origin story].

So far, [Product] has [metrics/traction].

I speak [describe your communication style].

[Add more about your beliefs, experiences, etc.]
```

### 4. Run the Tool

**Option A: Process a Video**
```bash
python main.py
# Choose 1, then enter path to your video
```

**Option B: Personalize a Script**
```bash
# Create/edit script.txt with reference script
python main.py
# Choose 2
```

### 5. Review Outputs

After processing:
- `video_analysis.json` - Scene-by-scene breakdown
- `screenshots/` - Scene screenshots  
- `script.txt` - Structured script
- `output.txt` - Personalized script

## 💡 Tips

### For Best Results

1. **Personality.txt should be detailed**
   - Include specific stories, not just facts
   - Add metrics and achievements
   - Describe your unique perspective

2. **Scene Detection Sensitivity**
   - Default threshold: 27.0
   - Lower = more scenes (more sensitive)
   - Higher = fewer scenes (less sensitive)
   - Adjust in `video_processor.py` if needed

3. **Video Quality**
   - Works with any format (mp4, mov, webm, etc.)
   - Audio quality matters for transcription
   - No resolution requirements

### Common Issues

**"Scene detection found too many/few scenes"**
- Adjust threshold in `detect_scenes()` function
- Try values between 20-35

**"Transcription is inaccurate"**
- Ensure clear audio in video
- Reduce background noise/music if possible
- Whisper is quite robust, but clear audio helps

**"Personalization is too generic"**
- Add more specific details to `personality.txt`
- Include concrete examples and stories
- Use your actual metrics and achievements

## 📊 Example Workflow

```bash
# 1. Process reference video
python main.py
> Enter choice: 1
> Enter path: examples/reference_ugc.mp4

# Processing happens automatically:
# ✓ Scene detection
# ✓ Screenshot extraction
# ✓ Visual analysis
# ✓ Transcription
# ✓ Alignment
# ✓ Personalization

# 2. Review outputs
cat output.txt  # Your personalized script

# 3. Iterate if needed
# Press 'y' to regenerate with different variations
```

## 🎯 Next Steps

1. **Test with a short video first** (30-60 seconds)
2. **Review the video_analysis.json** to understand scene detection
3. **Check screenshots/** to see if scenes are captured well
4. **Read output.txt** to see personalization quality
5. **Iterate on personality.txt** to improve results

## 🆘 Need Help?

Check the main [README.md](README.md) for:
- Detailed architecture explanation
- Customization options
- Cost estimates
- Full API reference

---

Built to make authentic founder-led content at scale 🚀

