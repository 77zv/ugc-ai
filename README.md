# AI Script Personalization Tool

An AI-powered content repurposing system that transforms reference marketing videos and scripts into personalized, authentic founder stories using RAG (Retrieval-Augmented Generation), GPT-4 Vision, Whisper, and semantic search.

## 🚀 What It Does

### Option 1: Video Processing (Full Pipeline)
Takes any UGC/marketing video and automatically:

1. **Detects scenes** - Identifies scene changes and transitions
2. **Extracts screenshots** - Captures key frames for each scene
3. **Describes visuals** - Uses GPT-4 Vision to understand what's happening on screen
4. **Transcribes audio** - Converts speech to text with Whisper API
5. **Aligns content** - Matches dialogue to corresponding scenes with timestamps
6. **Personalizes script** - Rewrites dialogue based on your real story

### Option 2: Script Personalization
Takes existing scripts and automatically personalizes them by:

1. **Analyzing** each line's rhetorical role (hook, backstory, credibility, proof, lesson, CTA, filler)
2. **Retrieving** relevant context from your personal memory database using semantic search
3. **Rewriting** content to reflect your authentic story while maintaining the original structure and intent

**Result:** Reduces scripting time by 80% while maintaining authenticity and brand consistency.

## 🛠️ Tech Stack

### Core AI
- **OpenAI GPT-4o** - Scene description (Vision) and content rewriting
- **OpenAI Whisper** - Audio transcription with timestamps
- **OpenAI Embeddings** (text-embedding-3-small) - Text vectorization
- **LangChain** - LLM orchestration and prompt templating
- **ChromaDB** - Vector database for semantic search

### Video Processing
- **PySceneDetect** - Scene change detection
- **OpenCV** - Frame extraction and image processing

## 📋 Prerequisites

- Python 3.10+
- OpenAI API key

## ⚙️ Setup

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd ugc-ai
```

2. **Create a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install langchain-openai langchain-text-splitters langchain-chroma langchain-core python-dotenv chromadb scenedetect[opencv] opencv-python openai
```

4. **Set up environment variables**

Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_api_key_here
```

5. **Create your personality file**

Edit `personality.txt` with your backstory, achievements, projects, beliefs, and voice. See the existing file for format examples.

## 🎯 Usage

### Option 1: Process a Video (Recommended)

1. **Run the tool**
```bash
python main.py
```

2. **Choose option 1** - Process video

3. **Provide video path** - Enter the path to your video file (mp4, mov, etc.)

4. **Wait for processing**
   - Scene detection (~10-30 seconds)
   - Screenshot extraction
   - GPT-4 Vision analysis
   - Whisper transcription
   - Script personalization

5. **Review outputs**
   - `video_analysis.json` - Full scene-by-scene breakdown
   - `script.txt` - Structured script with scenes, visuals, and dialogue
   - `output.txt` - Personalized script
   - `screenshots/` - Frame captures for each scene

### Option 2: Process Existing Script

1. **Add your reference script**

Create or edit `script.txt` with the reference marketing script you want to personalize.

2. **Run the tool**
```bash
python main.py
```

3. **Choose option 2** - Personalize existing script

4. **Review the output**

The personalized script will be:
- Displayed in the console
- Saved to `output.txt`

### Standalone Video Processing

You can also process videos directly without personalization:

```bash
python video_processor.py path/to/your/video.mp4
```

This generates:
- `video_analysis.json` - Structured scene data
- `script.txt` - Script format ready for personalization
- `screenshots/` - Scene screenshots

### Regeneration

After the first run, you can regenerate by typing `y` when prompted.

## 📁 Project Structure

```
ugc-ai/
├── main.py                 # Main application logic and CLI
├── video_processor.py      # Video analysis pipeline
├── requirements.txt        # Python dependencies
├── personality.txt         # Your personal backstory and brand voice
├── script.txt             # Input: reference script to personalize
├── output.txt             # Output: personalized script
├── video_analysis.json    # Output: structured video analysis
├── screenshots/           # Output: scene screenshots (auto-generated)
├── chroma_db/            # Vector database (auto-generated)
├── .env                  # Environment variables (create this)
└── venv/                 # Virtual environment
```

## 🧠 How It Works

### Video Processing Pipeline

1. **Scene Detection**
   - Analyzes video frame-by-frame for content changes
   - Detects cuts, fades, and transitions
   - Returns precise timestamps for each scene

2. **Frame Extraction**
   - Captures screenshot from middle of each scene
   - Saves as high-quality JPG
   - Organized in `screenshots/` directory

3. **Visual Analysis (GPT-4 Vision)**
   - Describes what's happening in each frame
   - Identifies setting, mood, actions
   - Generates concise 1-2 sentence descriptions

4. **Audio Transcription (Whisper)**
   - Transcribes speech with word-level timestamps
   - High accuracy across accents and audio quality
   - Handles background noise and music

5. **Scene Alignment**
   - Matches transcribed words to scenes based on timestamps
   - Groups dialogue by scene
   - Creates structured scene-by-scene breakdown

### Script Personalization Pipeline

1. **Personality Database Building**
   - Reads `personality.txt`
   - Splits into semantic chunks (400 chars, 60 char overlap)
   - Creates embeddings and stores in ChromaDB

2. **Script Analysis**
   - Classifies each line's rhetorical role
   - Generates targeted retrieval queries based on your real story
   - Avoids generic replacements by understanding context

3. **Content Retrieval**
   - Uses semantic search to find relevant personal facts
   - Retrieves top-k most relevant chunks from your memory DB
   - Returns empty context for filler lines

4. **Rewriting**
   - Grounds new content in retrieved facts
   - Maintains original rhetorical structure and intent
   - Applies your authentic voice and tone
   - Prevents fabrication or exaggeration

## 🎨 Customization

### Adjust Chunk Sizes
In `main.py`, modify:
```python
# For personality database
splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=60)

# For script processing
splitter = CharacterTextSplitter(chunk_size=160, chunk_overlap=0, separator="\n")
```

### Change LLM Model
```python
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
```

### Modify Retrieval Count
```python
def retrieve_context(db, retrieval_query: str, top_k: int = 4):
```

## 📝 Examples

### Video Processing Output

**video_analysis.json:**
```json
{
  "video": "reference_video.mp4",
  "total_scenes": 3,
  "scenes": [
    {
      "id": 1,
      "start": 0.0,
      "end": 5.2,
      "timestamp": "0:00-0:05",
      "screenshot": "screenshots/scene_001.jpg",
      "description": "Person speaking directly to camera in a bedroom setting with natural lighting, excited expression.",
      "dialogue": "I have 21 days to go insane building my startup if I want to meet the creator of ChatGPT."
    }
  ]
}
```

**script.txt (generated from video):**
```
[SCENE 1 | 0:00-0:05 | screenshots/scene_001.jpg]
Visual: Person speaking directly to camera in a bedroom setting with natural lighting, excited expression.
Dialogue: I have 21 days to go insane building my startup if I want to meet the creator of ChatGPT.

[SCENE 2 | 0:05-0:12 | screenshots/scene_002.jpg]
Visual: Close-up of laptop screen showing acceptance email.
Dialogue: I just got accepted into a startup program during which I have three weeks to deliver.
```

### Script Personalization

**Input (script.txt):**
```
I've been a fitness freak and gym bro for five years now,
And not once felt like I truly understood my current physique.
```

**Output (output.txt - personalized):**
```
I've been building social media products for three years now,
And never felt like platforms truly understood what users wanted to share.
```

## 💰 Cost Estimate (per video)

Based on a typical 1-minute UGC video with ~8 scenes:

- **Whisper transcription**: ~$0.006/minute = **$0.006**
- **GPT-4 Vision** (scene descriptions): ~$0.01/image × 8 = **$0.08**
- **GPT-4o-mini** (personalization): ~$0.02 = **$0.02**
- **Embeddings**: negligible

**Total: ~$0.10 per video**

Extremely affordable for high-quality, personalized UGC scripts!

## 🔒 Privacy

- All data stays local in `chroma_db/` and `screenshots/`
- Only API calls go to OpenAI (for transcription, vision, embeddings, and LLM inference)
- No data is stored by the tool beyond your local files
- Videos never leave your machine (only audio/frames sent to API)

## 🤝 Contributing

This is a personal project, but feel free to fork and adapt for your own use case!

## 📄 License

MIT License - feel free to use and modify for your projects.

## 👤 Author

**Antony Li**
- Building [Oppfy](https://oppfy.com) - social media made fun again
- Queen's University | Computer Science

---

Built to solve the problem of authenticity at scale in founder-led content marketing.

