# AI Script Personalization Tool

An AI-powered content repurposing system that transforms reference marketing scripts into personalized, authentic founder stories using RAG (Retrieval-Augmented Generation), semantic search, and LLM rewriting.

## 🚀 What It Does

Takes generic UGC/marketing scripts and automatically personalizes them based on your real story, achievements, and brand voice by:

1. **Analyzing** each line's rhetorical role (hook, backstory, credibility, proof, lesson, CTA, filler)
2. **Retrieving** relevant context from your personal memory database using semantic search
3. **Rewriting** content to reflect your authentic story while maintaining the original structure and intent

**Result:** Reduces scripting time by 80% while maintaining authenticity and brand consistency.

## 🛠️ Tech Stack

- **LangChain** - LLM orchestration and prompt templating
- **OpenAI GPT-4o-mini** - Content analysis and rewriting
- **ChromaDB** - Vector database for semantic search
- **OpenAI Embeddings** (text-embedding-3-small) - Text vectorization

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
pip install langchain-openai langchain-text-splitters langchain-chroma langchain-core python-dotenv chromadb
```

4. **Set up environment variables**

Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_api_key_here
```

5. **Create your personality file**

Edit `personality.txt` with your backstory, achievements, projects, beliefs, and voice. See the existing file for format examples.

## 🎯 Usage

1. **Add your reference script**

Create or edit `script.txt` with the reference marketing script you want to personalize.

2. **Run the tool**
```bash
python main.py
```

3. **Review the output**

The personalized script will be:
- Displayed in the console
- Saved to `output.txt`

4. **Regenerate (optional)**

After the first run, you can regenerate by typing `y` when prompted.

## 📁 Project Structure

```
ugc-ai/
├── main.py              # Main application logic
├── personality.txt      # Your personal backstory and brand voice
├── script.txt          # Input: reference script to personalize
├── output.txt          # Output: personalized script
├── chroma_db/          # Vector database (auto-generated)
├── .env                # Environment variables (create this)
└── venv/               # Virtual environment
```

## 🧠 How It Works

### 1. Personality Database Building
- Reads `personality.txt`
- Splits into semantic chunks (400 chars, 60 char overlap)
- Creates embeddings and stores in ChromaDB

### 2. Script Analysis
- Classifies each line's rhetorical role
- Generates targeted retrieval queries based on your real story
- Avoids generic replacements by understanding context

### 3. Content Retrieval
- Uses semantic search to find relevant personal facts
- Retrieves top-k most relevant chunks from your memory DB
- Returns empty context for filler lines

### 4. Rewriting
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

## 📝 Example

**Input (script.txt):**
```
I've been a fitness freak and gym bro for five years now,
And not once felt like I truly understood my current physique.
```

**Output (personalized):**
```
I've been building social media products for three years now,
And never felt like platforms truly understood what users wanted to share.
```

## 🔒 Privacy

- All data stays local in `chroma_db/`
- Only API calls go to OpenAI (for embeddings and LLM inference)
- No data is stored by the tool beyond your local files

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

