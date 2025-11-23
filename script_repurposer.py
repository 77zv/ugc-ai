import os
import json
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

# === Load API Key ===
load_dotenv()

DB_PATH = "chroma_db"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# === Prompt Templates ===

STRUCTURE_PROMPT = PromptTemplate(
    input_variables=["script"],
    template=(
        "You are analyzing a UGC (User Generated Content) script.\n\n"
        "Your job is to break this script into structured sections following the UGC format:\n\n"
        "1. 😰 RELATABLE PAIN HOOK (1 sentence)\n"
        "   - What specific pain point or problem does this address?\n"
        "   - Should grab attention and make viewer feel understood\n\n"
        "2. 🏠 BACKSTORY (2-4 short sentences)\n"
        "   - Personal story about the struggle\n"
        "   - Could be about: school, work, relationships, health, career, etc.\n"
        "   - Makes the creator relatable and authentic\n\n"
        "3. 💥 BREAKING POINT (1-2 sentences)\n"
        "   - The moment they decided to change\n"
        "   - What triggered the transformation\n"
        "   - The 'enough is enough' moment\n\n"
        "4. 📦 TAKEAWAY (2-3 sentences)\n"
        "   - The solution or lesson learned\n"
        "   - What changed after using the product/method\n"
        "   - The transformation or benefit\n\n"
        "5. 🤝 SOFT CTA (optional, 1 sentence)\n"
        "   - Gentle call to action\n"
        "   - Not pushy or salesy\n\n"
        "Original Script:\n"
        "{script}\n\n"
        "Analyze the script and identify each section. Return a JSON object with this structure:\n"
        "{{\n"
        "  \"sections\": [\n"
        "    {{\n"
        "      \"type\": \"hook\",\n"
        "      \"description\": \"Brief description of what this section is about\",\n"
        "      \"content\": \"The actual text from the script\"\n"
        "    }},\n"
        "    {{\n"
        "      \"type\": \"backstory\",\n"
        "      \"description\": \"e.g., 'Backstory about struggling in school'\",\n"
        "      \"content\": \"The actual text from the script\"\n"
        "    }},\n"
        "    // ... more sections\n"
        "  ]\n"
        "}}\n\n"
        "Valid section types: hook, backstory, breaking_point, takeaway, cta, transition\n"
        "Return ONLY valid JSON, no other text.\n"
    ),
)

CLASSIFY_PROMPT = PromptTemplate(
    input_variables=["chunk"],
    template=(
        "You are analyzing a line from a reference script.\n"
        "Line: \"{chunk}\"\n\n"
        "Your job:\n"
        "1. Identify the rhetorical_role as ONE of:\n"
        "   [hook, founder_backstory, credibility, proof, lesson, CTA, filler]\n"
        "2. Write retrieval_query: a short description of what information about ANTONY "
        "should be retrieved from his personal memory DB to REPLACE this line TRUTHFULLY.\n"
        "   - Focus on Antony's real backstory, projects, long-term obsessions, results, etc.\n"
        "   - If the line is clearly about fitness, but the product is a social/media/creator app,\n"
        "     then retrieval_query should point to Antony's authentic connection to social media,\n"
        "     online communities, building products, etc. NOT fitness.\n"
        "3. If the line is filler and doesn't need personalization, set rhetorical_role='filler'\n"
        "   and retrieval_query='none'.\n\n"
        "Return a JSON object ONLY, like:\n"
        "{{\"rhetorical_role\": \"founder_backstory\", \"retrieval_query\": \"Antony's long-term obsession with online communities and why he's building his current app.\"}}\n"
    ),
)

REWRITE_PROMPT = PromptTemplate(
    input_variables=["chunk", "context", "rhetorical_role", "extra_instructions"],
    template=(
        "You are Antony.\n"
        "You are rewriting one line from a reference script so it is 100% about YOU, "
        "your real story, and your current product.\n\n"
        "Context (true facts about you, your backstory, your work):\n"
        "{context}\n\n"
        "Rhetorical role of this line: {rhetorical_role}\n"
        "Original line (DO NOT COPY DETAILS FROM HERE, ONLY STRUCTURE & INTENT):\n"
        "{chunk}\n\n"
        "Rules:\n"
        "- Ground everything in the provided context. If the original mentions being a "
        "\"fitness freak\" or something unrelated, replace it with YOUR real, relevant story.\n"
        "- Never invent achievements or fake numbers.\n"
        "- Keep the same PURPOSE (hook / backstory / credibility / proof / lesson / CTA), "
        "but mapped to your real journey (e.g. social media, online communities, building apps).\n"
        "- Use your tone: fast, direct, and don't use words like \"yo\", \"fr\", \"deadass\", etc. No fluff.\n"
        "- 1–2 sentences max.\n"
        "- Output ONLY the rewritten line. No explanations.\n\n"
        "{extra_instructions}"
    ),
)

# === Database Functions ===

def build_personality_db():
    """
    Build a Chroma DB from personality.txt.
    personality.txt should contain your backstory, beliefs, projects, etc.
    We keep it simple: semantic chunks, no fancy metadata (optional later).
    """
    print("Building personality DB from personality.txt...")

    if not os.path.exists("personality.txt"):
        raise FileNotFoundError("personality.txt not found.")

    with open("personality.txt", "r", encoding="utf-8") as f:
        text = f.read()

    if not text.strip():
        raise ValueError("personality.txt is empty.")

    splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=60)
    chunks = [c.strip() for c in splitter.split_text(text) if c.strip()]

    db = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH,
    )
    print(f"Personality DB saved to {DB_PATH}/")


def load_db():
    """Load or build the personality database."""
    if not os.path.exists(DB_PATH):
        build_personality_db()
    return Chroma(persist_directory=DB_PATH, embedding_function=embeddings)


# === Script Processing Functions ===

def structure_script(script: str) -> list:
    """
    Use AI to analyze and break down the script into structured sections.
    
    Args:
        script: Raw script text
        
    Returns:
        List of section dictionaries with type, description, and content
    """
    print("\n🔍 Analyzing script structure...")
    
    try:
        resp = llm.invoke(STRUCTURE_PROMPT.format(script=script))
        text = resp.content.strip()
        
        # Remove markdown code blocks if present
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        data = json.loads(text)
        sections = data.get("sections", [])
        
        if not sections:
            print("⚠️  Could not identify sections, will process as one chunk")
            return [{
                "type": "full_script",
                "description": "Full script",
                "content": script
            }]
        
        print(f"✅ Identified {len(sections)} section(s):\n")
        for i, section in enumerate(sections, 1):
            section_type = section.get("type", "unknown")
            description = section.get("description", "")
            content_preview = section.get("content", "")[:60]
            print(f"   {i}. [{section_type.upper()}] {description}")
            print(f"      Preview: {content_preview}...")
        
        return sections
        
    except json.JSONDecodeError as e:
        print(f"⚠️  JSON parsing error: {e}")
        print("⚠️  Falling back to simple processing")
        return [{
            "type": "full_script",
            "description": "Full script",
            "content": script
        }]
    except Exception as e:
        print(f"⚠️  Error structuring script: {e}")
        print("⚠️  Falling back to simple processing")
        return [{
            "type": "full_script",
            "description": "Full script",
            "content": script
        }]


def analyze_chunk(chunk: str) -> dict:
    """
    Use the LLM to understand what this line is doing
    and what we should retrieve from the memory DB.
    """
    resp = llm.invoke(CLASSIFY_PROMPT.format(chunk=chunk))
    text = resp.content.strip()

    # Be defensive about malformed JSON
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: treat as filler with no retrieval
        data = {
            "rhetorical_role": "filler",
            "retrieval_query": "none"
        }

    role = data.get("rhetorical_role", "filler")
    query = data.get("retrieval_query", "none")

    # Normalize
    role = role.strip().lower()
    if not query or query.strip().lower() in ["none", "n/a"]:
        query = "none"

    return {
        "rhetorical_role": role,
        "retrieval_query": query
    }


def retrieve_context(db, retrieval_query: str, top_k: int = 4) -> str:
    """
    Use a purpose-built retrieval query instead of the raw line.
    If retrieval_query is 'none', return empty context.
    """
    if retrieval_query == "none":
        return ""

    docs = db.similarity_search(retrieval_query, k=top_k)
    if not docs:
        return ""

    # Join distinct chunks; keep it compact
    unique = []
    for d in docs:
        t = d.page_content.strip()
        if t and t not in unique:
            unique.append(t)

    return " | ".join(unique[:top_k])


def rewrite_chunk(chunk: str, db, extra_instructions: str = "") -> str:
    """
    Full pipeline for a single chunk:
    1. Classify its role & desired retrieval target.
    2. Retrieve relevant personal context.
    3. Rewrite grounded in that context.
    
    Args:
        chunk: Text chunk to rewrite
        db: Personality database
        extra_instructions: Optional extra instructions for rewriting
    """
    analysis = analyze_chunk(chunk)
    role = analysis["rhetorical_role"]
    retrieval_query = analysis["retrieval_query"]

    context = retrieve_context(db, retrieval_query)

    # Format extra instructions if provided
    extra_instructions_text = ""
    if extra_instructions and extra_instructions.strip():
        extra_instructions_text = f"Additional Instructions:\n{extra_instructions.strip()}"

    # If nothing retrieved, we still adapt style but avoid fake specifics.
    adapted = llm.invoke(
        REWRITE_PROMPT.format(
            chunk=chunk,
            context=context if context else "(no specific facts; stay vague but honest)",
            rhetorical_role=role,
            extra_instructions=extra_instructions_text,
        )
    ).content.strip()

    return adapted


def repurpose_script(input_file: str = "script.txt", output_file: str = "output.txt", extra_instructions: str = ""):
    """
    Main function to repurpose a script by personalizing it.
    
    Args:
        input_file: Path to input script file
        output_file: Path to save personalized output
        extra_instructions: Optional extra instructions for customizing the rewrite
    """
    print(f"\nLoading {input_file}...")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"{input_file} not found.")

    with open(input_file, "r", encoding="utf-8") as f:
        script = f.read().strip()

    if not script:
        print(f"{input_file} is empty!")
        return

    db = load_db()

    # Display extra instructions if provided
    if extra_instructions and extra_instructions.strip():
        print(f"\n📝 Extra Instructions: {extra_instructions.strip()}\n")

    # Step 1: Use AI to structure the script into sections
    sections = structure_script(script)
    
    print(f"\n{'='*60}")
    print("REPURPOSING SCRIPT BY SECTION")
    print(f"{'='*60}\n")

    all_results = []
    
    # Process each section
    for section_idx, section in enumerate(sections, 1):
        section_type = section.get("type", "unknown")
        section_description = section.get("description", "")
        section_content = section.get("content", "").strip()
        
        if not section_content:
            continue
        
        print(f"\n{'─'*60}")
        print(f"SECTION {section_idx}: [{section_type.upper()}]")
        if section_description:
            print(f"Description: {section_description}")
        print(f"{'─'*60}")
        
        # For longer sections, split into sentences
        # Otherwise, process as one chunk
        if len(section_content) > 300:
            # Split by sentences or newlines
            splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20, separator="\n")
            chunks = [c.strip() for c in splitter.split_text(section_content) if c.strip()]
        else:
            chunks = [section_content]
        
        section_results = []
        
        for chunk_idx, chunk in enumerate(chunks, 1):
            print(f"\n  Chunk {chunk_idx}/{len(chunks)}:")
            print(f"  IN:  {chunk[:100]}{'...' if len(chunk) > 100 else ''}")
            
            adapted = rewrite_chunk(chunk, db, extra_instructions)
            
            section_results.append(adapted)
            print(f"  OUT: {adapted}")
        
        # Join results for this section
        section_output = " ".join(section_results)
        all_results.append(section_output)
        
        print(f"\n✅ Section {section_idx} complete")

    # Join all sections with double newlines for readability
    full_output = "\n\n".join(all_results)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(full_output)

    print(f"\n{'='*60}")
    print(f"✅ FULL REPURPOSED SCRIPT SAVED TO: {output_file}")
    print(f"{'='*60}\n")
    print(full_output)
    print(f"\n{'='*60}")


def analyze_script_only(input_file: str, output_dir: str = "analyzed_scripts") -> str:
    """
    Analyze a script structure without repurposing it.
    Saves both JSON and human-readable formats.
    
    Args:
        input_file: Path to input script file
        output_dir: Directory to save analyzed scripts
        
    Returns:
        Path to the saved analyzed script
    """
    import os
    from datetime import datetime
    
    print(f"\n📖 Loading {input_file}...")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"{input_file} not found.")

    with open(input_file, "r", encoding="utf-8") as f:
        script = f.read().strip()

    if not script:
        print(f"{input_file} is empty!")
        return ""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename based on input filename
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"{base_name}_analyzed_{timestamp}"
    
    # Analyze the script structure
    sections = structure_script(script)
    
    # Save JSON format
    json_output_path = os.path.join(output_dir, f"{output_name}.json")
    analysis_data = {
        "original_file": input_file,
        "analyzed_at": datetime.now().isoformat(),
        "total_sections": len(sections),
        "sections": sections
    }
    
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(analysis_data, f, indent=2, ensure_ascii=False)
    
    # Save human-readable format
    txt_output_path = os.path.join(output_dir, f"{output_name}.txt")
    readable_lines = []
    readable_lines.append("=" * 60)
    readable_lines.append("SCRIPT STRUCTURE ANALYSIS")
    readable_lines.append("=" * 60)
    readable_lines.append(f"Original File: {input_file}")
    readable_lines.append(f"Analyzed At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    readable_lines.append(f"Total Sections: {len(sections)}")
    readable_lines.append("")
    
    for i, section in enumerate(sections, 1):
        section_type = section.get("type", "unknown")
        description = section.get("description", "")
        content = section.get("content", "")
        
        readable_lines.append("-" * 60)
        readable_lines.append(f"SECTION {i}: [{section_type.upper()}]")
        if description:
            readable_lines.append(f"Description: {description}")
        readable_lines.append("-" * 60)
        readable_lines.append(content)
        readable_lines.append("")
    
    readable_lines.append("=" * 60)
    
    with open(txt_output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(readable_lines))
    
    print(f"\n✅ Analysis complete!")
    print(f"   📄 JSON format: {json_output_path}")
    print(f"   📄 Readable format: {txt_output_path}")
    
    return txt_output_path

