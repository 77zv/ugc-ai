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

# === Build or Load Personality DB ===

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
    if not os.path.exists(DB_PATH):
        build_personality_db()
    return Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

# === Helpers ===

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
    input_variables=["chunk", "context", "rhetorical_role"],
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
        "- Output ONLY the rewritten line. No explanations.\n"
    ),
)

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


def rewrite_chunk(chunk: str, db) -> str:
    """
    Full pipeline for a single chunk:
    1. Classify its role & desired retrieval target.
    2. Retrieve relevant personal context.
    3. Rewrite grounded in that context.
    """
    analysis = analyze_chunk(chunk)
    role = analysis["rhetorical_role"]
    retrieval_query = analysis["retrieval_query"]

    context = retrieve_context(db, retrieval_query)

    # If nothing retrieved, we still adapt style but avoid fake specifics.
    adapted = llm.invoke(
        REWRITE_PROMPT.format(
            chunk=chunk,
            context=context if context else "(no specific facts; stay vague but honest)",
            rhetorical_role=role,
        )
    ).content.strip()

    return adapted


# === Repurpose Script ===

def repurpose():
    print("\nLoading script.txt...")
    if not os.path.exists("script.txt"):
        raise FileNotFoundError("script.txt not found.")

    with open("script.txt", "r", encoding="utf-8") as f:
        script = f.read().strip()

    if not script:
        print("script.txt is empty!")
        return

    db = load_db()

    # Smaller chunks for IG-style lines (often line-based)
    splitter = CharacterTextSplitter(chunk_size=160, chunk_overlap=0, separator="\n")
    chunks = [c.strip() for c in splitter.split_text(script) if c.strip()]

    print(f"\nRepurposing {len(chunks)} chunk(s)...\n")

    results = []
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:")
        print(f"   IN:  {chunk}")

        adapted = rewrite_chunk(chunk, db)

        results.append(adapted)
        print(f"   OUT: {adapted}\n")

    full_output = "\n".join(results)

    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(full_output)

    print("Full repurposed script saved to output.txt")
    print("\n" + "=" * 50)
    print(full_output)
    print("=" * 50)


# === MAIN ===

if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        build_personality_db()
    else:
        print(f"Personality DB found at {DB_PATH}/")

    repurpose()

    # Optional: quick regenerate loop
    while input("\nRegenerate? (y/n): ").strip().lower() == "y":
        repurpose()
