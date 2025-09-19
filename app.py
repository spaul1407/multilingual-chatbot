# chatbot_api.py
# Run:
#   pip install fastapi uvicorn[standard] sentence-transformers faiss-cpu transformers torch soundfile librosa pydantic
#   uvicorn chatbot_api:app --reload --host 0.0.0.0 --port 8000 --workers 1
HF_CACHE_DIR = "/data/hf_cache"

os.environ["HF_HOME"] = "/data/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/data/hf_cache"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/data/hf_cache"

os.makedirs("/data/hf_cache", exist_ok=True)


import os
import io
import json
import tempfile
import asyncio
from typing import List, Dict, Any, Optional
from threading import Lock

from fastapi import FastAPI, UploadFile, Form
from pydantic import BaseModel

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi import Request

import os

# Set cache directory to a local folder inside your repo

# ------------------------
# CONFIG
# ------------------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# ASR_MODEL = "openai/whisper-small"  # COMMENTED OUT
DEFAULT_CASUAL_LLM = "google/flan-t5-base"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3
RETRIEVAL_THRESHOLD = 0.30
MAX_CONTEXT_CHUNKS = 3
MEMORY_WINDOW_TURNS = 6
INDEX_FILE = "/tmp/faiss.index"
META_FILE = "docs_meta.json"

# ------------------------
# In-memory state + locks
# ------------------------
app = FastAPI(title="Audio+RAG Chatbot")

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body},
    )

index_lock = Lock()
embedder: Optional[SentenceTransformer] = None
# asr_pipeline = None  # COMMENTED OUT
llm_pipe = None
llm_tokenizer = None
llm_is_causal = False

docs_texts: List[str] = []
docs_meta: List[Dict[str, Any]] = []
index: Optional[faiss.IndexFlatIP] = None
doc_embeddings = None
user_memories: Dict[str, List[Dict[str, str]]] = {}

# ------------------------
# Persistence helpers
# ------------------------
def save_index_to_disk():
    global index, docs_meta
    with index_lock:
        if index is None:
            return False
        faiss.write_index(index, INDEX_FILE)
        with open(META_FILE, "w", encoding="utf-8") as f:
            json.dump(docs_meta, f, ensure_ascii=False, indent=2)
    return True

def load_index_from_disk():
    global index, docs_meta, docs_texts
    if not (os.path.exists(INDEX_FILE) and os.path.exists(META_FILE)):
        return False
    with index_lock:
        index = faiss.read_index(INDEX_FILE)
        with open(META_FILE, "r", encoding="utf-8") as f:
            docs_meta = json.load(f)
        docs_texts = [m.get("content", "") for m in docs_meta]
    return True

# ------------------------
# Model init
# ------------------------
def init_models():
    global embedder, llm_pipe, llm_tokenizer, llm_is_causal  # REMOVED asr_pipeline

    # ------------------------
    # Embeddings
    # ------------------------
    if embedder is None:
        print("Loading SentenceTransformer embedding model...")
        embedder = SentenceTransformer(EMBED_MODEL,cache_folder=HF_CACHE_DIR)

    # ------------------------
    # ASR (Whisper)
    # ------------------------
    # if asr_pipeline is None:
    #     print("Loading Whisper ASR pipeline...")
    #     asr_pipeline = pipeline(
    #         task="automatic-speech-recognition",
    #         model=ASR_MODEL,
    #         chunk_length_s=30,
    #         device=0,  # GPU if available in HF Spaces
    #         generate_kwargs={"language": "en", "task": "translate"},
    #         cache_dir=HF_CACHE_DIR# Always translate to English
    #     )

    # ------------------------
    # LLM (text2text RAG)
    # ------------------------
    if llm_pipe is None:
        try:
            print(f"Loading LLM: {DEFAULT_CASUAL_LLM}")
            llm_tokenizer = AutoTokenizer.from_pretrained(DEFAULT_CASUAL_LLM, cache_dir=HF_CACHE_DIR)
            llm_model = AutoModelForSeq2SeqLM.from_pretrained(DEFAULT_CASUAL_LLM, cache_dir=HF_CACHE_DIR)
            llm_pipe = pipeline("text2text-generation", model=llm_model, tokenizer=llm_tokenizer)
            llm_is_causal = False
        except Exception as e:
            print("Failed to load LLM:", e)
            raise e

# ------------------------
# Startup event
# ------------------------
@app.on_event("startup")
def startup_event():
    init_models()
    if load_index_from_disk():
        print("Loaded FAISS index and metadata.")
    else:
        print("No saved index found - starting empty.")

# ------------------------
# Chunking
# ------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
def chunk_document_text(text: str) -> List[str]:
    return splitter.split_text(text)

# ------------------------
# Embeddings & FAISS helpers
# ------------------------
async def embed_texts(texts: List[str]) -> np.ndarray:
    loop = asyncio.get_running_loop()
    emb = await loop.run_in_executor(None, lambda: embedder.encode(texts, convert_to_numpy=True).astype("float32"))
    faiss.normalize_L2(emb)
    return emb

def _create_index_if_missing(dim: int):
    global index
    if index is None:
        index = faiss.IndexFlatIP(dim)

# ------------------------
# Semantic search
# ------------------------
async def semantic_search(query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    if index is None or index.ntotal == 0:
        return []
    loop = asyncio.get_running_loop()
    q_emb = await loop.run_in_executor(None, lambda: embedder.encode([query], convert_to_numpy=True).astype("float32"))
    faiss.normalize_L2(q_emb)
    with index_lock:
        D, I = index.search(q_emb, k=top_k)
    results = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx < 0 or idx >= len(docs_texts):
            continue
        meta = docs_meta[int(idx)]
        results.append({
            "id": meta.get("id"),
            "source": meta.get("source"),
            "content": docs_texts[int(idx)],
            "score": float(score)
        })
    return results


# ------------------------
# Ingest endpoint
# ------------------------
from pydantic import BaseModel

class IngestRequest(BaseModel):
    id: str
    text: str
    source: str = "manual"

@app.post("/ingest")
async def ingest(req: IngestRequest):
    init_models()

    # 1. Chunk the input text
    chunks = chunk_document_text(req.text)
    if not chunks:
        return {"ok": False, "message": "No chunks produced."}

    # 2. Embed chunks
    embeddings = await embed_texts(chunks)

    # 3. Create index if missing
    _create_index_if_missing(embeddings.shape[1])

    # 4. Add to FAISS + metadata
    with index_lock:
        index.add(embeddings)
        for i, chunk in enumerate(chunks):
            docs_texts.append(chunk)
            docs_meta.append({
                "id": f"{req.id}_{i}",
                "source": req.source,
                "content": chunk
            })

    # 5. Persist
    saved = save_index_to_disk()

    return {
        "ok": True,
        "added_chunks": len(chunks),
        "total_chunks": len(docs_texts),
        "persisted": saved
    }

# ------------------------
# Prompt builder
# ------------------------
def build_prompt(user_id: str, user_message: str, retrieved: List[Dict[str,Any]], use_retrieval: bool):
    sys_instr = ("System: You are a concise, cautious medical assistant. "
                 "Do NOT provide definitive diagnoses. Keep answers short (2-4 sentences).")
    mem = user_memories.get(user_id, [])
    mem_window = mem[-(MEMORY_WINDOW_TURNS*2):]
    mem_text_parts = []
    for turn in mem_window:
        prefix = "User:" if turn["role"] == "user" else "Assistant:"
        mem_text_parts.append(f"{prefix} {turn['text']}")
    mem_text = "\n".join(mem_text_parts).strip()
    ctx_text = ""
    if use_retrieval and retrieved:
        ctx_text = "\n\n".join([f"source:{r['id']}\n{r['content']}" for r in retrieved[:MAX_CONTEXT_CHUNKS]])
    prompt_parts = [sys_instr]
    if mem_text:
        prompt_parts.append("Conversation history:\n" + mem_text)
    if ctx_text:
        prompt_parts.append("Retrieved context:\n" + ctx_text)
    prompt_parts.append(f"User: {user_message}\nAssistant:")
    return "\n\n".join(prompt_parts)

# ------------------------
# LLM call
# ------------------------
async def call_llm_async(prompt: str) -> str:
    loop = asyncio.get_running_loop()
    def _call():
        if llm_is_causal:
            out = llm_pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.3, top_p=0.95)[0]["generated_text"]
        else:
            out = llm_pipe(prompt, max_new_tokens=256)[0]["generated_text"]
        return out
    result = await loop.run_in_executor(None, _call)
    if result.startswith(prompt):
        return result[len(prompt):].strip()
    return result.strip()

# ------------------------
# Chat endpoint with Whisper ASR
# ------------------------
# from fastapi import FastAPI, UploadFile, Form
# from typing import Optional

# ------------------------
# Chat endpoint (text or audio) optimized for Flan-T5
# ------------------------
@app.post("/chat")
async def chat_endpoint(
    user_id: str = Form(...),
    user_input: Optional[str] = Form(None),
    # audio_file: Optional[UploadFile] = None,  # COMMENTED OUT
    top_k: int = Form(TOP_K),
    retrieval_threshold: float = Form(RETRIEVAL_THRESHOLD),
    target_language: Optional[str] = Form(None)  # NEW
):
    """
    Chat endpoint: accepts text input only.
    Returns: JSON with transcript (if audio), matched_ids, scores, used_retrieval, response.
    """
    init_models()

    # --- Determine final input (text) ---
    transcript = None
    # if audio_file is not None:
    #     # save temp file for ASR
    #     tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    #     try:
    #         content = await audio_file.read()
    #         tfile.write(content)
    #         tfile.flush()
    #         tfile.close()
    #         loop = asyncio.get_running_loop()
    #         asr_result = await loop.run_in_executor(None, lambda: asr_pipeline(tfile.name))
    #         transcript = asr_result.get("text", "").strip() if isinstance(asr_result, dict) else str(asr_result).strip()
    #         final_input = transcript
    #     finally:
    #         try:
    #             os.unlink(tfile.name)
    #         except Exception:
    #             pass
    if user_input:
        final_input = user_input.strip()
    else:
        return {"error": "Provide user_input."}

    if not final_input:
        return {"error": "Empty input after transcription."}

    # --- Semantic retrieval ---
    retrieved = await semantic_search(final_input, top_k=top_k)
    print([r["content"] for r in retrieved])
    filtered_retrieved = [r for r in retrieved[:MAX_CONTEXT_CHUNKS] if r["score"] >= retrieval_threshold]
    use_retrieval = bool(filtered_retrieved)
    print([r["content"] for r in filtered_retrieved])

    # --- Build prompt with memory + context ---
    sys_instr = (
        "You are a medical assistant. "
        "ONLY answer based on the provided context. "
        "Do NOT make up information. "
        "If you don't know, respond: 'I do not know, please consult a medical professional.' "
        "Answer concisely in 2-3 sentences."
    )

    # Windowed memory
    mem = user_memories.get(user_id, [])
    mem_window = mem[-(MEMORY_WINDOW_TURNS*2):]  # last N turns
    mem_text = "\n".join(
        f"{'User' if turn['role']=='user' else 'Assistant'}: {turn['text']}" for turn in mem_window
    )

    context_text = ""
    if use_retrieval:
        context_text = "Context from retrieved documents:\n" + "\n".join([r["content"] for r in filtered_retrieved])

    prompt_parts = [sys_instr]
    if mem_text:
        prompt_parts.append("Conversation history:\n" + mem_text)
    if context_text:
        prompt_parts.append(context_text)
    prompt_parts.append(f"Question: {final_input}")
    prompt = "\n\n".join(prompt_parts)

    # --- Append translation instruction if needed ---
    if target_language:
        prompt += f"\n\nTranslate the above response into {target_language}."

    # --- Call Flan-T5 asynchronously ---
    loop = asyncio.get_running_loop()
    response_text = await loop.run_in_executor(
        None,
        lambda: llm_pipe(
            prompt,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.4,
            top_p=0.85,
            repetition_penalty=2.0
        )[0]["generated_text"]
    )

    # --- Update memory ---
    mem = user_memories.setdefault(user_id, [])
    mem.append({"role": "user", "text": final_input})
    mem.append({"role": "assistant", "text": response_text})
    if len(mem) > MEMORY_WINDOW_TURNS * 2:
        user_memories[user_id] = mem[-(MEMORY_WINDOW_TURNS * 2):]

    matched_ids = [r["id"] for r in filtered_retrieved] if filtered_retrieved else []
    scores = [r["score"] for r in filtered_retrieved] if filtered_retrieved else []

    return {
        "user_id": user_id,
        "transcript": transcript,
        "input": final_input,
        "matched_ids": matched_ids,
        "scores": scores,
        "used_retrieval": use_retrieval,
        "response": response_text
    }

# ------------------------
# Memory endpoints
# ------------------------
@app.get("/memory/{user_id}")
async def view_memory(user_id: str):
    return {"user_id": user_id, "memory": user_memories.get(user_id, [])}

@app.post("/memory/clear")
async def clear_memory(user_id: str = Form(...)):
    user_memories.pop(user_id, None)
    return {"ok": True, "cleared_user": user_id}

# ------------------------
# Save/load FAISS endpoints
# ------------------------
@app.post("/save_index")
async def save_index_endpoint():
    return {"saved": save_index_to_disk()}

@app.post("/load_index")
async def load_index_endpoint():
    return {"loaded": load_index_from_disk()}

# ------------------------
# Health check
# ------------------------
@app.get("/health")
def health():
    return {"ok": True, "index_chunks": len(docs_texts)}
