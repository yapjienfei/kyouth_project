"""Career Pathway Navigator — Backend API (Flask)

Implements Claude1.md modules 1–6:
  Module 1 — Data pipeline (load JSON, external sources)
  Module 2 — Skill extraction (Gemini + taxonomy + keyword fallback)
  Module 3 — Deterministic calculations (skill gap, replaceability, salary delta, confidence)
  Module 4 — RAG retrieval (PDF text extraction + keyword scoring)
  Module 5 — Career path computation (graph + BFS)
  Module 6 — Explanation generation (template-based LLM, validated, cached)
"""
import json
import math
import os
import re
import sys
import tempfile
import time
from collections import defaultdict, deque
from pathlib import Path

import pdfplumber
from google import genai
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

sys.path.insert(0, str(Path(__file__).parent))
import db
import career_graph

try:
    import redis as redis_lib
except ImportError:
    redis_lib = None

load_dotenv(Path(__file__).parent.parent / ".env")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

DATA_DIR = Path(__file__).parent.parent / "data"
DOCS_DIR = DATA_DIR / "docs"

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1 — DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def _load_json(path: Path, default):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return default


def _load_jobs() -> tuple[list, bool]:
    snapshot = DATA_DIR / "jobs_snapshot.json"
    raw = DATA_DIR / "jobs.json"
    if snapshot.exists():
        jobs = _load_json(snapshot, [])
        print(f"[data] Loaded {len(jobs)} jobs from jobs_snapshot.json")
        return jobs, True
    if raw.exists():
        jobs = _load_json(raw, [])
        print(f"[data] Loaded {len(jobs)} raw jobs (no skills). Run scripts/extract_job_skills_with_gemini.py.")
        return jobs, False
    print("[data] No job data found!")
    return [], False


DB_AVAILABLE = db.init_db()
_db_jobs = db.load_jobs() if DB_AVAILABLE else None
if _db_jobs:
    JOBS_RAW = _db_jobs
    HAS_SKILLS = any(j.get("skills_ai") for j in JOBS_RAW)
    JOBS_SOURCE = "postgresql"
    print(f"[data] Loaded {len(JOBS_RAW)} jobs from PostgreSQL")
else:
    JOBS_RAW, HAS_SKILLS = _load_jobs()
    JOBS_SOURCE = "json"

SKILL_INTELLIGENCE: dict = _load_json(DATA_DIR / "skill_intelligence.json", {})
SKILL_LEARNING_TIMES: dict = _load_json(DATA_DIR / "skill_learning_times.json", {})

_taxonomy_data: dict = _load_json(DATA_DIR / "skill_taxonomy.json", {})
TAXONOMY_SKILLS: list[str] = _taxonomy_data.get("skills", [])
TAXONOMY_CATEGORIES: dict = _taxonomy_data.get("categories", {})

EXTERNAL_SOURCES: list[dict] = _load_json(DATA_DIR / "external_sources.json", [])

# Extract WEF replaceability map (0–100 scale)
WEF_REPLACEABILITY: dict[str, int] = {}
for _src in EXTERNAL_SOURCES:
    if "WEF" in _src.get("source_name", ""):
        WEF_REPLACEABILITY = _src.get("replaceability_scores", {})
        break

# Critical occupations from MyCOL
CRITICAL_OCCUPATIONS: set[str] = set()
for _src in EXTERNAL_SOURCES:
    if "MyCOL" in _src.get("source_name", "") or "TalentCorp" in _src.get("source_name", ""):
        for occ in _src.get("critical_occupations", []):
            CRITICAL_OCCUPATIONS.add(occ.lower())
        break

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4 — RAG RETRIEVER (PDF text extraction + keyword scoring)
# ─────────────────────────────────────────────────────────────────────────────

class RAGRetriever:
    def __init__(self):
        self.chunks: list[dict] = []
        self._ready = False
        # In Flask debug mode the reloader parent also imports this module —
        # only the serving child (WERKZEUG_RUN_MAIN=true) should index, or two
        # processes parse 500+ PDF pages at once and starve the container.
        is_reloader_parent = (
            os.environ.get("FLASK_ENV") == "development"
            and os.environ.get("WERKZEUG_RUN_MAIN") != "true"
        )
        if is_reloader_parent:
            print("[rag] reloader parent process — skipping PDF indexing")
            return
        # Load in background so Flask starts immediately
        import threading
        threading.Thread(target=self._index_pdfs, daemon=True).start()

    def _index_pdfs(self):
        if not DOCS_DIR.exists():
            print("[rag] No docs/ directory found — RAG context disabled.")
            return
        pdf_files = list(DOCS_DIR.glob("*.pdf"))
        if not pdf_files:
            print("[rag] No PDF files found in docs/ — RAG context disabled.")
            return

        # Bump CHUNK_VERSION whenever chunking parameters change so stale
        # caches (e.g. built with an old page cap) are rebuilt automatically
        CHUNK_VERSION = 2   # v2: all pages indexed (no MAX_PAGES cap)

        # Use cached text file if it exists (avoids re-parsing large PDFs)
        cache_path = DATA_DIR / "rag_cache.json"
        if cache_path.exists():
            try:
                payload = _load_json(cache_path, {})
                if isinstance(payload, dict) and payload.get("version") == CHUNK_VERSION:
                    cached = payload["chunks"]
                    for c in cached:
                        c["word_set"] = set(c.pop("word_list", []))
                    self.chunks = cached
                    self._ready = True
                    print(f"[rag] Loaded {len(self.chunks)} chunks from cache (v{CHUNK_VERSION})")
                    return
                print("[rag] Cache is from an older chunking version — re-indexing all pages")
            except Exception as e:
                print(f"[rag] Cache load failed ({e}), re-indexing PDFs")

        chunks = []
        for pdf_path in pdf_files:
            try:
                pdf_chunks_before = len(chunks)
                with pdfplumber.open(pdf_path) as pdf:
                    total_pages = len(pdf.pages)
                    for page_num, page in enumerate(pdf.pages, 1):
                        text = page.extract_text() or ""
                        page.flush_cache()   # cap memory: drop parsed page objects
                        time.sleep(0.01)     # yield the GIL so requests stay responsive
                        if len(text.strip()) < 80:
                            continue  # skip low-content pages (page-number spreads, image pages)
                        words = text.split()
                        for i in range(0, len(words), 250):
                            chunk_words = words[i: i + 300]
                            chunk_text = " ".join(chunk_words)
                            chunks.append({
                                "text": chunk_text,
                                "source": pdf_path.stem.replace("_", " "),
                                "page": page_num,
                                "word_set": set(w.lower() for w in chunk_words),
                            })
                        if page_num % 50 == 0:
                            print(f"[rag] {pdf_path.name}: {page_num}/{total_pages} pages…")
                new_chunks = len(chunks) - pdf_chunks_before
                print(f"[rag] {pdf_path.name}: {total_pages} pages → {new_chunks} chunks")
            except Exception as e:
                print(f"[rag] Failed to load {pdf_path.name}: {e}")

        self.chunks = chunks
        self._ready = True
        print(f"[rag] Total: {len(chunks)} chunks from {len(pdf_files)} PDFs")

        # Persist cache (convert sets to lists for JSON)
        try:
            serialisable = [
                {**{k: v for k, v in c.items() if k != "word_set"}, "word_list": list(c["word_set"])}
                for c in chunks
            ]
            with open(cache_path, "w") as f:
                json.dump({"version": CHUNK_VERSION, "chunks": serialisable}, f)
            print(f"[rag] Cache saved to {cache_path.name}")
        except Exception as e:
            print(f"[rag] Cache save failed: {e}")

    def retrieve(self, query_terms: list[str], top_k: int = 3) -> list[dict]:
        if not self.chunks or not query_terms:
            return []
        query_words = set(w.lower() for term in query_terms for w in term.split())
        scored = [
            (len(query_words & c["word_set"]) / max(len(query_words), 1), c)
            for c in self.chunks
            if query_words & c["word_set"]
        ]
        scored.sort(key=lambda x: -x[0])
        return [c for _, c in scored[:top_k]]


rag = RAGRetriever()

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 5 — CAREER GRAPH
# ─────────────────────────────────────────────────────────────────────────────

class JobNode:
    __slots__ = ("title", "company", "salary_min", "salary_max", "salary_median",
                 "salary_display", "skills")

    def __init__(self, d: dict):
        self.title = d["title"]
        self.company = d["company"]
        self.salary_min = d.get("salary_min", 0) or 0
        self.salary_max = d.get("salary_max", 0) or 0
        self.salary_median = (self.salary_min + self.salary_max) / 2 if self.salary_max > 0 else 0
        self.salary_display = d.get("salary_display", "Undisclosed")
        self.skills = frozenset(s.lower() for s in d.get("skills_ai", []))


job_nodes: list[JobNode] = [JobNode(j) for j in JOBS_RAW]
if HAS_SKILLS:
    _before = len(job_nodes)
    job_nodes = [n for n in job_nodes if n.skills]
    print(f"[graph] {len(job_nodes)} jobs with skills (filtered {_before - len(job_nodes)} empty)")

# ── Graph computed by the Celery worker, delivered via Redis ────────────────
GRAPH_KEY = "career:graph"
GRAPH_VERSION_KEY = "career:graph:version"

title_stats: dict[str, dict] = {}
edges: dict[str, list[tuple]] = {}
ALL_SKILLS: list[str] = []
GRAPH_SOURCE = "local"
_graph_version_seen = ""

_redis = None
if redis_lib is not None:
    try:
        _redis = redis_lib.from_url(REDIS_URL, socket_connect_timeout=2)
        _redis.ping()
    except Exception as e:
        print(f"[graph] Redis unavailable ({e})")
        _redis = None


def _apply_graph(data: dict, source: str):
    global title_stats, edges, ALL_SKILLS, GRAPH_SOURCE
    title_stats, edges = career_graph.deserialize_graph(data)
    ALL_SKILLS = sorted(set(s for st in title_stats.values() for s in st["skills"]))
    GRAPH_SOURCE = source
    print(f"[graph] {len(title_stats)} unique titles, "
          f"{sum(len(v) for v in edges.values())} edges (source: {source})")


def _load_graph_from_redis() -> bool:
    global _graph_version_seen
    if _redis is None:
        return False
    try:
        raw = _redis.get(GRAPH_KEY)
        if not raw:
            return False
        _apply_graph(json.loads(raw), "celery")
        _graph_version_seen = (_redis.get(GRAPH_VERSION_KEY) or b"").decode()
        return True
    except Exception as e:
        print(f"[graph] redis load failed ({e})")
        return False


def _init_graph():
    """Prefer the Celery-computed graph from Redis; enqueue a rebuild and wait
    briefly; compute locally as last resort so the API always starts."""
    if _redis is not None:
        try:
            from tasks import rebuild_graph
            rebuild_graph.delay()
            print("[graph] rebuild task enqueued")
        except Exception as e:
            print(f"[graph] could not enqueue celery task ({e})")
        for _ in range(20):  # wait up to ~10s for the worker
            if _load_graph_from_redis():
                return
            time.sleep(0.5)
    _apply_graph(career_graph.build_graph(JOBS_RAW), "local")


def _maybe_refresh_graph():
    """Pick up a newer worker-computed graph without restarting (throttled)."""
    global _graph_version_seen
    if _redis is None:
        return
    now = time.time()
    if now - getattr(_maybe_refresh_graph, "_last", 0) < 15:
        return
    _maybe_refresh_graph._last = now
    try:
        version = (_redis.get(GRAPH_VERSION_KEY) or b"").decode()
        if version and version != _graph_version_seen:
            _load_graph_from_redis()
    except Exception:
        pass


_init_graph()

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2 — SKILL EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _gemini_call(prompt: str, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            resp = gemini_client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            return resp.text.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 5 * (2 ** attempt)
            print(f"[gemini] retry {attempt + 1}: {e}, waiting {wait}s")
            time.sleep(wait)


def _parse_json(raw: str):
    for prefix in ("```json", "```"):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
    if raw.endswith("```"):
        raw = raw[:-3]
    return json.loads(raw.strip())


def _keyword_fallback(text: str) -> list[str]:
    text_lower = text.lower()
    return [s for s in TAXONOMY_SKILLS if s in text_lower]


# Surface forms a skill may take in the source text — Gemini normalizes these
# to the canonical name, so verification must accept them as evidence
_SKILL_SURFACE_FORMS: dict[str, list[str]] = {
    "kubernetes": ["k8s"],
    "javascript": ["js", "es6"],
    "typescript": ["ts"],
    "react": ["reactjs", "react.js"],
    "node.js": ["nodejs", "node js", "node"],
    "postgresql": ["postgres", "postgre"],
    "asp.net": [".net", "dotnet", "dot net"],
    "html": ["html5"],
    "css": ["css3"],
    "aws": ["amazon web services"],
    "gcp": ["google cloud", "google cloud platform"],
    "azure": ["microsoft azure"],
    "ci/cd": ["cicd", "ci cd", "ci-cd"],
    "machine learning": ["ml"],
    "artificial intelligence": ["ai"],
    "rest api": ["rest", "restful"],
}

# Server-side sanitization: lowercase tech-skill shaped, max 40 chars — blocks
# HTML/script payloads smuggled through a crafted "resume"
_VALID_SKILL_RE = re.compile(r"^[a-z0-9][a-z0-9 .+#/&'-]{0,39}$")

_RESUME_HINTS = ("experience", "education", "skill", "employment", "career",
                 "project", "certification", "university", "degree", "summary",
                 "objective", "intern", "graduate", "work history")


def _looks_like_resume(text: str) -> bool:
    tl = text.lower()
    return sum(1 for h in _RESUME_HINTS if h in tl) >= 2


def _skill_in_text(skill: str, text_lower: str) -> bool:
    forms = [skill] + _SKILL_SURFACE_FORMS.get(skill, [])
    return any(_mentions_skill(text_lower, f) for f in forms)


def _verify_skills(skills: list[str], text: str, max_skills: int = 60) -> list[str]:
    """Hallucination guard: keep only sanitized skills that verifiably appear
    in the source text (canonical name or a known surface form)."""
    text_lower = text.lower()
    out, seen = [], set()
    for s in skills:
        s = s.strip().lower()
        if not s or s in seen or not _VALID_SKILL_RE.match(s):
            continue
        if not _skill_in_text(s, text_lower):
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= max_skills:
            break
    return out


def extract_skills(text: str) -> list[str]:
    """Module 2.2/2.4: Gemini extraction with taxonomy normalization + keyword
    fallback. Output is verified against the source text so a skill can never
    be reported that the document doesn't actually mention."""
    taxonomy_hint = ", ".join(TAXONOMY_SKILLS[:70]) if TAXONOMY_SKILLS else ""
    prompt = f"""Extract only technical skills from the resume below.
Rules:
- Return a JSON array of lowercase strings.
- Normalize synonyms: K8s→kubernetes, JS→javascript, ReactJS→react, Postgres→postgresql,
  CSS3→css, HTML5→html, NodeJS→node.js, .NET→asp.net.
- Exclude ALL soft skills (communication, leadership, teamwork, problem-solving, etc.).
- Keep skills from this standard list where possible: {taxonomy_hint}
- Only list skills that are actually mentioned in the text. If the text mentions
  no technical skills, return [].
- Output ONLY valid JSON – no extra text.

Resume:
{text[:4000]}"""
    try:
        raw = _gemini_call(prompt)
        skills = _parse_json(raw)
        if not isinstance(skills, list):
            raise ValueError("not a list")
        skills = [s for s in skills if isinstance(s, str)]
    except Exception as e:
        print(f"[extract] Gemini failed ({e}), using keyword fallback")
        skills = _keyword_fallback(text)
    return _verify_skills(skills, text)

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3 — DETERMINISTIC CALCULATIONS
# ─────────────────────────────────────────────────────────────────────────────

def skill_gap(user_skills: set, required_skills: frozenset) -> tuple[frozenset, float]:
    """3.1 — Missing skills and gap percentage."""
    if not required_skills:
        return frozenset(), 0.0
    missing = required_skills - user_skills
    pct = len(missing) / len(required_skills) * 100
    return missing, round(pct, 1)


def ai_replaceability(role_skills: frozenset, user_skills: set) -> tuple[float, str]:
    """3.2 — Weighted AI replaceability score (0–100), personalised by user low-risk skills."""
    if not role_skills:
        return 50.0, "default (no skills data)"

    total = 0
    weighted_sum = 0.0
    has_low_risk = False

    for sk in role_skills:
        sk_lower = sk.lower()
        if sk_lower in WEF_REPLACEABILITY:
            score = WEF_REPLACEABILITY[sk_lower]        # 0–100 scale
        elif sk_lower in SKILL_INTELLIGENCE:
            score = SKILL_INTELLIGENCE[sk_lower].get("ai_replaceability_score", 5) * 10
        else:
            score = 50
        weighted_sum += score
        total += 1
        if score < 30 and sk_lower in user_skills:
            has_low_risk = True

    job_score = weighted_sum / total
    personalised = job_score * (1 - 0.2 * (1 if has_low_risk else 0))
    source = "WEF Future of Jobs Report 2025"
    return round(personalised, 1), source


def salary_delta(current_title: str, target_title: str) -> float | None:
    """3.3 — Salary delta between two roles (RM/month)."""
    cur = title_stats.get(current_title, {}).get("salary_median", 0)
    tgt = title_stats.get(target_title, {}).get("salary_median", 0)
    if not cur or not tgt:
        return None
    return round(tgt - cur, 0)


def confidence_score(num_postings: int, days_since_refresh: int = 0, gap_pct: float = 0.0) -> float:
    """3.4 — Composite confidence score 0–100."""
    score = (
        0.4 * min(1.0, num_postings / 100)
        + 0.3 * (1.0 - days_since_refresh / 365)
        + 0.3 * (1.0 - gap_pct / 100)
    ) * 100
    return round(score, 1)


def learning_weeks(skill: str) -> int:
    return SKILL_LEARNING_TIMES.get(skill.lower(), 4)


def total_learning_weeks(skills) -> int:
    return sum(learning_weeks(s) for s in skills)

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 6 — EXPLANATION GENERATION
# ─────────────────────────────────────────────────────────────────────────────

_explanation_cache: dict[str, dict] = {}
_CACHE_TTL = 86400  # 24 hours


def _validate_explanation(result: dict, missing_skills: list[str]) -> bool:
    """Hallucination guard: ensure skills list is non-empty and contains only known skills."""
    skills_out = result.get("skills", [])
    if not skills_out:
        return False
    returned_names = {item.get("skill", "").lower() for item in skills_out}
    known = set(s.lower() for s in missing_skills)
    # Allow if at least half the returned skills are from the known list
    overlap = returned_names & known
    return len(overlap) / max(len(returned_names), 1) >= 0.5


def _mentions_skill(text: str, skill: str) -> bool:
    """Word-boundary match so 'sql' doesn't match inside 'mysql'. Handles
    skills ending in non-word chars like 'c#' or 'c++'."""
    pattern = r"(?<!\w)" + re.escape(skill.lower()) + r"(?!\w)"
    return re.search(pattern, text.lower()) is not None


# Terms indicating a passage discusses market demand — used to rank evidence
_DEMAND_TERMS = {"demand", "shortage", "shortages", "critical", "talent", "hiring",
                 "skills", "skilled", "emerging", "growing", "growth", "required"}

_URL_RE = re.compile(r"https?://\S+|www\.\S+")


def _skill_outside_urls(text: str, skill: str) -> bool:
    """Match the skill with URLs removed first, so 'index.php?title=' can never
    count as a mention of 'php'."""
    return _mentions_skill(_URL_RE.sub(" ", text), skill)


def _evidence_score(chunk: dict, role_words: set) -> int:
    """Rank evidence pages: prefer ones that also mention the role's domain
    and demand/shortage language; penalise bibliography/reference pages,
    which are dense with URLs and citation noise."""
    words = chunk["word_set"]
    score = 2 * len(role_words & words) + len(_DEMAND_TERMS & words)
    score -= 5 * len(_URL_RE.findall(chunk["text"]))
    return score


def _supporting_sentence(text: str, skill: str) -> str | None:
    """The verbatim sentence from the page that mentions the skill —
    deterministic, so the quote can never be hallucinated. Skips citation
    runs (over-long 'sentences') and matches that only occur inside URLs."""
    for sent in re.split(r"(?<=[.!?])\s+", text):
        sent = sent.strip()
        if len(sent) > 450:
            continue  # un-punctuated reference lists masquerading as one sentence
        if _skill_outside_urls(sent, skill):
            return sent
    return None


def _best_evidence(skill: str, role_words: set) -> dict | None:
    """Single best document citation for a skill: the highest-ranked page that
    literally mentions it, with the verbatim supporting sentence and the full
    excerpt for highlighting in the UI."""
    if len(skill) < 3:
        # 1–2 letter skills ('r', 'c', 'go') false-match author initials and
        # common words in reports — citation evidence is unreliable for them
        return None
    hits = [c for c in rag.chunks if _skill_outside_urls(c["text"], skill)]
    if not hits:
        return None
    best = max(hits, key=lambda c: _evidence_score(c, role_words))
    quote = _supporting_sentence(best["text"], skill)
    if not quote:
        return None
    return {
        "source": best["source"],
        "page": best["page"],
        "quote": quote,
        "context": best["text"][:1500],
    }


def generate_explanation(target_title: str, missing_skills: list[str]) -> dict:
    """Module 6: Template-based LLM explanation with RAG context + hallucination guard."""
    cache_key = f"{target_title}|{'|'.join(sorted(missing_skills))}"
    now = time.time()
    if cache_key in _explanation_cache:
        cached = _explanation_cache[cache_key]
        if now - cached["ts"] < _CACHE_TTL:
            return cached["data"]

    # Module 4: retrieve relevant RAG context
    query_terms = [target_title] + missing_skills[:5]
    rag_chunks = rag.retrieve(query_terms, top_k=3)

    # Per-skill document evidence: best single page that literally mentions
    # the skill, ranked by role/demand relevance, with verbatim quote
    role_words = set(target_title.lower().split())
    skill_evidence: dict[str, dict] = {}
    for s in missing_skills[:10]:
        ev = _best_evidence(s.lower(), role_words)
        if ev:
            skill_evidence[s.lower()] = ev

    # Context for Gemini = role-level chunks + per-skill evidence quotes (deduped)
    context_chunks, seen_pages = [], set()
    for c in rag_chunks:
        key = (c["source"], c["page"])
        if key not in seen_pages:
            seen_pages.add(key)
            context_chunks.append(c)
    context_chunks = context_chunks[:5]
    context_text = ""
    if context_chunks:
        context_text = "\n\n".join(
            f"[{c['source']}, p.{c['page']}]\n{c['text'][:400]}" for c in context_chunks
        )

    evidence_text = "\n".join(
        f'- {s}: "{ev["quote"][:250]}" ({ev["source"]}, p.{ev["page"]})'
        for s, ev in skill_evidence.items()
    )

    skills_list = ", ".join(missing_skills[:10])
    prompt = f"""You are a career advisor for IT professionals in Malaysia.
Target role: {target_title}
Skills the user still needs to learn: {skills_list}
{"Context from industry reports:\n" + context_text if context_text else ""}
{"Documented Malaysian market evidence for specific skills (ground your explanation in these where relevant):\n" + evidence_text if evidence_text else ""}

For EACH skill listed above, output:
- skill: exact skill name from the list above
- importance_score: integer 1–10 (10 = absolutely critical for {target_title})
- explanation: one sentence (max 20 words) on WHY this skill is needed for {target_title}
- suggested_order: integer (1 = learn first, based on prerequisites)

Output ONLY valid JSON, no markdown:
{{
  "skills": [
    {{"skill": "skill_name", "importance_score": 8, "explanation": "Reason.", "suggested_order": 1}}
  ]
}}"""

    try:
        raw = _gemini_call(prompt)
        result = _parse_json(raw)
        if not _validate_explanation(result, missing_skills):
            raise ValueError("Explanation failed hallucination check")
        # Deterministic per-skill attribution: one best citation per skill,
        # with the verbatim sentence pulled from the page (never LLM-claimed)
        for sk in result.get("skills", []):
            name = str(sk.get("skill", "")).lower()
            ev = skill_evidence.get(name)
            if ev is None and name:
                ev = _best_evidence(name, role_words)
            sk["evidence"] = ev  # {source, page, quote, context} or None
        result["rag_sources"] = [
            {"source": c["source"], "page": c["page"]} for c in context_chunks
        ]
        _explanation_cache[cache_key] = {"ts": now, "data": result}
        return result
    except Exception as e:
        print(f"[explain] Error: {e}")
        return {"error": str(e), "skills": []}

# ─────────────────────────────────────────────────────────────────────────────
# CAREER PATH FINDER (Module 5 — BFS)
# ─────────────────────────────────────────────────────────────────────────────

def _format_step(title: str, curr_skills: set) -> dict:
    stats = title_stats[title]
    role_skills = stats["skills"]
    missing, gap_pct = skill_gap(curr_skills, role_skills)
    repl, repl_src = ai_replaceability(role_skills, curr_skills)
    conf = confidence_score(stats["count"], 0, gap_pct)
    mw = [{"skill": s, "weeks": learning_weeks(s)} for s in missing]
    return {
        "title": title,
        "companies": stats.get("companies", []),
        "salary_min": stats["salary_min"],
        "salary_max": stats["salary_max"],
        "salary_median": round(stats["salary_median"]),
        "job_count": stats["count"],
        "skills": list(role_skills),
        "missing_skills": list(missing),
        "missing_skills_with_weeks": mw,
        "missing_weeks": total_learning_weeks(missing),
        "skill_gap_pct": gap_pct,
        "ai_replaceability": repl,
        "replaceability_source": repl_src,
        "confidence_score": conf,
        "freshness": "Data from 2025 – 0 months old",
        "data_source": "JobStreet Job Listings 2025",
    }


MAX_PATH_DEPTH = 6   # shared by forward and backward search so results agree


def find_career_paths(
    user_skills: set,
    target_title: str,
    include_undisclosed: bool = False,
    max_depth: int = MAX_PATH_DEPTH,
) -> tuple[list | None, str | None]:
    if target_title not in title_stats:
        return None, "Target role not found in database"

    target_stats = title_stats[target_title]
    if not include_undisclosed and not target_stats["has_salary"]:
        return None, "Target role has undisclosed salary – enable 'include undisclosed' to proceed"

    paths: list[list[str]] = []
    # BFS queue: (current_combined_skills, path_so_far, visited_set)
    queue: deque = deque([(set(user_skills), [], set())])

    while queue and len(paths) < 3:
        curr_skills, path, visited = queue.popleft()
        if len(path) > max_depth:
            continue

        current_title = path[-1] if path else None

        if current_title == target_title:
            paths.append(path)
            continue

        # Find reachable next roles
        if current_title and current_title in edges:
            candidates = [(t, cost, m) for t, cost, m in edges[current_title]]
        else:
            # From user skills directly, find best matching roles
            candidates = []
            for title, stats in title_stats.items():
                if title in visited or not stats["skills"]:
                    continue
                missing_count = len(stats["skills"] - curr_skills)
                cost = missing_count / len(stats["skills"])
                if cost < 0.8:
                    candidates.append((title, cost, stats["skills"] - curr_skills))

        candidates.sort(key=lambda x: x[1])

        # Highest KNOWN salary anywhere earlier on this path — comparing only
        # adjacent steps would let salary drop across an undisclosed role
        # (e.g. 5k → undisclosed → 4k), since undisclosed (0) skips the check
        path_known_salary = max(
            (title_stats[t]["salary_median"] for t in path
             if title_stats[t]["salary_median"] > 0),
            default=0,
        )
        target_salary = target_stats["salary_median"]

        for next_title, cost, _ in candidates[:6]:
            if next_title in visited:
                continue
            next_stats = title_stats[next_title]
            if not include_undisclosed and not next_stats["has_salary"]:
                continue
            next_salary = next_stats["salary_median"]
            # Known salaries must be strictly ascending along the whole path,
            # even when undisclosed-salary roles sit in between
            if path_known_salary > 0 and next_salary > 0 and next_salary <= path_known_salary:
                continue
            # Intermediates must stay below the target's salary, or the path
            # could never ascend into the target
            if next_title != target_title and target_salary > 0 and next_salary >= target_salary:
                continue
            new_skills = curr_skills | next_stats["skills"]
            new_path = path + [next_title]
            new_visited = visited | {next_title}
            if next_title == target_title:
                paths.append(new_path)
            else:
                queue.append((new_skills, new_path, new_visited))

    # Direct path fallback if BFS found nothing
    if not paths:
        target_skills = target_stats["skills"]
        gap_pct = len(target_skills - user_skills) / max(len(target_skills), 1) * 100
        if gap_pct < 80:
            paths = [[target_title]]
        else:
            return None, f"No reachable path found to '{target_title}' with current skills"

    formatted = []
    for path in paths:
        steps = []
        curr = set(user_skills)
        for title in path:
            step = _format_step(title, curr)
            steps.append(step)
            curr |= title_stats[title]["skills"]
        formatted.append({"steps": steps, "total_weeks": sum(s["missing_weeks"] for s in steps)})

    formatted.sort(key=lambda x: x["total_weeks"])
    return formatted, None


def find_nearest_entry(
    user_skills: set,
    target_title: str,
    include_undisclosed: bool = False,
    max_depth: int = MAX_PATH_DEPTH,
) -> dict | None:
    """Backward BFS from the target: among all roles with a valid
    (ascending-salary, cost < 0.8) chain into the target, find the one the
    user is closest to entering today. Skill accumulation can't be computed
    backward, so the chosen chain is re-formatted forward with _format_step —
    the numbers shown match the forward search exactly."""
    if target_title not in title_stats:
        return None
    if not include_undisclosed and not title_stats[target_title]["has_salary"]:
        return None

    # Reverse adjacency: which roles have an edge INTO this one?
    reverse_adj: dict[str, list[str]] = defaultdict(list)
    for src, lst in edges.items():
        for dst, _cost, _missing in lst:
            reverse_adj[dst].append(src)

    succ: dict[str, str | None] = {target_title: None}   # role → next role toward target
    depth: dict[str, int] = {target_title: 0}
    q: deque = deque([target_title])
    while q:
        v = q.popleft()
        if depth[v] >= max_depth:
            continue
        for u in reverse_adj.get(v, []):
            if u in succ:
                continue
            if not include_undisclosed and not title_stats[u]["has_salary"]:
                continue
            succ[u] = v
            depth[u] = depth[v] + 1
            q.append(u)

    # Nearest entry = smallest user skill gap; ties go to fewer hops
    best: tuple | None = None
    for title in succ:
        skills = title_stats[title]["skills"]
        if not skills:
            continue
        gap = len(skills - user_skills) / len(skills) * 100
        cand = (round(gap, 1), depth[title], title)
        if best is None or cand < best:
            best = cand
    if best is None:
        return None
    gap_pct, hops, entry = best

    chain = []
    t: str | None = entry
    while t is not None:
        chain.append(t)
        t = succ[t]

    steps = []
    curr = set(user_skills)
    for title in chain:
        steps.append(_format_step(title, curr))
        curr |= title_stats[title]["skills"]

    return {
        "entry_title": entry,
        "entry_gap_pct": gap_pct,
        "hops_to_target": hops,
        "reachable_now": gap_pct < 80,
        "chain": chain,
        "steps": steps,
        "total_weeks": sum(s["missing_weeks"] for s in steps),
    }

# ─────────────────────────────────────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder=None)
CORS(app)


@app.before_request
def _refresh_graph_hook():
    _maybe_refresh_graph()


@app.route("/")
def index():
    return jsonify({"service": "Career Pathway Navigator API", "ui": "http://localhost:8080", "health": "/health"})


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "jobs_loaded": len(job_nodes),
        "jobs_source": JOBS_SOURCE,
        "unique_titles": len(title_stats),
        "has_skills": HAS_SKILLS,
        "rag_chunks": len(rag.chunks),
        "rag_ready": rag._ready,
        "rag_retrieval": "keyword overlap",
        "graph_source": GRAPH_SOURCE,
        "db_available": DB_AVAILABLE,
        "redis_available": _redis is not None,
        "skill_intelligence_loaded": bool(SKILL_INTELLIGENCE),
        "learning_times_loaded": bool(SKILL_LEARNING_TIMES),
        "setup_complete": HAS_SKILLS,
        "setup_instructions": (
            None if HAS_SKILLS
            else "Run: uv run python scripts/extract_job_skills_with_gemini.py"
        ),
    })


@app.route("/setup_status")
def setup_status():
    return jsonify({
        "jobs_snapshot": (DATA_DIR / "jobs_snapshot.json").exists(),
        "skill_intelligence": (DATA_DIR / "skill_intelligence.json").exists(),
        "skill_learning_times": (DATA_DIR / "skill_learning_times.json").exists(),
        "ready": HAS_SKILLS,
        "jobs_loaded": len(job_nodes),
        "unique_titles": len(title_stats),
        "rag_chunks": len(rag.chunks),
        "rag_ready": rag._ready,
    })


@app.route("/sources")
def sources():
    return jsonify(EXTERNAL_SOURCES)


MAX_RESUME_BYTES = 8 * 1024 * 1024   # 8 MB
MAX_RESUME_PAGES = 30


def _extract_response(text: str) -> dict:
    skills = extract_skills(text)
    resp = {"skills": skills, "text_length": len(text)}
    warnings = []
    if not _looks_like_resume(text):
        warnings.append("This document doesn't look like a resume/CV — "
                        "results may be incomplete or empty.")
    if not skills:
        warnings.append("No verifiable technical skills found in the text.")
    if warnings:
        resp["warning"] = " ".join(warnings)
    return resp


@app.route("/extract_skills", methods=["POST"])
def extract_skills_endpoint():
    # Accept either a multipart PDF upload or a JSON {"text": "..."} body
    if "resume" in request.files:
        f = request.files["resume"]
        if not f.filename:
            return jsonify({"error": "Empty filename"}), 400
        if request.content_length and request.content_length > MAX_RESUME_BYTES:
            return jsonify({"error": "PDF too large — maximum 8 MB"}), 413
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            f.save(tmp.name)
            tmp_path = tmp.name
        try:
            try:
                pdf = pdfplumber.open(tmp_path)
            except Exception:
                return jsonify({"error": "Could not read this file — it isn't a valid PDF, "
                                         "or it is password-protected."}), 400
            with pdf:
                n_pages = len(pdf.pages)
                text = "\n".join(page.extract_text() or "" for page in pdf.pages[:MAX_RESUME_PAGES])
            if not text.strip():
                return jsonify({"skills": [], "warning":
                                "No text found in the PDF — it may be a scanned image. "
                                "Try a text-based PDF, or type your skills manually."})
            resp = _extract_response(text)
            if n_pages > MAX_RESUME_PAGES:
                resp["warning"] = (resp.get("warning", "") +
                                   f" Only the first {MAX_RESUME_PAGES} of {n_pages} pages were read.").strip()
            return jsonify(resp)
        except Exception as e:
            print(f"[extract_skills] {e}")
            return jsonify({"skills": [], "error": "Failed to process the PDF."}), 500
        finally:
            os.unlink(tmp_path)
    else:
        data = request.get_json(silent=True) or {}
        text = str(data.get("text", "") or "").strip()
        if not text:
            return jsonify({"error": "Provide a 'resume' file or JSON body with 'text'"}), 400
        return jsonify(_extract_response(text[:100_000]))


@app.route("/job_titles")
def job_titles():
    return jsonify(sorted(title_stats.keys()))


@app.route("/all_skills")
def all_skills():
    return jsonify(ALL_SKILLS)


def _parse_skills(data: dict) -> set[str]:
    raw = data.get("skills")
    if not isinstance(raw, list):
        return set()
    return set(s.strip().lower() for s in raw if isinstance(s, str) and s.strip())


def _parse_int(value, default: int = 0, lo: int = 0, hi: int = 10**9) -> int:
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return default


@app.route("/suggest", methods=["POST"])
def suggest():
    data = request.get_json(silent=True) or {}
    user_skills = _parse_skills(data)
    if not user_skills:
        return jsonify({"error": "Provide a non-empty 'skills' list"}), 400
    salary_threshold = _parse_int(data.get("salary_threshold"), 0)
    include_undisclosed = bool(data.get("include_undisclosed", False))
    current_title = str(data.get("current_title", "") or "")

    user_id = data.get("user_id", "")
    if user_id and DB_AVAILABLE:
        db.log_search(user_id, "suggest", {
            "skills": sorted(user_skills),
            "salary_threshold": salary_threshold,
            "current_title": current_title,
        })
        db.snapshot_skills(user_id, list(user_skills))

    results = []
    for title, stats in title_stats.items():
        if not include_undisclosed and not stats["has_salary"]:
            continue
        if stats["has_salary"] and stats["salary_min"] < salary_threshold:
            continue
        if not stats["skills"]:
            continue

        missing, gap_pct = skill_gap(user_skills, stats["skills"])
        match_ratio = round(1 - gap_pct / 100, 3)
        repl, repl_src = ai_replaceability(stats["skills"], user_skills)
        conf = confidence_score(stats["count"], 0, gap_pct)
        sal_delta = salary_delta(current_title, title) if current_title else None
        mw = [{"skill": s, "weeks": learning_weeks(s)} for s in missing]
        is_critical = any(kw in title.lower() for kw in CRITICAL_OCCUPATIONS)

        # Trend data from skill intelligence
        skill_trends = []
        for sk in list(stats["skills"])[:5]:
            intel = SKILL_INTELLIGENCE.get(sk, {})
            if intel.get("trend") and intel["trend"] != "stable":
                skill_trends.append({"skill": sk, "trend": intel["trend"], "pct": intel.get("trend_pct", 0)})

        results.append({
            "title": title,
            "companies": stats.get("companies", []),
            "salary_min": stats["salary_min"],
            "salary_max": stats["salary_max"],
            "salary_median": round(stats["salary_median"]),
            "job_count": stats["count"],
            "match_ratio": match_ratio,
            "missing_skills": list(missing),
            "missing_skills_with_weeks": mw,
            "missing_weeks": total_learning_weeks(missing),
            "skill_gap_pct": gap_pct,
            "reachable": match_ratio >= 0.6,
            "undisclosed": not stats["has_salary"],
            "ai_replaceability": repl,
            "replaceability_source": repl_src,
            "confidence_score": conf,
            "salary_delta": sal_delta,
            "is_critical_occupation": is_critical,
            "skill_trends": skill_trends,
            "freshness": "Data from 2025 – 0 months old",
            "data_source": "JobStreet Job Listings 2025",
        })

    results.sort(key=lambda x: -x["match_ratio"])
    return jsonify(results)


@app.route("/find_paths", methods=["POST"])
def find_paths():
    data = request.get_json(silent=True) or {}
    user_skills = _parse_skills(data)
    if not user_skills:
        return jsonify({"error": "Provide a non-empty 'skills' list"}), 400
    target_title = str(data.get("target_title", "") or "").strip()
    if not target_title:
        return jsonify({"error": "Provide a 'target_title'"}), 400
    include_undisclosed = bool(data.get("include_undisclosed", False))

    user_id = data.get("user_id", "")
    if user_id and DB_AVAILABLE:
        db.log_search(user_id, "find_paths", {
            "skills": sorted(user_skills),
            "target_title": target_title,
        })
        db.snapshot_skills(user_id, list(user_skills))

    paths, error = find_career_paths(user_skills, target_title, include_undisclosed)
    nearest = find_nearest_entry(user_skills, target_title, include_undisclosed)
    if error and nearest is None:
        return jsonify({"error": error}), 404
    feedback = db.get_feedback(target_title) if DB_AVAILABLE else {}
    if error:
        # No complete forward path, but the backward search found guidance
        if nearest["hops_to_target"] == 0 and not nearest["reachable_now"]:
            msg = (
                f"No role in the dataset transitions into {target_title}, and your direct "
                f"skill gap is {nearest['entry_gap_pct']}%. The most impactful skills to "
                f"build first are shown below."
            )
        else:
            msg = (
                f"No complete path found with your current skills. The nearest entry role "
                f"toward {target_title} is '{nearest['entry_title']}' "
                f"(skill gap {nearest['entry_gap_pct']}%"
                + ("" if nearest["reachable_now"] else " — above the 80% transition threshold")
                + f", {nearest['hops_to_target']} hop(s) from the target)."
            )
        return jsonify({"paths": [], "target": target_title, "feedback": feedback,
                        "nearest_entry": nearest, "message": msg})
    return jsonify({"paths": paths, "target": target_title, "feedback": feedback,
                    "nearest_entry": nearest})


@app.route("/history", methods=["GET"])
def history():
    user_id = request.args.get("user_id", "")
    if not user_id:
        return jsonify({"error": "user_id required"}), 400
    return jsonify(db.get_history(user_id))


@app.route("/history/skills", methods=["POST"])
def history_skills():
    data = request.get_json(silent=True) or {}
    user_id = str(data.get("user_id", "") or "")
    skills = sorted(_parse_skills(data))
    if not user_id:
        return jsonify({"error": "user_id required"}), 400
    saved = db.snapshot_skills(user_id, skills) if DB_AVAILABLE else False
    return jsonify({"saved": saved})


@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json(silent=True) or {}
    user_id = str(data.get("user_id", "") or "")
    target_title = str(data.get("target_title", "") or "")
    path_titles = data.get("path_titles", [])
    rating = _parse_int(data.get("rating"), 99, lo=-1, hi=1)   # 0 = remove my vote
    if (not user_id or not target_title or rating not in (-1, 0, 1)
            or not isinstance(path_titles, list)
            or not all(isinstance(t, str) and t for t in path_titles)
            or not path_titles):
        return jsonify({"error": "user_id, target_title, path_titles (list of titles) and rating (1, -1, or 0 to remove) required"}), 400
    if not DB_AVAILABLE:
        return jsonify({"saved": False, "feedback": {}})
    if rating == 0:
        ok = db.delete_feedback(user_id, path_titles[:10])
    else:
        ok = db.save_feedback(user_id, target_title, path_titles[:10], rating)
    aggregate = db.get_feedback(target_title)
    return jsonify({"saved": ok, "feedback": aggregate})


@app.route("/recompute_graph", methods=["POST"])
def recompute_graph():
    try:
        from tasks import rebuild_graph
        task = rebuild_graph.delay()
        return jsonify({"enqueued": True, "task_id": task.id})
    except Exception as e:
        return jsonify({"enqueued": False, "error": str(e)}), 503


@app.route("/explain_path", methods=["POST"])
def explain_path():
    data = request.get_json(silent=True) or {}
    target_title = str(data.get("target_title", "") or "").strip()
    raw_missing = data.get("missing_skills", [])
    missing_skills = [s.strip().lower() for s in raw_missing
                      if isinstance(s, str) and s.strip()] if isinstance(raw_missing, list) else []
    if not target_title or not missing_skills:
        return jsonify({"error": "Missing target_title or missing_skills"}), 400
    result = generate_explanation(target_title, missing_skills[:15])
    if result.get("error") and not result.get("skills"):
        return jsonify(result), 500
    return jsonify(result)


@app.route("/skill_demand")
def skill_demand():
    limit = _parse_int(request.args.get("limit"), 30, lo=1, hi=200)
    threshold = _parse_int(request.args.get("threshold"), 7000)

    demand: dict[str, int] = defaultdict(int)
    for node in job_nodes:
        if node.salary_min >= threshold:
            for sk in node.skills:
                demand[sk] += 1

    result = []
    for skill, count in demand.items():
        intel = SKILL_INTELLIGENCE.get(skill, {})
        repl = WEF_REPLACEABILITY.get(skill, intel.get("ai_replaceability_score", 5) * 10)
        cat = TAXONOMY_CATEGORIES.get(skill, intel.get("skill_category", "Other"))
        trend = intel.get("trend", "stable")
        trend_pct = intel.get("trend_pct", 0)
        result.append({
            "skill": skill,
            "demand_count": count,
            "ai_replaceability_score": repl,
            "skill_category": cat,
            "trend": trend,
            "trend_pct": trend_pct,
            "freshness": "Data from 2025 – 0 months old",
            "source": "JobStreet Job Listings 2025",
        })

    result.sort(key=lambda x: -x["demand_count"])
    return jsonify({
        "skills": result[:limit],
        "total": len(result),
        "threshold": threshold,
        "source": "JobStreet Job Listings 2025",
        "wef_source": "WEF Future of Jobs Report 2025",
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=True)
