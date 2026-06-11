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
import tempfile
import time
from collections import defaultdict, deque
from pathlib import Path

import pdfplumber
from google import genai
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

load_dotenv(Path(__file__).parent.parent / ".env")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
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


JOBS_RAW, HAS_SKILLS = _load_jobs()

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

        # Use cached text file if it exists (avoids re-parsing large PDFs)
        cache_path = DATA_DIR / "rag_cache.json"
        if cache_path.exists():
            try:
                cached = _load_json(cache_path, [])
                for c in cached:
                    c["word_set"] = set(c.pop("word_list", []))
                self.chunks = cached
                self._ready = True
                print(f"[rag] Loaded {len(self.chunks)} chunks from cache")
                return
            except Exception as e:
                print(f"[rag] Cache load failed ({e}), re-indexing PDFs")

        MAX_PAGES = 100     # cap per PDF to limit first-run time

        chunks = []
        for pdf_path in pdf_files:
            try:
                pdf_chunks_before = len(chunks)
                with pdfplumber.open(pdf_path) as pdf:
                    pages_scanned = 0
                    for page_num, page in enumerate(pdf.pages[:MAX_PAGES], 1):
                        pages_scanned += 1
                        text = page.extract_text() or ""
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
                new_chunks = len(chunks) - pdf_chunks_before
                print(f"[rag] {pdf_path.name}: {pages_scanned} pages → {new_chunks} chunks")
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
                json.dump(serialisable, f)
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

# Aggregate per-title stats (multiple postings for same title)
_title_groups: dict[str, list[JobNode]] = defaultdict(list)
for n in job_nodes:
    _title_groups[n.title].append(n)

title_stats: dict[str, dict] = {}
for title, nodes in _title_groups.items():
    sal_nodes = [n for n in nodes if n.salary_median > 0]
    all_skills = frozenset().union(*(n.skills for n in nodes))
    title_stats[title] = {
        "count": len(nodes),
        "salary_median": sum(n.salary_median for n in sal_nodes) / len(sal_nodes) if sal_nodes else 0,
        "salary_min": min((n.salary_min for n in sal_nodes), default=0),
        "salary_max": max((n.salary_max for n in sal_nodes), default=0),
        "skills": all_skills,
        "has_salary": bool(sal_nodes),
    }

# Directed graph edges: A→B if B has higher median salary AND skill transition cost < 0.8
edges: dict[str, list[tuple]] = defaultdict(list)
for title_a, stats_a in title_stats.items():
    for title_b, stats_b in title_stats.items():
        if title_a == title_b:
            continue
        if stats_b["salary_median"] <= stats_a["salary_median"]:
            continue
        if not stats_b["skills"]:
            continue
        missing = stats_b["skills"] - stats_a["skills"]
        cost = len(missing) / len(stats_b["skills"])
        if cost < 0.8:
            edges[title_a].append((title_b, cost, missing))

ALL_SKILLS = sorted(set(s for st in title_stats.values() for s in st["skills"]))
print(f"[graph] {len(title_stats)} unique titles, {sum(len(v) for v in edges.values())} edges")

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


def extract_skills(text: str) -> list[str]:
    """Module 2.2/2.4: Gemini extraction with taxonomy normalization + keyword fallback."""
    taxonomy_hint = ", ".join(TAXONOMY_SKILLS[:70]) if TAXONOMY_SKILLS else ""
    prompt = f"""Extract only technical skills from the resume below.
Rules:
- Return a JSON array of lowercase strings.
- Normalize synonyms: K8s→kubernetes, JS→javascript, ReactJS→react, Postgres→postgresql,
  CSS3→css, HTML5→html, NodeJS→node.js, .NET→asp.net.
- Exclude ALL soft skills (communication, leadership, teamwork, problem-solving, etc.).
- Keep skills from this standard list where possible: {taxonomy_hint}
- Output ONLY valid JSON – no extra text.

Resume:
{text[:4000]}"""
    try:
        raw = _gemini_call(prompt)
        skills = _parse_json(raw)
        if not isinstance(skills, list):
            raise ValueError("not a list")
        return [s.lower() for s in skills if isinstance(s, str)]
    except Exception as e:
        print(f"[extract] Gemini failed ({e}), using keyword fallback")
        return _keyword_fallback(text)

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
    context_text = ""
    if rag_chunks:
        context_text = "\n\n".join(
            f"[{c['source']}, p.{c['page']}]\n{c['text'][:400]}" for c in rag_chunks
        )

    skills_list = ", ".join(missing_skills[:10])
    prompt = f"""You are a career advisor for IT professionals in Malaysia.
Target role: {target_title}
Skills the user still needs to learn: {skills_list}
{"Context from industry reports:\n" + context_text if context_text else ""}

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
        result["rag_sources"] = [
            {"source": c["source"], "page": c["page"]} for c in rag_chunks
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


def find_career_paths(
    user_skills: set,
    target_title: str,
    include_undisclosed: bool = False,
    max_depth: int = 4,
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

        for next_title, cost, _ in candidates[:6]:
            if next_title in visited:
                continue
            next_stats = title_stats[next_title]
            if not include_undisclosed and not next_stats["has_salary"]:
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

# ─────────────────────────────────────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder="../frontend/static", static_url_path="")
CORS(app)


@app.route("/")
def index():
    return send_from_directory("../frontend/static", "index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "jobs_loaded": len(job_nodes),
        "unique_titles": len(title_stats),
        "has_skills": HAS_SKILLS,
        "rag_chunks": len(rag.chunks),
        "rag_ready": rag._ready,
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


@app.route("/extract_skills", methods=["POST"])
def extract_skills_endpoint():
    # Accept either a multipart PDF upload or a JSON {"text": "..."} body
    if "resume" in request.files:
        f = request.files["resume"]
        if not f.filename:
            return jsonify({"error": "Empty filename"}), 400
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            f.save(tmp.name)
            tmp_path = tmp.name
        try:
            with pdfplumber.open(tmp_path) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            if not text.strip():
                return jsonify({"skills": [], "warning": "No text extracted from PDF"})
            skills = extract_skills(text)
            return jsonify({"skills": skills, "text_length": len(text)})
        except Exception as e:
            print(f"[extract_skills] {e}")
            return jsonify({"skills": [], "error": str(e)}), 500
        finally:
            os.unlink(tmp_path)
    else:
        data = request.get_json(silent=True) or {}
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "Provide a 'resume' file or JSON body with 'text'"}), 400
        skills = extract_skills(text)
        return jsonify({"skills": skills, "text_length": len(text)})


@app.route("/job_titles")
def job_titles():
    return jsonify(sorted(title_stats.keys()))


@app.route("/all_skills")
def all_skills():
    return jsonify(ALL_SKILLS)


@app.route("/suggest", methods=["POST"])
def suggest():
    data = request.get_json()
    user_skills = set(s.lower() for s in data.get("skills", []))
    salary_threshold = int(data.get("salary_threshold", 0))
    include_undisclosed = bool(data.get("include_undisclosed", False))
    current_title = data.get("current_title", "")

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
    data = request.get_json()
    user_skills = set(s.lower() for s in data.get("skills", []))
    target_title = data.get("target_title", "")
    include_undisclosed = bool(data.get("include_undisclosed", False))

    paths, error = find_career_paths(user_skills, target_title, include_undisclosed)
    if error:
        return jsonify({"error": error}), 404
    return jsonify({"paths": paths, "target": target_title})


@app.route("/explain_path", methods=["POST"])
def explain_path():
    data = request.get_json()
    target_title = data.get("target_title", "")
    missing_skills = data.get("missing_skills", [])
    if not target_title or not missing_skills:
        return jsonify({"error": "Missing target_title or missing_skills"}), 400
    result = generate_explanation(target_title, missing_skills)
    if result.get("error") and not result.get("skills"):
        return jsonify(result), 500
    return jsonify(result)


@app.route("/skill_demand")
def skill_demand():
    limit = int(request.args.get("limit", 30))
    threshold = int(request.args.get("threshold", 7000))

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
