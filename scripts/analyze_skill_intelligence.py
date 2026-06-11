#!/usr/bin/env python
"""
Module 3 (support): Analyze AI replaceability, category, and trend for each skill using Gemini.

Reads unique skills from data/jobs_snapshot.json, enriches with WEF-aligned AI
replaceability scores (1–10), skill category, and mock trend data.
Writes data/skill_intelligence.json.

Run after extract_job_skills_with_gemini.py.

Usage:
    uv run python scripts/analyze_skill_intelligence.py
"""
import json
import os
import time
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Set it in .env or environment.")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-3.1-flash-lite")

DATA_DIR = Path(__file__).parent.parent / "data"
SNAPSHOT_PATH = DATA_DIR / "jobs_snapshot.json"
OUTPUT_PATH = DATA_DIR / "skill_intelligence.json"

BATCH_SIZE = 20
MAX_RETRIES = 3
BASE_DELAY = 10

PROMPT_TEMPLATE = """You are a labour market analyst specializing in AI's impact on technical skills.
Based on the WEF Future of Jobs Report 2025 and industry trends, analyze each skill below.

For each skill, provide:
- ai_replaceability_score: integer 1–10 (1 = very hard to automate, 10 = easily automated)
  Examples: machine learning=2, html=8, python=4, sql=7, kubernetes=4
- skill_category: one of ["Programming Languages", "Web Frontend", "Web Backend", "Databases",
  "Cloud & DevOps", "Data Science & AI", "Mobile", "Security", "Architecture", "Testing", "Tools", "Other"]
- trend: one of ["growing", "stable", "declining"]
- trend_pct: integer percent change in demand year-over-year (positive=growing, negative=declining)
  Use realistic mock values based on known industry trends.

Output ONLY valid JSON in this format:
{{
  "skill_name": {{
    "ai_replaceability_score": 4,
    "skill_category": "Cloud & DevOps",
    "trend": "growing",
    "trend_pct": 22
  }},
  ...
}}
No extra text.

Skills to analyze: {skills}"""


def call_gemini_with_backoff(prompt: str) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = BASE_DELAY * (2 ** attempt)
            print(f"  Gemini error (attempt {attempt + 1}): {e}. Retrying in {wait}s...")
            time.sleep(wait)


def parse_json_response(raw: str) -> dict:
    if raw.startswith("```json"):
        raw = raw[7:]
    if raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    return json.loads(raw.strip())


# Sensible defaults so the app works even without Gemini enrichment
DEFAULTS = {
    "python": {"ai_replaceability_score": 4, "skill_category": "Programming Languages", "trend": "growing", "trend_pct": 18},
    "javascript": {"ai_replaceability_score": 6, "skill_category": "Programming Languages", "trend": "stable", "trend_pct": 5},
    "typescript": {"ai_replaceability_score": 5, "skill_category": "Programming Languages", "trend": "growing", "trend_pct": 25},
    "java": {"ai_replaceability_score": 5, "skill_category": "Programming Languages", "trend": "stable", "trend_pct": 2},
    "php": {"ai_replaceability_score": 6, "skill_category": "Programming Languages", "trend": "declining", "trend_pct": -8},
    "react": {"ai_replaceability_score": 6, "skill_category": "Web Frontend", "trend": "growing", "trend_pct": 15},
    "html": {"ai_replaceability_score": 8, "skill_category": "Web Frontend", "trend": "stable", "trend_pct": 0},
    "css": {"ai_replaceability_score": 8, "skill_category": "Web Frontend", "trend": "stable", "trend_pct": 1},
    "sql": {"ai_replaceability_score": 7, "skill_category": "Databases", "trend": "stable", "trend_pct": 3},
    "mysql": {"ai_replaceability_score": 6, "skill_category": "Databases", "trend": "stable", "trend_pct": -2},
    "postgresql": {"ai_replaceability_score": 6, "skill_category": "Databases", "trend": "growing", "trend_pct": 12},
    "mongodb": {"ai_replaceability_score": 6, "skill_category": "Databases", "trend": "stable", "trend_pct": 5},
    "aws": {"ai_replaceability_score": 4, "skill_category": "Cloud & DevOps", "trend": "growing", "trend_pct": 28},
    "azure": {"ai_replaceability_score": 4, "skill_category": "Cloud & DevOps", "trend": "growing", "trend_pct": 30},
    "docker": {"ai_replaceability_score": 4, "skill_category": "Cloud & DevOps", "trend": "growing", "trend_pct": 20},
    "kubernetes": {"ai_replaceability_score": 4, "skill_category": "Cloud & DevOps", "trend": "growing", "trend_pct": 32},
    "machine learning": {"ai_replaceability_score": 2, "skill_category": "Data Science & AI", "trend": "growing", "trend_pct": 45},
    "deep learning": {"ai_replaceability_score": 2, "skill_category": "Data Science & AI", "trend": "growing", "trend_pct": 50},
    "cybersecurity": {"ai_replaceability_score": 2, "skill_category": "Security", "trend": "growing", "trend_pct": 35},
    "git": {"ai_replaceability_score": 4, "skill_category": "Tools", "trend": "stable", "trend_pct": 2},
    "linux": {"ai_replaceability_score": 4, "skill_category": "Cloud & DevOps", "trend": "stable", "trend_pct": 5},
    "agile": {"ai_replaceability_score": 3, "skill_category": "Tools", "trend": "stable", "trend_pct": 2},
    "flutter": {"ai_replaceability_score": 5, "skill_category": "Mobile", "trend": "growing", "trend_pct": 22},
    "react native": {"ai_replaceability_score": 5, "skill_category": "Mobile", "trend": "stable", "trend_pct": 8},
}


def get_default(skill: str) -> dict:
    if skill in DEFAULTS:
        return DEFAULTS[skill]
    return {"ai_replaceability_score": 5, "skill_category": "Other", "trend": "stable", "trend_pct": 0}


def main():
    if not SNAPSHOT_PATH.exists():
        print(f"ERROR: {SNAPSHOT_PATH} not found. Run extract_job_skills_with_gemini.py first.")
        return

    print(f"Loading jobs from {SNAPSHOT_PATH}...")
    with open(SNAPSHOT_PATH) as f:
        jobs = json.load(f)

    all_skills = set()
    for job in jobs:
        all_skills.update(s.lower() for s in job.get("skills_ai", []))
    print(f"Found {len(all_skills)} unique skills.")

    # Load existing checkpoint
    existing = {}
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            existing = json.load(f)
        print(f"Checkpoint: {len(existing)} skills already processed.")

    remaining = [s for s in sorted(all_skills) if s not in existing]
    print(f"Need to process {len(remaining)} skills.")

    results = dict(existing)
    skill_list = list(remaining)
    total_batches = (len(skill_list) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_i in range(total_batches):
        batch = skill_list[batch_i * BATCH_SIZE: (batch_i + 1) * BATCH_SIZE]
        print(f"Batch {batch_i + 1}/{total_batches}: {batch}")

        try:
            prompt = PROMPT_TEMPLATE.format(skills=json.dumps(batch))
            raw = call_gemini_with_backoff(prompt)
            batch_result = parse_json_response(raw)

            for skill, data in batch_result.items():
                skill_lower = skill.lower()
                if isinstance(data, dict):
                    # Validate required fields
                    score = data.get("ai_replaceability_score", 5)
                    if not isinstance(score, int) or not (1 <= score <= 10):
                        score = 5
                    results[skill_lower] = {
                        "ai_replaceability_score": score,
                        "skill_category": data.get("skill_category", "Other"),
                        "trend": data.get("trend", "stable"),
                        "trend_pct": int(data.get("trend_pct", 0)),
                    }
        except Exception as e:
            print(f"  Batch failed: {e}. Using defaults.")
            for skill in batch:
                results[skill] = get_default(skill)

        with open(OUTPUT_PATH, "w") as f:
            json.dump(results, f, indent=2, sort_keys=True)
        time.sleep(5)

    # Fill any missing with defaults
    for skill in all_skills:
        if skill not in results:
            results[skill] = get_default(skill)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    print(f"\nDone! {len(results)} skill intelligence records saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
