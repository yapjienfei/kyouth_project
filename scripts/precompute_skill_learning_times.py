#!/usr/bin/env python
"""
Module 2 (support): Estimate weeks to learn each technical skill using Gemini.

Reads all unique skills from data/jobs_snapshot.json, queries Gemini for learning
time estimates, writes data/skill_learning_times.json.

Run after extract_job_skills_with_gemini.py.

Usage:
    uv run python scripts/precompute_skill_learning_times.py
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
OUTPUT_PATH = DATA_DIR / "skill_learning_times.json"

BATCH_SIZE = 20
MAX_RETRIES = 3
BASE_DELAY = 8

PROMPT_TEMPLATE = """You are a technical learning advisor. Estimate the number of weeks a competent IT professional
(with general programming knowledge) would need to reach basic job-readiness for each skill below.

Assume:
- The learner already knows how to code (has 1+ year general experience)
- "Job-ready" = can use the skill in a professional project
- Study time ≈ 2-3 hours per day

Return a JSON object where keys are the skill names (exactly as given) and values are integer weeks.
Typical ranges: simple tools (1-2 weeks), libraries (2-4 weeks), frameworks (4-8 weeks), cloud platforms (6-12 weeks), ML/AI (8-16 weeks).

Output ONLY valid JSON – no extra text.

Skills: {skills}"""


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


DEFAULT_WEEKS = {
    "python": 6, "javascript": 5, "typescript": 3, "java": 8, "php": 5,
    "c++": 10, "c#": 7, "go": 6, "rust": 10, "ruby": 5,
    "react": 6, "angular": 8, "vue": 5, "html": 2, "css": 2, "sass": 1,
    "next.js": 3, "tailwind": 1, "bootstrap": 1, "jquery": 2, "redux": 3,
    "node.js": 5, "flask": 3, "django": 5, "fastapi": 3, "express": 4,
    "spring boot": 8, "laravel": 5, "rails": 6, "asp.net": 7,
    "sql": 4, "postgresql": 3, "mysql": 3, "mongodb": 3, "redis": 2,
    "elasticsearch": 4, "cassandra": 5, "oracle": 5, "mssql": 3, "dynamodb": 3,
    "aws": 10, "azure": 10, "gcp": 10, "docker": 4, "kubernetes": 8,
    "terraform": 6, "ansible": 5, "jenkins": 4, "github actions": 2, "ci/cd": 4,
    "linux": 4, "bash": 3, "shell scripting": 3,
    "machine learning": 14, "deep learning": 16, "nlp": 12, "computer vision": 12,
    "tensorflow": 8, "pytorch": 8, "scikit-learn": 5, "pandas": 3, "numpy": 2,
    "apache spark": 8, "hadoop": 8, "tableau": 3, "power bi": 3,
    "android": 10, "ios": 10, "react native": 6, "flutter": 6, "swift": 8, "kotlin": 7,
    "cybersecurity": 12, "penetration testing": 14, "git": 2, "agile": 1, "scrum": 1,
    "microservices": 4, "rest api": 2, "graphql": 3, "kafka": 5, "rabbitmq": 4,
    "system design": 8, "algorithms": 10, "data structures": 8,
}


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

    # Load existing results for checkpointing
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
            for skill, weeks in batch_result.items():
                if isinstance(weeks, (int, float)):
                    results[skill.lower()] = int(weeks)
        except Exception as e:
            print(f"  Batch failed: {e}. Using defaults for this batch.")
            for skill in batch:
                results[skill] = DEFAULT_WEEKS.get(skill, 4)

        with open(OUTPUT_PATH, "w") as f:
            json.dump(results, f, indent=2, sort_keys=True)
        time.sleep(5)

    # Fill remaining with defaults
    for skill in all_skills:
        if skill not in results:
            results[skill] = DEFAULT_WEEKS.get(skill, 4)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    print(f"\nDone! {len(results)} skill learning times saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
