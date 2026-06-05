#!/usr/bin/env python
"""Precompute learning weeks for each technical skill using Gemini."""

import json
import os
import time
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
if not os.environ.get("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY not found in environment or .env file")

model = genai.GenerativeModel("gemini-3.1-flash-lite")  # or gemini-2.0-flash-lite

# Get all unique skills from the job snapshot
DATA_PATH = Path(__file__).parent.parent / "data" / "jobs_snapshot.json"
with open(DATA_PATH) as f:
    jobs = json.load(f)

all_skills = set()
for job in jobs:
    skills = job.get("skills_ai", [])
    all_skills.update(skills)
all_skills = sorted(all_skills)

print(f"Total unique skills: {len(all_skills)}")

# Batch size to avoid token limits
BATCH_SIZE = 30
output = {}
output_path = Path(__file__).parent.parent / "data" / "skill_learning_times.json"

# Load existing if any, to avoid recomputing everything
if output_path.exists():
    with open(output_path) as f:
        output = json.load(f)

skills_to_process = [s for s in all_skills if s not in output]
print(f"Skills already cached: {len(output)}")
print(f"Skills to process: {len(skills_to_process)}")

PROMPT_TEMPLATE = """
You are an expert in IT upskilling. For each skill listed below, estimate the number of weeks a full‑time working professional (studying 10 hours/week) needs to learn that skill from scratch to a job‑ready level.

Consider the skill's complexity, prerequisites, and typical learning curve. Return ONLY a JSON object where each key is the skill name and the value is an integer (weeks). Do NOT include any other text or explanations.

Skills: {skills}

Example output: {{"python": 4, "kubernetes": 8, "docker": 3}}
"""

for i in range(0, len(skills_to_process), BATCH_SIZE):
    batch = skills_to_process[i:i+BATCH_SIZE]
    prompt = PROMPT_TEMPLATE.format(skills=", ".join(batch))
    print(f"Processing batch {i//BATCH_SIZE + 1}: {len(batch)} skills...")
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        batch_result = json.loads(text)
        if isinstance(batch_result, dict):
            output.update(batch_result)
            # Save after each batch
            with open(output_path, "w") as f:
                json.dump(output, f, indent=2)
        else:
            print(f"  Unexpected output: {text[:200]}")
    except Exception as e:
        print(f"  Error on batch: {e}")
    time.sleep(2)  # be gentle to API

print(f"\n✅ Saved {len(output)} skill learning times to {output_path}")