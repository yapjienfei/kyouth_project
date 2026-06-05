#!/usr/bin/env python
"""Analyze skills for salary premium and AI replaceability using Gemini + market data."""

import json
import os
import time
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-3.1-flash-lite")

# Load all unique skills
DATA_PATH = Path(__file__).parent.parent / "data" / "jobs_snapshot.json"
with open(DATA_PATH) as f:
    jobs = json.load(f)

all_skills = set()
for job in jobs:
    skills = job.get("skills_ai", [])
    all_skills.update(skills)
all_skills = sorted(all_skills)

# Load existing cache if any
output_path = Path(__file__).parent.parent / "data" / "skill_intelligence.json"
output = {}
if output_path.exists():
    with open(output_path) as f:
        output = json.load(f)

skills_to_process = [s for s in all_skills if s not in output]
print(f"Skills to process: {len(skills_to_process)}")

BATCH_SIZE = 25
PROMPT_TEMPLATE = """
You are an IT career analyst for the Malaysian job market. For each skill listed below, provide:

1. **salary_premium** (RM per month): The estimated monthly salary increase a professional can expect after acquiring this skill in Malaysia's current market. Use market data: cloud/DevOps skills typically add RM 2,000–3,500, cybersecurity skills RM 1,500–3,000, data skills RM 1,500–2,500, basic coding skills RM 500–1,500. Be realistic.

2. **ai_replaceability_score** (1-10): 1 = highly likely to be automated by AI in the next 3-5 years, 10 = very AI-proof and human-centric (strategy, architecture, security, complex problem-solving).

3. **skill_category**: One of ["Cloud & Infrastructure", "Data & AI", "Cybersecurity", "Software Development", "Soft Skills & Architecture", "Other"]

Base your assessment on Malaysia's 2026 IT market trends.

Return ONLY valid JSON in this format:
{{"skills": [{{"skill": "skill_name", "salary_premium": 2500, "ai_replaceability_score": 8, "skill_category": "Cloud & Infrastructure"}}]}}

Skills: {skills}
"""

for i in range(0, len(skills_to_process), BATCH_SIZE):
    batch = skills_to_process[i:i+BATCH_SIZE]
    prompt = PROMPT_TEMPLATE.format(skills=", ".join(batch))
    print(f"Processing batch {i//BATCH_SIZE + 1}: {len(batch)} skills...")
    try:
        response = model.generate_content(prompt)
        raw = response.text.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.endswith("```"):
            raw = raw[:-3]
        result = json.loads(raw)
        if isinstance(result.get("skills"), list):
            for item in result["skills"]:
                output[item["skill"]] = {
                    "salary_premium": item["salary_premium"],
                    "ai_replaceability_score": item["ai_replaceability_score"],
                    "skill_category": item.get("skill_category", "Other")
                }
            with open(output_path, "w") as f:
                json.dump(output, f, indent=2)
    except Exception as e:
        print(f"Error on batch: {e}")
    time.sleep(2)

print(f"✅ Saved {len(output)} skills to {output_path}")