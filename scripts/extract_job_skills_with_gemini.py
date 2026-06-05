#!/usr/bin/env python
"""Extract technical skills from job descriptions using Gemini (free tier) – BATCH MODE."""

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

# Use a valid free-tier model (gemini-1.5-flash or gemini-2.0-flash-exp)
model = genai.GenerativeModel("gemini-3.1-flash-lite")

BATCH_SIZE = 5            # Number of jobs per API call
MAX_RETRIES = 3
BASE_DELAY = 10           # seconds, then exponential backoff

PROMPT_TEMPLATE = """
You are a technical skill extractor. For each job below, extract ONLY technical skills from its description.
Rules:
- Return a JSON object where each key is the job title and value is a list of lowercase skill strings.
- Normalise synonyms: "K8s" → "kubernetes", "JS" → "javascript", "ReactJS" → "react", "Postgres" → "postgresql".
- Exclude all soft skills (communication, leadership, problem-solving, team player, etc.).
- Exclude generic terms like "basic networking" – only concrete, specific technical skills.
- If a description contains no technical skills, return an empty list.

Jobs:
{job_list}

Output ONLY valid JSON, no extra text.
"""

def extract_skills_batch(jobs_batch):
    """
    jobs_batch: list of dicts with keys 'title' and 'description'
    Returns a dict: {title: [skills], ...}
    """
    # Build prompt
    job_text = ""
    for job in jobs_batch:
        title = job['title']
        desc = job.get('description', '')[:2000]   # truncate per job
        job_text += f"\n--- JOB TITLE: {title} ---\n{desc}\n"
    
    prompt = PROMPT_TEMPLATE.format(job_list=job_text)
    
    for attempt in range(MAX_RETRIES):
        try:
            response = model.generate_content(prompt)
            text = response.text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
            result = json.loads(text)
            # Ensure result is a dict
            if isinstance(result, dict):
                return result
            else:
                print(f"  Unexpected output format: {type(result)}. Retrying...")
                time.sleep(BASE_DELAY * (2 ** attempt))
                continue
        except Exception as e:
            print(f"  Gemini error: {e}")
            if "429" in str(e):   # rate limit
                wait = BASE_DELAY * (2 ** attempt)
                print(f"  Rate limit, waiting {wait}s before retry...")
                time.sleep(wait)
            else:
                # other error, return empty for all in batch
                return {job['title']: [] for job in jobs_batch}
    # after retries, return empty
    return {job['title']: [] for job in jobs_batch}

def main():
    data_path = Path(__file__).parent.parent / "data" / "jobs_snapshot.json"
    with open(data_path, "r", encoding="utf-8") as f:
        jobs = json.load(f)

    # Identify jobs that still need skills_ai
    jobs_to_process = [job for job in jobs if "skills_ai" not in job]
    total = len(jobs_to_process)
    print(f"Found {total} jobs without 'skills_ai'. Processing in batches of {BATCH_SIZE}.")

    # Process in batches
    for i in range(0, total, BATCH_SIZE):
        batch = jobs_to_process[i:i+BATCH_SIZE]
        print(f"\nBatch {i//BATCH_SIZE + 1}: processing {len(batch)} jobs...")
        
        # Extract skills for this batch
        skills_dict = extract_skills_batch(batch)
        
        # Assign back to original job objects
        for job in batch:
            title = job['title']
            skills = skills_dict.get(title, [])
            job["skills_ai"] = skills
            print(f"  {title}: {len(skills)} skills -> {skills[:5]}...")
        
        # Save after each batch to avoid losing progress
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(jobs, f, indent=2, ensure_ascii=False)
        
        # Wait between batches to respect rate limit (e.g., 10 seconds)
        if i + BATCH_SIZE < total:
            print(f"  Waiting 2s before next batch...")
            time.sleep(2)
    
    print(f"\n✅ Done! Updated {data_path} with 'skills_ai' fields for all jobs.")

if __name__ == "__main__":
    main()