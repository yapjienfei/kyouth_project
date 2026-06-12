#!/usr/bin/env python
"""
Module 1 / Module 2: Extract technical skills from job descriptions using Gemini.

Reads data/jobs.json, extracts skills in batches via Gemini, writes data/jobs_snapshot.json.
Run this once before starting the app.

Usage:
    uv run python scripts/extract_job_skills_with_gemini.py
"""
import json
import os
import time
import re
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Set it in .env or environment.")

# Use a valid model – override with env var if needed
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_PATH = DATA_DIR / "jobs.json"
OUTPUT_PATH = DATA_DIR / "jobs_snapshot.json"

BATCH_SIZE = 20
MAX_RETRIES = 3
BASE_DELAY = 10  # seconds

# Load taxonomy for normalization hints (not as a strict filter)
TAXONOMY_PATH = DATA_DIR / "skill_taxonomy.json"
TAXONOMY_SKILLS = []
if TAXONOMY_PATH.exists():
    with open(TAXONOMY_PATH) as f:
        tax = json.load(f)
    TAXONOMY_SKILLS = tax.get("skills", [])

# Comprehensive keyword set for fallback extraction – includes common tech skills
FALLBACK_SKILLS = set(TAXONOMY_SKILLS)
# Add common skills that might not be in taxonomy
FALLBACK_SKILLS.update([
    "flutter", "kotlin", "swift", "react native", "vue.js", "angular",
    "docker", "kubernetes", "k8s", "terraform", "ansible",
    "aws", "azure", "gcp", "cloud",
    "python", "java", "javascript", "typescript", "go", "rust", "c#", "c++",
    "spring boot", "django", "flask", "express", "fastapi",
    "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
    "graphql", "rest api", "grpc",
    "jenkins", "gitlab ci", "github actions", "circleci",
    "prometheus", "grafana", "datadog",
    "tensorflow", "pytorch", "scikit-learn",
    "selenium", "cypress", "jmeter", "postman",
    "workato", "power automate", "ui path",
    "linux", "bash", "powershell",
    "html", "css", "sass", "tailwind",
    "nixos", "cilium", "suricata", "nftables", "zeek",
    "fortinet", "cisco", "openvpn", "zscaler",
    "ibm websphere", "oracle weblogic", "jboss", "apache kafka",
    "phplens", "pl/sql", "oracle database",
])

PROMPT_TEMPLATE = """You are a technical skill extractor. For each job below, extract ONLY technical skills from its description.

Rules:
- Return a JSON object where each key is the job array index (0, 1, 2, ...) and value is a list of lowercase skill strings.
- Normalize synonyms using the provided taxonomy when possible, but you may extract any technical skill even if not in the taxonomy.
- Apply these normalizations:
  "K8s" → "kubernetes", "JS" → "javascript", "ReactJS" → "react", "React.js" → "react",
  "Postgres" → "postgresql", "CSS3" → "css", "HTML5" → "html", "NodeJS" → "node.js",
  "Net" → "asp.net", "MSSQL" → "mssql", "MS SQL" → "mssql",
  "Flutter" → "flutter", "Kotlin" → "kotlin", "Swift" → "swift",
  "RN" or "React Native" → "react native", "Vue" → "vue.js",
  "SpringBoot" → "spring boot", "Docker" → "docker", "K8s" → "kubernetes",
  "Terraform" → "terraform", "AWS" → "aws", "Azure" → "azure", "GCP" → "gcp".
- Exclude ALL soft skills (communication, teamwork, leadership, problem-solving, analytical skills, etc.).
- Exclude vague terms like "basic networking", "IT knowledge", "computer skills".
- If no technical skills are found, return an empty list for that job.

Taxonomy reference (use for normalization, not restriction):
{taxonomy}

Jobs:
{jobs}"""


def call_gemini_with_backoff(prompt: str) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = BASE_DELAY * (2 ** attempt)
            print(f"  Gemini error (attempt {attempt + 1}/{MAX_RETRIES}): {e}. Retrying in {wait}s...")
            time.sleep(wait)


def parse_json_response(raw: str) -> dict:
    # Remove markdown fences if present
    raw = re.sub(r"^```json\s*", "", raw.strip())
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"```$", "", raw)
    return json.loads(raw)


def keyword_fallback(description: str) -> list[str]:
    """Extract skills using simple keyword matching – used when Gemini fails or returns empty."""
    if not description:
        return []
    text = description.lower()
    found = set()
    for skill in FALLBACK_SKILLS:
        # Use word boundaries to avoid false positives (e.g., "rest" in "restaurant")
        # But many skills have spaces, so simple regex
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text):
            found.add(skill)
    # Additional heuristic: catch common patterns like "Python", "Java" even without boundaries
    for skill in ["python", "java", "javascript", "react", "aws"]:
        if skill in text:
            found.add(skill)
    return sorted(list(found))


# Surface forms Gemini normalizes away — accepted as evidence during verification
SURFACE_FORMS = {
    "kubernetes": ["k8s"], "javascript": ["js", "es6"], "typescript": ["ts"],
    "react": ["reactjs", "react.js"], "node.js": ["nodejs", "node js", "node"],
    "postgresql": ["postgres"], "asp.net": [".net", "dotnet"], "html": ["html5"],
    "css": ["css3"], "aws": ["amazon web services"], "gcp": ["google cloud"],
    "azure": ["microsoft azure"], "ci/cd": ["cicd", "ci cd"],
    "machine learning": ["ml"], "artificial intelligence": ["ai"],
    "rest api": ["rest", "restful"],
}


def verify_skills(skills: list[str], description: str) -> list[str]:
    """Hallucination guard: drop any skill Gemini returned that does not
    actually appear in the job description (word-boundary match, including
    known surface forms). A job with no real skills stays empty."""
    if not description:
        return []
    text = description.lower()
    kept = []
    for s in skills:
        if not isinstance(s, str):
            continue
        s = s.strip().lower()
        forms = [s] + SURFACE_FORMS.get(s, [])
        for form in forms:
            if re.search(r"(?<!\w)" + re.escape(form) + r"(?!\w)", text):
                kept.append(s)
                break
    return kept


def extract_batch(jobs_batch: list[dict]) -> dict[int, list[str]]:
    """Send a batch of jobs to Gemini and return a dict mapping batch index -> skills."""
    jobs_text = ""
    for i, job in enumerate(jobs_batch):
        desc = job.get("description", "")[:4000]  # Increased from 800 to 4000 characters
        jobs_text += f"\n[{i}] Title: {job['title']}\nDescription: {desc}\n"

    # Provide a reasonable sample of taxonomy (first 100 skills) for normalization
    taxonomy_sample = ", ".join(TAXONOMY_SKILLS[:100]) if TAXONOMY_SKILLS else "python, javascript, react, aws, docker"
    prompt = PROMPT_TEMPLATE.format(taxonomy=taxonomy_sample, jobs=jobs_text)

    raw = call_gemini_with_backoff(prompt)
    result = parse_json_response(raw)
    # Ensure keys are integers
    return {int(k): v for k, v in result.items() if isinstance(v, list)}


def main():
    print(f"Loading jobs from {INPUT_PATH}...")
    with open(INPUT_PATH, encoding="utf-8") as f:
        jobs = json.load(f)
    print(f"Loaded {len(jobs)} jobs.")

    # Load existing snapshot for checkpointing
    existing_skills = {}
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            snapshot = json.load(f)
        for job in snapshot:
            key = (job.get("title", ""), job.get("company", ""))
            existing_skills[key] = job.get("skills_ai", [])
        print(f"Checkpoint: {len(existing_skills)} jobs already processed.")

    results = []
    total_batches = (len(jobs) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(total_batches):
        start = batch_idx * BATCH_SIZE
        end = start + BATCH_SIZE
        batch = jobs[start:end]

        # Separate jobs in this batch that need extraction
        needs_extraction = []
        need_indices_in_batch = []  # original indices within this batch (0..len(batch)-1)
        for i, job in enumerate(batch):
            key = (job.get("title", ""), job.get("company", ""))
            if key not in existing_skills:
                needs_extraction.append(job)
                need_indices_in_batch.append(i)

        # Create a copy of the batch with skills filled (from checkpoint or new extraction)
        processed_batch = []
        for i, job in enumerate(batch):
            key = (job.get("title", ""), job.get("company", ""))
            job_copy = dict(job)
            if key in existing_skills:
                job_copy["skills_ai"] = existing_skills[key]
            else:
                # Will be filled after extraction; placeholder for now
                job_copy["skills_ai"] = []
            job_copy["salary_display"] = (
                f"RM {job_copy['salary_min']:,} – RM {job_copy['salary_max']:,}"
                if job_copy.get("salary_min") and job_copy["salary_min"] > 0
                else "Undisclosed"
            )
            processed_batch.append(job_copy)

        # If there are jobs to extract in this batch
        if needs_extraction:
            print(f"Batch {batch_idx + 1}/{total_batches}: extracting {len(needs_extraction)} jobs...")
            try:
                extracted = extract_batch(needs_extraction)
            except Exception as e:
                print(f"  Batch failed: {e}. Using keyword fallback for all jobs in this batch.")
                extracted = {i: keyword_fallback(job.get("description", "")) for i, job in enumerate(needs_extraction)}

            # Map extracted skills back to the correct positions in processed_batch
            for offset, (orig_index, job) in enumerate(zip(need_indices_in_batch, needs_extraction)):
                skills = extracted.get(offset, [])
                # Hallucination guard: only keep skills the description mentions
                verified = verify_skills(skills, job.get("description", ""))
                if len(verified) < len(skills):
                    dropped = sorted(set(s.lower() for s in skills if isinstance(s, str)) - set(verified))
                    print(f"    Dropped {len(skills) - len(verified)} hallucinated skill(s) "
                          f"for '{job['title']}': {dropped[:6]}")
                skills = verified
                # If nothing verifiable came back, try keyword fallback
                if not skills:
                    fallback_skills = keyword_fallback(job.get("description", ""))
                    if fallback_skills:
                        print(f"    Gemini gave no verifiable skills for '{job['title']}', using fallback: {fallback_skills}")
                        skills = fallback_skills
                processed_batch[orig_index]["skills_ai"] = skills

            # Save checkpoint after each batch
            # First, update results with all previous batches + this processed_batch
            # But results currently contains only previous batches? Actually we append later.
            # Better: accumulate results as we go.
        else:
            print(f"Batch {batch_idx + 1}/{total_batches}: all jobs already have skills (skipping API call).")

        # Append this batch's processed jobs to results
        results.extend(processed_batch)

        # Write checkpoint after each batch (idempotent)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"  Saved checkpoint ({len(results)} jobs processed).")
        time.sleep(5)  # rate limiting

    print(f"\nDone! {len(results)} jobs saved to {OUTPUT_PATH}")
    jobs_with_skills = sum(1 for j in results if j.get("skills_ai"))
    print(f"  {jobs_with_skills} jobs have at least one skill extracted.")

    # Optional: print a few examples to verify
    print("\nSample extractions:")
    for job in results[:5]:
        if job.get("skills_ai"):
            print(f"  {job['title']}: {job['skills_ai'][:5]}")


if __name__ == "__main__":
    main()