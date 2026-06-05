# 🧭 Malaysian Career Navigator

> An AI-powered career pivot tool that extracts your skills from a resume, maps them against real Malaysian IT job listings, and generates a personalised upskilling roadmap — ranked by speed and salary gain.

---

## Problem Statement

Malaysia's IT job market is fast-moving, but most career advice tools are generic or built for Western markets. Fresh graduates and mid-career professionals face three compounding problems:

1. **They don't know which skills they already have** that are market-relevant.
2. **They can't see which roles are realistically reachable** given their current profile.
3. **They have no structured, time-bound plan** for bridging the skill gap.

Existing job portals (JobStreet, MyFutureJobs) show listings but offer no personalised guidance. Career coaches are expensive and scarce. The result: people either stay stuck or make uninformed pivots.

---

## Target Users

- **Fresh IT graduates** mapping their first job options
- **Mid-career professionals** considering a domain switch (e.g. developer → cloud engineer)
- **Bootcamp or self-taught learners** who want to validate their skill set against live demand
- **Career advisors** at universities or workforce agencies (HRDF-registered entities, etc.)

---

## System Goal

Given a resume (PDF), the system must:

1. Extract the candidate's technical skills automatically
2. Match them against a snapshot of real Malaysian IT job listings
3. Score every role by reachability (skill overlap) and salary
4. Find multi-step career paths through the job graph to a target role
5. Generate a prioritised upskilling roadmap with realistic learning time estimates (weeks)
6. Explain *why* each skill matters, using AI-generated context grounded in Malaysia's 2026 market

---

## System Architecture

### Data Flow

```
[User uploads PDF resume]
        ↓
[pdfplumber extracts raw text]
        ↓
[Gemini Flash: identify technical skills from resume text]
        ↓
[Skill Matching Engine]
   ├── Match ratio against all job_nodes (set intersection / union)
   ├── Filter by salary threshold + disclosed/undisclosed toggle
   └── Score & rank suggestions
        ↓
[Job Graph (directed, salary-ordered edges)]
   └── BFS/path search from user skill set → target role
        ↓
[Upskilling Roadmap]
   ├── Missing skills per step, ordered by learning weeks
   └── Gemini Flash: explain each skill's importance + suggested learning order
        ↓
[Frontend renders: suggestions, paths, skill demand table, explanations]
```

### Module Breakdown

| Module | Location | Responsibility |
|---|---|---|
| **Flask API** | `backend/app.py` | Core server: routing, skill matching, graph traversal, Gemini calls |
| **Job Graph** | `backend/app.py` (runtime) | Directed graph of `JobNode` objects; edges represent reachable transitions |
| **PDF Extractor** | `backend/app.py` → `pdfplumber` | Converts uploaded resume PDF to raw text |
| **Skill Intelligence** | `data/skill_intelligence.json` | Pre-computed salary premium, AI replaceability score, skill category per skill |
| **Learning Times** | `data/skill_learning_times.json` | Pre-computed weeks-to-learn per skill (Gemini-generated, cached) |
| **Job Snapshot** | `data/jobs_snapshot.json` | Static snapshot of Malaysian IT job listings with AI-extracted skills |
| **Frontend** | `frontend/static/index.html` | Single-page app: resume upload, skill selector, suggestions, path explorer |
| **Nginx** | `frontend/nginx.conf` | Reverse proxy: routes `/extract_skills`, `/suggest`, `/find_paths`, etc. to backend |
| **Skill Extractor Script** | `scripts/extract_job_skills_with_gemini.py` | One-time pipeline: batch-extracts skills from job descriptions using Gemini |
| **Skill Intelligence Script** | `scripts/analyze_skill_intelligence.py` | One-time pipeline: enriches each skill with salary premium + AI replaceability |
| **Learning Times Script** | `scripts/precompute_skill_learning_times.py` | One-time pipeline: estimates learning weeks per skill via Gemini |

---

## Setup & Installation

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose
- A **Gemini API key** (free tier works — [get one here](https://aistudio.google.com/app/apikey))

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd career-navigator
```

### 2. Configure environment

```bash
cp .env.example .env
```

Open `.env` and fill in your Gemini API key:

```env
GEMINI_API_KEY=your_key_here
```

### 3. Build and run with Docker

```bash
make build
make up
```

Or without Make:

```bash
docker compose build
docker compose up -d
```

### 4. Open the app

Navigate to **[http://localhost:8080](http://localhost:8080)** in your browser.

### Useful Make commands

| Command | Description |
|---|---|
| `make up` | Start all services (detached) |
| `make down` | Stop all services |
| `make logs` | Tail logs from all containers |
| `make shell-backend` | Open a shell in the backend container |
| `make restart` | Full clean rebuild and restart |

### Running the data pipeline scripts (optional)

These scripts were used to build the pre-computed data files. Only run them if you want to refresh the job snapshot or skill data:

```bash
# Install dependencies (Python 3.14+)
pip install -e .

# Step 1: Extract skills from job descriptions
python scripts/extract_job_skills_with_gemini.py

# Step 2: Precompute learning times
python scripts/precompute_skill_learning_times.py

# Step 3: Generate skill intelligence (salary premium, AI replaceability)
python scripts/analyze_skill_intelligence.py
```

> ⚠️ These scripts make batched Gemini API calls and include rate-limit backoff. Running them on a large dataset may take several minutes.

### Dependencies

| Package | Version | Purpose |
|---|---|---|
| `flask` | 3.1.0 | Backend web framework |
| `flask-cors` | 5.0.0 | Cross-origin requests between frontend and backend |
| `pdfplumber` | 0.11.4 | PDF text extraction |
| `google-generativeai` | ≥0.8.0 | Gemini API client |
| `python-dotenv` | 1.0.1 | `.env` file loading |
| `requests` | 2.32.3 | HTTP client (utility) |
| `nginx:alpine` | — | Frontend static file serving + reverse proxy |

---

## Features

### 1. PDF Resume Upload & Skill Extraction

Upload a PDF resume and the system uses `pdfplumber` to extract the full text, then sends it to Gemini Flash with a structured prompt to identify only technical skills (normalised and deduplicated — e.g. "K8s" → `kubernetes`). Soft skills and generic terms are excluded. The extracted skills are returned as a clickable tag cloud for the user to review and adjust.

### 2. Manual Skill Selector

Every unique skill found across the job dataset is listed as a toggleable chip. Users can manually add skills that weren't on their resume, or deselect skills they consider outdated. Selections update all downstream results in real time.

### 3. Job Suggestions with Match Scoring

All jobs in the dataset are scored against the user's current skill set using a **match ratio** (overlap / total job skills required). Results are:
- Filtered by a configurable **minimum salary threshold** (RM slider)
- Togglable to include or exclude jobs with undisclosed salaries
- Paginated (10 per page) and sorted by match percentage
- Each card shows missing skills and estimated weeks to fill the gap

### 4. Career Path Finder (Graph BFS)

Users select a target role and the system runs a **breadth-first search** through a directed job graph. Edges only exist between roles where the target pays more *and* the skill overlap is ≥ 20% (transition cost < 0.8). Paths are ranked by total upskilling weeks (fastest first), showing:
- Each intermediate role with its salary range
- The specific skills to learn at each step, with week estimates
- Multiple path options (direct vs. stepped via intermediate roles)

### 5. AI Path Explanation

Each career path has an "Explain this path" button that calls Gemini Flash with the missing skills and target role. Gemini returns a structured JSON response with per-skill:
- **Importance score** (1–10) for the target role
- **Plain-language explanation** of why the skill matters
- **Suggested learning order** (which to tackle first)

High-importance skills (≥7) are highlighted in green, mid-range in yellow, low in grey.

### 6. Skill Demand Table

A live view of all skills in the dataset enriched with pre-computed intelligence:
- **Salary premium** (estimated RM/month uplift in the Malaysian market)
- **AI replaceability score** (1 = highly automatable, 10 = AI-proof)
- **Skill category** (Cloud & Infrastructure, Data & AI, Cybersecurity, etc.)

This table helps users prioritise *which* skills to learn based on long-term career resilience, not just immediate job matching.

### 7. Salary Threshold Filter

A range slider lets users set a minimum acceptable salary (RM). All suggestions and path results are filtered dynamically — roles below the threshold are excluded from both suggestions and graph traversal, keeping results focused on genuine career improvements.

---

## Technical Decisions

### Static job snapshot instead of live scraping

**Decision:** The job dataset (`jobs_snapshot.json`) is a pre-fetched static file, not a live scrape.

**Rationale:** Live scraping in a prototype introduces fragile dependencies on third-party sites (rate limits, HTML structure changes, legal grey areas). A static snapshot keeps the system self-contained and demo-reliable. The data pipeline scripts (`extract_job_skills_with_gemini.py`) can be re-run periodically to refresh the snapshot independently of the web app.

### Graph traversal for career paths instead of direct LLM generation

**Decision:** Career paths are computed by BFS on a directed graph of `JobNode` objects, not by asking Gemini to "suggest a career path."

**Rationale:** LLM-generated paths are non-deterministic and uncheckable against real data. The graph approach guarantees that every suggested intermediate role and salary figure actually exists in the dataset. Gemini is used only for *explanation* (why a skill matters), where nuanced language is valuable and factual grounding is less critical.

### Pre-computed skill metadata

**Decision:** Learning times (`skill_learning_times.json`) and skill intelligence (`skill_intelligence.json`) are computed once and stored as flat JSON, not generated per-request.

**Rationale:** Calling Gemini for every skill lookup would add ~1–2 seconds of latency per path step and consume significant API quota. Pre-computation makes runtime responses fast and predictable. The data is refreshable by re-running the pipeline scripts.

### Gemini Flash Lite as the AI model

**Decision:** `gemini-3.1-flash-lite` is used throughout (skills extraction, path explanation, skill intelligence).

**Rationale:** The free tier of Gemini Flash Lite is sufficient for structured JSON extraction tasks and short explanations. It avoids any cost for a prototype and fits the constraint of using free/open-source resources where possible.

### Single-container architecture (Flask serves static files + API)

**Decision:** The backend Flask app can also serve the frontend static files directly (`static_folder='../frontend/static'`), but Docker Compose also provides a separate Nginx container.

**Rationale:** The dual setup allows simple single-process local development (just `python backend/app.py`) while the Docker Compose stack uses Nginx as a proper reverse proxy for production-like behaviour, with `client_max_body_size 10M` for PDF uploads.

---

## Limitations

### Known Issues

- **Gemini model string `gemini-3.1-flash-lite` may not be a valid public model ID.** The official free-tier model name as of mid-2025 is `gemini-1.5-flash` or `gemini-2.0-flash-lite`. If API calls fail, update the model string in `backend/app.py` and all three scripts.
- **Job graph is O(n²) at startup.** Building edges between all pairs of `JobNode` objects is quadratic. For the current dataset size this is acceptable, but it will become slow with > 5,000 jobs.
- **Skills are matched by exact lowercase string.** Synonyms not caught by the extraction normalisation (e.g. "react.js" vs "react") will cause missed matches. There is no fuzzy matching or embedding-based similarity.
- **Resume extraction quality depends on PDF structure.** Heavily styled, image-based, or multi-column PDFs may produce garbled text from `pdfplumber`, leading to missed skills.
- **No user authentication or session persistence.** Skill selections and results are in-memory only; refreshing the page resets everything.
- **Salary data quality varies.** Many Malaysian job listings omit salary ranges. The `include_undisclosed` toggle is a workaround, but path quality degrades when intermediate roles have no salary data to order edges by.

### Future Improvements

- **Replace static snapshot with a scheduled scraper** that refreshes `jobs_snapshot.json` weekly, keeping the job graph current without requiring manual script runs.
- **Embedding-based skill matching** using a lightweight sentence transformer to handle synonyms, typos, and skill variants more robustly than exact string comparison.
- **User accounts and saved roadmaps** so users can track progress, mark skills as learned, and watch how their match scores improve over time.
- **Integrate free learning resources** — automatically link each missing skill to a free course (Coursera audit, freeCodeCamp, YouTube playlist) so the roadmap is immediately actionable.
- **Localise salary data by state** — IT salaries in KL/Selangor differ significantly from Penang or Johor; per-region filtering would improve advice quality.
- **Export to PDF/Word** — allow users to download their personalised upskilling roadmap as a shareable document for career counselling sessions or HRDF grant applications.
- **Graph optimisation** — move from O(n²) edge building to a clustered or indexed approach to support larger job datasets.

---

## Project Structure

```
career-navigator/
├── backend/
│   ├── app.py              # Flask API + job graph + Gemini integration
│   └── Dockerfile
├── data/
│   ├── jobs_snapshot.json          # Malaysian IT job listings (with AI-extracted skills)
│   ├── skill_intelligence.json     # Salary premium + AI replaceability per skill
│   └── skill_learning_times.json   # Weeks-to-learn per skill
├── frontend/
│   ├── static/
│   │   └── index.html      # Single-page app (vanilla JS)
│   ├── Dockerfile
│   └── nginx.conf          # Reverse proxy config
├── scripts/
│   ├── extract_job_skills_with_gemini.py       # Data pipeline: skill extraction
│   ├── analyze_skill_intelligence.py           # Data pipeline: skill enrichment
│   └── precompute_skill_learning_times.py      # Data pipeline: learning time estimation
├── .env.example
├── docker-compose.yml
├── Makefile
└── pyproject.toml
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/extract_skills` | Upload PDF resume; returns extracted skill list |
| `POST` | `/suggest` | Given skills + filters; returns ranked job suggestions |
| `POST` | `/find_paths` | Given skills + target role; returns career paths with week estimates |
| `POST` | `/explain_path` | Given target role + missing skills; returns AI explanation per skill |
| `GET` | `/all_skills` | Returns full list of unique skills in the dataset |
| `GET` | `/job_titles` | Returns all available job titles for the target role selector |
| `GET` | `/skill_demand` | Returns skill intelligence table (premium, replaceability, category) |
