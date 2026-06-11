# Claude.md – Career Pathway Navigator (Malaysia)
## Project Overview
Build a **career pathway navigator** that helps Malaysian job seekers (IT/tech) discover realistic next roles based on their existing resume. The system extracts skills from resumes and job postings, then recommends career transitions with:
- Skill gap analysis
- AI replaceability score (personalised)
- Salary progression
- Confidence score & data freshness indicators(Mock data)
**Core philosophy:** Hybrid system – LLM only for parsing unstructured text; all numerical calculations deterministic. Every output must be traceable to a source.
## Tech Stack Recommendation
- **Backend:** Python (FastAPI/Flask) + PostgreSQL
- **LLM:** gemini-3.1-flash-lite (via API)
- **RAG:** Chroma / FAISS + embeddings (for retrieving job postings & reports)
- **Frontend:** React + Tailwind
- **Evaluation:** DeepEval (hallucination detection)
- **Tech Stack addition:** `pypdf` or `pdfplumber` for PDF text extraction.

## Data Sources (Download First)

| Source | Key data | URL / Access |
|--------|----------|---------------|
| WEF Future of Jobs Report 2025 | Global job growth/decline, skill trends, AI impact by sector | weforum.org |
| TalentCorp MyCOL 2024/2025 | Malaysia critical occupations, shortage indicators | talentcorp.com.my |
| JobStreet Hiring & Compensation Report 2025 | Malaysian salary ranges by role, hiring trends | jobstreet.com.my |
| DOSM Labour Force Survey | Unemployment rate, employment trends | dosm.gov.my (OpenDOSM) |

## Note
No scraping is done, I only have a json file 300+ jobs, each has the below format
  {
    "title": "PHP Web Developer",
    "company": "ETCTECH GLOBAL SDN BHD",
    "description": "  Involve in typical software lifecycle design, coding, testing, debugging..."
    "salary_min": 4000,
    "salary_max": 4000
  }
Any freshness indicator or data caluculated that involves time of posting etc will have to use mock data.
You can treat the current data as 0 days since refresh.

## System Architecture (High-Level)
1. User uploads resume in pdf, parse pdf to text, LLM extract skills(strict).
2. In existing json for each job posting, there is company, job title, job description, salary. Store into database, then LLM extract role skills from job description, and add to the row.
3. Deterministic Engine for skill gap, job AI replaceability, salary delta, confidence levels.
4. Load External Reports into RAG, RAG retrieve relevant docs for template LLM Explanation.
5. UI with sources & confidence.

## Module Breakdown (Implement in Order)
### Module 1: Data Pipeline
- **1.1** There exists a json file with company, job title, description, and salary. If salary is not available, it is set to 0. Do not discard the posting. For salary calculations, only use postings with non‑zero salaries. For this json file, store it into postgres database with respective fields.
Use job listing salaries as your primary numeric source. Keep external reports for RAG explanations, trend validation, and fallback only when salary data is absent.
- **1.2** Create `external_sources` table with `source_name`, `data_year`, `replaceability_scores` (JSON)
Example
{
  "source_name": "WEF Future of Jobs Report 2025",
  "data_year": 2025,
  "replaceability_scores": {
    "HTML/CSS": 80,
    "JavaScript": 60,
    "Python": 50,
    "Team collaboration": 10,
    "Creative thinking": 15,
    "AWS": 45,
    "Kubernetes": 35
  }
}

### Module 2: Skill Extraction (LLM + Taxonomy)
- **2.1** Build skill taxonomy JSON (start with 100 common IT skills: Python, AWS, React, etc.)
- **2.2** Try to refine this prompt to also normalize skills. Write LLM prompt for extracting skills from resume:
You are a skill extractor. Extract technical and soft skills from the resume below.
Return only a JSON array of skill names. Use the standard names from the taxonomy provided.
Do NOT invent skills not clearly present. Do NOT add explanations.
Taxonomy: {taxonomy_list}
Resume: {resume_text}
- **2.3** Same prompt for job descriptions (aggregate multiple postings for a role)
- **2.4** Add fallback: if LLM returns invalid JSON or skills outside taxonomy, use keyword matching
**2.5** Implement exponential backoff for API calls. 

### Module 3: Deterministic Calculation Engine
**3.1 Skill Gap**
def skill_gap(resume_skills, required_skills):
  missing = set(required_skills) - set(resume_skills)
  overlap = len(set(resume_skills) & set(required_skills))
  percentage_gap = len(missing) / len(required_skills) * 100
  return missing, percentage_gap
**3.2** AI Replaceability Score
Pre-load skill-level base scores from WEF (manual mapping for 50-100 skills).
If a skill is not in the mapping, assign default 50% and log a warning.
Example mapping:
json
{"HTML/CSS": 80, "JavaScript": 60, "Python": 50, "Team collaboration": 10, "Creative thinking": 15}
For each target role, compute weighted average:
job_score = sum(skill_replaceability[skill] * weight[skill]) / sum(weights)
where weight[skill] = frequency of that skill in job ads for that role (normalised)
Personalised score: personalised = job_score * (1 - 0.2*user_proficiency_in_low_risk_skills)
User proficiency is a binary flag: 1 if the user’s resume explicitly lists any skill whose replaceability score is <30% (low risk), otherwise 0.
If data older than 12 months, multiply confidence by 0.7
3.3 Salary Delta
From data, compute median salary for current role and target role
Delta = target_median - current_median
3.4 Confidence Score
confidence = (
    0.4 * min(1, num_job_postings / 100)   # data density
  + 0.3 * (1 - days_since_refresh / 365)   # recency
  + 0.3 * (1 - skill_gap_percentage / 100) # ease of transition
) * 100

### Module 4: RAG & Retrieval
Embed all job postings + external report excerpts (using gemini-embedding-2)
When user uploads resume, retrieve top-k similar job postings + relevant WEF sections
Use retrieved context as ground truth for LLM explanations (prevents hallucination)

### Module 5: Career Path Computation
Build graph: nodes = job titles, edges = possible transition (if skill overlap > 30%)
An edge exists from Job A to Job B if the skill set of Job B has at least 30% overlap (Jaccard similarity) with the skill set of Job A.
Implement BFS to find paths from start role to end role, optionally including intermediate roles
Return top 3 shortest paths (by number of transitions)

### Module 6: Explanation & UI
Use template-based LLM generation – never let LLM invent numbers
Template example for skill gap:
python
explanation = f"""
To move from {start_role} to {target_role}, you need to learn {len(missing_skills)} new skills:
{', '.join(missing_skills[:5])}.
"""
Display every number with source badge and freshness indicator

### Module 7: Frontend Components
Resume upload
Role selector (start role, target role, optional intermediate roles)
Results page showing:
Career path diagram (node-link)
For each step: skill gap, salary delta, AI replaceability score
Expandable "Why?" panel with sources and confidence score
"Re-validate with latest data" button(Mock button)
Collapsible "Limitations & Caveats" section

### Validation & Hallucination Prevention
Skill extraction evaluation: Run 50 test resumes with known skill lists. Aim for >90% precision/recall. If lower, refine prompt.

Numerical checks: After LLM explanation, run regex to extract numbers and compare with deterministic values. If mismatch, reject and regenerate.
Source citation: Every output must include a source field (e.g., "source": "JobStreet report 2025").
Fallback messaging: If any data is missing or stale, show: "Insufficient reliable data to compute this recommendation. Please check back after data refresh."
Use DeepEval to automatically detect hallucinations in LLM-generated text.

### Freshness & Dynamic Market Handling
Component	Implementation
Data timestamp  Every external report has published_year.
Freshness badge	Display "Data from {date} – {age} months old" next to each number. (Mock data)
Confidence decay	Multiply confidence by max(0.5, 1 - age_in_days/365). (Mock data)
Trend indicators	Compare skill frequency in last 3 months vs last 12 months; show ⬆️/X% or ⬇️/X%. (Mock Data)
Manual refresh	UI button triggers re-scrape of target roles.(This is just mock button that does not do anything).
Expiry warnings	If a report is >2 years old, show warning: "Based on 2025 report – newer data not available.

For a user who is a Frontend Developer, here's how these files power the output:
Extract Skills from Resume: The LLM parses the user's resume to identify their current skills (e.g., "JavaScript", "React").
Retrieve Relevant Career Data: The RAG system then fetches data from your four sources. It finds that "Cloud Engineer" is on the MyCOL list, retrieves an average salary from the "Hiring, Compensation and Benefits Report", and extracts key skill recommendations from the WEF Report.
Determine the Skill Gap: The system compares the user's skills to the "Cloud Engineer" requirements, identifying they would need to learn "AWS" and "Kubernetes".
Calculate a Score & Recommendation: The RAG-retrieved data feeds into the deterministic engine:
Confidence Score: This would be high, as the MyCOL list confirms it's a critical role.
Salary Delta: The system calculates the potential salary increase based on the report data.
AI Replaceability Score: This would be low, based on WEF projections for skilled cloud roles.
Generate Final Output: The system uses the LLM to synthesize this analysis into a clear, human-readable output, complete with source citations.

The four files for RAG has the following purpose:
MY Hiring Report	Salary Delta calculation	Salary benchmarks for 2025
MyCOL (Critical Occupations)	Confidence Score weighting	Flags high-demand roles in Malaysia
WEF Future of Jobs Report	AI Replaceability Score and skill trends	Global, forward-looking data on growth & decline
Labour Force Survey (DOSM)	Macroeconomic context for Confidence Score	Official, local labour market health data