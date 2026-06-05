import json
import tempfile
import os
import time
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pdfplumber
from collections import defaultdict, deque
import google.generativeai as genai

genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
gemini_model = genai.GenerativeModel("gemini-3.1-flash-lite")

app = Flask(__name__, static_folder='../frontend/static', static_url_path='')
CORS(app)

# Load job data
JOBS_PATH = Path(__file__).parent.parent / "data" / "jobs_snapshot.json"
with open(JOBS_PATH) as f:
    JOBS = json.load(f)

# Load pre‑computed skill learning times (weeks)
LEARNING_TIMES_PATH = Path(__file__).parent.parent / "data" / "skill_learning_times.json"
if LEARNING_TIMES_PATH.exists():
    with open(LEARNING_TIMES_PATH) as f:
        skill_learning_times = json.load(f)
else:
    skill_learning_times = {}
    print("Warning: skill_learning_times.json not found, defaulting to 4 weeks per skill.")

def get_learning_weeks(skill):
    """Return estimated weeks to learn a skill, default 4."""
    return skill_learning_times.get(skill.lower(), 4)

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

# Preprocess jobs
class JobNode:
    def __init__(self, job_dict):
        self.title = job_dict["title"]
        self.company = job_dict["company"]
        self.salary_min = job_dict.get("salary_min", 0)
        self.salary_max = job_dict.get("salary_max", 0)
        self.salary_median = (self.salary_min + self.salary_max) / 2 if self.salary_max > 0 else 0
        self.salary_display = job_dict.get("salary_display", "Undisclosed")
        self.skills = set(job_dict.get("skills_ai", []))
        self.raw = job_dict

job_nodes = [JobNode(job) for job in JOBS]

# Remove jobs with no skills extracted
original_count = len(job_nodes)
job_nodes = [node for node in job_nodes if node.skills]
print(f"Loaded {len(job_nodes)} jobs with skills (filtered out {original_count - len(job_nodes)})")

# Rebuild job_index
job_index = {node.title: node for node in job_nodes}

# Build job graph (directed edges)
edges = defaultdict(list)
for a in job_nodes:
    for b in job_nodes:
        if a is b:
            continue
        if b.salary_median <= a.salary_median:
            continue
        if not b.skills:
            continue
        missing_skills = b.skills - a.skills
        transition_cost = len(missing_skills) / len(b.skills)
        if transition_cost < 0.8:
            edges[a.title].append((b.title, transition_cost, missing_skills))

ALL_SKILLS = sorted(set(skill for node in job_nodes for skill in node.skills))

def match_ratio(user_skills, job_skills):
    if not job_skills:
        return 0
    overlap = len(user_skills & job_skills)
    return overlap / len(job_skills)

def total_learning_weeks(skills_list):
    """Sum of weeks for a list of skills."""
    return sum(get_learning_weeks(s) for s in skills_list)

def find_paths_to_target(user_skills, target_title, salary_threshold=0, include_undisclosed=False, max_depth=5):
    if target_title not in job_index:
        return None, "Target role not found in database"
    target_node = job_index[target_title]
    # Check target salary
    if target_node.salary_min == 0:
        if not include_undisclosed:
            return None, "Target role has undisclosed salary – enable 'include undisclosed' to proceed"
    else:
        if target_node.salary_min < salary_threshold:
            return None, f"Target role salary ({target_node.salary_min}) below threshold"

    user_skills_set = set(user_skills)
    target_skills = target_node.skills
    missing_to_target = target_skills - user_skills_set
    direct_weeks = total_learning_weeks(missing_to_target)

    direct_path = {
        "steps": [
            {
                "title": "Current (your skills)",
                "missing": list(missing_to_target),
                "missing_with_weeks": [{'skill': s, 'weeks': get_learning_weeks(s)} for s in missing_to_target],
                "weeks": direct_weeks,
                "salary": None
            },
            {
                "title": target_title,
                "missing": [],
                "missing_with_weeks": [],
                "weeks": 0,
                "salary": target_node.salary_display,
                "salary_min": target_node.salary_min,
                "salary_max": target_node.salary_max
            }
        ],
        "total_missing_skills": len(missing_to_target),
        "total_weeks": direct_weeks,
        "final_salary": target_node.salary_min,
        "type": "direct"
    }

    # Find starting jobs reachable with at least 30% match
    reachable_starts = []
    for node in job_nodes:
        if node.salary_min == 0:
            if not include_undisclosed:
                continue
        else:
            if node.salary_min < salary_threshold:
                continue
        ratio = match_ratio(user_skills_set, node.skills)
        if ratio >= 0.3:
            missing = node.skills - user_skills_set
            reachable_starts.append((node.title, missing))

    queue = deque()
    for start_title, missing in reachable_starts:
        weeks = total_learning_weeks(missing)
        step = {
            "title": start_title,
            "missing": list(missing),
            "missing_with_weeks": [{'skill': s, 'weeks': get_learning_weeks(s)} for s in missing],
            "weeks": weeks,
            "salary": job_index[start_title].salary_display,
            "salary_min": job_index[start_title].salary_min,
            "salary_max": job_index[start_title].salary_max
        }
        cumulative_skills = user_skills_set.union(missing)
        total_weeks = weeks
        queue.append((start_title, [step], cumulative_skills, total_weeks))

    found_paths = []
    seen_signatures = set()

    while queue and len(found_paths) < 10:
        current_title, steps, cumulative_skills, total_weeks = queue.popleft()
        if len(steps) > max_depth:
            continue

        if current_title == target_title:
            sig = tuple(step["title"] for step in steps)
            if sig not in seen_signatures:
                seen_signatures.add(sig)
                found_paths.append({
                    "steps": steps,
                    "total_missing_skills": sum(len(step["missing"]) for step in steps),
                    "total_weeks": total_weeks,
                    "final_salary": job_index[current_title].salary_min,
                    "type": "multi_step"
                })
            continue

        for neighbor_title, cost, _ in edges.get(current_title, []):
            neighbor_node = job_index[neighbor_title]
            if neighbor_node.salary_min == 0:
                if not include_undisclosed:
                    continue
            else:
                if neighbor_node.salary_min < salary_threshold:
                    continue
            if any(step["title"] == neighbor_title for step in steps):
                continue
            missing = neighbor_node.skills - cumulative_skills
            if missing and len(missing) / len(neighbor_node.skills) > 0.9:
                continue
            weeks = total_learning_weeks(missing)
            new_skills = cumulative_skills.union(missing)
            new_weeks = total_weeks + weeks
            new_step = {
                "title": neighbor_title,
                "missing": list(missing),
                "missing_with_weeks": [{'skill': s, 'weeks': get_learning_weeks(s)} for s in missing],
                "weeks": weeks,
                "salary": neighbor_node.salary_display,
                "salary_min": neighbor_node.salary_min,
                "salary_max": neighbor_node.salary_max
            }
            new_steps = steps + [new_step]
            queue.append((neighbor_title, new_steps, new_skills, new_weeks))

    multi_paths = found_paths
    multi_paths.sort(key=lambda x: x["total_weeks"])

    result_paths = []
    result_paths.extend(multi_paths[:3])
    direct_sig = (target_title,)
    if not any(tuple(step["title"] for step in p["steps"]) == direct_sig for p in result_paths):
        result_paths.append(direct_path)

    if not result_paths:
        result_paths = [direct_path]

    return result_paths[:3], None

# --- Explanation cache (in‑memory, 24h TTL) ---
explanation_cache = {}
CACHE_TTL = 86400

@app.route('/explain_path', methods=['POST'])
def explain_path():
    data = request.get_json()
    target_title = data.get('target_title')
    missing_skills = data.get('missing_skills', [])
    if not target_title or not missing_skills:
        return jsonify({'error': 'Missing target_title or missing_skills'}), 400

    cache_key = f"{target_title}_{'_'.join(sorted(missing_skills))}"
    now = time.time()
    if cache_key in explanation_cache:
        cached = explanation_cache[cache_key]
        if now - cached['timestamp'] < CACHE_TTL:
            return jsonify(cached['data'])

    skills_list = ", ".join(missing_skills)
    prompt = f"""
You are a career advisor for IT professionals in Malaysia. For the target role **{target_title}**, the user is missing the following skills: {skills_list}.

For each missing skill, provide:
- importance_score: integer from 1 to 10 (10 = critical for this role)
- explanation: one short sentence explaining WHY this skill is needed for {target_title} (max 20 words)
- suggested_order: integer (1 = learn first, then 2, etc.) based on logical prerequisites

Output ONLY valid JSON in this format:
{{
  "skills": [
    {{"skill": "skill_name", "importance_score": 8, "explanation": "Needed for infrastructure automation.", "suggested_order": 1}},
    ...
  ]
}}
No extra text, no URLs.
"""
    try:
        response = gemini_model.generate_content(prompt)
        raw = response.text.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.endswith("```"):
            raw = raw[:-3]
        result = json.loads(raw)
        if 'skills' in result and isinstance(result['skills'], list):
            explanation_cache[cache_key] = {'timestamp': now, 'data': result}
            return jsonify(result)
        else:
            return jsonify({'error': 'Invalid AI response format'}), 500
    except Exception as e:
        print(f"Explain path error: {e}")
        return jsonify({'error': str(e)}), 500

# --- Flask endpoints ---
@app.route('/')
def index():
    return send_from_directory('../frontend/static', 'index.html')

@app.route('/extract_skills', methods=['POST'])
def extract_skills_endpoint():
    if 'resume' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['resume']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name
    try:
        text = extract_text_from_pdf(tmp_path)
        if not text.strip():
            return jsonify({'skills': []})
        prompt = f"""
Extract only **technical skills** from the following resume text.
Rules:
- Return as a JSON list of strings, e.g., ["python", "docker", "aws"].
- Normalise synonyms: "K8s" → "kubernetes", "JS" → "javascript", "ReactJS" → "react", "Postgres" → "postgresql","CSS3" → "css", "HTML5" → "html", "css3" → "css", "html5" → "html".
- Exclude all soft skills (e.g., communication, leadership, problem-solving, team player).
- Keep skills lowercased.
- Output ONLY valid JSON – no extra text.

Resume text:
{text[:4000]}
"""
        response = gemini_model.generate_content(prompt)
        raw = response.text.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.endswith("```"):
            raw = raw[:-3]
        skills = json.loads(raw)
        if not isinstance(skills, list):
            skills = []
        return jsonify({'skills': skills})
    except Exception as e:
        print(f"Gemini error: {e}")
        return jsonify({'skills': []})
    finally:
        os.unlink(tmp_path)

@app.route('/job_titles', methods=['GET'])
def job_titles():
    titles = sorted(job_index.keys())
    return jsonify(titles)

@app.route('/all_skills', methods=['GET'])
def all_skills():
    return jsonify(ALL_SKILLS)

@app.route('/suggest', methods=['POST'])
def suggest():
    data = request.get_json()
    user_skills = set(data.get('skills', []))
    salary_threshold = data.get('salary_threshold', 0)
    include_undisclosed = data.get('include_undisclosed', False)
    all_roles = []
    for node in job_nodes:
        if node.salary_min == 0:
            if not include_undisclosed:
                continue
        else:
            if node.salary_min < salary_threshold:
                continue

        job_skills = node.skills
        if not job_skills:
            continue
        overlap = len(user_skills & job_skills)
        ratio = overlap / len(job_skills) if job_skills else 0
        missing = job_skills - user_skills
        missing_weeks = total_learning_weeks(missing)
        all_roles.append({
            'title': node.title,
            'company': node.company,
            'salary_min': node.salary_min,
            'salary_max': node.salary_max,
            'match_ratio': ratio,
            'missing_skills': list(missing),
            'missing_skills_with_weeks': [{'skill': s, 'weeks': get_learning_weeks(s)} for s in missing],
            'missing_weeks': missing_weeks,
            'reachable': ratio >= 0.6,
            'undisclosed': node.salary_min == 0
        })
    all_roles.sort(key=lambda x: -x['match_ratio'])
    return jsonify(all_roles)

@app.route('/find_paths', methods=['POST'])
def find_paths():
    data = request.get_json()
    user_skills = set(data.get('skills', []))
    target_title = data.get('target_title', '')
    salary_threshold = data.get('salary_threshold', 0)
    include_undisclosed = data.get('include_undisclosed', False)
    paths, error = find_paths_to_target(user_skills, target_title, salary_threshold, include_undisclosed)
    if error:
        return jsonify({'error': error}), 404
    return jsonify({'paths': paths})

# Load skill intelligence data
INTELLIGENCE_PATH = Path(__file__).parent.parent / "data" / "skill_intelligence.json"
if INTELLIGENCE_PATH.exists():
    with open(INTELLIGENCE_PATH) as f:
        skill_intelligence = json.load(f)
else:
    skill_intelligence = {}
    print("Warning: skill_intelligence.json not found, intelligence features disabled.")

@app.route('/skill_demand', methods=['GET'])
def skill_demand():
    """Return top skills by demand in jobs with salary >= threshold."""
    sort_by = request.args.get('sort', 'demand')   # only 'demand' now
    limit = int(request.args.get('limit', 30))
    threshold = int(request.args.get('threshold', 7000))

    demand = defaultdict(int)
    for node in job_nodes:
        if node.salary_min >= threshold:
            for skill in node.skills:
                demand[skill] += 1

    result = []
    for skill, count in demand.items():
        intel = skill_intelligence.get(skill, {})
        result.append({
            'skill': skill,
            'demand_count': count,
            'ai_replaceability_score': intel.get('ai_replaceability_score', 5),
            'skill_category': intel.get('skill_category', 'Other')
        })

    result.sort(key=lambda x: -x['demand_count'])
    return jsonify({
        'skills': result[:limit],
        'total': len(result),
        'sort_by': sort_by,
        'threshold': threshold
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=True)