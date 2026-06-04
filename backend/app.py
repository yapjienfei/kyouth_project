import json
import tempfile
import os
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
        self.salary_median = (self.salary_min + self.salary_max) / 2
        # Use pre‑computed skills_ai; fallback to empty list if missing
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
        if transition_cost < 0.6:
            edges[a.title].append((b.title, transition_cost, missing_skills))

ALL_SKILLS = sorted(set(skill for node in job_nodes for skill in node.skills))

def match_ratio(user_skills, job_skills):
    if not job_skills:
        return 0
    overlap = len(user_skills & job_skills)
    return overlap / len(job_skills)

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
    direct_match_ratio = match_ratio(user_skills_set, target_skills)

    direct_path = {
        "steps": [
            {"title": "Current (your skills)", "missing": list(missing_to_target), "weeks": len(missing_to_target)*2},
            {"title": target_title, "missing": [], "weeks": 0}
        ],
        "total_missing_skills": len(missing_to_target),
        "total_weeks": len(missing_to_target)*2,
        "final_salary": target_node.salary_min,
        "type": "direct"
    }

    if direct_match_ratio >= 0.6:
        return [direct_path], None

    # Find starting jobs that are reachable with at least 40% match
    reachable_starts = []
    for node in job_nodes:
        if node.salary_min == 0:
            if not include_undisclosed:
                continue
        else:
            if node.salary_min < salary_threshold:
                continue
        ratio = match_ratio(user_skills_set, node.skills)
        if ratio >= 0.4:
            # Compute missing from current to this job
            missing = node.skills - user_skills_set
            reachable_starts.append((node.title, missing))

    if not reachable_starts:
        return [direct_path], None

    from collections import deque
    queue = deque()
    for start_title, missing in reachable_starts:
        # Create first step: from current to start job
        step = {"title": start_title, "missing": list(missing), "weeks": len(missing)*2}
        cumulative_skills = user_skills_set.union(missing)
        total_weeks = len(missing)*2
        queue.append((start_title, [step], cumulative_skills, total_weeks))

    found_paths = []
    seen_signatures = set()

    while queue and len(found_paths) < 10:
        current_title, steps, cumulative_skills, total_weeks = queue.popleft()
        if len(steps) > max_depth:
            continue

        if current_title == target_title:
            # Reached target; steps already include the target as the last step? Actually we add target when we transition,
            # so the last step in steps is the target job. Good.
            # Create signature: titles of jobs in steps (excluding any "Current" which we don't have)
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

        # Expand to neighbors
        for neighbor_title, cost, _ in edges.get(current_title, []):  # we ignore precomputed missing; recompute
            neighbor_node = job_index[neighbor_title]
            if neighbor_node.salary_min == 0:
                if not include_undisclosed:
                    continue
            else:
                if neighbor_node.salary_min < salary_threshold:
                    continue
            # Avoid cycles
            if any(step["title"] == neighbor_title for step in steps):
                continue
            # Compute missing skills from current cumulative skills to neighbor
            missing = neighbor_node.skills - cumulative_skills
            # Only allow transition if missing is not too large? We can use a threshold, e.g., missing <= 70% of neighbor skills
            if missing and len(missing) / len(neighbor_node.skills) > 0.7:
                continue  # too big gap, skip
            new_skills = cumulative_skills.union(missing)
            new_weeks = total_weeks + len(missing)*2
            new_step = {"title": neighbor_title, "missing": list(missing), "weeks": len(missing)*2}
            new_steps = steps + [new_step]
            queue.append((neighbor_title, new_steps, new_skills, new_weeks))

    # Separate multi-step and direct
    multi_paths = found_paths
    multi_paths.sort(key=lambda x: x["total_weeks"])

    result_paths = []
    result_paths.extend(multi_paths[:3])
    # Add direct path if not already present as a single-step path
    direct_sig = (target_title,)  # direct path has only one job step after current
    if not any(tuple(step["title"] for step in p["steps"]) == direct_sig for p in result_paths):
        result_paths.append(direct_path)

    return result_paths[:3], None

# Flask endpoints
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
        # Call Gemini
        prompt = f"""
Extract only **technical skills** from the following resume text.
Rules:
- Return as a JSON list of strings, e.g., ["python", "docker", "aws"].
- Normalise synonyms: "K8s" → "kubernetes", "JS" → "javascript", "ReactJS" → "react", "Postgres" → "postgresql".
- Exclude all soft skills (e.g., communication, leadership, problem-solving, team player).
- Keep skills lowercased.
- Output ONLY valid JSON – no extra text.

Resume text:
{text[:4000]}   # truncate
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
        return jsonify({'skills': []})   # fallback to empty list
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
        # Salary filter:
        # - If job has salary_min > 0, require >= threshold
        # - If job has salary_min == 0 (undisclosed), include only if include_undisclosed is True
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
        all_roles.append({
            'title': node.title,
            'company': node.company,
            'salary_min': node.salary_min,
            'salary_max': node.salary_max,
            'match_ratio': ratio,
            'missing_skills': list(job_skills - user_skills),
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
    include_undisclosed = data.get('include_undisclosed', False)   # new
    paths, error = find_paths_to_target(user_skills, target_title, salary_threshold, include_undisclosed)
    if error:
        return jsonify({'error': error}), 404
    return jsonify({'paths': paths})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=True)