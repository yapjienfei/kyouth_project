"""Career graph computation — pure functions shared by the Flask app and the
Celery worker. All structures are JSON-serializable so the worker can hand
results to the API process through Redis."""
from collections import defaultdict


def build_graph(jobs: list[dict]) -> dict:
    """Compute per-title stats and ascending-salary edges from raw job rows.

    Returns a JSON-serializable dict:
      {"title_stats": {title: {...skills: [...]}}, "edges": {title: [[next, cost, missing], ...]}}
    """
    nodes = []
    for j in jobs:
        skills = [s.lower() for s in (j.get("skills_ai") or [])]
        if not skills:
            continue
        salary_min = j.get("salary_min") or 0
        salary_max = j.get("salary_max") or 0
        nodes.append({
            "title": j["title"],
            "company": (j.get("company") or "").strip(),
            "salary_min": salary_min,
            "salary_max": salary_max,
            "salary_median": (salary_min + salary_max) / 2 if salary_max > 0 else 0,
            "skills": set(skills),
        })

    groups: dict[str, list[dict]] = defaultdict(list)
    for n in nodes:
        groups[n["title"]].append(n)

    title_stats: dict[str, dict] = {}
    for title, grp in groups.items():
        sal = [n for n in grp if n["salary_median"] > 0]
        all_skills = set().union(*(n["skills"] for n in grp))
        title_stats[title] = {
            "count": len(grp),
            "salary_median": sum(n["salary_median"] for n in sal) / len(sal) if sal else 0,
            "salary_min": min((n["salary_min"] for n in sal), default=0),
            "salary_max": max((n["salary_max"] for n in sal), default=0),
            "skills": sorted(all_skills),
            "has_salary": bool(sal),
            "companies": sorted({n["company"] for n in grp if n["company"]}),
        }

    # Directed edges: A→B if B has higher median salary AND transition cost < 0.8
    edges: dict[str, list] = defaultdict(list)
    for title_a, stats_a in title_stats.items():
        skills_a = set(stats_a["skills"])
        for title_b, stats_b in title_stats.items():
            if title_a == title_b:
                continue
            if stats_b["salary_median"] <= stats_a["salary_median"]:
                continue
            if not stats_b["skills"]:
                continue
            skills_b = set(stats_b["skills"])
            missing = skills_b - skills_a
            cost = len(missing) / len(skills_b)
            if cost < 0.8:
                edges[title_a].append([title_b, cost, sorted(missing)])

    return {"title_stats": title_stats, "edges": dict(edges)}


def deserialize_graph(data: dict) -> tuple[dict, dict]:
    """Convert the JSON form back to runtime types (skill sets, tuples)."""
    title_stats = {}
    for title, st in data["title_stats"].items():
        st = dict(st)
        st["skills"] = frozenset(st["skills"])
        title_stats[title] = st
    edges = {
        title: [(t, cost, frozenset(missing)) for t, cost, missing in lst]
        for title, lst in data["edges"].items()
    }
    return title_stats, edges
