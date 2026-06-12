"""Celery tasks — graph computation runs in the worker, results land in Redis.

The Flask app reads the computed graph from Redis (key career:graph) and
re-checks the version key periodically, so a recompute becomes visible
without restarting the API.
"""
import json
import os
import sys
import time
from pathlib import Path

from celery import Celery

sys.path.insert(0, str(Path(__file__).parent))

import db
import career_graph

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery("career", broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.update(task_serializer="json", result_serializer="json", accept_content=["json"])

GRAPH_KEY = "career:graph"
GRAPH_VERSION_KEY = "career:graph:version"


@celery_app.task(name="career.rebuild_graph")
def rebuild_graph() -> dict:
    """Load jobs (postgres first, JSON fallback), compute the career graph,
    store it in Redis for the API process."""
    import redis as redis_lib

    jobs = db.load_jobs()
    source = "postgres"
    if jobs is None:
        snapshot = Path(__file__).parent.parent / "data" / "jobs_snapshot.json"
        with open(snapshot) as f:
            jobs = json.load(f)
        source = "json"

    t0 = time.time()
    graph = career_graph.build_graph(jobs)
    elapsed = round(time.time() - t0, 2)

    r = redis_lib.from_url(REDIS_URL)
    r.set(GRAPH_KEY, json.dumps(graph))
    version = str(time.time())
    r.set(GRAPH_VERSION_KEY, version)

    n_edges = sum(len(v) for v in graph["edges"].values())
    print(f"[task] graph rebuilt from {source}: {len(graph['title_stats'])} titles, "
          f"{n_edges} edges in {elapsed}s")
    return {"titles": len(graph["title_stats"]), "edges": n_edges,
            "seconds": elapsed, "source": source, "version": version}
