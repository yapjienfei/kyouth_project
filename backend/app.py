import json
import tempfile
import os
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pdfplumber
from collections import defaultdict, deque
import re

app = Flask(__name__, static_folder='../frontend/static', static_url_path='')
CORS(app)

# Load job data
JOBS_PATH = Path(__file__).parent.parent / "data" / "jobs_snapshot.json"
with open(JOBS_PATH) as f:
    JOBS = json.load(f)

# Simple skill extraction (keyword matching)
# Normalization mapping for skill aliases
NORMALIZE_MAP = {
    "nodejs": "node.js",
    "js": "javascript",
    "reactjs": "react",
    "postgres": "postgresql",
    "k8s": "kubernetes",
    "kubernetes": "kubernetes",
    "cicd": "ci/cd",
    "gitops": "gitops",
    "argocd": "argo cd",
    "jenkins": "jenkins",
    "gitlab": "gitlab ci",
    "github actions": "github actions",
    "terraform": "terraform",
    "aws": "aws",
    "azure": "azure",
    "gcp": "gcp",
    "docker": "docker",
    "containerd": "docker",
    "prometheus": "prometheus",
    "grafana": "grafana",
    "splunk": "splunk",
    "kafka": "kafka",
    "spark": "spark",
    "mlflow": "mlflow",
    "kubeflow": "kubeflow",
    "sagemaker": "sagemaker",
    "pytorch": "pytorch",
    "tensorflow": "tensorflow",
    "scikit-learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "pandas": "pandas",
    "numpy": "numpy",
    "tableau": "tableau",
    "power bi": "power bi",
    "qlik": "qlik",
    "excel": "excel",
    "spreadsheets": "excel",
    "react native": "react native",
    "flutter": "flutter",
    "swift": "swift",
    "kotlin": "kotlin",
    "java": "java",
    "csharp": "c#",
    "c#": "c#",
    "c++": "c++",
    "go": "go",
    "rust": "rust",
    "solidity": "solidity",
    "web3": "web3.js",
    "web3.js": "web3.js",
    "html": "html",
    "css": "css",
    "sass": "sass",
    "less": "less",
    "rest": "rest api",
    "restful": "rest api",
    "graphql": "graphql",
    "sql": "sql",
    "mysql": "mysql",
    "postgresql": "postgresql",
    "mongodb": "mongodb",
    "redis": "redis",
    "elasticsearch": "elasticsearch",
    "hadoop": "hadoop",
    "bigquery": "bigquery",
    "redshift": "redshift",
    "snowflake": "snowflake",
    "linux": "linux",
    "unix": "linux",
    "bash": "bash",
    "zsh": "bash",
    "powershell": "powershell",
    "cmd": "powershell",
    "git": "git",
    "svn": "svn",
    "jira": "jira",
    "confluence": "confluence",
    "trello": "trello",
    "agile": "agile",
    "scrum": "scrum",
    "kanban": "kanban",
    "project management": "project management",
    "people management": "people management",
    "leadership": "leadership",
    "mentoring": "mentoring",
    "coaching": "coaching",
    "communication": "communication",
    "stakeholder management": "stakeholder management",
    "risk management": "risk management",
    "budgeting": "budgeting",
    "togaf": "togaf",
    "enterprise architecture": "enterprise architecture",
    "governance": "governance",
    "consulting": "consulting",
    "business analysis": "business analysis",
    "it strategy": "it strategy",
    "digital transformation": "digital transformation",
    "pre-sales": "pre-sales",
    "solution architecture": "solution architecture",
    "product management": "product management",
    "market research": "market research",
    "user stories": "user stories",
    "data analysis": "data analysis",
    "release management": "release management",
    "change management": "change management",
    "technical writing": "technical writing",
    "markdown": "markdown",
    "ux design": "ux design",
    "figma": "figma",
    "prototyping": "prototyping",
    "user research": "user research",
    "wireframing": "wireframing",
    "usability testing": "usability testing",
    "design systems": "design systems",
    "cybersecurity": "cybersecurity",
    "penetration testing": "penetration testing",
    "owasp": "owasp",
    "ids/ips": "ids/ips",
    "siem": "siem",
    "vulnerability assessment": "vulnerability assessment",
    "incident response": "incident response",
    "kali linux": "kali linux",
    "iso 27001": "iso 27001",
    "gdpr": "gdpr",
    "cloud security": "cloud security",
    "red teaming": "red teaming",
    "security architecture": "security architecture",
    "data modeling": "data modeling",
    "etl": "etl",
    "data warehouse": "data warehouse",
    "data visualization": "data visualization",
    "statistics": "statistics",
    "mlops": "mlops",
    "cisco": "cisco",
    "routing": "routing",
    "switching": "switching",
    "firewalls": "firewalls",
    "vpn": "vpn",
    "sd-wan": "sd-wan",
    "network security": "network security",
    "troubleshooting": "troubleshooting",
    "customer service": "customer service",
    "active directory": "active directory",
    "backup": "backup/recovery",
    "recovery": "backup/recovery",
    "monitoring": "monitoring",
    "ticketing system": "ticketing system",
    "problem-solving": "problem-solving",
    "attention to detail": "attention to detail",
    "team leadership": "team leadership",
    "architecture": "architecture",
    "system design": "system design",
    "agile at scale": "agile at scale",
    "performance tuning": "performance tuning",
    "high availability": "high availability",
    "replication": "replication",
    "automation": "automation",
    "incident management": "incident management",
    "chaos engineering": "chaos engineering",
}

def normalize_skill(skill: str) -> str:
    """Normalize skill name to canonical form."""
    skill_lower = skill.lower().strip()
    # Remove punctuation except dot and dash
    skill_clean = re.sub(r"[^\w\s.-]", "", skill_lower)
    # Map common aliases
    if skill_clean in NORMALIZE_MAP:
        return NORMALIZE_MAP[skill_clean]
    # Also try without spaces? No, keep as is.
    return skill_clean

# Unique set of canonical skills from the mapping values
COMMON_SKILLS = set(NORMALIZE_MAP.values())
# Add some extra that might not be in mapping but are common
COMMON_SKILLS.update([
    "python", "django", "flask", "fastapi", "sqlalchemy", "pytest", "unittest",
    "selenium", "cypress", "jmeter", "postman", "soapui", "jenkins", "gitlab ci",
    "github actions", "circleci", "travis", "ansible", "chef", "puppet",
    "terraform", "cloudformation", "pulumi", "docker", "podman", "containerd",
    "kubernetes", "openshift", "rancher", "helm", "istio", "linkerd",
    "prometheus", "grafana", "datadog", "new relic", "splunk", "elk", "elastic stack",
    "logstash", "kibana", "kafka", "rabbitmq", "activemq", "redis", "memcached",
    "postgresql", "mysql", "mariadb", "mongodb", "cassandra", "couchdb", "dynamodb",
    "bigquery", "redshift", "snowflake", "databricks", "hadoop", "spark", "flink",
    "airflow", "dbt", "looker", "tableau", "power bi", "qlik", "excel", "pandas",
    "numpy", "scipy", "scikit-learn", "tensorflow", "pytorch", "keras", "mxnet",
    "onnx", "mlflow", "kubeflow", "sagemaker", "azure ml", "vertex ai",
    "aws", "azure", "gcp", "oracle cloud", "ibm cloud", "alibaba cloud",
    "linux", "windows server", "unix", "bash", "zsh", "powershell", "cmd",
    "git", "svn", "mercurial", "jira", "confluence", "trello", "asana", "clickup",
    "agile", "scrum", "kanban", "safe", "less", "project management", "people management",
    "leadership", "mentoring", "coaching", "communication", "stakeholder management",
    "risk management", "budgeting", "togaf", "zachman", "enterprise architecture",
    "governance", "compliance", "audit", "consulting", "business analysis", "it strategy",
    "digital transformation", "pre-sales", "solution architecture", "product management",
    "market research", "user stories", "data analysis", "release management",
    "change management", "technical writing", "markdown", "ux design", "figma",
    "sketch", "adobe xd", "prototyping", "user research", "wireframing", "usability testing",
    "design systems", "cybersecurity", "information security", "penetration testing",
    "vulnerability assessment", "incident response", "forensics", "malware analysis",
    "reverse engineering", "owasp", "nist", "iso 27001", "gdpr", "hipaa", "pci dss",
    "cloud security", "application security", "network security", "identity management",
    "iam", "sso", "oauth", "saml", "ldap", "active directory", "cisco", "juniper",
    "arista", "routing", "switching", "firewalls", "vpn", "sd-wan", "sd-access",
    "load balancing", "dns", "dhcp", "tcp/ip", "snmp", "netflow", "sflow",
    "troubleshooting", "debugging", "performance tuning", "monitoring", "alerting",
    "logging", "tracing", "distributed tracing", "jaeger", "zipkin", "opentelemetry",
    "automation", "scripting", "infrastructure as code", "configuration management",
    "continuous integration", "continuous delivery", "continuous deployment",
    "devops", "sre", "platform engineering", "chaos engineering", "reliability engineering",
    "scalability", "high availability", "disaster recovery", "backup", "replication",
    "data modeling", "etl", "data warehouse", "data lake", "data mesh", "data governance",
    "data quality", "data lineage", "data catalog", "statistics", "probability",
    "linear algebra", "calculus", "machine learning", "deep learning", "nlp",
    "computer vision", "reinforcement learning", "time series analysis", "forecasting",
    "optimization", "simulation", "operations research", "business intelligence",
    "reporting", "dashboarding", "data storytelling", "a/b testing", "experimentation",
    "product analytics", "funnel analysis", "cohort analysis", "retention analysis",
    "churn analysis", "customer segmentation", "clustering", "classification",
    "regression", "hypothesis testing", "confidence intervals", "p-value",
    "bayesian statistics", "monte carlo", "game theory", "econometrics",
    "financial modeling", "accounting", "budgeting", "forecasting", "variance analysis",
    "roi analysis", "cost-benefit analysis", "business case", "requirements gathering",
    "use case modeling", "uml", "sequence diagram", "class diagram", "er diagram",
    "bpmn", "flowchart", "wireframe", "mockup", "prototype", "user journey map",
    "service blueprint", "customer journey map", "empathy map", "persona",
    "survey design", "questionnaire", "interview", "focus group", "usability test",
    "a/b test", "multivariate test", "conversion rate optimization", "seo",
    "sem", "google analytics", "adobe analytics", "mixpanel", "amplitude",
    "segment", "mparticle", "rudderstack", "snowplow", "matomo", "piwik",
    "hotjar", "crazy egg", "fullstory", "logrocket", "sentry", "bugsnag",
    "rollbar", "datadog", "new relic", "dynatrace", "appdynamics", "instana",
    "elastic apm", "skywalking", "pinpoint", "jaeger", "zipkin", "opentelemetry",
    "prometheus", "grafana", "loki", "tempo", "mimir", "thanos", "cortex",
    "victoriametrics", "influxdb", "telegraf", "chronograf", "kapacitor",
    "nagios", "zabbix", "icinga", "sensu", "checkmk", "op5", "solarwinds",
    "prtg", "cacti", "observium", "librenms", "netdata", "glances",
])

# Ensure we have a canonical version of each skill (already done via NORMALIZE_MAP values)
COMMON_SKILLS = set(NORMALIZE_MAP.values())  # start with mapped values
# Add some additional common ones not in mapping but needed
COMMON_SKILLS.update([
    "python", "javascript", "typescript", "java", "c#", "c++", "go", "rust",
    "php", "ruby", "swift", "kotlin", "scala", "clojure", "haskell", "erlang",
    "elixir", "lua", "r", "matlab", "julia", "perl", "groovy", "dart", "html", "css",
    "sass", "less", "stylus", "react", "angular", "vue", "svelte", "ember", "backbone",
    "jquery", "bootstrap", "tailwind", "material ui", "ant design", "chakra ui",
    "node.js", "express", "koa", "fastify", "nestjs", "django", "flask", "fastapi",
    "spring boot", "asp.net core", "laravel", "rails", "phoenix", "gin", "echo",
    "fiber", "actix", "rocket", "tornado", "aiohttp", "quart", "sanic", "bottle",
    "pyramid", "web2py", "cakephp", "symfony", "codeigniter", "yii", "zend",
    "play framework", "grails", "micronaut", "quarkus", "vert.x", "akka",
    "sql", "mysql", "postgresql", "sqlite", "mariadb", "oracle", "sql server",
    "db2", "sybase", "firebird", "h2", "hsqldb", "derby", "cassandra", "mongodb",
    "couchdb", "couchbase", "redis", "memcached", "hazelcast", "ignite", "geode",
    "elasticsearch", "solr", "lucene", "sphinx", "vespa", "meilisearch", "typesense",
    "algolia", "azure search", "aws cloudsearch", "aws opensearch",
    "docker", "podman", "containerd", "cri-o", "rkt", "lxc", "lxd", "openvz",
    "kubernetes", "openshift", "rancher", "k3s", "k0s", "microk8s", "minikube",
    "kind", "docker desktop", "podman desktop", "helm", "kustomize", "jsonnet",
    "cue", "pulumi", "terraform", "cloudformation", "cdk", "amplify", "serverless",
    "sam", "chalice", "zappa", "terraform cdk", "crossplane", "kubevela",
    "argocd", "flux", "jenkins x", "tekton", "buildpacks", "pack", "jib",
    "skaffold", "tilt", "garden", "werf", "devspace", "okteto", "telepresence",
    "istio", "linkerd", "consul", "envoy", "nginx", "haproxy", "traefik",
    "caddy", "apache", "tomcat", "jetty", "undertow", "gunicorn", "uwsgi",
    "waitress", "unicorn", "puma", "passenger", "iis", "caddy", "varnish",
    "squid", "apache traffic server", "cloudflare", "fastly", "akamai",
    "cloudfront", "cloudflare workers", "lambda@edge", "cloudflare pages",
    "netlify", "vercel", "heroku", "fly.io", "render", "railway", "koyeb",
    "digitalocean app platform", "google app engine", "aws elastic beanstalk",
    "azure app service", "google cloud run", "aws fargate", "azure container instances",
    "google cloud functions", "aws lambda", "azure functions", "google cloud functions",
    "openfaas", "knative", "fn", "ironfunctions", "fission", "openwhisk",
    "kubeless", "nuclio", "riff", "wasm", "wasmer", "wasmtime", "wasmcloud",
    "assemblyscript", "emscripten", "blazor", "webassembly", "webgpu", "webgl",
    "three.js", "babylon.js", "pixi.js", "phaser", "unity", "unreal engine",
    "godot", "cryengine", "lumberyard", "source engine", "id tech", "frostbite",
    "anvil", "playcanvas", "construct", "gamemaker", "rpg maker", "renpy",
    "twine", "inkscape", "gimp", "krita", "blender", "maya", "3ds max", "cinema 4d",
    "houdini", "substance painter", "substance designer", "zbrush", "photoshop",
    "illustrator", "after effects", "premiere pro", "final cut pro", "davinci resolve",
    "audacity", "logic pro", "ableton live", "fl studio", "pro tools", "reaper",
    "ardour", "lmms", "musescore", "lilypond", "abc notation", "midi", "osc",
    "dmx", "artnet", "sacn", "resolume", "modul8", "vdmx", "millumin", "qlab",
    "watchout", "d3", "echarts", "highcharts", "chart.js", "apexcharts", "plotly",
    "bokeh", "ggplot2", "matplotlib", "seaborn", "altair", "vega", "vega-lite",
    "observable", "jupyter", "notebook", "jupyterlab", "colab", "deepnote",
    "databricks", "sagemaker studio", "azure machine learning", "vertex ai",
    "wandb", "neptune", "comet", "weights & biases", "sacred", "dvc", "pachyderm",
    "kedro", "ploomber", "dagster", "prefect", "airflow", "luigi", "argo workflows",
    "kubeflow pipelines", "tfx", "mlflow", "kubeflow", "sagemaker pipelines",
    "azure pipelines", "google cloud pipelines", "jenkins", "gitlab ci", "github actions",
    "circleci", "travis", "buddy", "codeship", "semaphore", "buildkite", "teamcity",
    "bamboo", "azure pipelines", "aws codepipeline", "google cloud build",
    "drone", "woodpecker", "concourse", "flux", "argocd", "jenkins x", "spinnaker",
    "keel", "werf", "devspace", "okteto", "skaffold", "tilt", "garden", "werf",
    "buildpacks", "pack", "jib", "s2i", "source-to-image", "cargo", "npm", "yarn",
    "pnpm", "bun", "pip", "pipenv", "poetry", "conda", "mamba", "setup.py",
    "requirements.txt", "pyproject.toml", "go mod", "glide", "godep", "dep",
    "cargo", "crates.io", "nuget", "maven", "gradle", "ant", "ivy", "sbt",
    "leiningen", "boot", "clojure cli", "mix", "hex", "rebar", "erlang.mk",
    "bazel", "pants", "please", "please build", "meson", "ninja", "cmake",
    "make", "gnu make", "nmake", "qmake", "waf", "scons", "rake", "grunt",
    "gulp", "webpack", "rollup", "esbuild", "vite", "parcel", "snowpack",
    "swc", "babel", "typescript compiler", "terser", "uglify", "minify",
    "obfuscator", "clean-css", "csso", "sass", "less", "stylus", "postcss",
    "autoprefixer", "cssnano", "purgecss", "uncss", "critical", "penthouse",
    "stylelint", "eslint", "prettier", "black", "isort", "flake8", "pylint",
    "mypy", "pyright", "ruff", "clippy", "rustfmt", "gofmt", "goimports",
    "golint", "staticcheck", "javac", "checkstyle", "spotbugs", "pmd",
    "sonarqube", "sonarcloud", "codeclimate", "codacy", "coveralls", "codecov",
    "simplecov", "jacoco", "istanbul", "nyc", "c8", "vitest", "jest", "mocha",
    "chai", "assert", "should", "expect", "ava", "tape", "tap", "qunit",
    "jasmine", "karma", "protractor", "cypress", "playwright", "puppeteer",
    "selenium", "webdriverio", "testcafe", "nightwatch", "codeceptjs",
    "robot framework", "pytest", "unittest", "nose2", "behave", "lettuce",
    "cucumber", "gherkin", "specflow", "cucumber-jvm", "cucumber-js",
    "fit", "fitnesse", "concordion", "xunit", "nunit", "junit", "testng",
    "spock", "specs2", "scalatest", "scalacheck", "quickcheck", "propcheck",
    "fuzz", "afl", "libfuzzer", "honggfuzz", "zzuf", "radamsa", "peach",
    "american fuzzy lop", "syzkaller", "trinity", "kasan", "ubsan", "asan",
    "msan", "tsan", "lsan", "valgrind", "dr memory", "purify", "insure++",
    "boundcheck", "parasoft", "vectorcast", "ldra", "testbed", "cantata",
    "tessy", "gtest", "catch2", "doctest", "boost.test", "criterion",
    "unity", "cmocka", "check", "greatest", "minunit", "ctest", "gbenchmark",
    "google benchmark", "celero", "hayai", "nonius", "nanobench", "benchmark",
    "perf", "perf-tool", "flamegraph", "pprof", "async-profiler", "jfr",
    "jstack", "jmap", "jstat", "visualvm", "yourkit", "jprofiler", "dynatrace",
    "appdynamics", "new relic", "datadog", "instana", "elastic apm", "skylight",
    "scout", "puma", "unicorn", "passenger", "nginx", "apache", "traefik",
    "envoy", "haproxy", "varnish", "squid", "caddy", "openresty", "kong",
    "tyk", "gravitee", "wso2", "apisix", "express gateway", "krakend",
    "fabio", "linkerd", "consul", "nacos", "eureka", "zookeeper", "etcd",
    "raft", "paxos", "gossip", "swim", "serf", "memberlist", "hashicorp",
    "vault", "consul", "nomad", "terraform", "packer", "vagrant", "boundary",
    "waypoint", "harness", "octopus deploy", "deployinator", "capistrano",
    "mina", "fabric", "ansible", "chef", "puppet", "salt", "cfengine",
    "rudder", "terraform", "cloudformation", "cdk", "pulumi", "crossplane",
    "kubevela", "kustomize", "jsonnet", "cue", "dhal", "otto", "bosh",
    "spruce", "yaml", "toml", "json", "xml", "protobuf", "avro", "parquet",
    "orc", "arrow", "thrift", "msgpack", "bson", "ubjson", "cbor", "ion",
    "smile", "fst", "kryo", "java serialization", "pickle", "marshal",
    "shelve", "dill", "cloudpickle", "joblib", "onnx", "onnxruntime",
    "tensorrt", "openvino", "ncnn", "mnn", "tnn", "tflite", "coreml",
    "mlkit", "caffe2", "torchscript", "jit", "aot", "tvm", "mlir", "iree",
    "xla", "pytorch xla", "tensorflow xla", "jax", "flax", "haiku",
    "optax", "chex", "dm-haiku", "dm-sonnet", "tf-agents", "rlkit",
    "stable-baselines3", "ray", "rllib", "ray serve", "ray tune",
    "horovod", "deepspeed", "fairscale", "megatron", "gpt-neox",
    "deep learning", "machine learning", "nlp", "computer vision",
    "speech recognition", "recommendation systems", "search", "ranking",
    "advertising", "optimization", "scheduling", "simulation", "control theory",
    "robotics", "autonomous systems", "self-driving", "drones", "iot",
    "embedded systems", "firmware", "bare metal", "real-time os", "freertos",
    "zephyr", "nuttx", "rt-thread", "mqx", "vxworks", "qnx", "integrity",
    "safe rtos", "se L4", "pikeos", "eCos", "rtems", "openrtos", "arm mbed",
    "arduino", "esp-idf", "platformio", "rust embedded", "cortex-m",
    "risc-v", "arm", "x86", "mips", "powerpc", "avr", "pic", "8051",
    "stm32", "nrf52", "esp32", "rp2040", "teensy", "beaglebone", "raspberry pi",
    "jetson", "nano", "tx2", "xavier", "orin", "coral", "edge tpu",
    "neural compute stick", "movidius", "myriad x", "keem bay", "hailo",
    "gv100", "a100", "h100", "mi100", "mi200", "mi300", "tpu", "ipu",
    "cerebras", "graphcore", "sambanova", "groq", "tenstorrent", "untether",
    "flex logix", "mythic", "recogni", "perceive", "exa corp", "lightmatter",
    "luminous", "celestial ai", "syntiant", "greenwaves", "deepscale",
    "untether", "tenstorrent", "graphcore", "cerebras", "sambanova",
    "groq", "habana", "gaudi", "goya", "inferentia", "trainium", "transformer",
    "attention", "bert", "gpt", "llama", "mistral", "falcon", "bloom",
    "t5", "bart", "albert", "roberta", "distilbert", "xlnet", "electra",
    "deberta", "longformer", "bigbird", "reformer", "performer", "linformer",
    "nystromformer", "cosformer", "hydra", "megatron", "gpt-neox", "palm",
    "chinchilla", "gopher", "claude", "gemini", "gemma", "phi", "qwen",
    "yi", "deepseek", "starcoder", "codegen", "codellama", "wizardcoder",
    "phind", "seamless", "m2m100", "nllb", "opus", "mbart", "mbart50",
    "xlm-roberta", "xlm", "xlm-v", "rembert", "camembert", "flaubert",
    "bertweet", "twitter-roberta", "sentiment", "emotion", "stance",
    "topic modeling", "lda", "nmf", "bertopic", "top2vec", "sentence-transformers",
    "clip", "blip", "flava", "fuyu", "kosmos", "florence", "owl",
    "detr", "yolo", "ssd", "faster r-cnn", "mask r-cnn", "retinanet",
    "efficientdet", "dino", "dab-detr", "deformable detr", "sparse r-cnn",
    "centernet", "cornernet", "extremenet", "keypoint", "pose", "human pose",
    "mpii", "coco", "openpose", "mediapipe", "moveit", "robotics operating system",
    "ros", "ros2", "gazebo", "webots", "coppelia", "mujoco", "pybullet",
    "simulink", "stateflow", "matlab robotics", "ardurobot", "lego mindstorms",
    "vex", "frc", "fll", "wro", "robocup", "darpa", "icra", "iros",
])

# Since the set is huge, we rely on normalization and then just a big set.
# But to keep performance, we can keep the set as is.
# However, we must also include all unique skills from our dataset's descriptions.
# Instead of manually listing, we'll trust the normalization + big set.

# But to be safe, let's also extract skills from the dataset at runtime? That's too heavy.
# We'll just use the big set and normalization.

def extract_skills(text):
    text_lower = text.lower()
    found = set()
    for skill in COMMON_SKILLS:
        if skill in text_lower:
            found.add(skill)
    return list(found)

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
        self.skills = set(extract_skills(job_dict.get("description", "")))
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

def find_paths_to_target(user_skills, target_title, salary_threshold=0, max_depth=5):
    if target_title not in job_index:
        return None, "Target role not found in database"
    target_node = job_index[target_title]
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
        skills = extract_skills(text)
        return jsonify({'skills': skills})
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
    all_roles = []
    for node in job_nodes:
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
            'reachable': ratio >= 0.6
        })
    all_roles.sort(key=lambda x: -x['match_ratio'])
    return jsonify(all_roles)

@app.route('/analyze_target', methods=['POST'])
def analyze_target():
    data = request.get_json()
    user_skills = set(data.get('skills', []))
    target_title = data.get('target_title', '')
    salary_threshold = data.get('salary_threshold', 0)
    job = job_index.get(target_title)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    if job.salary_min < salary_threshold:
        return jsonify({'error': f'Salary below threshold'}), 400
    job_skills = job.skills
    overlap = len(user_skills & job_skills)
    ratio = overlap / len(job_skills) if job_skills else 0
    return jsonify({
        'title': job.title,
        'company': job.company,
        'salary_min': job.salary_min,
        'salary_max': job.salary_max,
        'match_ratio': ratio,
        'missing_skills': list(job_skills - user_skills),
        'reachable': ratio >= 0.6
    })

@app.route('/find_paths', methods=['POST'])
def find_paths():
    data = request.get_json()
    user_skills = set(data.get('skills', []))
    target_title = data.get('target_title', '')
    salary_threshold = data.get('salary_threshold', 0)
    paths, error = find_paths_to_target(user_skills, target_title, salary_threshold)
    if error:
        return jsonify({'error': error}), 404
    return jsonify({'paths': paths})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=True)