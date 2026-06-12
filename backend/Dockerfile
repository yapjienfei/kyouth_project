FROM python:3.12-slim

WORKDIR /app

# Install dependencies with pip — avoids uv downloading Python 3.14 inside the container
COPY pyproject.toml ./
RUN pip install --no-cache-dir \
    flask==3.1.0 \
    flask-cors==5.0.0 \
    requests==2.32.3 \
    pdfplumber==0.11.4 \
    python-dotenv==1.0.1 \
    "google-genai>=1.0.0" \
    "psycopg2-binary>=2.9" \
    "celery[redis]>=5.4" \
    "redis>=5.0"

COPY backend/ ./backend/
COPY data/ ./data/
COPY scripts/ ./scripts/

EXPOSE 8001

CMD ["python", "backend/app.py"]
