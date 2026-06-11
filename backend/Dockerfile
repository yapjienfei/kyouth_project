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
    "google-genai>=1.0.0"

COPY backend/ ./backend/
COPY frontend/static/ ./frontend/static/
COPY data/ ./data/
COPY scripts/ ./scripts/

EXPOSE 8001

CMD ["python", "backend/app.py"]
