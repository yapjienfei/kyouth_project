FROM python:3.14-slim

WORKDIR /app

# Install system dependencies for pdfplumber
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml .python-version ./

# Install all Python dependencies explicitly (including google-generativeai)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        flask==3.1.0 \
        flask-cors==5.0.0 \
        requests==2.32.3 \
        pdfplumber==0.11.4 \
        python-dotenv==1.0.1 \
        google-generativeai>=0.8.0

# Copy backend code and data
COPY backend/ ./backend/
COPY data/ ./data/

EXPOSE 8001

CMD ["python", "backend/app.py"]