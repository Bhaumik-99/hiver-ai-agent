FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY scripts ./scripts
COPY data ./data
COPY results ./results
COPY .env.example ./.env.example

# The application uses Groq by default. A local Ollama backend is not
# installed in this image; pass GROQ_API_KEY at runtime or point the client
# at an externally reachable Ollama instance if using --local.

CMD ["python", "scripts/demo.py", "--help"]
