FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

COPY pyproject.toml README.md ./
COPY app.py ./app.py
COPY src ./src
COPY knowledge ./knowledge

RUN python -m pip install --no-cache-dir . \
    && useradd --create-home --shell /usr/sbin/nologin routearena \
    && mkdir -p /app/runs /app/outputs \
    && chown -R routearena:routearena /app

USER routearena

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
