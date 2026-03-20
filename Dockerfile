FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

RUN python -m pip install --upgrade pip \
    && python -m pip install fastapi uvicorn pydantic pymongo

COPY src ./src
COPY run_api.py README_MVP.md ./

EXPOSE 8000

CMD ["python", "run_api.py"]
