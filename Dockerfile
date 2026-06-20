FROM python:3.12-slim

WORKDIR /app

# Core deps only (voice/ML extras are optional and omitted from the image).
COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi uvicorn[standard] httpx pydantic python-dotenv Faker numpy

COPY . .

# Bake the synthetic datasets into the image so services are ready on boot.
RUN python -m data.generate

EXPOSE 8101 8102 8103 8104 8105

# Default: run a single platform (overridden per-service in docker-compose).
CMD ["python", "-m", "scripts.run_platforms"]
