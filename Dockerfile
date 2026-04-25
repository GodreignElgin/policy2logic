FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY policy_to_logic_env/server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the package
COPY policy_to_logic_env/ /app/policy_to_logic_env/
COPY main.py /app/main.py
COPY inference.py /app/inference.py

# Expose the HF Spaces port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run the server
CMD ["python", "-m", "uvicorn", "policy_to_logic_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
