FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire package
COPY . /app/policy_to_logic_env/server/

# Copy models.py and __init__.py to the right place
COPY ../models.py /app/policy_to_logic_env/models.py
COPY ../__init__.py /app/policy_to_logic_env/__init__.py
COPY ../client.py /app/policy_to_logic_env/client.py

# Expose the port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run the server
CMD ["uvicorn", "policy_to_logic_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
