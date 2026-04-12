FROM python:3.12-slim

WORKDIR /app

# Install uv for fast, reproducible installs (openenv validate checks uv.lock)
RUN pip install --no-cache-dir uv

# Copy project files
COPY . .

# Install all dependencies using uv (honours uv.lock for reproducibility)
RUN uv pip install --system --no-cache .

EXPOSE 8000

CMD ["uvicorn", "execucode.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
