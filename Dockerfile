FROM python:3.11-slim

# Prevent python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (build-essential, curl, git)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy codebase
COPY . .

# Ensure uploads and outputs directories exist
RUN mkdir -p uploads outputs

# Expose ports: 7860 (UI) and 8000 (MCP Server)
EXPOSE 7860
EXPOSE 8000

# Make start_services.sh executable inside image
RUN chmod +x start_services.sh

# Run startup script by default
CMD ["./start_services.sh"]