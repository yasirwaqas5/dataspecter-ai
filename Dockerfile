# Use Python 3.13 slim base
FROM python:3.13-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (cache optimization)
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project
COPY . .

# Streamlit default port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
