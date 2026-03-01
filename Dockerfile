FROM python:3.11-slim

# Working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create docs directory if it doesn't exist
RUN mkdir -p docs

# Build local index from docs/
RUN python indexer.py || true

# Expose port
EXPOSE 8000

# Run the backend server
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]
