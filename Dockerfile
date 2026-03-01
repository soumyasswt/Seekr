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

# Expose port
EXPOSE 8000

# Run the backend server and test connectivity
CMD ["/bin/bash", "-c", "uvicorn backend:app --host 0.0.0.0 --port 8000 & sleep 10 && curl -v http://localhost:8000/wake"]
