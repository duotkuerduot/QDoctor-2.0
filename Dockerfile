# Use Python 3.10 slim for a smaller, faster image
FROM python:3.10-slim

# Set environment variables to prevent Python from writing .pyc and buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies for FAISS and PDF processing
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user (Required by Hugging Face)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy requirements and install
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the application
COPY --chown=user . .

# Ensure storage directory exists and has permissions
RUN mkdir -p $HOME/app/storage/qbrain_faiss_index && \
    chmod -R 777 $HOME/app/storage

# Expose the default Hugging Face port
EXPOSE 7860

# Start the FastAPI app using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]