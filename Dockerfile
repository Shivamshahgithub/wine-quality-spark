FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install Java (required by PySpark)
RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-21-jre-headless procps && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-arm64
ENV PATH="$JAVA_HOME/bin:${PATH}"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code, model, and test data into the image
COPY predict.py .
COPY ValidationDataset.csv .
COPY model_lr ./model_lr

# When the container runs, execute predict.py on the local model & CSV
ENTRYPOINT ["python", "predict.py", "model_lr", "ValidationDataset.csv"]
