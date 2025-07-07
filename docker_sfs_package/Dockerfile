FROM python:3.11-slim

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 300 --index-url https://pypi.org/simple/ -r requirements.txt

COPY sfs.py .
COPY *.IMG ./
COPY *.json ./

RUN mkdir -p /output

ENTRYPOINT ["python", "sfs.py"]
CMD ["--help"]
