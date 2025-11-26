# Use an official slim Python image
FROM python:3.11-slim

# Prevent Python from writing .pyc and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install OS dependencies required by Playwright + common libs
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
       ca-certificates \
       curl \
       wget \
       gnupg \
       libnss3 \
       libatk1.0-0 \
       libatk-bridge2.0-0 \
       libx11-xcb1 \
       libcups2 \
       libxcomposite1 \
       libxrandr2 \
       libasound2 \
       libgbm1 \
       libpangocairo-1.0-0 \
       libxss1 \
       fonts-liberation \
  && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers
RUN python -m playwright install --with-deps

# Copy app code
COPY . .

# Expose port 8000 (Render provides PORT env var)
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]


