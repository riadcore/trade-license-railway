# ---------------- Base Image ----------------
FROM python:3.11-slim


# ---------------- System Packages ----------------
# - tesseract + ben/eng language packs
# - locales for Bangla UTF-8
# - image libs so Pillow renders like local (jpeg/png/tiff/openjp2)
# - curl/ca-certificates to fetch tessdata_best
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-ben \
    tesseract-ocr-eng \
    libtesseract-dev \
    libgl1 \
    libglib2.0-0 \
    locales \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libtiff-dev \
    libopenjp2-7-dev \
    curl \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# ---------------- Locale ----------------
RUN locale-gen bn_BD.UTF-8
ENV LANG=bn_BD.UTF-8
ENV LC_ALL=bn_BD.UTF-8

# ---------------- High-quality tessdata (ben + eng) ----------------
# Use the official tessdata_best models for higher accuracy
RUN mkdir -p /usr/share/tesseract-ocr/4.00/tessdata_best \
    && curl -L -o /usr/share/tesseract-ocr/4.00/tessdata_best/ben.traineddata \
         https://github.com/tesseract-ocr/tessdata_best/raw/main/ben.traineddata \
    && curl -L -o /usr/share/tesseract-ocr/4.00/tessdata_best/eng.traineddata \
         https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata

# Tell Tesseract to use tessdata_best by default
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata_best

# Deterministic performance on small vCPUs
ENV OMP_THREAD_LIMIT=1

# (Optional but helpful) unbuffered Python logs, no .pyc
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ---------------- App Setup ----------------
WORKDIR /app

# Install Python deps (layer-cached)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy source AFTER deps for faster rebuilds
COPY . /app

# ---------------- Runtime ----------------
EXPOSE 8080
ENV GRADIO_SERVER_PORT=8080
CMD ["python", "-u", "app.py"]
