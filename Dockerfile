# ---------------- Base Image ----------------
ARG BASE=python:3.11-slim
FROM ${BASE}

# ---------------- System Dependencies (Tesseract + Bengali/English) ----------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-ben \
    tesseract-ocr-eng \
    libtesseract-dev \
    libgl1 \
    libglib2.0-0 \
    locales \
 && rm -rf /var/lib/apt/lists/*

# ---------------- Locale & OCR Environment ----------------
RUN locale-gen bn_BD.UTF-8
ENV LANG=bn_BD.UTF-8
ENV LC_ALL=bn_BD.UTF-8
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata
ENV OMP_THREAD_LIMIT=2

# ---------------- App Setup ----------------
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy your source AFTER installing deps (better layer caching)
COPY . /app

# ---------------- Runtime ----------------
EXPOSE 8080
ENV GRADIO_SERVER_PORT=8080
# Override distro-provided models with your exact local ones
RUN mkdir -p /usr/share/tesseract-ocr/4.00/tessdata
COPY tessdata/*.traineddata /usr/share/tesseract-ocr/4.00/tessdata/
CMD ["python", "-u", "app.py"]
