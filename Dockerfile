ARG BASE=python:3.11-slim
FROM ${BASE}

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-ben \
    libgl1 libglib2.0-0 locales \
 && sed -i 's/# \(en_US\.UTF-8\)/\1/' /etc/locale.gen \
 && sed -i 's/# \(bn_BD\.UTF-8\)/\1/' /etc/locale.gen \
 && locale-gen \
 && rm -rf /var/lib/apt/lists/*

ENV LANG=bn_BD.UTF-8
ENV LC_ALL=bn_BD.UTF-8
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8080
ENV GRADIO_SERVER_PORT=8080

CMD ["python", "app.py"]
