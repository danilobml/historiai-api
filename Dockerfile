# Use Python 3.10 as the base image
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y tesseract-ocr

RUN apt-get install -y python3-opencv

COPY . /app/

EXPOSE 5000

ENTRYPOINT FLASK_APP=/app/main.py flask run --host=0.0.0.0