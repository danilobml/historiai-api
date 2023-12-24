FROM python:3-slim-buster as python-base
 
WORKDIR /flask-app

RUN python3 -m venv venv
ENV VIRTUAL_ENV=/flask-app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt /flask-app/

RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y tesseract-ocr
RUN apt-get install -y python3-opencv

COPY . /flask-app/

ENV FLASK_APP=/flask-app/main.py

EXPOSE 5000

CMD ["python3", "-m" , "flask", "run", "--host=0.0.0.0"]