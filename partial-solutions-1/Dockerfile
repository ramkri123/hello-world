FROM python:3.11

WORKDIR /app

ADD . /app

COPY . /app

RUN pip install -r requirements.txt

CMD ["python", "test.py"]
