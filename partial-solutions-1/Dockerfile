FROM python:3.11

WORKDIR /app

ADD . /app

COPY . /app

RUN pip install pykwalify PyYAML
RUN pip install -U "ray[data,serve]"
RUN pip install -U pyspark
RUN pip install "pyspark[sql]"
RUN pip install "pyspark[pandas_on_spark]"
RUN pip install "pyspark[connect]"
RUN pip install -r requirements.txt
RUN pip install -e .
RUN pip install text_extractor
RUN pip install text_chunker
RUN pip install nltk
RUN pip install spacy
RUN pip install sentence-transformers
RUN pip install spacy-universal-sentence-encoder[multi]


#CMD ["python", "test.py"]
CMD ["python", "app.py"]
