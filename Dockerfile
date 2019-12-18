FROM python:3.6

COPY . /the/workdir/text_render
WORKDIR /the/workdir/text_render

RUN pip3 install -r requirements.txt

CMD ["python3", "main.py"]