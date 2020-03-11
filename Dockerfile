FROM python:3.6

COPY . /the/workdir/text_render
WORKDIR /the/workdir/text_render

RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

CMD ["python3", "main.py"]