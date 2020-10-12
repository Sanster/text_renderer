FROM python:3.6

COPY . /the/workdir/text_render
WORKDIR /the/workdir/text_render

RUN apt-get update && apt-get install libgl1-mesa-glx -y
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

CMD ["python3", "main.py"]