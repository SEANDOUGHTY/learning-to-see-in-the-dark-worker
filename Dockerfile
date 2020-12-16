FROM python:3.8-alpine

RUN apt-get update

WORKDIR /app
COPY requirements.txt /app

RUN pip install -r requirements.txt
RUN pip install torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN apt-get install -y vim libgl1-mesa-glx git
COPY . /app

CMD ["python", "main.py"]
