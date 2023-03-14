FROM ubuntu:latest

ADD requirements.txt .
ADD epub2tts.py .

RUN apt-get update && sudo apt install espeak-ng ffmpeg
RUN pip3 install -r requirements.txt

CMD ["python", "./epub2tts.py", "$1"]