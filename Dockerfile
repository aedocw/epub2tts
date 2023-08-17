FROM ubuntu:latest

ENV BOOK=mybook.epub

RUN mkdir /opt/epub2tts && \
          apt-get update && \
          apt -y install espeak-ng ffmpeg python3 python3-pip
ADD requirements.txt /opt/epub2tts/.
ADD epub2tts.py /opt/epub2tts/.
RUN pip3 install -r /opt/epub2tts/requirements.txt

ENTRYPOINT ["python3", "/opt/epub2tts/epub2tts.py"]

CMD ["--help"]
