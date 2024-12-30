FROM ubuntu:24.04

ENV BOOK=mybook.epub

SHELL ["/bin/bash", "-c"]

RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && \
    apt -y install espeak-ng ffmpeg python3 python3-pip python3-venv rustc-1.80

COPY requirements.txt epub2tts.py setup.py /app/

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m venv .venv && \
    source .venv/bin/activate && \
    pip install .
    
ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["python", "epub2tts.py"]

CMD ["--help"]
