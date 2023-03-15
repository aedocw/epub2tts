This script takes an epub and creates mp3's of the chapters, read by https://github.com/coqui-ai/TTS

I have not had good luck getting Coqui-AI to run on an M1 mac but it runs great on Linux. You CAN however run this on a mac with docker. Works great with [LIMA](https://github.com/lima-vm/lima), just remember to [make your home directory writable](https://github.com/lima-vm/lima#filesystem-is-not-writable) for the docker template, otherwise saving the voice model will fail.

I recognize this is not very user friendly, but I wanted to share in case folks thought it was useful. If there are a few more people than myself that find this is useful I will keep working on turning it into something that could be used by someone without dev experience.

## Docker usage:
This image is HUGE, I think mainly from all the stuff TTS pulls in. When I build locally it's 3gb!

From one directory below epub2tts repo, build the docker image with `docker build -t epub2tts:latest epub2tts`

Voice models will be saved locally in `~/.local/share/tts`

Change to directory containning your epub and run with:
```
docker run -v "$PWD:$PWD" -v ~/.local/share/tts:/root/.local/share/tts -w "$PWD" -e BOOK=your-book.epub epub2tts
```

## INSTALLATION:

For  now I've only tested this on a linux machine (Ubuntu 22 in my case)

```
#clone the repo
git clone https://github.com/aedocw/epub2tts
cd epub2tts
#create a virtual environment
python3 -m venv .venv
#activate the virtual environment
source .venv/bin/activate
#install dependencies
sudo apt install espeak-ng ffmpeg
pip3 install -r requirements.txt
```

Usage: `python3 epub2tts.py my-book.epub`
