This script takes an epub and reads it to an mp3, using TTS by https://github.com/coqui-ai/TTS

I recognize this is not very user friendly, but I wanted to share in case folks thought it was useful. If there are a few more people than myself that find this is useful I will keep working on turning it into something that could be used by someone without dev experience.

## MAC INSTALLATION:
This installation requires Python 3.10 and [Homebrew](https://brew.sh/) (I use homebrew to install espeak, pyenv and ffmpeg).

```
#install dependencies
brew install espeak pyenv ffmpeg
git clone https://github.com/aedocw/epub2tts
cd epub2tts
pyenv install 3.10.11
pyenv local 3.10.11
#OPTIONAL - install this in a virtual environment
python -m venv .venv && source .venv/bin/activate
pip install .
```
Usage: `epub2tts my-book.epub`

## LINUX INSTALLATION:

For  now I've only tested this on a linux machine (Ubuntu 22 in my case). Ensure you have `ffmpeg` installed before use.

```
#install dependencies
sudo apt install espeak-ng ffmpeg
#clone the repo
git clone https://github.com/aedocw/epub2tts
cd epub2tts
pip install .
```

Usage: `epub2tts my-book.epub`

## Docker usage:
Voice models will be saved locally in `~/.local/share/tts`

Change to directory containing your epub and run with:
```
docker run -v "$PWD:$PWD" -v ~/.local/share/tts:/root/.local/share/tts -w "$PWD" -e BOOK=your-book.epub ghcr.io/aedocw/epub2tts:release
```
## DEVELOPMENT INSTALL:

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
pip install -r requirements.txt
```

Usage: `python3 epub2tts.py my-book.epub`