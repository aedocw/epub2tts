This script takes an epub (or text file) and reads it to an m4b audiobook file, using TTS by https://github.com/coqui-ai/TTS or OpenAI. The audiofiles are created in discrete chunks then transcribed using whisper speech-to-text. The transcription is compared to the original text, and if they don't match well it tries again. Finally all silence longer than a second is removed from all audio segments, and the audio is cleaned up before being combined into an m4b audiobook file.

I recognize this is not very user friendly, but I wanted to share in case folks thought it was useful. If there are a few more people than myself that find this is useful I will keep working on turning it into something that could be used by someone without dev experience.

**NOTE: BIG UPDATE for XTTS!** The Coqui team released v2 of their XTTS model and the quality is amazing! This latest release includes significant refactoring, and uses streaming inference for XTTS. Suggested usage is to include up to three wav file speaker samples, up to 30 seconds each. Check out the XTTS sample to get an idea of the quality you can expect. Also take a look in the utils directory for notes on finetuning your model for exceptional results.

Example usage: `epub2tts my-book.epub --start 4 --end 20 --xtts shadow-1.wav,shadow-2.wav,shadow-3.wav`

**NOTE:** Now with [OpenAI TTS](https://platform.openai.com/docs/guides/text-to-speech) support! It's not free, but the average cost for a few books I tested was around $7. If you use `--openai <API key>` flag epub2tts will provide a cost estimate and prompt you to approve before continuing.

**NOTE:** HUGE thanks to a recent PR from [wonka929](https://github.com/wonka929), epub2tts now recognizes when a CUDA GPU is available and will use it automatically. In a brief test I did, the speedup was incredible!

## USAGE:
Usage: 

  EPUB: `epub2tts my-book.epub`

  TEXT: `epub2tts my-book.txt`

To use Coqui XTTS, add: `--xtts <sample-1.wav>,<sample-2.wav>,<sample-3.wav> --language 'en' book.epub` (slow but sounds amazing!)

To use OpenAI TTS, add: `--openai <your API key>` (Use speaker option to specify voice other than onyx: `--speaker shimmer`)

To change speaker (ex p307 for a good male voice w/Coqui TTS), add: `--speaker p307`

To skip reading any links, add: `--skiplinks`

Using `--scan` will list excerpts of each chapter, then exit. This is helpful for finding which chapter to start and end on if you want to skip bibliography, TOC, etc.

To specify which chapter to start on (ex 3): `--start 3`

To specify which chapter to end on (ex 20): `--end 20`

To specify bitrate (ex 30k): `--bitrate 30k`

If epub2tts is interrupted or crashes, you can run it again with the same parameters and it will pick up where it left off, assuming it made it far enough to save some WAV files. If you want to start fresh, be sure to delete any of the wav files (with the same name as the epub) in the working directory before running again.

## DOCKER INSTRUCTIONS:
Voice models will be saved locally in `~/.local/share/tts`

For *Linux and MacOS*:
```
alias epub2tts='docker run -v "$PWD:$PWD" -v ~/.local/share/tts:/root/.local/share/tts -w "$PWD" ghcr.io/aedocw/epub2tts:release'
```

For *Windows*:
Pre-requisites:
* Install Docker Desktop
* From PowerShell run "mkdir ~/.local/share/tts"

```
#Example for running scan of "mybook.epub"
docker run -v ${PWD}/.local/share/tts:/root/.local/share/tts -v ${PWD}:/root -w /root ghcr.io/aedocw/epub2tts:release mybook.epub --scan

#Example for reading parts 3 through 15 of "mybook.epub"
docker run -v ${PWD}/.local/share/tts:/root/.local/share/tts -v ${PWD}:/root -w /root ghcr.io/aedocw/epub2tts:release mybook.epub --start 3 --end 15
```

## MAC INSTALLATION:
This installation requires Python < 3.12 and [Homebrew](https://brew.sh/) (I use homebrew to install espeak, [pyenv](https://stackoverflow.com/questions/36968425/how-can-i-install-multiple-versions-of-python-on-latest-os-x-and-use-them-in-par) and ffmpeg). Per [this bug](https://github.com/coqui-ai/TTS/issues/2052), mecab should also be installed via homebrew.

Voice models will be saved locally in `~/.local/share/tts`
```
#install dependencies
brew install espeak pyenv ffmpeg mecab
#install epub2tts
git clone https://github.com/aedocw/epub2tts
cd epub2tts
pyenv install 3.11
pyenv local 3.11
#OPTIONAL - install this in a virtual environment
python -m venv .venv && source .venv/bin/activate
pip install .
```

## LINUX INSTALLATION:

These instructions are for Ubuntu, but should work (with appropriate package installer mods) for just about any repo. Ensure you have `ffmpeg` installed before use.

Voice models will be saved locally in `~/.local/share/tts`

```
#install dependencies
sudo apt install espeak-ng ffmpeg
#clone the repo
git clone https://github.com/aedocw/epub2tts
cd epub2tts
pip install .
```

## DEVELOPMENT INSTALL:

```
#clone the repo
git clone https://github.com/aedocw/epub2tts
cd epub2tts
#create a virtual environment
python -m venv .venv
#activate the virtual environment
source .venv/bin/activate
#install dependencies
sudo apt install espeak-ng ffmpeg
pip install -r requirements.txt
```

