> epub2tts is a free and open source python app to easily create a full-featured audiobook from an epub or text file using realistic text-to-speech from [Coqui AI TTS](https://github.com/coqui-ai/TTS), OpenAI or MS Edge.

## 🚀 Features

- [x] Creates standard format M4B audiobook file
- [x] Automatic chapter break detection
- [x] Embeds cover art if specified
- [x] Can use MS Edge for free cloud-based TTS
- [x] Easy voice cloning with Coqui XTTS model
- [x] 58 studio quality voices from Coqui AI
- [x] Uses deepspeed if available for faster processing
- [x] Resumes where it left off if interrupted
- [x] NOTE: epub file must be DRM-free


## 📖 Usage
<details>
<summary> Usage instructions</summary>

## Extract epub contents to text:
1. `epub2tts mybook.epub --export txt`
2. **edit mybook.txt**, replacing `# Part 1` etc with desired chapter names, and removing front matter like table of contents and anything else you do not want read. **Note:** First two lines can be Title: and Author: to use that in audiobook metadata.
3. The speaker can be set to change per chapter by appending `% <speaker>` after the chapter name, for instance `# Chapter One % en-US-AvaMultilingualNeural`. See the file `multi-speaker-sample-edge.txt` for an example. **Note:** Only works with Coqui TTS multi-speaker engine (default) or `--engine edge`.

## Default audiobook, fairly quick:
Using VITS model, all defaults, no GPU required:

* `epub2tts mybook.epub` (To change speaker (ex p307 for a good male voice w/Coqui TTS), add: `--speaker p307`)

## MS Edge Cloud TTS:
Uses [Microsoft Edge TTS](https://github.com/rany2/edge-tts/) in the cloud, FREE, only minimal CPU required, and it's pretty fast (100 minutes for 7hr book for instance). Many voices and languages to choose from, and the quality is really good (listen to `sample-en-US-AvaNeural-edge.m4b` for an example).

* List available voices with `edge-tts --list-voices`, default speaker is `en-US-AndrewNeural` if `--speaker` is not specified.
* `epub2tts mybook.txt --engine edge --speaker en-US-AvaNeural --cover cover-image.jpg --sayparts`

## XTTS with Coqui Studio voice:
1. Choose a studio voice, [samples here](https://github.com/rejuce/CoquiTTS_XTTS_Examples)
2. `epub2tts mybook.txt --engine xtts --speaker "Damien Black" --cover cover-image.jpg --sayparts`

## XTTS using your own voice clone:
1. `epub2tts mybook.epub --scan`, determine which part to start and end on so you can skip TOC, etc.
2. Secure 1-3 30 second clips of a speaker you really like (`voice-1.wav``, etc)
3. `epub2tts my-book.epub --start 4 --end 20 --xtts voice-1.wav,voice-2.wav,voice-3.wav --cover cover-image.jpg`

## All options
* -h, --help - show this help message and exit
* --engine [ENGINE] - Which TTS engine to use [tts|xtts|openai|edge]
* --xtts [sample-1.wav,sample-2.wav] - Sample wave/mp3 file(s) for XTTS v2 training separated by commas
* --openai OPENAI_API_KEY - OpenAI API key if engine is OpenAI
* --model [MODEL] - TTS model to use, default: tts_models/en/vctk/vits
* --speaker SPEAKER - Speaker to use (examples: p335 for VITS, onyx for OpenAI, "Damien Black" for XTTS v2, en-US-EricNeural for edge)
* --scan - Scan the epub to show beginning of chapters, then exit
* --start [START] - Chapter/part to start from
* --end [END] - Chapter/part to end with
* --language [LANGUAGE] - Language of the epub, default: en
* --minratio [MINRATIO] - Minimum match ratio between text and transcript, 0 to disable whisper
* --skiplinks - Skip reading any HTML links
* --skipfootnotes - Try to skip reading footnotes
* --skip-cleanup - Do not replace special characters with ","
* --sayparts - Say each part number at start of section
* --bitrate [BITRATE] - Specify bitrate for output file
* --debug  - Enable debug output
* --export txt - Export epub contents to file (txt, md coming soon)
* --no-deepspeed - Disable deepspeed
* --cover image.jpg - jpg image to use for cover

</details>

## 🐞 Reporting bugs
<details>
<summary>How to report bugs/issues</summary>

Thank you in advance for reporting any bugs/issues you encounter! If you are having issues, first please [search existing issues](https://github.com/aedocw/epub2tts/issues) to see if anyone else has run into something similar previously.

If you've found something new, please open an issue and be sure to include:
1. The full command you executed
2. The platform (Linux, Windows, OSX, Docker)
3. Your Python version if not using Docker
4. Try running the command again with `--debug --minratio 0` added on, to get more information
5. Relevant output around the crash, including the sentence (should be in debug output) if it crashed during a TTS step

</details>

## 🗒️ Release notes
<details>
<summary>Release notes </summary>

* 20240403: Added support for specifying speaker per chapter, https://github.com/aedocw/epub2tts/issues/229
* 20240320: Added MS Edge cloud TTS support
* 20240301: Added `--skip-cleanup` option to skip replacement of special characters with ","
* 20240222: Implemented pause between sentences, https://github.com/aedocw/epub2tts/issues/208 and https://github.com/aedocw/epub2tts/issues/153
* 20240131: [Repaired missing pause between chapters](https://github.com/aedocw/epub2tts/issues/204)
* 20240114: Updated README
* 20240111: Added support for Title & Author in text files
* 20240110: Added support for "--cover image.jpg"

</details>

## Performance
<details>
<summary>Some benchmarks</summary>
VITS model is the fastest, does not require GPU, but does not sound as good as using XTTS. We have not done any comparative benchmarks with that model.

Typical inference times for xtts_v2 averaged over 4 processing chunks (about 4 sentences each) that can be expected:

```
| Hardware                            | Inference Time |
|-------------------------------------|----------------|
| 20x CPU Xeon E5-2630 (without AVX)  | 3.7x realtime  |
| 20x CPU Xeon Silver 4214 (with AVX) | 1.7x realtime  |
| 8x CPU Xeon Silver 4214 (with AVX)  | 2.0x realtime  |
| 2x CPU Xeon Silver 4214 (with AVX)  | 2.9x realtime  |
| Intel N4100 Atom (NAS)              | 4.7x realtime  |
| GPU RTX A2000 4GB (w/o deepspeed)   | 0.4x realtime  |
| GPU RTX A2000 4GB (w deepspeed)     | 0.15x realtime |
```
</details>

## 📦 Install

Required Python version is 3.11.

<details>
<summary>MAC INSTALLATION</summary>

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
</details>

<details>
<summary>LINUX INSTALLATION</summary>

These instructions are for Ubuntu 22.04 (20.04 showed some depedency issues), but should work (with appropriate package installer mods) for just about any repo. Ensure you have `ffmpeg` installed before use. If you have an NVIDIA GPU you should also [install CUDA toolkit](https://developer.nvidia.com/cuda-downloads) to make use of deepspeed.

Voice models will be saved locally in `~/.local/share/tts`

```
#install dependencies
sudo apt install espeak-ng ffmpeg
#If you have a CUDA-compatible GPU, run:
sudo apt install nvidia-cuda-toolkit
#clone the repo
git clone https://github.com/aedocw/epub2tts
cd epub2tts
pip install .
```

**NOTE:** If you have deepspeed installed, it may be detected but not work properly, causing errors. Try [installing CUDA toolkit](https://developer.nvidia.com/cuda-downloads) to see if that resolves the issue. If that does not fix it, add `--no-deepspeed` and it will not be used. Also in that case, open an issue with your details and we will look into it.

</details>

<details>
<summary>WINDOWS INSTALLATION</summary>

Runnig epub2tts in WSL2 with Ubuntu 22 is the easiest approach, but these steps should work for running directly in windows.

1. Install Microsoft C++ Build Tools. Download the installer from https://visualstudio.microsoft.com/visual-cpp-build-tools/ then run the downloaded file `vs_BuildTools.exe` and select the "C++ Buld tools" checkbox leaving all options at their default value. **Note:** This will require about 7 GB of space on C drive.
2. Install espeak-ng from https://github.com/espeak-ng/espeak-ng/releases/latest
3. [Install chocolaty](https://chocolatey.org/install)
4. Install ffmpeg with the command `choco install ffmpeg`, make sure you are in an elevated powershell session.
5. Install python 3.11 with the command `choco install python311`
6. Install git with the command `choco install git`.
7. Decide where you want your epub2tts project to live, documents is a common place. Once you've found a directory you're happy with, clone the project with `git clone https://github.com/aedocw/epub2tts` and cd epub2tts so you're now in your working directory.
8. There are probably a few different ways you can go here, I personally opted for a venv to keep everything organized. Create a venv with the command `python -m venv .venv`
9. Activate the venv, on windows the command is slightly different as you issue `.venv\scripts\activate`
10. Install epub2tts along with the requirements with the command `pip install .`

11. If all goes well, you should be able to call epub2tts from within your venv and update it from this directory going forward. To update, use `git pull` and then `pip install . --upgrade`

**Some errors you may encounter**
* Encountered error while trying to install package lxml
  * Run `pip install lxml` to install the latest version manually then re-run `pip install .`
* ffmpeg not found
  * Rerun the command `choco install ffmpeg``, making sure you are in an elevated powershell session, outside of the virtual environment
* NLTK: punkt not found
  * Run the following to install it: `python -c "import nltk"` then `python -m nltk.downloader punkt`
* Torch not compiled with CUDA enabled
  * `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
* If you have deepspeed installed, it may be detected but not work properly, causing errors. If that is the case, add `--no-deepspeed` and it will not be used.

</details>

<details>
<summary>DOCKER</summary>

Voice models will be saved locally in `~/.local/share/tts`

Docker usage does not reliably utilize GPU, if someone wants to work on improving this your PR will be very welcome!

For *Linux and MacOS*:
```
alias epub2tts='docker run -e COQUI_TOS_AGREED=1 -v "$PWD:$PWD" -v ~/.local/share/tts:/root/.local/share/tts -w "$PWD" ghcr.io/aedocw/epub2tts:release'
```

For *Windows*:
Pre-requisites:
* Install Docker Desktop
* From PowerShell run "mkdir ~/.local/share/tts"

```
#Example for running scan of "mybook.epub"
docker run -e COQUI_TOS_AGREED=1 -v ${PWD}/.local/share/tts:/root/.local/share/tts -v ${PWD}:/root -w /root ghcr.io/aedocw/epub2tts:release mybook.epub --scan

#Example for reading parts 3 through 15 of "mybook.epub"
docker run -e COQUI_TOS_AGREED=1 -v ${PWD}/.local/share/tts:/root/.local/share/tts -v ${PWD}:/root -w /root ghcr.io/aedocw/epub2tts:release mybook.epub --start 3 --end 15
```
</details>

<details>
<summary>DEVELOPMENT INSTALL</summary>

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
</details>


## Updating

<details>
<summary>UPDATING YOUR INSTALLATION</summary>

1. cd to repo directory
2. `git pull`
3. Activate virtual environment you installed epub2tts in if you installed in a virtual environment
4. `pip install . --upgrade`
</details>


## Author

👤 **Christopher Aedo**

- Website: [aedo.dev](https://aedo.dev)
- GitHub: [@aedocw](https://github.com/aedocw)
- LinkedIn: [@aedo](https://linkedin.com/in/aedo)

👥 **Contributors**

[![Contributors](https://contrib.rocks/image?repo=aedocw/epub2tts)](https://github.com/aedocw/epub2tts/graphs/contributors)

## 🤝 Contributing

Contributions, issues and feature requests are welcome!\
Feel free to check the [issues page](https://github.com/aedocw/epub2tts/issues) or [discussions page](https://github.com/aedocw/epub2tts/discussions).

## Show your support

Give a ⭐️ if this project helped you!
