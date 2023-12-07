This is for fine tuning an XTTS v2 model.

* Set up a virtual environment for fine tuning
* * `python -m venv .venv; source .venv/bin/activate`
* Clone the right branch and install requirements
* * `sh ./install_reqs.sh`
* Run gradio
* * `sh ./run_gradio.sh`

I've had the best results using about an 8 minute audio sample. When you have a model you are happy with, run `python copy_model.py` and it will be copied to `~/.local/share/tts/voice-TIMESTAMP`. I usually then rename that directory to something I can easily remember.

To use with epub2tts, just add `--model my-model` when creating an audiobook with XTTS v2.

Reference this YouTube video: https://www.youtube.com/watch?v=8tpDiiouGxc to see how to do this in google colab.