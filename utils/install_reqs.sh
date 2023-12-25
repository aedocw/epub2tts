#NOTE, this should probably be done in a virtual environment
#python -m venv .venv; source .venv/bin/activate
rm -rf TTS/ # delete repo to be able to reinstall if needed
git clone --branch xtts_demo https://github.com/coqui-ai/TTS.git
pip install --use-deprecated=legacy-resolver -e TTS
pip install --use-deprecated=legacy-resolver -r TTS/TTS/demos/xtts_ft_demo/requirements.txt
pip install typing_extensions==4.8 numpy==1.26.2
