head -n 10 README.md > /tmp/testing.txt
python epub2tts.py /tmp/testing.txt
python epub2tts.py /tmp/testing.txt --speaker p307
python epub2tts.py /tmp/testing.txt --xtts voices/adam-1.wav --model adamwhite
python epub2tts.py /tmp/testing.txt --xtts voices/adam-1.wav --model adamwhite --language it
python epub2tts.py /tmp/testing.txt --engine xtts --speaker "Damien Black"
python epub2tts.py /tmp/testing.txt --engine xtts --speaker "Damien Black" --language it
