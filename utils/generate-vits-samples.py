import torch
from TTS.api import TTS

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

tts = TTS("tts_models/en/vctk/vits").to(device)
speakers = []
for x in tts.speakers:
    if x.startswith('p'):
        speakers.append(x)

for speaker in speakers:
    text = "My name is " + speaker + ". It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent."
    output = f"speaker-{speaker}.wav"
    print("Generating " + output)

    tts.tts_to_file(text=text, speaker=speaker, file_path=output)

