import argparse
import os
import re
import string
import subprocess
import sys
import time
import warnings
import wave

from bs4 import BeautifulSoup
import ebooklib
from ebooklib import epub
from fuzzywuzzy import fuzz
from newspaper import Article
import noisereduce
from openai import OpenAI
from pedalboard import Pedalboard, Compressor, Gain, NoiseGate, LowShelfFilter
from pedalboard.io import AudioFile
from pydub import AudioSegment
from pydub.silence import split_on_silence
from nltk.tokenize import sent_tokenize
import requests
import torch, gc
import torchaudio
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.generic_utils import get_user_data_dir
from tqdm import tqdm
import whisper


class EpubToAudiobook:
    def __init__(self, source, start, end, skiplinks, engine, minratio, model_name, debug):
        self.source = source
        self.bookname = os.path.splitext(os.path.basename(source))[0]
        self.start = start - 1
        self.end = end
        self.skiplinks = skiplinks
        self.engine = engine
        self.minratio = minratio
        self.debug = debug
        self.output_filename = self.bookname + ".m4b"
        self.chapters = []
        self.chapters_to_read = []
        if source.endswith('.epub'):
            self.book = epub.read_epub(source)
            self.sourcetype = 'epub'
        elif source.endswith('.txt'):
            self.sourcetype = 'txt'
        else:
            print("Can only handle epub or txt as source.")
            sys.exit()
        self.tts_dir = str(get_user_data_dir("tts"))
        if model_name == 'tts_models/en/vctk/vits':
            self.xtts_model = self.tts_dir + "/tts_models--multilingual--multi-dataset--xtts_v2"
        else:
            self.xtts_model = self.tts_dir + "/" + model_name
        self.whispermodel = whisper.load_model("tiny")
        self.ffmetadatafile = "FFMETADATAFILE"
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

    def generate_metadata(self, files, title, author):
        chap = 1
        start_time = 0
        with open(self.ffmetadatafile, "w") as file:
            file.write(";FFMETADATA1\n")
            file.write("ARTIST=" + str(author) + "\n")
            file.write("ALBUM=" + str(title) + "\n")
            for file_name in files:
                duration = self.get_wav_duration(file_name)
                file.write("[CHAPTER]\n")
                file.write("TIMEBASE=1/1000\n")
                file.write("START=" + str(start_time) + "\n")
                file.write("END=" + str(start_time + duration) + "\n")
                file.write("title=Part " + str(chap) + "\n")
                chap += 1
                start_time += duration

    def get_wav_duration(self, file_path):
        with wave.open(file_path, 'rb') as wav_file:
            num_frames = wav_file.getnframes()
            frame_rate = wav_file.getframerate()
            duration = num_frames / frame_rate
            duration_milliseconds = duration * 1000
            return int(duration_milliseconds)

    def get_length(self, start, end, chapters_to_read):
        total_chars = 0
        for i in range(start, end):
            total_chars += len(chapters_to_read[i])
        return (total_chars)

    def chap2text(self, chap):
        blacklist = ['[document]', 'noscript', 'header', 'html', 'meta', 'head', 'input', 'script']
        output = ''
        soup = BeautifulSoup(chap, 'html.parser')
        if self.skiplinks:
            # Remove everything that is an href
            for a in soup.findAll('a', href=True):
                a.extract()
        # Always skip reading links that are just a number (footnotes)
        for a in soup.findAll('a', href=True):
            if a.text.isdigit():
                a.extract()
        text = soup.find_all(string=True)
        for t in text:
            if t.parent.name not in blacklist:
                output += '{} '.format(t)
        return output

    def get_chapters_epub(self):
        for item in self.book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                self.chapters.append(item.get_content())

        for i in range(len(self.chapters)):
            #strip some characters that might have caused TTS to choke
            text = self.chap2text(self.chapters[i])
            text = text.replace("—", ", ").replace("--", ", ").replace(";", ", ").replace(":", ", ").replace("''", ", ")
            allowed_chars = string.ascii_letters + string.digits + "-,.!?' "
            text = ''.join(c for c in text if c in allowed_chars)
            if len(text) < 150:
                #too short to bother with
                continue
            print("Length: " + str(len(text)))
            print("Part: " + str(len(self.chapters_to_read) + 1))
            print(text[:256])
            self.chapters_to_read.append(text)  # append the last piece of text (shorter than max_len)
        print("Number of chapters to read: " + str(len(self.chapters_to_read)))
        if self.end == 999:
            self.end = len(self.chapters_to_read)

    def get_chapters_text(self):
        with open(self.source, 'r') as file:
            text = file.read()
        print(text[:256])
        self.chapters_to_read.append(text)
        self.end = len(self.chapters_to_read)

    def read_chunk_xtts(self, sentences, wav_file_path):
        #takes list of sentences to read, reads through them and saves to wave file
        t0 = time.time()
        wav_chunks = []
        sentence_list = sent_tokenize(sentences)
        for i, sentence in enumerate(sentence_list):
            # Run TTS for each sentence
            print(sentence) if self.debug else None
            chunks = self.model.inference_stream(
                sentence,
                "en",
                self.gpt_cond_latent,
                self.speaker_embedding,
                stream_chunk_size=60,
                temperature=0.60,
                repetition_penalty=10.0,
                enable_text_splitting=True
            )
            for j, chunk in enumerate(chunks):
                if i == 0:
                    print(f"Time to first chunck: {time.time() - t0}") if self.debug else None
                print(f"Received chunk {i} of audio length {chunk.shape[-1]}") if self.debug else None
                wav_chunks.append(chunk.to(device=self.device))  # Move chunk to available device
            # Add a short pause between sentences (e.g., X.XX seconds of silence)
            if i < len(sentence_list) - 1:
                silence_duration = int(24000 * 1.0)
                silence = torch.zeros((silence_duration,), dtype=torch.float32,
                                      device=self.device)  # Move silence tensor to available device
                wav_chunks.append(silence)
        wav = torch.cat(wav_chunks, dim=0)
        torchaudio.save(wav_file_path, wav.squeeze().unsqueeze(0).cpu(), 24000)
        with AudioFile(wav_file_path).resampled_to(24000) as f:
            audio = f.read(f.frames)
        reduced_noise = noisereduce.reduce_noise(y=audio, sr=24000, stationary=True, prop_decrease=0.75)
        board = Pedalboard([
            NoiseGate(threshold_db=-30, ratio=1.5, release_ms=250),
            Compressor(threshold_db=12, ratio=2.5),
            LowShelfFilter(cutoff_frequency_hz=400, gain_db=5, q=1),
            Gain(gain_db=0)
        ])
        result = board(reduced_noise, 24000)
        with AudioFile(wav_file_path, 'w', 24000, result.shape[0]) as f:
            f.write(result)

    def compare(self, text, wavfile):
        result = self.whispermodel.transcribe(wavfile)
        text = re.sub(' +', ' ', text).lower().strip()
        ratio = fuzz.ratio(text, result["text"].lower())
        print("Transcript: " + result["text"].lower()) if self.debug else None
        print("Text to transcript comparison ratio: " + str(ratio)) if self.debug else None
        return (ratio)

    def combine_sentences(self, sentences, length=1000):
        combined = ""
        for sentence in sentences:
            if len(combined) + len(sentence) <= length:
                combined += sentence + " "
            else:
                yield combined
                combined = sentence
        yield combined

    def read_book(self, voice_samples, engine, openai, model_name, speaker, bitrate):
        self.model_name = model_name
        self.openai = openai
        if engine == 'xtts':
            self.voice_samples = []
            for f in voice_samples.split(","):
                self.voice_samples.append(os.path.abspath(f))
            voice_name = "-" + re.split('-|\d+|\.', os.path.basename(self.voice_samples[0]))[0]
        elif engine == 'openai':
            if speaker == 'p335':
                speaker = 'onyx'
            voice_name = "-" + speaker
        else:
            voice_name = "-" + speaker
        self.output_filename = re.sub('.m4b', voice_name + ".m4b", self.output_filename)
        print("Saving to " + self.output_filename)
        total_chars = self.get_length(self.start, self.end, self.chapters_to_read)
        print("Total characters: " + str(total_chars))
        if engine == "xtts":
            print("Loading model: " + self.xtts_model)
            #This will trigger model load even though we won't use tts object later
            tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
            tts = ''
            #Don't think the next two lines are needed, but couldn't hurt just in case
            gc.collect()
            torch.cuda.empty_cache()
            config = XttsConfig()
            model_json = self.xtts_model + "/config.json"
            config.load_json(model_json)
            self.model = Xtts.init_from_config(config)
            self.model.load_checkpoint(config,
                                checkpoint_dir=self.xtts_model,
                                use_deepspeed=False)
            self.model.cuda()
            print("Computing speaker latents...")
            self.gpt_cond_latent, self.speaker_embedding = self.model.get_conditioning_latents(
                audio_path=self.voice_samples)
        elif engine == "openai":
            while True:
                openai_sdcost = (total_chars/1000) * 0.015
                print("OpenAI TTS SD Cost: $" + str(openai_sdcost))
                user_input = input("This will not be free, continue? (y/n): ")
                if user_input.lower() not in ['y', 'n']:
                    print("Invalid input. Please enter y for yes or n for no.")
                elif user_input.lower() == 'n':
                    sys.exit()
                else:
                    print("Continuing...")
                    break
            client = OpenAI(api_key=self.openai)
        else:
            print("Engine is TTS, model is " + model_name)
            self.tts = TTS(model_name).to(self.device)

        files = []
        position = 0
        start_time = time.time()
        print("Reading from " + str(self.start + 1) + " to " + str(self.end))
        for i in range(self.start, self.end):
            outputwav = self.bookname + "-" + str(i+1) + ".wav"
            if os.path.isfile(outputwav):
                print(outputwav + " exists, skipping to next chapter")
            else:
                #print("Debug is " + str(self.debug))
                tempfiles = []
                #segmenter = pysbd.Segmenter(language="en", clean=True)
                #sentences = segmenter.segment(self.chapters_to_read[i])
                sentences = sent_tokenize(self.chapters_to_read[i])
                sentence_groups = list(self.combine_sentences(sentences))
                for x in tqdm(range(len(sentence_groups))):
                    retries = 1
                    tempwav = "temp" + str(x) + ".wav"
                    if os.path.isfile(tempwav):
                        print(tempwav + " exists, skipping to next chunk")
                    else:
                        while retries > 0:
                            try:
                                if engine == "xtts":
                                    self.read_chunk_xtts(sentence_groups[x], tempwav)
                                elif engine == "openai":
                                    response = client.audio.speech.create( model="tts-1", voice=speaker, input=sentence_groups[x])
                                    response.stream_to_file(tempwav)
                                elif engine == "tts":
                                    if model_name == 'tts_models/en/vctk/vits':
                                        #assume we're using a multi-speaker model
                                        print(sentence_groups[x]) if self.debug else None
                                        self.tts.tts_to_file(text = sentence_groups[x], speaker = speaker, file_path = tempwav)
                                    else:
                                        print(sentence_groups[x]) if self.debug else None
                                        self.tts.tts_to_file(text = sentence_groups[x], file_path = tempwav)
                                ratio = self.compare(sentence_groups[x], tempwav)
                                if ratio < self.minratio:
                                    raise Exception("Spoken text did not sound right - " +str(ratio))
                                break
                            except Exception as e:
                                retries -= 1
                                print(f"Error: {str(e)} ... Retrying ({retries} retries left)")
                        if retries == 0:
                            print("Something is wrong with the audio (" + str(ratio) + "): " + tempwav)
                            #sys.exit()
                    tempfiles.append(tempwav)
                tempwavfiles = [AudioSegment.from_mp3(f"{f}") for f in tempfiles]
                concatenated = sum(tempwavfiles)
                concatenated.export(outputwav, format="wav")
                for f in tempfiles:
                    os.remove(f)
            files.append(outputwav)
            position += len(self.chapters_to_read[i])
            percentage = (position / total_chars) * 100
            print(f"{percentage:.2f}% spoken so far.")
            elapsed_time = time.time() - start_time
            chars_remaining = total_chars - position
            estimated_total_time = elapsed_time / position * total_chars
            estimated_time_remaining = estimated_total_time - elapsed_time
            print(f"Elapsed: {int(elapsed_time / 60)} minutes, ETA: {int((estimated_time_remaining) / 60)} minutes")
            gc.collect()
            torch.cuda.empty_cache()
        # Load all WAV files and concatenate into one object
        wav_files = [AudioSegment.from_wav(f"{f}") for f in files]
        one_sec_silence = AudioSegment.silent(duration=1000)
        concatenated = AudioSegment.empty()
        print("Replacing silences longer than one second with one second of silence (" + str(len(wav_files)) + " files)")
        for audio in wav_files:
            # Split audio into chunks where detected silence is longer than one second
            chunks = split_on_silence(audio, min_silence_len=1000, silence_thresh=-50)
            # Iterate through each chunk
            for i, chunk in enumerate(tqdm(chunks)):
                concatenated += chunk
                concatenated += one_sec_silence
        outputm4a = self.output_filename.replace("m4b", "m4a")
        concatenated.export(outputm4a, format="ipod", bitrate=bitrate)
        if self.sourcetype == 'epub':
            author = self.book.get_metadata('DC', 'creator')[0][0]
            title = self.book.get_metadata('DC', 'title')[0][0]
        else:
            author = "Unknown"
            title = self.bookname
        self.generate_metadata(files, title, author)
        ffmpeg_command = ["ffmpeg","-i",outputm4a,"-i",self.ffmetadatafile,"-map_metadata","1","-codec","copy",self.output_filename]
        subprocess.run(ffmpeg_command)
        os.remove(self.ffmetadatafile)
        os.remove(outputm4a)
        for f in files:
            os.remove(f)
        print(self.output_filename + " complete")

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    parser = argparse.ArgumentParser(
                        prog='EpubToAudiobook',
                        description='Read an epub (or other source) to audiobook format')
    parser.add_argument('sourcefile', type=str, help='The epub or text file to process')
    parser.add_argument('--engine', type=str, default='tts', nargs='?', const='tts', help='Which TTS to use [tts|xtts|openai]')
    parser.add_argument('--xtts', type=str, nargs='?', const="zzz", default="zzz", help='Sample wave file(s) for XTTS v2 training separated by commas')
    parser.add_argument('--openai', type=str, nargs='?', const="zzz", default="zzz", help='OpenAI API key if engine is OpenAI')
    parser.add_argument('--model', type=str, nargs='?', const='tts_models/en/vctk/vits', default='tts_models/en/vctk/vits', help='TTS model to use, default: tts_models/en/vctk/vits')
    parser.add_argument('--speaker', type=str, default='p335', nargs='?', const='p335', help='Speaker to use (ex p335 for VITS, or onyx for OpenAI)')
    parser.add_argument("--scan", action='store_true', help='Scan the epub to show beginning of chapters, then exit')
    parser.add_argument('--start', type=int, nargs='?', const=1, default=1, help='Chapter/part to start from')
    parser.add_argument('--end', type=int, nargs='?', const=999, default=999, help='Chapter/part to end with')
    parser.add_argument('--minratio', type=int, nargs='?', const=88, default=88, help='Minimum match ratio between text and transcript')
    parser.add_argument('--skiplinks', action='store_true', help='Skip reading any HTML links')
    parser.add_argument('--bitrate', type=str, nargs='?', const="69k", default="69k", help="Specify bitrate for output file")
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()
    print(args)

    if args.openai != "zzz":
        args.engine = "openai"
    if args.xtts != "zzz":
        args.engine = "xtts"
    mybook = EpubToAudiobook(source=args.sourcefile, start=args.start, end=args.end, skiplinks=args.skiplinks, engine=args.engine, minratio=args.minratio, model_name=args.model, debug=args.debug)
    if mybook.sourcetype == 'epub':
        mybook.get_chapters_epub()
    else:
        mybook.get_chapters_text()
    if args.scan:
        sys.exit()
    mybook.read_book(voice_samples=args.xtts, engine=args.engine, openai=args.openai, model_name=args.model, speaker=args.speaker, bitrate=args.bitrate)


if __name__ == '__main__':
    main()
