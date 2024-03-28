import argparse
import asyncio
import os
import pkg_resources
import re
import subprocess
import sys
import time
import warnings

from bs4 import BeautifulSoup
import ebooklib
from ebooklib import epub
import edge_tts
from fuzzywuzzy import fuzz
from mutagen import mp4
import noisereduce
from openai import OpenAI
from pedalboard import Pedalboard, Compressor, Gain, NoiseGate, LowShelfFilter
from pedalboard.io import AudioFile
from pydub import AudioSegment
from pydub.silence import split_on_silence
import nltk
from nltk.tokenize import sent_tokenize
import torch, gc
import torchaudio
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.generic_utils import get_user_data_dir
from tqdm import tqdm
import whisper


class EpubToAudiobook:
    def __init__(
        self,
        source,
        start,
        end,
        skiplinks,
        engine,
        minratio,
        model_name,
        debug,
        language,
        skipfootnotes,
        sayparts,
        no_deepspeed,
        skip_cleanup,
        audioformat,
    ):
        self.source = source
        self.bookname = os.path.splitext(os.path.basename(source))[0]
        self.start = start - 1
        self.end = end
        self.language = language
        self.skiplinks = skiplinks
        self.skipfootnotes = skipfootnotes
        self.sayparts = sayparts
        self.engine = engine
        self.minratio = minratio
        self.debug = debug
        self.output_filename = self.bookname + ".m4b"
        self.chapters = []
        self.chapters_to_read = []
        self.section_names = []
        self.no_deepspeed = no_deepspeed
        self.skip_cleanup = skip_cleanup
        self.title = self.bookname
        self.author = "Unknown"
        self.audioformat = audioformat.lower()
        if source.endswith(".epub"):
            self.book = epub.read_epub(source)
            self.sourcetype = "epub"
        elif source.endswith(".txt"):
            self.sourcetype = "txt"
        else:
            print("Can only handle epub or txt as source.")
            sys.exit()
        self.tts_dir = str(get_user_data_dir("tts"))
        if model_name == "tts_models/en/vctk/vits":
            self.xtts_model = (
                self.tts_dir + "/tts_models--multilingual--multi-dataset--xtts_v2"
            )
        else:
            self.xtts_model = f"{self.tts_dir}/{model_name}"
        self.whispermodel = whisper.load_model("tiny")
        self.ffmetadatafile = "FFMETADATAFILE"
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        # Make sure we've got nltk punkt
        self.ensure_punkt()

    def ensure_punkt(self):
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")

    def is_installed(self, package_name):
        package_installed = False
        try:
            pkg_resources.get_distribution(package_name)
            package_installed = True
        except pkg_resources.DistributionNotFound:
            pass
        return package_installed

    def generate_metadata(self, files):
        chap = 1
        start_time = 0
        with open(self.ffmetadatafile, "w") as file:
            file.write(";FFMETADATA1\n")
            file.write(f"ARTIST={self.author}\n")
            file.write(f"ALBUM={self.title}\n")
            file.write("DESCRIPTION=Made with https://github.com/aedocw/epub2tts\n")
            for file_name in files:
                duration = self.get_duration(file_name)
                file.write("[CHAPTER]\n")
                file.write("TIMEBASE=1/1000\n")
                file.write(f"START={start_time}\n")
                file.write(f"END={start_time + duration}\n")
                if len(self.section_names) > 0:
                    file.write(f"title={self.section_names[chap-1]}\n")
                else:
                    file.write(f"title=Part {chap}\n")
                chap += 1
                start_time += duration

    def get_duration(self, file_path):
        audio = AudioSegment.from_file(file_path)
        duration_milliseconds = len(audio)
        return duration_milliseconds

    def get_length(self, start, end, chapters_to_read):
        total_chars = 0
        for i in range(start, end):
            total_chars += len(chapters_to_read[i])
        return total_chars

    def chap2text(self, chap):
        blacklist = [
            "[document]",
            "noscript",
            "header",
            "html",
            "meta",
            "head",
            "input",
            "script",
        ]
        output = ""
        soup = BeautifulSoup(chap, "html.parser")
        if self.skiplinks:
            # Remove everything that is an href
            for a in soup.findAll("a", href=True):
                a.extract()
        # Always skip reading links that are just a number (footnotes)
        for a in soup.findAll("a", href=True):
            if not any(char.isalpha() for char in a.text):
                a.extract()
        text = soup.find_all(string=True)
        for t in text:
            if t.parent.name not in blacklist:
                output += "{} ".format(t)
        return output

    def prep_text(self, text_in):
        # Replace some chars with comma to improve TTS by introducing a pause
        text = (
            text_in.replace("--", ", ")
            .replace("—", ", ")
            .replace(";", ", ")
            .replace(":", ", ")
            .replace("''", ", ")
            .replace("’", "'")
            .replace('“', '"')
            .replace('”', '"')
            .replace("◇", "")
            .replace(" . . . ", ", ")
            .replace("... ", ", ")
            .replace("«", " ")
            .replace("»", " ")
            .replace("[", "")
            .replace("]", "")
            .replace("&", " and ")
            .replace(" GNU ", " new ")
            .replace("\n", " \n")
            .replace("*", " ")
            .strip()
        )
        return text

    def exclude_footnotes(self, text):
        pattern = r'\s*\d+\.\s.*$'  # Matches lines starting with numbers followed by a dot and whitespace
        return re.sub(pattern, '', text, flags=re.MULTILINE)

    def get_chapters_epub(self):
        for item in self.book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                self.chapters.append(item.get_content())
        self.author = self.book.get_metadata("DC", "creator")[0][0]
        self.title = self.book.get_metadata("DC", "title")[0][0]

        for i in range(len(self.chapters)):
            if self.skip_cleanup:
                text = self.chap2text(self.chapters[i])
            else:
                text = self.prep_text(self.chap2text(self.chapters[i]))
            if len(text) < 150:
                # too short to bother with
                continue
            print(f"Length: {len(text)}")
            print(f"Part: {len(self.chapters_to_read) + 1}")
            if self.skipfootnotes:
                text = self.exclude_footnotes(text)
                #This drops everything after "Skip Notes" in a chapter
                text = text.split("Skip Notes")[0].strip()
            if self.skipfootnotes and text.startswith("Footnotes"):
                continue
            print(text[:256])
            self.chapters_to_read.append(text)
        print(f"Number of chapters to read: {len(self.chapters_to_read)}")
        if self.end == 999:
            self.end = len(self.chapters_to_read)

    def get_chapters_text(self):
        with open(self.source, "r") as file:
            text = file.read()
        metadata, text = self.extract_title_author(text)
        if metadata.get("Title") != None:
            self.title = metadata.get("Title")
        if metadata.get("Author") != None:
            self.author = metadata.get("Author")
        if self.skip_cleanup:
            pass
        else:
            text = self.prep_text(text)
        max_len = 50000
        lines_with_hashtag = [line for line in text.splitlines() if line.startswith("# ")]
        if lines_with_hashtag:
            for line in lines_with_hashtag:
                self.section_names.append(line.lstrip("# ").strip())
            sections = re.split(r"\n(?=#\s)", text)
            sections = [section.strip() for section in sections if section.strip()]
            for i, section in enumerate(sections):
                lines = section.splitlines()
                section = "\n".join(lines[1:])
                self.chapters_to_read.append(section.strip())
                print(f"Part: {len(self.chapters_to_read)}")
                print(f"{self.section_names[i]}")
                print(str(self.chapters_to_read[-1])[:256])
        else:
            while len(text) > max_len:
                pos = text.rfind(" ", 0, max_len)  # find the last space within the limit
                self.chapters_to_read.append(text[:pos])
                print(f"Part: {len(self.chapters_to_read)}")
                print(str(self.chapters_to_read[-1])[:256])
                text = text[pos + 1 :]  # +1 to avoid starting the next chapter with a space
            self.chapters_to_read.append(text)
        if self.end == 999:
            self.end = len(self.chapters_to_read)
        print(f"Section names: {self.section_names}") if self.debug else None

    def read_chunk_xtts(self, sentences, wav_file_path):
        # takes list of sentences to read, reads through them and saves to file
        t0 = time.time()
        wav_chunks = []
        sentence_list = sent_tokenize(sentences)
        for i, sentence in enumerate(sentence_list):
            # Run TTS for each sentence
            if self.debug:
                print(
                    sentence
                )
                with open("debugout.txt", "a") as file: file.write(f"{sentence}\n")
            chunks = self.model.inference_stream(
                sentence,
                self.language,
                self.gpt_cond_latent,
                self.speaker_embedding,
                stream_chunk_size=60,
                temperature=0.60,
                repetition_penalty=20.0,
                enable_text_splitting=True,
            )
            for j, chunk in enumerate(chunks):
                if i == 0:
                    print(
                        f"Time to first chunck: {time.time() - t0}"
                    ) if self.debug else None
                print(
                    f"Received chunk {i} of audio length {chunk.shape[-1]}"
                ) if self.debug else None
                wav_chunks.append(
                    chunk.to(device=self.device)
                )  # Move chunk to available device
            # Add a short pause between sentences (e.g., X.XX seconds of silence)
            if i < len(sentence_list):
                silence_duration = int(24000 * .6)
                silence = torch.zeros(
                    (silence_duration,), dtype=torch.float32, device=self.device
                )  # Move silence tensor to available device
                wav_chunks.append(silence)
        wav = torch.cat(wav_chunks, dim=0)
        torchaudio.save(wav_file_path, wav.squeeze().unsqueeze(0).cpu(), 24000)
        with AudioFile(wav_file_path).resampled_to(24000) as f:
            audio = f.read(f.frames)
        reduced_noise = noisereduce.reduce_noise(
            y=audio, sr=24000, stationary=True, prop_decrease=0.75
        )
        board = Pedalboard(
            [
                NoiseGate(threshold_db=-30, ratio=1.5, release_ms=250),
                Compressor(threshold_db=12, ratio=2.5),
                LowShelfFilter(cutoff_frequency_hz=400, gain_db=5, q=1),
                Gain(gain_db=0),
            ]
        )
        result = board(reduced_noise, 24000)
        with AudioFile(wav_file_path, "w", 24000, result.shape[0]) as f:
            f.write(result)

    def compare(self, text, wavfile):
        result = self.whispermodel.transcribe(wavfile)
        text = re.sub(" +", " ", text).lower().strip()
        ratio = fuzz.ratio(text, result["text"].lower())
        print(f"Transcript: {result['text'].lower()}") if self.debug else None
        print(
            f"Text to transcript comparison ratio: {ratio}"
        ) if self.debug else None
        return ratio

    def combine_sentences(self, sentences, length=1000):
        for sentence in sentences:
            yield sentence

    def export(self, format):
        allowed_formats = ["txt"]
        #this should probably be a try/except, fix later
        if format not in allowed_formats:
            print(f"{format} not allowed export format")
            sys.exit()
        outputfile = f"{self.bookname}.{format}"
        self.check_for_file(outputfile)
        print(f"Exporting parts {self.start + 1} to {self.end} to {outputfile}")
        with open(outputfile, "w") as file:
            file.write(f"Title: {self.title}\n")
            file.write(f"Author: {self.author}\n\n")
            for partnum, i in enumerate(range(self.start, self.end)):
                file.write(f"\n# Part {partnum + 1}\n\n")
                file.write(self.chapters_to_read[i] + "\n")

    def check_for_file(self, filename):
        if os.path.isfile(filename):
            print(f"The file '{filename}' already exists.")
            overwrite = input("Do you want to overwrite the file? (y/n): ")
            if overwrite.lower() != 'y':
                print("Exiting without overwriting the file.")
                sys.exit()
            else:
                os.remove(filename)

    def add_cover(self, cover_img):
        if os.path.isfile(cover_img):
            m4b = mp4.MP4(self.output_filename)
            cover_image = open(cover_img, "rb").read()
            m4b["covr"] = [mp4.MP4Cover(cover_image)]
            m4b.save()
        else:
            print(f"Cover image {cover_img} not found")

    async def edgespeak(self, sentence, speaker, filename):
        communicate = edge_tts.Communicate(sentence, speaker)
        await communicate.save(filename)

    def extract_title_author(self, text):
        lines = text.split('\n')
        metadata = {}

        # A copy of the list for iteration
        lines_copy = lines[:]

        for line in lines_copy[:2]:  # We check only the first two lines
            if line.startswith('Title: '):
                metadata['Title'] = line.replace('Title: ', '').strip()
                lines.remove(line)  # Remove line from the original list
            elif line.startswith('Author: '):
                metadata['Author'] = line.replace('Author: ', '').strip()
                lines.remove(line)  # Remove line from the original list

        text = '\n'.join(lines)   # Join the lines back
        return metadata, text

    def read_book(self, voice_samples, engine, openai, model_name, speaker, bitrate):
        self.model_name = model_name
        self.openai = openai
        if engine == "xtts":
            if voice_samples != None:
                self.voice_samples = []
                for f in voice_samples.split(","):
                    self.voice_samples.append(os.path.abspath(f))
                voice_name = (
                    "-" + re.split("-|\d+|\.", os.path.basename(self.voice_samples[0]))[0]
                )
            else:
                voice_name = "-" + speaker.replace(" ", "-").lower()
        elif engine == "openai":
            if speaker == None:
                speaker = "onyx"
            voice_name = "-" + speaker
        elif engine == "edge":
            if speaker == None:
                speaker = "en-US-AndrewNeural"
            voice_name = "-" + speaker
        elif engine == "tts":
            if speaker == None:
                speaker = "p335"
            voice_name = "-" + speaker
        else:
            voice_name = "-" + speaker
        self.output_filename = re.sub(".m4b", voice_name + ".m4b", self.output_filename)
        print(f"Saving to {self.output_filename}")
        self.check_for_file(self.output_filename)
        total_chars = self.get_length(self.start, self.end, self.chapters_to_read)
        print(f"Total characters: {total_chars}")
        if engine == "xtts":
            if (
                torch.cuda.is_available()
                and torch.cuda.get_device_properties(0).total_memory > 3500000000
            ):
                print("Using GPU")
                print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory}")
                self.device = "cuda"
            else:
                print("Not enough VRAM on GPU or CUDA not found. Using CPU")
                self.device = "cpu"

            print("Loading model: " + self.xtts_model)
            # This will trigger model load even though we might not use tts object later
            tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
            tts = ""
            config = XttsConfig()
            model_json = self.xtts_model + "/config.json"
            config.load_json(model_json)
            self.model = Xtts.init_from_config(config)
            if self.no_deepspeed:
                use_deepspeed = False
            else:
                use_deepspeed = self.is_installed("deepspeed")
            self.model.load_checkpoint(
                config, checkpoint_dir=self.xtts_model, use_deepspeed=use_deepspeed
            )

            if self.device == "cuda":
                print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory}")
                self.model.cuda()

            print("Computing speaker latents...")
            if speaker == None:
                (
                self.gpt_cond_latent,
                self.speaker_embedding,
                ) = self.model.get_conditioning_latents(audio_path=self.voice_samples)
            else: #using Coqui speaker
                (
                self.gpt_cond_latent,
                self.speaker_embedding,
                ) = self.model.speaker_manager.speakers[speaker].values()

        elif engine == "openai":
            while True:
                openai_sdcost = (total_chars / 1000) * 0.015
                print("OpenAI TTS SD Cost: $" + str(openai_sdcost))
                user_input = input("This will not be free, continue? (y/n): ")
                if user_input.lower() not in ["y", "n"]:
                    print("Invalid input. Please enter y for yes or n for no.")
                elif user_input.lower() == "n":
                    sys.exit()
                else:
                    print("Continuing...")
                    break
            client = OpenAI(api_key=self.openai)
        elif engine == "edge":
            print("Engine is Edge TTS")
        else:
            print(f"Engine is TTS, model is {model_name}")
            self.tts = TTS(model_name).to(self.device)

        files = []
        position = 0
        start_time = time.time()
        print(f"Reading from {self.start + 1} to {self.end}")
        for partnum, i in enumerate(range(self.start, self.end)):
            outputwav = f"{self.bookname}-{i + 1}.wav"
            if os.path.isfile(outputwav):
                print(f"{outputwav} exists, skipping to next chapter")
            else:
                tempfiles = []
                if self.sayparts and len(self.section_names) == 0:
                    chapter = "Part " + str(partnum + 1) + ". " + self.chapters_to_read[i]
                elif self.sayparts and len(self.section_names) > 0:
                    chapter = self.section_names[i].strip() + ".\n" + self.chapters_to_read[i]
                else:
                    chapter = self.chapters_to_read[i]
                sentences = sent_tokenize(chapter)
                #Drop any items that do NOT have at least one letter or number
                sentences = [s for s in sentences if any(c.isalnum() for c in s)]
                if engine == "tts" and model_name == "tts_models/multilingual/multi-dataset/xtts_v2":
                    #we are using coqui voice, so make smaller chunks
                    length = 500
                else:
                    length = 1000
                sentence_groups = list(self.combine_sentences(sentences, length))

                for x in tqdm(range(len(sentence_groups))):
                    #skip if item is empty
                    if len(sentence_groups[x]) == 0:
                        continue
                    #skip if item has no characters or numbers
                    if not any(char.isalnum() for char in sentence_groups[x]):
                        continue
                    retries = 2
                    tempwav = "temp" + str(x) + ".wav"
                    tempflac = tempwav.replace("wav", "flac")
                    if os.path.isfile(tempwav):
                        print(tempwav + " exists, skipping to next chunk")
                    else:
                        while retries > 0:
                            try:
                                if engine == "xtts":
                                    if self.language != "en":
                                            sentence_groups[x] = sentence_groups[x].replace(".", ",")
                                    self.read_chunk_xtts(sentence_groups[x], tempwav)
                                elif engine == "openai":
                                    self.minratio = 0
                                    response = client.audio.speech.create(
                                        model="tts-1",
                                        voice=speaker.lower(),
                                        input=sentence_groups[x],
                                    )
                                    response.stream_to_file(tempwav)
                                elif engine == "edge":
                                    self.minratio = 0
                                    if self.debug:
                                        print(
                                            sentence_groups[x]
                                        )
                                    asyncio.run(self.edgespeak(sentence_groups[x], speaker, tempwav))
                                elif engine == "tts":
                                    if model_name == "tts_models/en/vctk/vits":
                                        self.minratio = 0
                                        # assume we're using a multi-speaker model
                                        if self.debug:
                                            print(
                                                sentence_groups[x]
                                            )
                                            with open("debugout.txt", "a") as file: file.write(f"{sentence_groups[x]}\n")
                                        self.tts.tts_to_file(
                                            text=sentence_groups[x],
                                            speaker=speaker,
                                            file_path=tempwav,
                                        )
                                    else:
                                        if self.debug:
                                            print(
                                                sentence_groups[x]
                                            )
                                            with open("debugout.txt", "a") as file: file.write(f"{sentence_groups[x]}\n")
                                        self.tts.tts_to_file(
                                            text=sentence_groups[x], file_path=tempwav
                                        )
                                if self.minratio == 0:
                                    print("Skipping whisper transcript comparison") if self.debug else None
                                    ratio = self.minratio
                                else:
                                    ratio = self.compare(sentence_groups[x], tempwav)
                                if ratio < self.minratio:
                                    raise Exception(
                                        f"Spoken text did not sound right - {ratio}"
                                    )
                                break
                            except Exception as e:
                                retries -= 1
                                print(
                                    f"Error: {e} ... Retrying ({retries} retries left)"
                                )
                        if retries == 0:
                            print(
                                f"Something is wrong with the audio ({ratio}): {tempwav}"
                            )
                        if engine == "openai" or engine == "edge":
                            temp = AudioSegment.from_mp3(tempwav)
                        else:
                            temp = AudioSegment.from_wav(tempwav)
                    tempfiles.append(tempwav)
                tempwavfiles = [AudioSegment.from_file(f"{f}") for f in tempfiles]
                concatenated = sum(tempwavfiles)
                # remove silence, then export to wav
                print(
                    f"Replacing silences longer than one second with one second of silence ({outputwav})"
                )
                one_sec_silence = AudioSegment.silent(duration=1000)
                two_sec_silence = AudioSegment.silent(duration=2000)
                # This AudioSegment is dedicated for each file.
                audio_modified = AudioSegment.empty()
                # Split audio into chunks where detected silence is longer than one second
                chunks = split_on_silence(
                    concatenated, min_silence_len=1000, silence_thresh=-50
                )
                # Iterate through each chunk
                for chunkindex, chunk in enumerate(tqdm(chunks)):
                    audio_modified += chunk
                    audio_modified += one_sec_silence
                # add extra 2sec silence at the end of each part/chapter
                audio_modified += two_sec_silence
                # Write modified audio to the final audio segment
                audio_modified.export(outputwav, format="wav")
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
            print(
                f"Elapsed: {int(elapsed_time / 60)} minutes, ETA: {int((estimated_time_remaining) / 60)} minutes"
            )
            gc.collect()
            torch.cuda.empty_cache()
        outputm4a = self.output_filename.replace("m4b", "m4a")
        filelist = "filelist.txt"
        with open(filelist, "w") as f:
            for filename in files:
                filename = filename.replace("'", "'\\''")
                f.write(f"file '{filename}'\n")

        if self.audioformat == "wav":
            outputm4a = outputm4a.replace(".m4a", "_without_metadata.wav")
            self.output_filename = self.output_filename.replace(".m4b", ".wav")
            ffmpeg_command = [
                "ffmpeg",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                filelist,
                outputm4a,
            ]
        elif self.audioformat == "flac":
            outputm4a = outputm4a.replace(".m4a", "_without_metadata.flac")
            self.output_filename = self.output_filename.replace(".m4b", ".flac")
            ffmpeg_command = [
                "ffmpeg",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                filelist,
                outputm4a,
            ]
        else:
            ffmpeg_command = [
                "ffmpeg",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                filelist,
                "-codec:a",
                "aac",
                "-b:a",
                bitrate,
                "-f",
                "ipod",
                outputm4a,
            ]
        subprocess.run(ffmpeg_command)
        self.generate_metadata(files)
        ffmpeg_command = [
            "ffmpeg",
            "-i",
            outputm4a,
            "-i",
            self.ffmetadatafile,
            "-map_metadata",
            "1",
            "-codec",
            "copy",
            self.output_filename,
        ]
        subprocess.run(ffmpeg_command)
        if not self.debug: # Leave the files if debugging
            os.remove(filelist)
            os.remove(self.ffmetadatafile)
            os.remove(outputm4a)
            for f in files:
                os.remove(f)
        print(self.output_filename + " complete")


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    parser = argparse.ArgumentParser(
        prog="EpubToAudiobook",
        description="Read an epub (or other source) to audiobook format",
    )
    parser.add_argument("sourcefile", type=str, help="The epub or text file to process")
    parser.add_argument(
        "--engine",
        type=str,
        default="tts",
        nargs="?",
        const="tts",
        help="Which TTS to use [tts|xtts|openai|edge]",
    )
    parser.add_argument(
        "--xtts",
        type=str,
        help="Sample wave/mp3 file(s) for XTTS v2 training separated by commas",
    )
    parser.add_argument("--openai", type=str, help="OpenAI API key if engine is OpenAI")
    parser.add_argument(
        "--model",
        type=str,
        nargs="?",
        const="tts_models/en/vctk/vits",
        default="tts_models/en/vctk/vits",
        help="TTS model to use, default: tts_models/en/vctk/vits",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        help="Speaker to use (ex p335 for VITS, or onyx for OpenAI)",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Scan the epub to show beginning of chapters, then exit",
    )
    parser.add_argument(
        "--start",
        type=int,
        nargs="?",
        const=1,
        default=1,
        help="Chapter/part to start from",
    )
    parser.add_argument(
        "--end",
        type=int,
        nargs="?",
        const=999,
        default=999,
        help="Chapter/part to end with",
    )
    parser.add_argument(
        "--language",
        type=str,
        nargs="?",
        const="en",
        default="en",
        help="Language of the epub, default: en",
    )
    parser.add_argument(
        "--minratio",
        type=int,
        nargs="?",
        const=93,
        default=88,
        help="Minimum match ratio between text and transcript, 0 to disable whisper",
    )
    parser.add_argument(
        "--skiplinks", 
        action="store_true", 
        help="Skip reading any HTML links"
    )
    parser.add_argument(
        "--skipfootnotes", 
        action="store_true", 
        help="Try to skip reading footnotes"
    )
    parser.add_argument(
        "--sayparts", 
        action="store_true", 
        help="Say each part number at start of section"
    )
    parser.add_argument(
        "--audioformat", 
        type=str,
        default="m4b",
        help="Audio format of the output file (m4b [default], wav, flac)"
    )
    parser.add_argument(
        "--bitrate",
        type=str,
        nargs="?",
        const="69k",
        default="69k",
        help="Specify bitrate for output file",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
    )
    parser.add_argument(
        "--export",
        type=str,
        help="Export epub contents to file (txt, md coming soon)"
    )
    parser.add_argument(
        "--no-deepspeed",
        action="store_true",
        help="Disable deepspeed",
    )
    parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="Skip text cleanup",
    )
    parser.add_argument(
        "--cover",
        type=str,
        help="jpg image to use for cover",
    )

    args = parser.parse_args()
    print(args)

    if args.openai:
        args.engine = "openai"
    elif args.xtts:
        args.engine = "xtts"
    mybook = EpubToAudiobook(
        source=args.sourcefile,
        start=args.start,
        end=args.end,
        skiplinks=args.skiplinks,
        engine=args.engine,
        minratio=args.minratio,
        model_name=args.model,
        debug=args.debug,
        language=args.language,
        skipfootnotes=args.skipfootnotes,
        sayparts=args.sayparts,
        no_deepspeed=args.no_deepspeed,
        skip_cleanup=args.skip_cleanup,
        audioformat=args.audioformat,
    )

    print(f"Language selected: {mybook.language}")

    if mybook.sourcetype == "epub":
        mybook.get_chapters_epub()
    else:
        mybook.get_chapters_text()
    if args.scan:
        sys.exit()
    if args.export is not None:
        mybook.export(
            format=args.export,
        )
        sys.exit()
    mybook.read_book(
        voice_samples=args.xtts,
        engine=args.engine,
        openai=args.openai,
        model_name=args.model,
        speaker=args.speaker,
        bitrate=args.bitrate,
    )
    if args.cover is not None:
        mybook.add_cover(args.cover)


if __name__ == "__main__":
    main()
