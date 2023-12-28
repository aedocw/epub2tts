import argparse
import os
import re
import subprocess
import sys
import time
import warnings

from bs4 import BeautifulSoup
import ebooklib
from ebooklib import epub
from fuzzywuzzy import fuzz
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
    ):
        self.source = source
        self.bookname = os.path.splitext(os.path.basename(source))[0]
        self.start = start - 1
        self.end = end
        self.language = language
        self.skiplinks = skiplinks
        self.skipfootnotes = skipfootnotes
        self.engine = engine
        self.minratio = minratio
        self.debug = debug
        self.output_filename = self.bookname + ".m4b"
        self.chapters = []
        self.chapters_to_read = []
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
            self.xtts_model = self.tts_dir + "/" + model_name
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

    # Call the function to ensure punkt is downloaded

    def generate_metadata(self, files, title, author):
        chap = 1
        start_time = 0
        with open(self.ffmetadatafile, "w") as file:
            file.write(";FFMETADATA1\n")
            file.write("ARTIST=" + str(author) + "\n")
            file.write("ALBUM=" + str(title) + "\n")
            for file_name in files:
                duration = self.get_duration(file_name)
                file.write("[CHAPTER]\n")
                file.write("TIMEBASE=1/1000\n")
                file.write("START=" + str(start_time) + "\n")
                file.write("END=" + str(start_time + duration) + "\n")
                file.write("title=Part " + str(chap) + "\n")
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
            if a.text.isdigit():
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
            .replace(" . . . ", ", ")
            .replace("... ", ", ")
            .replace("«", " ")
            .replace("»", " ")
            .replace("&", " and ")
            .replace(" GNU ", " new ")
            .replace("\n", " \n")
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

        for i in range(len(self.chapters)):
            text = self.prep_text(self.chap2text(self.chapters[i]))
            if len(text) < 150:
                # too short to bother with
                continue
            print("Length: " + str(len(text)))
            print("Part: " + str(len(self.chapters_to_read) + 1))
            if self.skipfootnotes:
                text = self.exclude_footnotes(text)
            if self.skipfootnotes and text.startswith("Footnotes"):
                continue
            print(text[:256])
            self.chapters_to_read.append(text)
        print("Number of chapters to read: " + str(len(self.chapters_to_read)))
        if self.end == 999:
            self.end = len(self.chapters_to_read)

    def get_chapters_text(self):
        with open(self.source, "r") as file:
            text = file.read()
        text = self.prep_text(text)
        max_len = 50000
        while len(text) > max_len:
            pos = text.rfind(" ", 0, max_len)  # find the last space within the limit
            self.chapters_to_read.append(text[:pos])
            print("Part: " + str(len(self.chapters_to_read)))
            print(str(self.chapters_to_read[-1])[:256])
            text = text[pos + 1 :]  # +1 to avoid starting the next chapter with a space
        self.chapters_to_read.append(text)
        self.end = len(self.chapters_to_read)

    def read_chunk_xtts(self, sentences, wav_file_path):
        # takes list of sentences to read, reads through them and saves to file
        t0 = time.time()
        wav_chunks = []
        sentence_list = sent_tokenize(sentences)
        for i, sentence in enumerate(sentence_list):
            # Run TTS for each sentence
            print(sentence) if self.debug else None
            chunks = self.model.inference_stream(
                sentence,
                self.language,
                self.gpt_cond_latent,
                self.speaker_embedding,
                stream_chunk_size=60,
                temperature=0.60,
                repetition_penalty=10.0,
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
            if i < len(sentence_list) - 1:
                silence_duration = int(24000 * 1.0)
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
        print("Transcript: " + result["text"].lower()) if self.debug else None
        print(
            "Text to transcript comparison ratio: " + str(ratio)
        ) if self.debug else None
        return ratio

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
        if engine == "xtts":
            if voice_samples != '':
                self.voice_samples = []
                for f in voice_samples.split(","):
                    self.voice_samples.append(os.path.abspath(f))
                voice_name = (
                    "-" + re.split("-|\d+|\.", os.path.basename(self.voice_samples[0]))[0]
                )
            else:
                voice_name = speaker.replace(" ", "-").lower()
        elif engine == "openai":
            if speaker == "p335":
                speaker = "onyx"
            voice_name = "-" + speaker
        else:
            voice_name = "-" + speaker
        self.output_filename = re.sub(".m4b", voice_name + ".m4b", self.output_filename)
        print("Saving to " + self.output_filename)
        total_chars = self.get_length(self.start, self.end, self.chapters_to_read)
        print("Total characters: " + str(total_chars))
        if engine == "xtts":
            if (
                torch.cuda.is_available()
                and torch.cuda.get_device_properties(0).total_memory > 3500000000
            ):
                print("Using GPU")
                print("VRAM: " + str(torch.cuda.get_device_properties(0).total_memory))
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
            self.model.load_checkpoint(
                config, checkpoint_dir=self.xtts_model, use_deepspeed=False
            )

            if self.device == "cuda":
                print("VRAM: " + str(torch.cuda.get_device_properties(0).total_memory))
                self.model.cuda()

            print("Computing speaker latents...")
            (
                self.gpt_cond_latent,
                self.speaker_embedding,
            ) = self.model.get_conditioning_latents(audio_path=self.voice_samples)
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
        else:
            print("Engine is TTS, model is " + model_name)
            self.tts = TTS(model_name).to(self.device)

        files = []
        position = 0
        start_time = time.time()
        print("Reading from " + str(self.start + 1) + " to " + str(self.end))
        for i in range(self.start, self.end):
            outputflac = self.bookname + "-" + str(i + 1) + ".flac"
            if os.path.isfile(outputflac):
                print(outputflac + " exists, skipping to next chapter")
            else:
                tempfiles = []
                sentences = sent_tokenize(self.chapters_to_read[i])
                if engine == "tts" and model_name == "tts_models/multilingual/multi-dataset/xtts_v2":
                    #we are using coqui voice, so make smaller chunks
                    length = 500
                else:
                    length = 1000
                sentence_groups = list(self.combine_sentences(sentences, length))
                for x in tqdm(range(len(sentence_groups))):
                    retries = 1
                    tempwav = "temp" + str(x) + ".wav"
                    tempflac = tempwav.replace("wav", "flac")
                    if os.path.isfile(tempflac):
                        print(tempflac + " exists, skipping to next chunk")
                    else:
                        while retries > 0:
                            try:
                                if engine == "xtts":
                                    if self.language != "en":
                                            sentence_groups[x] = sentence_groups[x].replace(".", ",")
                                    self.read_chunk_xtts(sentence_groups[x], tempwav)
                                elif engine == "openai":
                                    response = client.audio.speech.create(
                                        model="tts-1",
                                        voice=speaker,
                                        input=sentence_groups[x],
                                    )
                                    response.stream_to_file(tempwav)
                                elif engine == "tts":
                                    if model_name == "tts_models/en/vctk/vits":
                                        # assume we're using a multi-speaker model
                                        print(
                                            sentence_groups[x]
                                        ) if self.debug else None
                                        self.tts.tts_to_file(
                                            text=sentence_groups[x],
                                            speaker=speaker,
                                            file_path=tempwav,
                                        )
                                    elif model_name == "tts_models/multilingual/multi-dataset/xtts_v2":
                                        if self.language != "en":
                                            sentence_groups[x] = sentence_groups[x].replace(".", ",")
                                        print(
                                            "text to read: " +
                                            sentence_groups[x]
                                        ) if self.debug else None
                                        self.tts.tts_to_file(
                                            text=sentence_groups[x],
                                            speaker=speaker,
                                            language=self.language,
                                            file_path=tempwav,
                                        )
                                    else:
                                        print(
                                            sentence_groups[x]
                                        ) if self.debug else None
                                        self.tts.tts_to_file(
                                            text=sentence_groups[x], file_path=tempwav
                                        )
                                if self.minratio == 0 or model_name == "tts_models/en/vctk/vits":
                                    print("Skipping whisper transcript comparison") if self.debug else None
                                    ratio = self.minratio
                                else:
                                    ratio = self.compare(sentence_groups[x], tempwav)
                                if ratio < self.minratio:
                                    raise Exception(
                                        "Spoken text did not sound right - "
                                        + str(ratio)
                                    )
                                break
                            except Exception as e:
                                retries -= 1
                                print(
                                    f"Error: {str(e)} ... Retrying ({retries} retries left)"
                                )
                        if retries == 0:
                            print(
                                "Something is wrong with the audio ("
                                + str(ratio)
                                + "): "
                                + tempwav
                            )
                            # sys.exit()
                        temp = AudioSegment.from_wav(tempwav)
                        temp.export(tempflac, format="flac")
                        os.remove(tempwav)
                    tempfiles.append(tempflac)
                tempflacfiles = [AudioSegment.from_file(f"{f}") for f in tempfiles]
                concatenated = sum(tempflacfiles)
                # remove silence, then export to flac
                print(
                    "Replacing silences longer than one second with one second of silence ("
                    + outputflac
                    + ")"
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
                # add extra 2sec silence at the end of each part/chapter if it's an epub
                if self.sourcetype == "epub":
                    audio_modified += two_sec_silence
                # Write modified audio to the final audio segment
                audio_modified.export(outputflac, format="flac")
                for f in tempfiles:
                    os.remove(f)
            files.append(outputflac)
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
        # Load all FLAC files and concatenate into one object
        flac_files = [AudioSegment.from_file(f"{f}") for f in files]
        concatenated = AudioSegment.empty()
        for audio in flac_files:
            concatenated += audio
        outputm4a = self.output_filename.replace("m4b", "m4a")
        concatenated.export(outputm4a, format="ipod", bitrate=bitrate)
        if self.sourcetype == "epub":
            author = self.book.get_metadata("DC", "creator")[0][0]
            title = self.book.get_metadata("DC", "title")[0][0]
        else:
            author = "Unknown"
            title = self.bookname
        self.generate_metadata(files, title, author)
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
        os.remove(self.ffmetadatafile)
        os.remove(outputm4a)
        if not self.debug: # Leave the files if debugging
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
        help="Which TTS to use [tts|xtts|openai]",
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
        default="p335",
        nargs="?",
        const="p335",
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
        const=88,
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
        "--bitrate",
        type=str,
        nargs="?",
        const="69k",
        default="69k",
        help="Specify bitrate for output file",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()
    print(args)

    if args.openai:
        args.engine = "openai"
    elif args.xtts:
        args.engine = "xtts"
    elif args.speaker != "" and args.engine == "xtts" and args.model != "":
        #we are using a Coqui XTTS voice
        args.engine = "tts"
        args.model = "tts_models/multilingual/multi-dataset/xtts_v2"
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
    )

    print("Language selected: " + mybook.language)

    if mybook.sourcetype == "epub":
        mybook.get_chapters_epub()
    else:
        mybook.get_chapters_text()
    if args.scan:
        sys.exit()
    mybook.read_book(
        voice_samples=args.xtts,
        engine=args.engine,
        openai=args.openai,
        model_name=args.model,
        speaker=args.speaker,
        bitrate=args.bitrate,
    )


if __name__ == "__main__":
    main()
