import argparse
import re
import asyncio
import os
import copy
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]="1"
import pkg_resources
import multiprocessing as mp
import re
import subprocess
import sys
import time
import warnings
import zipfile
from pathlib import Path

from bs4 import BeautifulSoup
from ebooklib import epub
from fuzzywuzzy import fuzz
from lxml import etree
from mutagen import mp4
from nltk.tokenize import sent_tokenize
from openai import OpenAI
from pedalboard import Pedalboard, Compressor, Gain, NoiseGate, LowShelfFilter
from pedalboard.io import AudioFile
from PIL import Image
from pydub import AudioSegment
from pydub.silence import split_on_silence
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.generic_utils import get_user_data_dir
import ebooklib
import edge_tts
import nltk
import noisereduce
import torch, gc
import torchaudio
import whisper

namespaces = {
   "calibre":"http://calibre.kovidgoyal.net/2009/metadata",
   "dc":"http://purl.org/dc/elements/1.1/",
   "dcterms":"http://purl.org/dc/terms/",
   "opf":"http://www.idpf.org/2007/opf",
   "u":"urn:oasis:names:tc:opendocument:xmlns:container",
   "xsi":"http://www.w3.org/2001/XMLSchema-instance",
}

class Text2WaveFile:
    whispermodel = None
    debug = False
    def __init__(self, config = {}):
        """
        initalizes a Text 2 Wave File class
        This might mean loading the ML model used for speech syntesis or setting up other stuff
        """
        self.config = config

    def proccess_text(self, text, wave_file_name):
        """
        takes a pice of text and generates audio from it then saves that audio in wave_file_name
        returns True if successfull
        """

    def compare(self, text, wavfile):
        if self.whispermodel is None:
            self.whispermodel = whisper.load_model("tiny")
        
        result = self.whispermodel.transcribe(wavfile)
        text = re.sub(" +", " ", text).lower().strip()
        ratio = fuzz.ratio(text, result["text"].lower())
        print(f"Transcript: {result['text'].lower()}") if self.debug else None
        print(f"Text to transcript comparison ratio: {ratio}") if self.debug else None
        return ratio, result['text']

    
    def proccess_text_retry(self, text, wave_file_name):
        retries = 2
        while retries > 0:
            self.proccess_text(text, wave_file_name)
            result_text = ""
            if self.config['minratio'] == 0:
                print("Skipping whisper transcript comparison") if self.config['debug'] else None
                ratio = self.config['minratio']
            else:
                ratio, result_text = self.compare(text, wave_file_name)
            if ratio < self.config['minratio']:
                print(f"Spoken text did not sound right after control with whisper - {ratio}\nInput: {text}\nOutput: {result_text}")
            else:
                break
            retries -= 1
        if retries == 0:
            print(f"Something is wrong with the audio acording to whisper ({ratio}): {tempwav}")

class EdgeTTS(Text2WaveFile):
    def __init__(self, config = {}):
        if 'speaker' not in config:
            raise Exception('no speeker configured')
        self.config = config

    def proccess_text(self, text, wave_file_name):

        asyncio.run(self.edgespeak(text, wave_file_name))

        if os.path.exists(wave_file_name):
            return True
        return False

    async def edgespeak(self, text, wave_file_name):
        communicate = edge_tts.Communicate(text, self.config['speaker'])
        await communicate.save(wave_file_name)

class OpenAI_TTS(Text2WaveFile):
    def __init__(self, config = {}):
        if 'api_key' not in config:
            raise Exception('no api_key given')
        if 'speaker' not in config:
            raise Exception('no speeker configured')
        self.config = config
        self.client = OpenAI(api_key=config['api_key'])

    def proccess_text(self, text, wave_file_name):
        self.client.audio.speech.create(
            model="tts-1",
            voice=self.config['speaker'].lower(),
            input=text,
        )
        response.stream_to_file(wave_file_name)

        if os.path.exists(wave_file_name):
            return True
        return False

class XTTS(Text2WaveFile):
    def __init__(self, config = {}):
        if 'speaker' not in config:
            raise Exception('no speeker configured')

        if 'language' not in config:
            raise Exception('no language configured')

        if 'xtts_model' not in config:
            raise Exception('no xtts_model configured')

        if 'debug' not in config:
            self.debug = False
        else:
            self.debug = config['debug']

        self.config = config

        if 'voice_samples' in config:
            self.voice_samples = self.config['voice_samples']

        self.language = self.config['language']
        self.xtts_model = self.config['xtts_model']

        if (
            torch.cuda.is_available()
            and torch.cuda.get_device_properties(0).total_memory > 3500000000
        ):
            print("Using GPU")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory}")
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            print("Using GPU")
            self.device = "mps"
            self.config['no_deepspeed'] = True
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
        if self.config['no_deepspeed']:
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
        if self.config['speaker'] == None:
            (
            self.gpt_cond_latent,
            self.speaker_embedding,
            ) = self.model.get_conditioning_latents(audio_path=self.voice_samples)
        else: #using Coqui speaker
            (
            self.gpt_cond_latent,
            self.speaker_embedding,
            ) = self.model.speaker_manager.speakers[self.config['speaker']].values()

    def is_installed(self, package_name):
        package_installed = False
        try:
            pkg_resources.get_distribution(package_name)
            package_installed = True
        except pkg_resources.DistributionNotFound:
            pass
        return package_installed

    def proccess_text(self, text, wave_file_name):
        if self.language != "en":
            text = text.replace(".", ",")
        self.read_chunk_xtts(text, wave_file_name)

    def read_chunk_xtts(self, sentences, wav_file_path):
        # takes list of sentences to read, reads through them and saves to file
        t0 = time.time()
        wav_chunks = []
        sentence_list = sent_tokenize(sentences)
        for i, sentence in enumerate(sentence_list):
            # Run TTS for each sentence
            if self.debug:
                print(sentence)
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
                    print(f"Time to first chunk: {time.time() - t0}") if self.debug else None
                print(f"Received chunk {i} of audio length {chunk.shape[-1]}") if self.debug else None
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

class org_TTS(Text2WaveFile):
    def __init__(self, config = {}):
        if 'model_name' not in config:
            raise Exception('no model_name configured')
        if 'device' not in config:
            raise Exception('no device configured')

        if 'debug' not in config:
            self.debug = False
        else:
            self.debug = config['debug']

        self.config = config

        self.model_name = self.config['model_name']
        self.device = self.config['device']
        self.tts = TTS(self.config['model_name']).to(self.device)


    def proccess_text(self, text, wave_file_name):
        if self.model_name == "tts_models/en/vctk/vits":
            self.minratio = 0
            # assume we're using a multi-speaker model
            if self.debug:
                print(text)
                with open("debugout.txt", "a") as file: file.write(f"{text}\n")
            self.tts.tts_to_file(
                text=text,
                speaker=self.config['speaker'],
                file_path=wave_file_name,
            )
        else:
            if self.debug:
                print(text)
                with open("debugout.txt", "a") as file: file.write(f"{text}\n")
            self.tts.tts_to_file(
                text=text, file_path=wave_file_name
            )

def join_temp_files_to_chapter(tempfiles, outputwav):
    tempwavfiles = [AudioSegment.from_file(f"{f}") for f in tempfiles]
    concatenated = sum(tempwavfiles)
    # remove silence, then export to wav
    #print(f"Replacing silences longer than one second with one second of silence ({outputwav})")
    one_sec_silence = AudioSegment.silent(duration=1000)
    two_sec_silence = AudioSegment.silent(duration=2000)
    # This AudioSegment is dedicated for each file.
    audio_modified = AudioSegment.empty()
    # Split audio into chunks where detected silence is longer than one second
    chunks = split_on_silence(
        concatenated, min_silence_len=1000, silence_thresh=-50
    )
    # Iterate through each chunk
    for chunkindex, chunk in enumerate(chunks):
        audio_modified += chunk
        audio_modified += one_sec_silence
    # add extra 2sec silence at the end of each part/chapter
    audio_modified += two_sec_silence
    # Write modified audio to the final audio segment
    audio_modified.export(outputwav, format="wav")
    for f in tempfiles:
        os.remove(f)

def process_book_chapter(dat):
    print("initiating chapter: ", dat['chapter'])
    tts_engine = dat['config']['engine_cl'](dat['config'])
    for text, file_name in dat['sentene_job_que']:
        tts_engine.proccess_text_retry(text, file_name)
    join_temp_files_to_chapter(dat['tempfiles'], dat['outputwav'])
    print("done chapter: ", dat['chapter'])
    return dat['outputwav']



class EpubToAudiobook:
    def __init__(
        self,
        source,
        start,
        threads,
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
        self.threads = threads
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
        self.section_speakers = []
        self.no_deepspeed = no_deepspeed
        self.skip_cleanup = skip_cleanup
        self.title = self.bookname
        self.author = "Unknown"
        self.audioformat = [i.lower() for i in audioformat.split(",")]
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
        
        self.ffmetadatafile = "FFMETADATAFILE"
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
             self.device = "mps"
        else:
            self.device = "cpu"
        # Make sure we've got nltk punkt
        self.ensure_punkt()

    def ensure_punkt(self):
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab")


    def generate_metadata(self, files):
        chap = 1
        start_time = 0
        with open(self.ffmetadatafile, "w", encoding='utf8') as file:
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
                    file.write(f"title={self.section_names[self.start+chap-1]}\n")
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

    def chap2text(self, chap, element_id = None, end_element_id = None):
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
        skip_epub_types = [
           "pagebreak", #contains the page number we dont need to read the alloud
           "annotation", #Contains stuff like table descriptions (ie a text saying: "this table has 3 columns and 4 rows")
           #"sidebar", # contains the side bar if there is one (We keep it but it  might not be desirable)
           #"chapter", # we definetly want to keep the chapters
           "index", #this will be an audiobook we dont need the index (especially since we dont include the page numbers)
        ]
        output = ""
        if type(chap) == BeautifulSoup:
            soup = chap
        else:
            soup = BeautifulSoup(chap, "html.parser")
        if element_id is not None:
            soup = soup.find(id=element_id).parent

        #We copy the html tree before modifining it
        soup = copy.deepcopy(soup)
        if self.skiplinks:
            # Remove everything that is an href
            for a in soup.findAll("a", href=True):
                a.extract()
        # Always skip reading links that are just a number (footnotes)
        for a in soup.findAll("a", href=True):
            if not any(char.isalpha() for char in a.text):
                a.extract()

        for elm in soup.findAll("[epub:type]"):
            elm_epub_type = elm.get('epub:type')
            if elm_epub_type is not None and elm_epub_type in skip_epub_types: #Dont read the page numbers or annotations
                elm.extract()


        #remove all elements after the end element(if we have a end element)
        remove = False
        for elm in soup.find_all(True):
            if not remove and end_element_id is not None and elm.get('id') == end_element_id:
                remove = True
            if remove:
                elm.extract()

        #TODO: render the HTML using a real HTML renderer
        #append enters after newline elements
        for elm in soup.find_all(True):
            if elm.name in ('p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'div', 'li', 'ol', 'ul', 'table', 'th', 'tr', 'br', 'pre', 'form'):#insert enters where there are new linebreaking elements
                elm.append("HTMLENTER_MAGIC_STR_ENTER")

        raw_text = soup.get_text().strip().replace("\n"," ") #remove enters in strings HTML renderers does not show these
        output = re.sub(r'\s{1,}', ' ', raw_text) #remove any place with more than one space HTML only renders one space even if there are many
        output = output.replace("HTMLENTER_MAGIC_STR_ENTER", "\n") #insert enters that come from HTML block elements
        output = re.sub(r'\s{3,}', "\n\n", output) #Remove execive nr of newlines
        output = output.replace("\n ", "\n").strip() #Remove space in beginin of newline and strip whitespace in the ends of the string
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

    def get_chapters_epub(self, speaker):

        self.author = self.book.get_metadata("DC", "creator")[0][0]
        self.title = self.book.get_metadata("DC", "title")[0][0]

        one_chapter_per_file = False
        chaper_file_index = {}
        for item in self.book.get_items():
            if type(item) == ebooklib.epub.EpubNcx:
                relative_file_dir =  str(Path(item.get_name()).parent)
                if relative_file_dir == '.':
                    relative_file_dir = ''
                else:
                    relative_file_dir += '/'
                root = etree.fromstring(item.get_content())
                navMap = root.find('.//{*}navMap')
                nav_points = navMap.findall('.//{*}navPoint')

                #extract part description and start and end positions
                part_list = []
                for nav_point in nav_points:
                    chapter_location = nav_point.find('.//{*}content').get("src")
                    chapter_desc = nav_point.find('.//{*}text').text
                    chapter_src = chapter_location.split("#")
                    if len(chapter_src) > 1:
                       chapter_file, chapter_id = chapter_src
                    else:
                       chapter_file, chapter_id = chapter_location, None

                    chapter_file = relative_file_dir+chapter_file
                    if len(part_list) != 0 and part_list[len(part_list)-1]['chapter_file'] == chapter_file:
                        part_list[len(part_list)-1]['chapter_end_id'] = chapter_id
                    part_list.append({'chapter_desc': chapter_desc, 'chapter_file': chapter_file, 'chapter_id': chapter_id, 'chapter_end_id': None})


                #extract part text from start to end
                for i, part in enumerate(part_list):
                    if part['chapter_file'] not in chaper_file_index:
                        chaper_file_index[part['chapter_file']] =  BeautifulSoup(self.book.get_item_with_href(part['chapter_file']).get_content(), "html.parser")
                    chapter_text = self.chap2text(chaper_file_index[part['chapter_file']], part['chapter_id'], part['chapter_end_id'])
                    self.chapters.append((chapter_text, part['chapter_desc']))

        #if there was no ncx file we asume the one file per chaper style of epub
        if len(chaper_file_index) == 0:
            one_chapter_per_file = True

        if one_chapter_per_file:
            for item in self.book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    self.chapters.append((self.chap2text(item.get_content()), None))

        for i in range(len(self.chapters)):
            text, header = self.chapters[i]

            if not self.skip_cleanup:
                text = self.prep_text(text)

            if len(text) < 150:
                # too short to bother with
                continue
            if header is not None:
                print(f"Part: {header}")
            print(f"Part No: {len(self.chapters_to_read) + 1}")
            print(f"Length: {len(text)}")
            if self.skipfootnotes:
                text = self.exclude_footnotes(text)
                #This drops everything after "Skip Notes" in a chapter
                text = text.split("Skip Notes")[0].strip()
            if self.skipfootnotes and text.startswith("Footnotes"):
                continue
            print(text[:256]+"\n")
            self.chapters_to_read.append(text)
            if header is not None:
                self.section_names.append(header)
            self.section_speakers.append(speaker)
        print(f"Number of chapters to read: {len(self.chapters_to_read)}")
        if self.end == 999:
            self.end = len(self.chapters_to_read)

    def get_chapters_text(self, speaker):
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
                if '%' in line: # Was a new speaker specified?
                    parts = line.split('%')
                    if speaker != parts[1].strip():
                        speaker = parts[1].strip()
                    self.section_speakers.append(speaker)
                    self.section_names.append(parts[0].lstrip("# ").strip())
                else:
                    self.section_speakers.append(speaker)
                    self.section_names.append(line.lstrip("# ").strip())
                print(f"Section speakers: {self.section_speakers}") if self.debug else None
                print(f"Section names: {self.section_names}") if self.debug else None
            sections = re.split(r"\n(?=#\s)", text)
            sections = [section.strip() for section in sections if section.strip()]
            for i, section in enumerate(sections):
                lines = section.splitlines()
                section = "\n".join(lines[1:])
                self.chapters_to_read.append(section.strip())
                print(f"Part: {len(self.chapters_to_read)}")
                print(f"{self.section_names[i]}")
                print(f"Speaker: {self.section_speakers[i]}")
                print(str(self.chapters_to_read[-1])[:256])
        else:
            self.section_speakers.append(speaker)
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

    
    def combine_sentences(self, sentences, length=1000):
        for sentence in sentences:
            yield sentence

    def export(self, format):
      allowed_formats = ["txt"]
      try:
        if format not in allowed_formats:
            raise ValueError(f"{format} not allowed export format")
        file_path = os.path.abspath(self.source)
        cover_image = self.get_epub_cover(file_path)
        image_path = None
        if cover_image is not None:
            image = Image.open(cover_image)
            image_filename = self.bookname + ".png"
            image_path = os.path.join(image_filename)
            image.save(image_path)
            print(f"Cover image saved to {image_path}")
        outputfile = f"{self.bookname}.{format}"
        self.check_for_file(outputfile)
        print(f"Exporting parts {self.start + 1} to {self.end} to {outputfile}")
        with open(outputfile, "w") as file:
            file.write(f"Title: {self.title}\n")
            file.write(f"Author: {self.author}\n\n")
            for partnum, i in enumerate(range(self.start, self.end)):
                file.write(f"\n# Part {partnum + 1}\n\n")
                file.write(self.chapters_to_read[i] + "\n")
      except ValueError as e:
        print(e)
        sys.exit()

    def get_epub_cover(self, epub_path):
        try:
            with zipfile.ZipFile(epub_path) as z:
                t = etree.fromstring(z.read("META-INF/container.xml"))
                rootfile_path =  t.xpath("/u:container/u:rootfiles/u:rootfile",
                                                    namespaces=namespaces)[0].get("full-path")

                t = etree.fromstring(z.read(rootfile_path))
                cover_meta = t.xpath("//opf:metadata/opf:meta[@name='cover']",
                                            namespaces=namespaces)
                if not cover_meta:
                    print("No cover image found.")
                    return None
                cover_id = cover_meta[0].get("content")

                cover_item = t.xpath("//opf:manifest/opf:item[@id='" + cover_id + "']",
                                                namespaces=namespaces)
                if not cover_item:
                    print("No cover image found.")
                    return None
                cover_href = cover_item[0].get("href")
                cover_path = os.path.join(os.path.dirname(rootfile_path), cover_href)

                return z.open(cover_path)
        except FileNotFoundError:
            print(f"Could not get cover image of {epub_path}")

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
        self.voice_samples = None
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
        else:
            voice_name = "-" + speaker
        self.output_filename = re.sub(".m4b", voice_name + ".m4b", self.output_filename)
        print(f"Saving to {self.output_filename}")
        self.check_for_file(self.output_filename)
        total_chars = self.get_length(self.start, self.end, self.chapters_to_read)
        print(f"Total characters: {total_chars}")
        if engine == "openai":
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

        if engine == "tts" and model_name == "tts_models/multilingual/multi-dataset/xtts_v2":
            #we are using coqui voice, so make smaller chunks
            sentance_chunk_length = 500
        else:
            sentance_chunk_length = 1000

        files = []
        position = 0
        start_time = time.time()
        print(f"Reading from {self.start + 1} to {self.end}")
        chapter_job_que = []
        for partnum, i in enumerate(range(self.start, self.end)):
            sentene_job_que = []
            outputwav = f"{self.bookname}-{i + 1}.wav"
            files.append(outputwav)
            if os.path.isfile(outputwav):
                print(f"{outputwav} exists, skipping to next chapter")
            else:
                tempfiles = []
                chapter_name = "Part " + str(partnum + 1)
                if len(self.section_names) > 0:
                    chapter_name = self.section_names[i].strip()

                if self.sayparts and len(self.section_names) == 0:
                    chapter = chapter_name + ". " + self.chapters_to_read[i]
                elif self.sayparts and len(self.section_names) > 0:
                    chapter = chapter_name + ".\n" + self.chapters_to_read[i]
                else:
                    chapter = self.chapters_to_read[i]

                if self.section_speakers[i] != None:
                    speaker = self.section_speakers[i]

                config = {
                    'speaker': speaker,
                    'language': self.language,
                    'model_name': model_name,
                    'debug': self.debug,
                    'device': self.device,
                    'minratio': self.minratio,
                    'engine_cl': None
                }

                if engine == "xtts":
                    config['voice_samples'] = self.voice_samples
                    config['xtts_model'] = self.xtts_model
                    config['no_deepspeed'] = self.no_deepspeed
                    config['engine_cl'] = XTTS

                elif engine == "openai":
                    config['api_key'] = self.openai
                    config['engine_cl'] = OpenAI_TTS
                    config['minratio'] = 0

                elif engine == "edge":
                    config['engine_cl'] = EdgeTTS
                    config['minratio'] = 0

                elif engine == "tts":
                    config['engine_cl'] = org_TTS

                    if config['model_name'] == "tts_models/en/vctk/vits":
                        config['minratio'] = 0


                sentences = sent_tokenize(chapter)
                #Drop any items that do NOT have at least one letter or number
                sentences = [s for s in sentences if any(c.isalnum() for c in s)]
                sentence_groups = list(self.combine_sentences(sentences, sentance_chunk_length))


                #tts_engine = config['engine_cl'](config)

                for x in range(len(sentence_groups)):
                    #skip if item is empty
                    if len(sentence_groups[x]) == 0:
                        continue
                    #skip if item has no characters or numbers
                    if not any(char.isalnum() for char in sentence_groups[x]):
                        continue
                    retries = 2
                    tempwav = "temp"+ str(partnum)+ "_" + str(x) + ".wav"
                    tempflac = tempwav.replace("wav", "flac")

                    if os.path.isfile(tempwav):
                        print(tempwav + " exists, skipping to next chunk")
                    else:
                        sentene_job_que.append((sentence_groups[x], tempwav))
                    tempfiles.append(tempwav)
                chapter_job_que.append(({'config': config, 'tempfiles': tempfiles, 'sentene_job_que': sentene_job_que, 'outputwav': outputwav, 'chapter': chapter_name}))

        print("initiating work:")
        
        if self.device == 'cuda':
            map_result = list(map(process_book_chapter, chapter_job_que))
        else:
            pool = mp.Pool(processes=self.threads)
            pool.map(process_book_chapter, chapter_job_que)
        files2 =[]
        for filename in files:
            if os.path.isfile(filename):
                files2.append(filename)
        files = files2
        outputm4a = self.output_filename.replace("m4b", "m4a")
        filelist = "filelist.txt"
        with open(filelist, "w") as f:
            for filename in files:
                filename = filename.replace("'", "'\\''")
                f.write(f"file '{filename}'\n")

        for i in self.audioformat:
            if i == "wav":
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
            elif i == "flac":
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
            elif i == "m4b":
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
                os.remove(outputm4a)
        if not self.debug: # Leave the files if debugging
            os.remove(filelist)
            os.remove(self.ffmetadatafile)
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
        "--threads",
        type=int,
        default=2,
        help="Number of threads to use, if using cuda threading is disabled as it does not make things faster since you are limited by the GPU",
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
        help="One or multiple audio format separate by comma for the output file (m4b [default], wav, flac)"
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

    xtts_arg_present = False
    if args.openai:
        args.engine = "openai"
    elif args.xtts:
        args.engine = "xtts"
        xtts_arg_present = True
    mybook = EpubToAudiobook(
        source=args.sourcefile,
        start=args.start,
        threads=args.threads,
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
    if args.engine == "openai" and args.speaker == None:
        speaker = "onyx"
    elif args.engine == "edge" and args.speaker == None:
        speaker = "en-US-AndrewNeural"
    elif args.engine == "tts" and args.speaker == None:
        speaker = "p335"
    elif args.engine == "xtts" and args.speaker == None and not xtts_arg_present:
        speaker = "Damien Black"
    else:
        speaker = args.speaker
    print(f"in main, Speaker is {speaker}")

    if mybook.sourcetype == "epub":
        mybook.get_chapters_epub(
            speaker=speaker,
        )
    else:
        mybook.get_chapters_text(
            speaker=speaker,
        )
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
        speaker=speaker,
        bitrate=args.bitrate,
    )
    if args.cover is not None:
        mybook.add_cover(args.cover)


if __name__ == "__main__":
    main()
