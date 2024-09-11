import argparse
import asyncio
import multiprocessing as mp
import os
import pkg_resources
import re
import subprocess
import sys
import time
import warnings
import zipfile

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
from tqdm import tqdm
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


async def edgespeak(sentence, speaker, filename):
    communicate = edge_tts.Communicate(sentence, speaker)
    await communicate.save(filename)

whispermodel = whisper.load_model("tiny")

def compare(text, wavfile, debug):
    result = whispermodel.transcribe(wavfile)
    text = re.sub(" +", " ", text).lower().strip()
    ratio = fuzz.ratio(text, result["text"].lower())
    print(f"Transcript: {result['text'].lower()}") if debug else None
    print(
        f"Text to transcript comparison ratio: {ratio}"
    ) if debug else None
    return ratio

def read_chunk_xtts(sentences, wav_file_path, debug, modl, language):
    # takes list of sentences to read, reads through them and saves to file
    t0 = time.time()
    wav_chunks = []
    sentence_list = sent_tokenize(sentences)
    for i, sentence in enumerate(sentence_list):
        # Run TTS for each sentence
        if debug:
            print(
                sentence
            )
            with open("debugout.txt", "a") as file: file.write(f"{sentence}\n")
        chunks = modl['model'].inference_stream(
            sentence,
            language,
            modl['gpt_cond_latent'],
            modl['speaker_embedding'],
            stream_chunk_size=60,
            temperature=0.60,
            repetition_penalty=20.0,
            enable_text_splitting=True,
        )
        for j, chunk in enumerate(chunks):
            if i == 0:
                print(
                    f"Time to first chunck: {time.time() - t0}"
                ) if debug else None
            print(
                f"Received chunk {i} of audio length {chunk.shape[-1]}"
            ) if debug else None
            wav_chunks.append(
                chunk.to(device=modl['device'])
            )  # Move chunk to available device
        # Add a short pause between sentences (e.g., X.XX seconds of silence)
        if i < len(sentence_list):
            silence_duration = int(24000 * .6)
            silence = torch.zeros(
                (silence_duration,), dtype=torch.float32, device=modl['device']
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



def load_engine(engine, xtts_model, use_deepspeed, speaker, voice_samples, model_name, openai):
    ret = {}
    if torch.cuda.is_available():
        ret['device'] = "cuda"
    else:
        ret['device'] = "cpu"

    if engine == "xtts":
        if (
            torch.cuda.is_available()
            and torch.cuda.get_device_properties(0).total_memory > 3500000000
        ):
            print("Using GPU")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory}")
            ret['device'] = "cuda"
        else:
            print("Not enough VRAM on GPU or CUDA not found. Using CPU")
            ret['device'] = "cpu"

        print("Loading model: " + xtts_model)
        # This will trigger model load even though we might not use tts object later
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(ret['device'])
        tts = ""
        config = XttsConfig()
        model_json = xtts_model + "/config.json"
        config.load_json(model_json)
        ret['model'] = Xtts.init_from_config(config)
        ret['model'].load_checkpoint(
            config, checkpoint_dir=xtts_model, use_deepspeed=use_deepspeed
        )

        if ret['device'] == "cuda":
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory}")
            ret['model'].cuda()

        print("Computing speaker latents...")
        if speaker == None:
            ret['gpt_cond_latent'], ret['speaker_embedding'] = ret['model'].get_conditioning_latents(audio_path=voice_samples)
        else: #using Coqui speaker
            ret['gpt_cond_latent'], ret['speaker_embedding'] = ret['model'].speaker_manager.speakers[speaker].values()

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
        client = OpenAI(api_key=openai)
    elif engine == "edge":
        print("Engine is Edge TTS")
    else:
        print(f"Engine is TTS, model is {model_name}")
        ret['tts'] = TTS(model_name).to(ret['device'])
    return ret




def process_book_part(partnum, i, sentence_groups, length, outputwav, engine, model_name, minratio, debug, speaker, language, xtts_model, use_deepspeed, voice_samples, openai):

    modl = load_engine(engine, xtts_model, use_deepspeed, speaker, voice_samples, model_name, openai)
    print("process_book_part", partnum, i, sentence_groups, length, outputwav)
    tempfiles = []
    for x in tqdm(range(len(sentence_groups))):
        #skip if item is empty
        if len(sentence_groups[x]) == 0:
            continue
        #skip if item has no characters or numbers
        if not any(char.isalnum() for char in sentence_groups[x]):
            continue
        retries = 2
        tempwav = "part-"+str(partnum)+"-temp" + str(x) + ".wav"
        tempflac = tempwav.replace("wav", "flac")
        if os.path.isfile(tempwav):
            print(tempwav + " exists, skipping to next chunk")
        else:
            ratio = 0
            while retries > 0:
                try:
                    if engine == "xtts":
                        if language != "en":
                                sentence_groups[x] = sentence_groups[x].replace(".", ",")
                        read_chunk_xtts(sentence_groups[x], tempwav, debug, modl, language)
                    elif engine == "openai":
                        minratio = 0
                        response = client.audio.speech.create(
                            model="tts-1",
                            voice=speaker.lower(),
                            input=sentence_groups[x],
                        )
                        response.stream_to_file(tempwav)
                    elif engine == "edge":
                        minratio = 0
                        if debug:
                            print(
                                sentence_groups[x]
                            )
                        asyncio.run(edgespeak(sentence_groups[x], speaker, tempwav))
                    elif engine == "tts":
                        print("tts_to_file", modl['device'])
                        if model_name == "tts_models/en/vctk/vits":
                            minratio = 0
                            # assume we're using a multi-speaker model
                            if debug:
                                print(
                                    sentence_groups[x]
                                )
                                with open("debugout.txt", "a") as file: file.write(f"{sentence_groups[x]}\n")
                            modl['tts'].tts_to_file(
                                text=sentence_groups[x],
                                speaker=speaker,
                                file_path=tempwav,
                            )
                        else:
                            if debug:
                                print(
                                    sentence_groups[x]
                                )
                                with open("debugout.txt", "a") as file: file.write(f"{sentence_groups[x]}\n")
                            modl['tts'].tts_to_file(
                                text=sentence_groups[x], file_path=tempwav
                            )
                    if minratio == 0:
                        print("Skipping whisper transcript comparison") if debug else None
                        ratio = minratio
                    else:
                        ratio = compare(sentence_groups[x], tempwav, debug)
                    if ratio < minratio:
                        raise Exception(
                            f"Spoken text did not sound right - {ratio}"
                        )
                    break
                except Exception as e:
                    print(e)
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
    return outputwav
    #files.append(outputwav)
    #position += len(self.chapters_to_read[i])
    #percentage = (position / total_chars) * 100
    #print(f"{percentage:.2f}% spoken so far.")
    #elapsed_time = time.time() - start_time
    #chars_remaining = total_chars - position
    #estimated_total_time = elapsed_time / position * total_chars
    #estimated_time_remaining = estimated_total_time - elapsed_time
    #print(
    #    f"Elapsed: {int(elapsed_time / 60)} minutes, ETA: {int((estimated_time_remaining) / 60)} minutes"
    #)
    #gc.collect()
    #torch.cuda.empty_cache()



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
        if self.skiplinks:
            # Remove everything that is an href
            for a in soup.findAll("a", href=True):
                a.extract()
        # Always skip reading links that are just a number (footnotes)
        for a in soup.findAll("a", href=True):
            if not any(char.isalpha() for char in a.text):
                a.extract()
        text = soup.find_all(string=True)
        last_paragraph = None
        children_2_skip = None
        for t in text:
            if end_element_id is not None and t.parent.get('id') == end_element_id:
                break

            #skip if element is child of a previously skiped element
            if children_2_skip is not None and t.parent in children_2_skip:
                continue

            elm_epub_type = t.parent.get('epub:type')
            if elm_epub_type is not None and elm_epub_type in skip_epub_types: #Dont read the page numbers or annotations
                children_2_skip = t.parent.find_all(True)
                continue

            if t.parent.name not in blacklist:
                txt = "{}".format(t).strip()
                if txt != "":
                    output += txt+" "

            if t.parent.name in ('p', 'h1', 'h2', 'h3', 'h4', 'h5', 'div', 'li', 'ul', 'tr'):#insert enters where there are new linebreaking elements
                if last_paragraph is not None and last_paragraph != t.parent and len(output) > 0 and output[-1] != "\n":
                    output += "\n"
                last_paragraph = t.parent

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
                root = etree.fromstring(item.get_content())
                navMap = root.find('.//{*}navMap')
                nav_points = navMap.findall('.//{*}navPoint')

                #extract part description and start and end positions
                part_list = []
                for nav_point in nav_points:
                    chapter_location = nav_point.find('.//{*}content').get("src")
                    chapter_desc = nav_point.find('.//{*}text').text
                    chapter_file, chapter_id = chapter_location.split("#")
                    if len(part_list) != 0 and part_list[len(part_list)-1]['chapter_file'] == chapter_file:
                        part_list[len(part_list)-1]['chapter_end_id'] = chapter_id
                    part_list.append({'chapter_desc': chapter_desc, 'chapter_file': chapter_file, 'chapter_id': chapter_id, 'chapter_end_id': None})


                #extract part text from start to end
                for i, part in enumerate(part_list):
                    if part['chapter_file'] not in chaper_file_index:
                        chaper_file_index[part['chapter_file']] =  BeautifulSoup(self.book.get_item_with_href("Content/"+part['chapter_file']).get_content(), "html.parser")
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
        files = []
        position = 0
        start_time = time.time()
        print(f"Reading from {self.start + 1} to {self.end}")

        voice_samples = None
        if hasattr(self, 'voice_samples'):
            voice_samples = self.voice_sample
        openai = None
        if hasattr(self, 'openai'):
            openai = self.openai

        if self.no_deepspeed:
            use_deepspeed = False
        else:
            use_deepspeed = self.is_installed("deepspeed")
        job_que = []
        for partnum, i in enumerate(range(self.start, self.end)):#this should be able to  be paralized
            outputwav = f"{self.bookname}-{i + 1}.wav"
            if os.path.isfile(outputwav):
                print(f"{outputwav} exists, skipping to next chapter")
            else:
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

                speaker = None
                if self.section_speakers[i] != None:
                    speaker = self.section_speakers[i]

                job_que.append((partnum, i, sentence_groups, length, outputwav, engine, model_name, self.minratio, self.debug, speaker, self.language, self.xtts_model, use_deepspeed, voice_samples, openai))
                #job_que.append((partnum, i, length, outputwav, engine, model_name))

        print("job que created, start jobs:")
        print(job_que[:1])
        mp.set_start_method('spawn')
        nr_procceses = os.cpu_count()
        nr_procceses = 4
        pool = mp.Pool(processes=nr_procceses)
        files = pool.starmap(process_book_part, job_que)
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
    if args.engine == "openai" and args.speaker == None:
        speaker = "onyx"
    elif args.engine == "edge" and args.speaker == None:
        speaker = "en-US-AndrewNeural"
    elif args.engine == "tts" and args.speaker == None:
        speaker = "p335"
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
