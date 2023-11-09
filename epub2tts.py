# Inspired by this medium article:
# https://medium.com/@zazazakaria18/turn-your-ebook-to-text-with-python-in-seconds-2a1e42804913
# and this post which just cleaned up what was in the medium article:
# https://xwiki.recursos.uoc.edu/wiki/mat00001ca/view/Research%20on%20Translation%20Technologies/Working%20with%20PDF%20files%20using%20Python/
#
# Usage: `epub2tts my-book.epub`
# To change speaker (ex p307 for a good male voice), add: `--speaker p307`
# To output in mp3 format instead of m4b, add: `--mp3`
# To skip reading any links, add: `--skip-links`
# Using `--scan` will list excerpts of each chapter, then exit. This is helpful
# for finding which chapter to start and end on if you want to skip bibliography, TOC, etc.
# To specify which chapter to start on (ex 3): `--start 3`
# To specify which chapter to end on (ex 20): `--end 20`
# To specify bitrate: --bitrate 30k
# Output will be an m4b or mp3 with each chapter read by Coqui TTS: https://github.com/coqui-ai/TTS

import os
import requests
import subprocess
import sys
import time
import wave


from bs4 import BeautifulSoup
import ebooklib
from ebooklib import epub
from newspaper import Article
from pydub import AudioSegment
from TTS.api import TTS
import torch, gc

# Verify if CUDA or mps is available and select it
if torch.cuda.is_available():
    device = "cuda" 
#except mps doesn't work right for this yet :(
#elif torch.backends.mps.is_available():
#    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

model_name = "tts_models/en/vctk/vits"
blacklist = ['[document]', 'noscript', 'header', 'html', 'meta', 'head', 'input', 'script']
ffmetadatafile = "FFMETADATAFILE"

usage = """
Usage: 
  EPUB: epub2tts my-book.epub
  TEXT: epub2tts my-book.txt
  URL:  epub2tts --url https://www.example.com/page --name example-page

Adding --scan will list excerpts of each chapter, then exit. This is
helpful for finding which chapter to start and end on if you want to
skip TOC, bibliography, etc.

To change speaker (ex p307 for a good male voice), add: --speaker p307
To output in mp3 format instead of m4b, add: --mp3
To skip reading any links, add: --skip-links
To specify which chapter to start on (ex 3): --start 3
To specify which chapter to end on (ex 20): --end 20
To specify bitrate (ex 30k): --bitrate 30k
"""

def chap2text(chap):
    output = ''
    soup = BeautifulSoup(chap, 'html.parser')
    if "--skip-links" in sys.argv:
        # Remove everything that is an href
        for a in soup.findAll('a', href=True):
            a.extract()
    #Always skip reading links that are just a number (footnotes)
    for a in soup.findAll('a', href=True):
        if a.text.isdigit():
            a.extract()
    text = soup.find_all(string=True)
    for t in text:
        if t.parent.name not in blacklist:
            output += '{} '.format(t)
    return output


def get_wav_duration(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        num_frames = wav_file.getnframes()
        frame_rate = wav_file.getframerate()
        duration = num_frames / frame_rate
        duration_milliseconds = duration * 1000
        return int(duration_milliseconds)
    

def gen_ffmetadata(files, title, author):
    chap = 1
    start_time = 0
    with open(ffmetadatafile, "w") as file:
        file.write(";FFMETADATA1\n")
        file.write("ARTIST=" + str(author) + "\n")
        file.write("ALBUM=" + str(title) + "\n")
        for file_name in files:
            duration = get_wav_duration(file_name)
            file.write("[CHAPTER]\n")
            file.write("TIMEBASE=1/1000\n")
            file.write("START=" + str(start_time) + "\n")
            file.write("END=" + str(start_time + duration) + "\n")
            file.write("title=Part " + str(chap) + "\n")
            chap += 1
            start_time += duration

def get_bookname():
    bookname = ''
    for i, arg in enumerate(sys.argv):
        if arg.endswith('.txt') or arg.endswith('.epub'):
            bookname = arg
    if ("--url" in sys.argv) and ("--name" in sys.argv):
        index = sys.argv.index("--name")
        bookname = sys.argv[index + 1] + ".url"
    if len(bookname) > 0:
        print(f"Book filename: {bookname}")
        return(bookname)
    elif ("--url" in sys.argv) and ("--name" in sys.argv):
        return(".url")
    else:
        print(usage)
        sys.exit()

def get_url():
    index = sys.argv.index("--url")
    url = sys.argv[index + 1]
    return(url)

def get_speaker():
    if "--speaker" in sys.argv:
        index = sys.argv.index("--speaker")
        speaker_used = sys.argv[index + 1]    
    else:
        speaker_used = "p335"
    print(f"Speaker: {speaker_used}")
    return(speaker_used)

def get_bitrate():
    if "--bitrate" in sys.argv:
        index = sys.argv.index("--bitrate")
        bitrate = sys.argv[index + 1]    
    else:
        bitrate = "69k"
    print(f"Bitrate: {bitrate}")
    return(bitrate)

def get_chapters_epub(book, bookname):
    chapters = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            chapters.append(item.get_content())

    chapters_to_read = []
    for i in range(len(chapters)):
        #strip some characters that might have caused TTS to choke
        text = chap2text(chapters[i])
        text = text.translate({ord(c): None for c in '[]*“”"\''})
        if len(text) < 150:
            #too short to bother with
            continue
        outputwav = str(i)+"-"+bookname.split(".")[0]+".wav"
        print(outputwav + " Length: " + str(len(text)))
        print("Part: " + str(len(chapters_to_read)+1))
        print(text[:256])
        chapters_to_read.append(text)  # append the last piece of text (shorter than max_len)
    print("Number of chapters to read: " + str(len(chapters_to_read)))
    if "--scan" in sys.argv:
        sys.exit()
    return(chapters_to_read)

def get_chapters_text(text):
    chapters_to_read = []
    max_len = 50000
    while len(text) > max_len:
        pos = text.rfind(' ', 0, max_len)  # find the last space within the limit
        chapters_to_read.append(text[:pos])
        print("Part: " + str(len(chapters_to_read)))
        print(str(chapters_to_read[-1])[:256])
        text = text[pos+1:]  # +1 to avoid starting the next chapter with a space
    chapters_to_read.append(text)
    return(chapters_to_read)

def get_text(bookname):
    with open(bookname, 'r') as file:
        text = file.read()
    return(text)

def get_url_text(url):
    article = Article(url)
    article.download()
    article.parse()
    return(article.text)

def get_length(start, end, chapters_to_read):
    total_chars = 0
    for i in range(start, end):
        total_chars += len(chapters_to_read[i])
    return(total_chars)

def get_start():
# There are definitely better ways to handle arguments, this should be fixed
    if "--start" in sys.argv:
        start = int(sys.argv[sys.argv.index("--start") + 1]) - 1
    else:
        start = 0
    return(start)

def get_end(chapters_to_read):
# There are definitely better ways to handle arguments, this should be fixed
    if "--end" in sys.argv:
        end = int(sys.argv[sys.argv.index("--end") + 1])
    else:
        end = len(chapters_to_read)
    return(end)

def main():
    bookname = get_bookname() #detect .txt, .epub or https
    booktype = bookname.split('.')[-1]
    speaker_used = get_speaker()
    if booktype == "epub":
        book = epub.read_epub(bookname)
        chapters_to_read = get_chapters_epub(book, bookname)
    elif booktype == "txt":
        print("Detected TEXT for file type, --scan, --start and --end will be ignored")
        text = get_text(bookname)
        chapters_to_read = get_chapters_text(text)
    elif booktype == "url":
        print("Detected URL for file type, --scan, --start and --end will be ignored")
        url = get_url()
        text = get_url_text(url)
        print("Name: " + bookname)
        print(text)
        while True:
            user_input = input("Look good, continue? (y/n): ")
            if user_input.lower() not in ['y', 'n']:
                print("Invalid input. Please enter y for yes or n for no.")
            elif user_input.lower() == 'n':
                sys.exit()
            else:
                print("Continuing...")
                break
        chapters_to_read = get_chapters_text(text)
    start = get_start()
    end = get_end(chapters_to_read)
    total_chars = get_length(start, end, chapters_to_read)
    files = []
    position = 0
    start_time = time.time()
    tts = TTS(model_name).to(device)
    for i in range(start, end):
        outputwav = bookname.split(".")[0]+"-"+str(i+1)+".wav"
        print("Reading " + str(i))
        if os.path.isfile(outputwav):
            print(outputwav + " exists, skipping to next chapter")
        else:
            tts.tts_to_file(text = chapters_to_read[i], speaker = speaker_used, file_path = outputwav)
        files.append(outputwav)
        position += len(chapters_to_read[i])
        percentage = (position / total_chars) *100
        print(f"{percentage:.2f}% spoken so far.")
        elapsed_time = time.time() - start_time
        chars_remaining = total_chars - position
        estimated_total_time = elapsed_time / position * total_chars
        estimated_time_remaining = estimated_total_time - elapsed_time
        print(f"Elapsed: {int(elapsed_time / 60)} minutes, ETA: {int((estimated_time_remaining) / 60)} minutes")

        # Clean GPU cache to have it all available for next step
        if device == 'cuda':
            gc.collect()
            torch.cuda.empty_cache()
        else:
            pass


    #Load all WAV files and concatenate into one object
    wav_files = [AudioSegment.from_wav(f"{f}") for f in files]
    concatenated = sum(wav_files)
    if "--mp3" in sys.argv:
        outputmp3 = bookname.split(".")[0]+"-"+speaker_used+".mp3"
        concatenated.export(outputmp3, format="mp3", parameters=["-write_xing", "0", "-filter:a", "speechnorm=e=6.25:r=0.00001:l=1"])
    else:
        outputm4a = bookname.split(".")[0]+"-"+speaker_used+".m4a"
        outputm4b = outputm4a.replace("m4a", "m4b")
        bitrate = get_bitrate()
        concatenated.export(outputm4a, format="ipod", bitrate=bitrate)
        if booktype == 'epub':
            author = book.get_metadata('DC', 'creator')[0][0]
            title = book.get_metadata('DC', 'title')[0][0]
        else:
            author = "Unknown"
            title = bookname
        gen_ffmetadata(files, title, author)
        ffmpeg_command = ["ffmpeg","-i",outputm4a,"-i",ffmetadatafile,"-map_metadata","1","-codec","copy",outputm4b]
        subprocess.run(ffmpeg_command)
        os.remove(ffmetadatafile)
        os.remove(outputm4a)
    #cleanup, delete the wav files we no longer need
    for f in files:
        os.remove(f)

if __name__ == '__main__':
    main()
