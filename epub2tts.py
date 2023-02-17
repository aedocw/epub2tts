#Inspired by this medium article:
#https://medium.com/@zazazakaria18/turn-your-ebook-to-text-with-python-in-seconds-2a1e42804913
#and this post which just cleaned up what was in the medium article:
#https://xwiki.recursos.uoc.edu/wiki/mat00001ca/view/Research%20on%20Translation%20Technologies/Working%20with%20PDF%20files%20using%20Python/
#
#This script takes in an epub as only argument, then previews the first 256 characters
#from each chapter - enter y to include that chapter, n to skip, and q when you are
#done adding chapters. This preview step is required because most ebooks have a ton
#of content up front that you don't want read to you.
#Output will be mp3's for each chapter, read by Coqui TTS: https://github.com/coqui-ai/TTS

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import sys
import subprocess
import pydub
from pydub import AudioSegment

#From https://github.com/coqui-ai/TTS
from TTS.api import TTS
model_name = "tts_models/en/vctk/vits"

def chap2text(chap):
    output = ''
    soup = BeautifulSoup(chap, 'html.parser')
    text = soup.find_all(text=True)
    for t in text:
        if t.parent.name not in blacklist:
            output += '{} '.format(t)
    return output
    
blacklist = [   '[document]',   'noscript', 'header',   'html', 'meta', 'head','input', 'script',   ]

#TODO: accept URL to fetch book directly from project gutenberg
try:
    bookname=sys.argv[1]
except:
    print("Please specify epub to read as first argument")
    exit()

book = epub.read_epub(bookname)

chapters = []
for item in book.get_items():
    if item.get_type() == ebooklib.ITEM_DOCUMENT:
        chapters.append(item.get_content())

chapters_to_read = []
for i in range(len(chapters)):
    text=chap2text(chapters[i])
    outputwav=str(i)+"-"+bookname.split(".")[0]+".wav"
    print(outputwav + " Length: " + str(len(text)))
    print(text[:256])
    include = input("\nInclude? (y/n/q): ")
    if include == 'y':
        chapters_to_read.append(text)
    if include == 'q':
        break

print("Number of chapters to read: " + str(len(chapters_to_read)))

# Init TTS
tts = TTS(model_name)

for i in range(len(chapters_to_read)):
    text=chap2text(chapters_to_read[i])
    outputwav=str(i+1)+"-"+bookname.split(".")[0]+".wav"
    outputmp3=str(i+1)+"-"+bookname.split(".")[0]+".mp3"
    tts.tts_to_file(text=chapters_to_read[i], speaker='p307', file_path=outputwav)
    #Seems TTS can only output in wav? convert to mp3 aftwarwards
    wav = AudioSegment.from_file(outputwav)
    wav.export(outputmp3, format="mp3")
    subprocess.call(['rm', '-f', outputwav])