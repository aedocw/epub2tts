# Inspired by this medium article:
# https://medium.com/@zazazakaria18/turn-your-ebook-to-text-with-python-in-seconds-2a1e42804913
# and this post which just cleaned up what was in the medium article:
# https://xwiki.recursos.uoc.edu/wiki/mat00001ca/view/Research%20on%20Translation%20Technologies/Working%20with%20PDF%20files%20using%20Python/
#
# This script takes in an epub as only argument, then previews the first 256 characters
# from each chapter - enter y to include that chapter, n to skip, and q when you are
# done adding chapters. This preview step is required because most ebooks have a ton
# of content up front that you don't want read to you.
# Output will be mp3's for each chapter, read by Coqui TTS: https://github.com/coqui-ai/TTS

import os
import sys


from bs4 import BeautifulSoup
import ebooklib
from ebooklib import epub
from pydub import AudioSegment
from TTS.api import TTS


model_name = "tts_models/en/vctk/vits"
blacklist = ['[document]', 'noscript', 'header', 'html', 'meta', 'head', 'input', 'script']

def chap2text(chap):
    output = ''
    soup = BeautifulSoup(chap, 'html.parser')
    # Remove everything that is an href
    for a in soup.findAll('a', href=True):
        a.extract()
    text = soup.find_all(text=True)
    for t in text:
        if t.parent.name not in blacklist:
            output += '{} '.format(t)
    return output


def main():

    # TODO: accept URL to fetch book directly from project gutenberg
    try:
        bookname = sys.argv[1]
    except:
        print("Please specify epub to read as first argument")
        print("Optionally, add a second argument specifying speaker to use")
        sys.exit()

    book = epub.read_epub(bookname)

    chapters = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            chapters.append(item.get_content())

    chapters_to_read = []
    for i in range(len(chapters)):
        #strip some characters that might have caused TTS to choke
        text = chap2text(chapters[i])
        text = text.translate({ord(c): None for c in '[]'})
        if len(text) < 150:
            #too short to bother with
            continue
        outputwav = str(i)+"-"+bookname.split(".")[0]+".wav"
        print(outputwav + " Length: " + str(len(text)))
        print(text[:256])
        if len(text) > 100000:
            # too long, split in four
            # TODO: Find what size actually causes problems, and chunk this up
            # into appropriate sizes rather than just blindly chopping into 1/4ths
            q = len(text)//4
            chapters_to_read.append(text[:q])
            chapters_to_read.append(text[q:q*2])
            chapters_to_read.append(text[q*2:q*3])
            chapters_to_read.append(text[q*3:])
        else:
            chapters_to_read.append(text)

    print("Number of chapters to read: " + str(len(chapters_to_read)))

    files = []
    # check for optional speaker parameter
    speakers = ["p261", "p225", "p294", "p347", "p238", "p234", "p248", "p335", "p245", "p326","p302"]
    speaker_used = "p335"
    if len(sys.argv) >= 3:
        speaker_param = sys.argv[2]
        if speaker_param in speakers:
            speaker_used = speaker_param

    for i in range(len(chapters_to_read)):
        tts = TTS(model_name)
        text = chap2text(chapters_to_read[i])
        outputwav = bookname.split(".")[0]+"-"+str(i+1)+".wav"
        print("Reading " + str(i))
        # Seems TTS can only output in wav? convert to mp3 aftwarwards
        tts.tts_to_file(text = chapters_to_read[i], speaker = speaker_used, file_path = outputwav)
        files.append(outputwav)

    #Load all WAV files and concatenate into one mp3
    wav_files = [AudioSegment.from_wav(f"{f}") for f in files]
    concatenated = sum(wav_files)
    outputmp3=bookname.split(".")[0]+"-"+speaker_used+".mp3"
    concatenated.export(outputmp3, format="mp3")
    #cleanup, delete the wav files we no longer need
    for f in files:
        os.remove(f)

if __name__ == '__main__':
    main()
