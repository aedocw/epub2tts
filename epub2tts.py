#Source: https://medium.com/@zazazakaria18/turn-your-ebook-to-text-with-python-in-seconds-2a1e42804913
#and https://xwiki.recursos.uoc.edu/wiki/mat00001ca/view/Research%20on%20Translation%20Technologies/Working%20with%20PDF%20files%20using%20Python/
#which is just a ripoff from medium article
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import sys
import codecs


def chap2text(chap):
    output = ''
    soup = BeautifulSoup(chap, 'html.parser')
    text = soup.find_all(text=True)
    for t in text:
        if t.parent.name not in blacklist:
            output += '{} '.format(t)
    return output
    
blacklist = [   '[document]',   'noscript', 'header',   'html', 'meta', 'head','input', 'script',   ]

bookname=sys.argv[1]
outputname=sys.argv[2]

#outputfile=codecs.open(outputname,"w",encoding="utf-8")

book = epub.read_epub(bookname)

chapters = []
for item in book.get_items():
    if item.get_type() == ebooklib.ITEM_DOCUMENT:
        chapters.append(item.get_content())

#for chapter in chapters:
#    text=chap2text(chapter)
#    outputfile.write(text+"\n")
for i in range(len(chapters)):
    text=chap2text(chapters[i])
    outputname=str(i)+".txt"
    outputfile=codecs.open(outputname,"w",encoding="utf-8")
    outputfile.write(text+"\n")
