from bs4 import BeautifulSoup
from sys import argv
import re
import codecs

soup = BeautifulSoup(codecs.open(argv[1], 'r', 'utf-8'), 'lxml')
text = soup.get_text()
text = re.sub(r'\n+', '\n', text, flags = re.MULTILINE)

f = codecs.open(argv[2], 'w', 'utf-8')
f.write(text)
f.close()
