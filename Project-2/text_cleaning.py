#Removing HTML Tags
from bs4 import BeautifulSoup

def remove_html_tags(text):
    print('Removing HTML Tags, the text can be as big as entire wepage')
    return BeautifulSoup(text, 'html.parser').get_text()


#Removing Accented characters
import unicodedata

def remove_accented_chars(text):
    print("Removing accented characters, which convert résumé to resumve")
    new_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return new_text

#Expanding Contractions
from contractions import CONTRACTION_MAP # from contractions.py
import re 
# function to expand contractions
def expand_contractions(text, map=CONTRACTION_MAP):
    pattern = re.compile('({})'.format('|'.join(map.keys())), flags=re.IGNORECASE|re.DOTALL)
    def get_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded = map.get(match) if map.get(match) else map.get(match.lower())
        expanded = first_char+expanded[1:]
        return expanded 
    new_text = pattern.sub(get_match, text)
    new_text = re.sub("'", "", new_text)
    return new_text


# "Well this was fun! See you at 7:30, What do you think!!? #$@@9318@ 🙂🙂🙂" ==> 'Well this was fun See you at  What do you think  '


import re

def remove_special_characters(text, remove_digits=False):
    print("Removing special characters like smileys from the text")
    pattern = r'[^a-zA-Z0-9\s]' if not remove_digits else r'[^a-zA-Z\s]'
    text = re.sub(pattern, '', text)
    return text

