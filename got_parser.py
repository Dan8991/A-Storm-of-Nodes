import PyPDF2 as pdf
import enchant
import sys
from functools import reduce
import re


class Chapter(object):

    def __init__(self, title, words):
        self.title = title
        self.words = words

    '''
    the title of the chapter in the text must be upper_case
    text = array of strings containing one line in every element of the array
    '''

    def __str__(self):
        return "chapter name = {}\nchapter length = {}".format(self.title, len(self.words))

    def get_non_english_words(self, non_names=[], bastard_names=[]):
        dictionary = enchant.Dict("en_UK")

        words = self.words.split()
        non_english_words = []
        last_non_eng = False

        for word in words:
            if (word in bastard_names) or ((not dictionary.check(word.lower())) and word[0].isupper() and (word not in non_names)):
                if last_non_eng:
                    non_english_words[-1] += " " + word
                else:
                    non_english_words.append(word)
                last_non_eng = True
            else:
                last_non_eng = False
        return set(non_english_words)


def get_chapters_from_text(text):

    # initializing data supposing that the first line is a title
    title = ""
    words = ""
    chapters = []

    for line in text:
        if line.isupper() and (len(line.split()) == 1):
            title = line
            chapters.append(Chapter(title, words))
            words = ""
        else:
            words += line

    return chapters


def parse_punctuation(s):

    # changing all puntuation to a comma and separating it from words
    parsed = re.sub(r"[\.!?,”“…;]", " , ", s)

    # removing multiple spaces
    parsed = " ".join(parsed.split()) + "\n"

    parsed = re.sub(r"Storm End", "StormEnd", parsed)

    # removing saxons genitive and other similar things because it is annoying for parsing
    parsed = re.sub(r"(’s|’n|’d|’i)", "", parsed)

    parsed = re.sub(r"—", " ", parsed)

    parsed = re.sub(r"(-\n|—\n)", "", parsed)

    return parsed


def get_names(file_name):
    return set([line.replace("\n", "") for line in open(file_name, 'r')])


if (len(sys.argv) > 1) and (sys.argv[1] == "parse"):

    with open("storm_of_swords.txt", "r") as f:
        lines = [parse_punctuation(line) for line in f]

    with open("storm_of_swords.txt", "w") as f:
        f.writelines(lines)

pdf_file = open("storm_of_swords.txt", 'r')

chapters = get_chapters_from_text([line for line in pdf_file])

non_names = get_names("non_character_names.txt")
bastard_names = get_names("extra_names.txt")

if sys.argv[-1] == "-1":
    unique = list(reduce(lambda x, y: x.union(y), [chapter.get_non_english_words(
        non_names=non_names, bastard_names=bastard_names) for chapter in chapters]))
else:
    unique = list(chapters[int(sys.argv[-1])].get_non_english_words(
        non_names=non_names, bastard_names=bastard_names))

unique.sort()
print("Unique characters found:", len(unique))