import PyPDF2 as pdf
import enchant
import sys
from functools import reduce
import re
import json


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

    def get_non_english_words(self, non_names=[], extra_names=[], return_word_numbers=False):
        dictionary = enchant.Dict("en_UK")

        words = self.words.split()
        non_english_words = []
        last_non_eng = False

        word_count = 0
        word_numbers = []

        for word in words:
            word_count += 1
            if (word in extra_names) or ((not dictionary.check(word.lower())) and word[0].isupper() and (word not in non_names)):
                if last_non_eng:
                    non_english_words[-1] += " " + word
                else:
                    non_english_words.append(word)
                    word_numbers.append(word_count)
                last_non_eng = True
            else:
                last_non_eng = False

        if return_word_numbers:
            return non_english_words, word_numbers
        else:
            return non_english_words

    '''
        characters = dictionary that has as keys all the extra names and as value the characters that theese keys refer to
        characters_id = dictionary that has the character name as key and it's unique id as value
        link_length = maximum distance in words for which two characters can be considered linked
        non_names = words that are not in the vocabulary that are not actually characters names
        extra_names = words that are in the vocabulary that are actually characters names
        return = all characters connections in the chapter
    '''

    def get_chapter_connections(self, characters, characters_id, link_length=100, non_names=[], extra_names=[]):

        extra_words, word_numbers = self.get_non_english_words(
            non_names=non_names, extra_names=extra_names, return_word_numbers = True)

        valid_char = [(characters_id[c], word_num) for char, word_num in zip(
            extra_words, word_numbers) if char in characters for c in characters[char]]

        protagonist_name = re.sub(r"\n", "", self.title[0] + self.title[1:].lower())

        protagonist_id = -1
        if protagonist_name != "Prologue" and protagonist_name != "Epilogue": 
            protagonist_id = characters_id[characters[protagonist_name][0]]

        current_index = 0
        connections = []
    
        for i in range(len(valid_char)):
            
            if protagonist_id != -1:
                connections.append((protagonist_id, valid_char[i][0]))

            while (valid_char[i][1] - link_length) > valid_char[current_index][1]:
                current_index += 1

            for j in range(current_index, i):
                connections.append((valid_char[j][0], valid_char[i][0]))
            

        return connections


def get_chapters_from_text(text):

    # initializing data supposing that the first line is a title
    title = "PROLOGUE"
    words = ""
    chapters = []
    i = 1
    for line in text[1:]:
        if line.isupper() and (len(line.split()) == 1):
            i+=1
            chapters.append(Chapter(title, words))
            title = line
            words = ""
        else:
            words += line

    chapters.append(Chapter(title, words))
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

# this function is needed in order to get a basic idea of which characters are actually the same one


def compile_dict(character, characters_dict):
    words_list = character.split()
    for word in words_list:
        if word in characters_dict:
            characters_dict[word].append(character)
        else:
            characters_dict[word] = [character]


def convert_dictionary(dictionary):
    converted = {}

    houses = set([line.replace("\n", "") for line in open('houses.txt', 'r')])

    for key in dictionary:
        if key in houses:
            converted[key] = set(dictionary[key])
        else:
            for value in dictionary[key]:
                if value in converted:
                    converted[value].union(set(dictionary[key]))
                else:
                    converted[value] = set(dictionary[key])

    for key in converted:
        converted[key] = list(converted[key])

    return converted


def get_characters_dictionary(dictionary):
    ret = {}
    for key in dictionary:
        for val in dictionary[key]:
            if val not in ret:
                ret[val] = [key]
            else:
                ret[val].append(key)
    return ret


if (len(sys.argv) > 1) and (sys.argv[1] == "parse"):

    with open("storm_of_swords.txt", "r") as f:
        lines = [parse_punctuation(line) for line in f]

    with open("storm_of_swords.txt", "w") as f:
        f.writelines(lines)

pdf_file = open("storm_of_swords.txt", 'r')

chapters = get_chapters_from_text([line for line in pdf_file])

non_names = get_names("non_character_names.txt")
extra_names = get_names("extra_names.txt")

if sys.argv[-1] == "-1":
    uniques = list(set(reduce(lambda x, y: set(x).union(set(y)), [chapter.get_non_english_words(
        non_names=non_names, extra_names=extra_names) for chapter in chapters])))
else:
    uniques = list(set(chapters[int(sys.argv[-1])].get_non_english_words(
        non_names=non_names, extra_names=extra_names)))

uniques.sort()
print("Unique characters found:", len(uniques))
print(uniques)


characters_nicknames = json.load(open("characters.json", "r"))
nicknames_characters = get_characters_dictionary(characters_nicknames)

print("Total characters found:", len(characters_nicknames.keys()))

characters_id = {key: value for value, key in enumerate(characters_nicknames.keys())}

if sys.argv[-1] == "-1":
    connections = reduce(lambda x, y: x + y, [x.get_chapter_connections(
        nicknames_characters, characters_id, non_names=non_names, extra_names=extra_names) for x in chapters])
else:
    connections = chapters[int(sys.argv[-1])].get_chapter_connections(
        nicknames_characters, characters_id, non_names=non_names, extra_names=extra_names)

print(connections[:10])