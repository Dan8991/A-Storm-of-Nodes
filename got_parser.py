import PyPDF2 as pdf
import enchant
import sys
from functools import reduce
import re
import json


class Chapter(object):

    '''
    the title of the chapter in the text must be upper_case
    text = array of strings containing one line in every element of the array
    words = text of the chapter
    '''
    def __init__(self, title, words):
        self.title = title
        self.words = words

    '''
    used for debugging sice it prints useful infos
    '''
    def __str__(self):
        return "chapter name = {}\nchapter length = {}".format(self.title, len(self.words))

    '''
    non_names = city names, ... these are the names that are not character names
    extra_names = character names that are in the vocabulary
    return_word_numbers = True if you want it to return where a name appeard in the text as an index starting from 0 
    at the beginning of the chapter
    return = names found in the chapter that are not in the dictionary and are not in non_names or are in extra_names
    '''
    def get_non_english_words(self, non_names=[], extra_names=[], return_word_numbers=False):
        #english vocaboulary
        dictionary = enchant.Dict("en_UK")

        #getting single words and initializing variables
        words = self.words.split()
        non_english_words = []
        last_non_eng = False
        word_count = 0
        word_numbers = []

        for word in words:
            word_count += 1
            #checking if the word is a character name
            if (word in extra_names) or ((not dictionary.check(word.lower())) and word[0].isupper() and (word not in non_names)):
                #checking if there was another name preceding this one meaning that this is probably the surname
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
        get_end = if True an end string is added at the end of the array
        return = all characters connections in the chapter
    '''

    def get_chapter_connections(self, characters, characters_id, link_length=100, non_names=[], extra_names=[], get_end=False):

        #getting names from the book and their offset from the beginning of the chapter
        extra_words, word_numbers = self.get_non_english_words(
            non_names=non_names, extra_names=extra_names, return_word_numbers=True)

        #searching only valid character names
        valid_char = [(characters_id[c], word_num) for char, word_num in zip(
            extra_words, word_numbers) if char in characters for c in characters[char] if len(characters[char]) == 1]

        #getting the name of the protagonist for the chapter
        protagonist_name = re.sub(
            r"\n", "", self.title[0] + self.title[1:].lower())

        #prologue and epilogue are assigned to their protagonis manually
        if protagonist_name == "Prologue":
            protagonist_id = characters_id[characters["Chett"][0]]
        elif protagonist_name == "Epilogue":
            protagonist_id = characters_id[characters["Merrett"][0]]
        else:
            protagonist_id = characters_id[characters[protagonist_name][0]]

        current_index = 0
        connections = []

        for i in range(len(valid_char)):

            #used to evade self loops 
            if protagonist_id != valid_char[i][0]:
                connections.append((protagonist_id, valid_char[i][0]))

            #checking if more than link_length words were passed, if so just move forward
            while (valid_char[i][1] - link_length) > valid_char[current_index][1]:
                current_index += 1

            #adding all good connections to the array
            for j in range(current_index, i):
                if valid_char[j][0] != valid_char[i][0]:
                    connections.append((valid_char[j][0], valid_char[i][0]))

        if get_end:
            connections.append("end")

        return connections

"""
text = book in string format
return = array of chapters
"""
def get_chapters_from_text(text):

    # initializing data supposing that the first line is a title
    title = "PROLOGUE"
    words = ""
    chapters = []

    for line in text[1:]:
        if line.isupper() and (len(line.split()) == 1):

            chapters.append(Chapter(title, words))
            title = line
            words = ""
        else:
            words += line

    chapters.append(Chapter(title, words))
    return chapters

"""
s = string to be parsed
return = string without punctuation and stuff like that
"""
def parse_punctuation(s):

    # changing all puntuation to a comma and separating it from words
    parsed = re.sub(r"[\.!?,”“…;]", " , ", s)

    # removing multiple spaces
    parsed = " ".join(parsed.split()) + "\n"

    #used to prevent bad character recognition
    parsed = re.sub(r"Storm End", "StormEnd", parsed)

    # removing saxons genitive and other similar things because it is annoying for parsing
    parsed = re.sub(r"(’s|’n|’d|’i)", "", parsed)

    parsed = re.sub(r"—", " ", parsed)

    parsed = re.sub(r"(-\n|—\n)", "", parsed)

    return parsed


def get_names(file_name):
    return set([line.replace("\n", "") for line in open(file_name, 'r')])

# this function is needed in order to get a basic idea of which characters are actually the same one

#theese two functions were used to build the dictionaries but a lot of the work was done
#manually so they are not really important and actually at the moment they are not used
#they are here just to give an idea on how the dataset was created but they were used to put all the
#names in a file with possible nicknames associated so i could manually select the correct nicknames
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

def connections_to_string(x):
    if x == "end":
        return "end\n"
    return "{},{}\n".format(*x)

parse_file = False
parse_file_name = ""
parse_specific = False
parse_chapter = -1
add_end = False
verbose = False

if len(sys.argv) > 1:
    if "-p" in sys.argv:
        parse_file = True
        parse_file_name = sys.argv[sys.argv.index("-p") + 1]
    if "-n" in sys.argv:
        parse_specific = True
        parse_chapter = sys.argv[sys.argv.index("-n") + 1]
        if parse_chapter.isdigit():
            parse_chapter = int(parse_chapter)
        else:
            sys.exit(1)
    if "-e" in sys.argv:
        add_end = True
    if "-v" in sys.argv:
        verbose = True

if parse_file:
    with open(parse_file_name, "r") as f:
        lines = [parse_punctuation(line) for line in f]

    with open(parse_file_name, "w") as f:
        f.writelines(lines)
    
    sys.exit(0)

pdf_file = open("storm_of_swords.txt", 'r')

chapters = get_chapters_from_text([line for line in pdf_file])

non_names = get_names("non_character_names.txt")
extra_names = get_names("extra_names.txt")

if parse_specific:
    uniques = list(set(chapters[parse_chapter].get_non_english_words(
        non_names=non_names, extra_names=extra_names)))
else:
    uniques = list(set(reduce(lambda x, y: set(x).union(set(y)), [chapter.get_non_english_words(
        non_names=non_names, extra_names=extra_names) for chapter in chapters])))

uniques.sort()

characters_nicknames = json.load(open("characters.json", "r"))
nicknames_characters = get_characters_dictionary(characters_nicknames)

if verbose:
    print(characters_nicknames.keys())
print("Total characters found:", len(characters_nicknames.keys()))

characters_id = {key: value for value,
                 key in enumerate(characters_nicknames.keys())}

if parse_specific:
    connections = chapters[parse_chapter].get_chapter_connections(
        nicknames_characters, characters_id, non_names=non_names, extra_names=extra_names, link_length=20, get_end=add_end)
else:
    connections = reduce(lambda x, y: x + y, [x.get_chapter_connections(
        nicknames_characters, characters_id, non_names=non_names, extra_names=extra_names, link_length=20, get_end=add_end) 
        for x in chapters])

if add_end:
    with open("connections_for_temp.csv", "w") as fp:
        fp.writelines(map(connections_to_string, connections))

else:
    with open("connections.csv", "w") as fp:
        fp.writelines(map(connections_to_string, connections))

with open("nodes.csv", "w") as fp:
    for key in characters_id:
        fp.write("{},{}\n".format(key, characters_id[key]))