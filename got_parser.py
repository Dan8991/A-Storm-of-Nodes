import PyPDF2 as pdf
import enchant
from functools import reduce
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

    def get_non_english_words(self):
        
        dictionary = enchant.Dict("en_UK")
        words = self.words.split()
        non_english_words = []
        last_non_eng = False
        for word in words:
            if not dictionary.check(word) and word[0].isupper():
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
    title = text[0]
    words = ""
    chapters = []

    for line in text[1:]:
        if line.isupper():
            chapters.append(Chapter(title, words))
            title = line
            words = ""
        else:
            words += line

    return chapters


pdf_file = open("storm_of_swords.txt", 'r')

chapters = get_chapters_from_text([line for line in pdf_file])

unique = list(reduce(lambda x, y: x.union(y), [chapter.get_non_english_words() for chapter in chapters]))
print(len(unique))