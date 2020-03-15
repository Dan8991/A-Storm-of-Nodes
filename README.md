# Intro

This was the final project for the network science course at Unipd and it conists in applying algorithms to a social network in order to find out it's properties, in particular I chose to study the network formed in "A Storm of Swords" because I really like the books.

All the code is provided and an explanation about what I did can be found in homework1.pdf and homework2.pdf, in paricular homework2 might be more interesting as its results are clearly explained by the story of the book.


### Generic Info

characters.json is a file json containing a key-value realtion between characters and nicknames

connections.csv is a file that contains the dataset

connections_for_temp.csv is a file that contains the dataset with chapters separated by the word "end"

extra_names.txt and non_names.txt were used to build the dataset and contain exceptions.

got_parser.py is the script used for parsing and generating the dataset.

nodes.csv contains the name of the characters and their id

in order to use got_parser to generate the dataset just use python got_parser.py, it also supports some flags:

-p "filename" parses the file "filename"

-n "chapter_number" generates the dataset only for a specific chapter

-e creates the connections_for_temp.csv, this file is not created by default

-v is verbose mode and prints out the characters that it found
