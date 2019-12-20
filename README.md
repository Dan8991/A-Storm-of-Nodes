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