import nltk
import sys
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
from nltk          import RegexpParser

import aymara.lima
nlp = aymara.lima.Lima("ud-eng")




########## VARIABLES GLOBALES ##########
SENTENCES = []
REF_PTB   = {}
PTB_UNIV  = {}
TAG_LIMA  = []

TEXT_REF_FILE     = "pos_reference.txt"
TEXT_RES_FILE     = "pos_test.txt"
POSTAGS_REF_PTB   = "POSTags_REF_PTB.txt" 
POSTAGS_PTB_UNIV  = "POSTags_PTB_Universal.txt"
POSTAGS_UNIV_FILE = "pos_reference.txt.univ"
POSTAG_NLTK_FILE  = "pos_test.txt.pos.nltk"
POSTAG_LIMA_FILE  = "pos_test.txt.pos.lima"
POSTAG_UNIV_NLTK_FILE = "pos_test.txt.pos.nltk.univ" 
POSTAG_UNIV_LIMA_FILE = "pos_test.txt.pos.lima.univ"

########## FONCTIONS GLOBALES ##########
def extract_postags_ref_ptb_files(ref_ptb_file):
    try:
        with open(ref_ptb_file, 'r') as file:
            for line in file:
                l = line.split()
                REF_PTB[l[0]] = l[1]
    except FileNotFoundError:
        print(f"File \"{ref_ptb_file}\" not found.")

def extract_postags_ptb_univ_files(ptb_univ_file):
    try:
        with open(ptb_univ_file, 'r') as file:
            for line in file:
                l = line.split()
                PTB_UNIV[l[0]] = l[1]
    except FileNotFoundError:
        print(f"File \"{ptb_univ_file}\" not found.")

def create_dic_ref_universal(ref_ptb_file, ptb_univ_file):
    extract_postags_ref_ptb_files(ref_ptb_file)
    extract_postags_ptb_univ_files(ptb_univ_file)

def extract_sentences_file(text_ref_file):
    sentence = ""
    try:
        with open(text_ref_file, 'r') as file:
            for line in file:
                if len(line) == 1:
                    continue
                else:
                    l = line.split("\t")
                    if l[0] == ".":
                        sentence = sentence[:-1] + l[0]
                        SENTENCES.append(sentence)
                        sentence = ""
                    elif l[0] == ",":
                        sentence = sentence[:-1] + l[0] + " "
                    else:
                        sentence += l[0] + " "
    except FileNotFoundError:
        print(f"File \"{text_ref_file}\" not found.")

def open_file(text_file):
    try:
        with open(text_file, 'r') as file:
            text = file.read()
        print(f"Text file \"{text_file}\" loaded.")
    except FileNotFoundError:
        print(f"File \"{text_file}\" not found.")
    return text

def write_sentences_file(text_res_file):
    with open(text_res_file, 'w') as file:
        for sentence in SENTENCES:
            file.write(sentence + "\n")
    print(f"Constructing sentences for text : File \"{text_res_file}\" created.")

def write_postags_universal_file(postags_reference_file, postags_universal_file, tag):
    try:
        with open(postags_reference_file, 'r') as input_file, open(postags_universal_file, 'w') as output_file:
            for line in input_file:
                if len(line) == 1:
                    output_file.write("\n")
                else:
                    l = line.split("\t")
                    if tag == "REF":
                        output_file.write(l[0] + "\t" + PTB_UNIV[REF_PTB[l[1][:-1]]] + "\n")
                    elif tag == "PTB":
                        output_file.write(l[0] + "\t" + PTB_UNIV[l[1][:-1]] + "\n")
    except FileNotFoundError:
        print(f"File \"{postags_reference_file}\" not found.")
    print(f"Universal Postags : File \"{postags_universal_file}\" created.")

def extract_tag_lima_conll(text_res_file):
    try:
        with open(text_res_file, 'r') as file:
            for line in file:
                print(file)
                sentence = nlp(line)
                for s in sentence:
                    for token in s:
                        TAG_LIMA.append((token.form, token.upos))
    except FileNotFoundError:
        print(f"File \"{text_res_file}\" not found.")


def write_tag_file(token_tag_file, tag, postagger):
    with open(token_tag_file, 'w') as file:
        for token in tag:
            file.write(token[0] + "\t" + token[1] + '\n')
            if token[0] == ".":
                file.write("\n")
    print(f"Tokenisation {postagger} achieved : File \"{token_tag_file}\" created.")


##### MAIN #####
extract_sentences_file(TEXT_REF_FILE)
write_sentences_file(TEXT_RES_FILE)
create_dic_ref_universal(POSTAGS_REF_PTB, POSTAGS_PTB_UNIV)
write_postags_universal_file(TEXT_REF_FILE, POSTAGS_UNIV_FILE, "REF")

text   = open_file(TEXT_RES_FILE)
tokens = word_tokenize(text)
# extract_tag_lima_conll(TEXT_RES_FILE)
tag_nltk = nltk.pos_tag(tokens)

write_tag_file(POSTAG_NLTK_FILE, tag_nltk, "NLTK")
write_postags_universal_file(POSTAG_NLTK_FILE, POSTAG_UNIV_NLTK_FILE, "PTB")

print(TAG_LIMA)
# write_tag_file(POSTAG_LIMA_FILE, TAG_LIMA, "LIMA")