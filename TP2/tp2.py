import nltk
import sys
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
from nltk          import RegexpParser

# LIGNE DE COMMANDE :
# python3 tp2.py wsj_0010_sample.txt structures_rules.txt





########## VARIABLES GLOBALES ##########
TEXT_FILE      = sys.argv[1]
STRUCT_FILE    = sys.argv[2]
UNIV_TAG       = {} 
PATTERNS       = {}
UNIV_NAMED_ENT = {}

TOKEN_TAG_FILE      = "wsj_0010_sample.txt.pos.nltk"
UNIV_TOKEN_TAG_FILE = "wsj_0010_sample.txt.pos.univ"
CHUNK_MATCH_FILE    = "wsj_0010_sample.txt.chk.nltk"
NAMED_ENT_FILE      = "wsj_0010_sample.txt.ne.nltk"
UNIV_NAMED_ENT_FILE = "wsj_0010_sample.txt.ne.univ"

UNIV_TAG_FILE      = "POSTags_PTB_Universal_Linux.txt"
UNIV_ENTITIES_FILE = "NERTags_PTB_Universal_Linux.txt"


########## FONCTIONS GLOBALES ##########
def open_file(text_file):
    try:
        with open(text_file, 'r') as file:
            text = file.read()
        print(f"Text file \"{text_file}\" loaded.")
    except FileNotFoundError:
        print(f"File \"{text_file}\" not found.")
    return text
    
def extract_named_entities(named_tree):
    result = ""
    for token in named_tree.leaves():
        result += token[0] + " "
    result = result[:-1]
    result += "\t" + named_tree.label() + "\n"
    return result

def extract_named_entities_univ(named_tree):
    result = ""
    for token in named_tree.leaves():
        result += token[0] + " "
    result = result[:-1]
    result += "\t" + UNIV_NAMED_ENT[named_tree.label()] + "\n"
    return result

def extract_chunk(chunk):
    result = ""
    for token in chunk:
        result += token[0] + " "
    return result

def create_univ_tag_dico(univ_tag_file):
    try:
        with open(univ_tag_file, 'r') as file:
            for line in file:
                l = line.split()
                UNIV_TAG[l[0]] = l[1]
    except FileNotFoundError:
        print(f"File \"{univ_tag_file}\" not found.")

def create_patterns_dico(struct_file):
    try:
        with open(struct_file, 'r') as file:
            for line in file:
                l = line.strip().split("    ")
                PATTERNS[l[1]] = l[0]
    except FileNotFoundError:
        print(f"File \"{struct_file}\" not found.")

def create_named_entities_dico(univ_named_ent_file):
    try:
        with open(univ_named_ent_file, 'r') as file:
            for line in file:
                l = line.split()
                UNIV_NAMED_ENT[l[0]] = l[1]
    except FileNotFoundError:
        print(f"File \"{univ_named_ent_file}\" not found.")

def write_tag_file(token_tag_file, tag):
    with open(token_tag_file, 'w') as file:
        for token in tag:
            file.write(token[0] + "    " + token[1] + '\n')
    print(f"Tokenisation achieved : File \"{token_tag_file}\" created.")

def write_univ_tag_file(univ_tag_file, tag):
    with open(univ_tag_file, 'w') as file:
        for token in tag:
            file.write(token[0] + "\t" + UNIV_TAG[token[1]] + '\n')
    print(f"Universal tokenisation achieved : File \"{univ_tag_file}\" created.")

def write_chunk_match(chunk_match_file, patterns):
    with open(chunk_match_file, 'w') as fichier:
        fichier.write("")

    for key in patterns:
        chunker = RegexpParser(key.replace("\"\"\"", ""))
        output  = chunker.parse(tag)
        matching_chunks = [subtree for subtree in output.subtrees() if subtree.label() == key.replace("\"\"\"", "").split(":")[0]]

        with open(chunk_match_file, 'a') as fichier:
            fichier.write(patterns[key] + " :" + "\n")
            for chunk in matching_chunks:
                fichier.write(extract_chunk(chunk) + '\n')
            fichier.write('\n')
    print(f"Parsing achieved : File \"{chunk_match_file}\" created.")

def write_entities(entities_file, named_entities):    
    with open(entities_file, 'w') as file:
        for tree in named_entities:
            file.write(extract_named_entities(tree))
    print(f"Named entities achieved : File \"{entities_file}\" created.")

def write_univ_entities(univ_entities_file, named_entities):    
    with open(univ_entities_file, 'w') as file:
        for tree in named_entities:
            file.write(extract_named_entities_univ(tree))
    print(f"Universal named entities achieved : File \"{univ_entities_file}\" created.")


##### MAIN #####
text   = open_file(TEXT_FILE)
tokens = word_tokenize(text)

# ANALYSE MORPHO-SYNTAXIQUE
tag = nltk.pos_tag(tokens)
create_univ_tag_dico(UNIV_TAG_FILE)
write_tag_file(TOKEN_TAG_FILE, tag)
write_univ_tag_file(UNIV_TOKEN_TAG_FILE, tag)

# ANALYSE SYNTAXIQUE
create_patterns_dico(STRUCT_FILE)
write_chunk_match(CHUNK_MATCH_FILE, PATTERNS)

# EXTRACTION ENTITES NOMMEES
entities = nltk.ne_chunk(tag, binary=False)
create_named_entities_dico(UNIV_ENTITIES_FILE)
named_entities = [subtree for subtree in entities.subtrees() if subtree.label() != "S"]
write_entities(NAMED_ENT_FILE, named_entities)
write_univ_entities(UNIV_NAMED_ENT_FILE, named_entities)