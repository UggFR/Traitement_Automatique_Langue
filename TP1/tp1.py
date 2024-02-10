import nltk
import sys
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
from nltk          import RegexpParser

# LIGNE DE COMMANDE :
# python3 tp1.py wsj_0010_sample.txt structures_rules.txt Determinant-Adjectif-Nom





########## VARIABLES GLOBALES ##########
TEXT_FILE        = sys.argv[1]
STRUCT_FILE      = sys.argv[2]
STRUCT_MATCH     = sys.argv[3]
TOKEN_TAG_FILE   = "wsj_0010_sample.txt.pos.nltk"
CHUNK_MATCH_FILE = "wsj_0010_sample.txt.chk.nltk"
NAMED_ENT_FILE   = "wsj_0010_sample.txt.ne.nltk"


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

def pattern_chunk(struct_file):
    try:
        with open(struct_file, 'r') as file:
            for line in file:
                l = line.strip().split("    ")
                if l[0] == STRUCT_MATCH:
                    pattern = l[1]
        print(f"Pattern {STRUCT_MATCH} loaded.")
        return pattern
    except FileNotFoundError:
        print(f"File \"{struct_file}\" not found.")

def extract_chunk(chunk):
    result = ""
    for token in chunk:
        result += token[0] + " "
    return result

def write_tag_file(token_tag_file, tag):
    with open(token_tag_file, 'w') as file:
        for token in tag:
            file.write(token[0] + "    " + token[1] + '\n')
    print(f"Tokenisation achieved : File \"{token_tag_file}\" created.")

def write_chunk_match(chunk_file, matching_chunks):
    with open(chunk_file, 'w') as file:
        file.write(STRUCT_MATCH + " :" + "\n")
        for chunk in matching_chunks:
            file.write(extract_chunk(chunk) + '\n')
    print(f"Parsing achieved : File \"{chunk_file}\" created.")

def write_entities(entities_file, named_entities):    
    with open(entities_file, 'w') as file:
        for tree in named_entities:
            file.write(extract_named_entities(tree))
    print(f"Named entities achieved : File \"{entities_file}\" created.")


##### MAIN #####
text   = open_file(TEXT_FILE)
tokens = word_tokenize(text)

# ANALYSE MORPHO-SYNTAXIQUE
tag    = nltk.pos_tag(tokens)
write_tag_file(TOKEN_TAG_FILE, tag)

# ANALYSE SYNTAXIQUE
pattern = pattern_chunk(STRUCT_FILE)

chunker = RegexpParser(pattern.replace("\"\"\"", ""))
output  = chunker.parse(tag)
matching_chunks = [subtree for subtree in output.subtrees() if subtree.label() == pattern.replace("\"\"\"", "").split(":")[0]]
write_chunk_match(CHUNK_MATCH_FILE, matching_chunks)

# EXTRACTION ENTITES NOMMEES
entities = nltk.ne_chunk(tag, binary=False)
named_entities = [subtree for subtree in entities.subtrees() if subtree.label() != "S"]
write_entities(NAMED_ENT_FILE, named_entities)