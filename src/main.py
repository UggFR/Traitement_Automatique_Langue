import nltk
import sys
import os
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
from nltk.tokenize import word_tokenize
from nltk          import RegexpParser
from nltk.tag import StanfordPOSTagger
from nltk.tag import StanfordNERTagger

java_path = "C:/Program Files/Java/jdk-18.0.1.1/bin/java.exe"
os.environ['JAVAHOME'] = java_path

stanford_pos_jar   = '../data/stanford-postagger.jar'
stanford_pos_model = '../data/english-bidirectional-distsim.tagger'
st_pos = StanfordPOSTagger(stanford_pos_model, stanford_pos_jar)

stanford_ne_jar   = '../data/stanford-ner.jar'
stanford_ne_model = '../data/english.all.3class.distsim.crf.ser.gz'
st_ner = StanfordNERTagger(stanford_ne_model, stanford_ne_jar, encoding='utf-8')





########## VARIABLES GLOBALES ##########
SENTENCES_TAG = []
SENTENCES_NE  = []
TAG_STANFORD  = []
REF_PTB       = {}
PTB_UNIV      = {"HYPH":".", "ADD":".", "NFP":"NOUN", "-LRB-":".","-RRB-":".", "AFX":"ADJ"}
C0NLL2003     = { "PERSON": "PER",
                  "ORGANIZATION": "ORG",
                  "LOCATION": "LOC" }

########## FICHIERS ANALYSE MORPHO-SYNTAXIQUE ##########
TEXT_REF_FILE_TAG = "../data/pos_reference.txt"
TEXT_RES_FILE_TAG = "../data/pos_test.txt"
POSTAGS_REF_PTB   = "../data/POSTags_REF_PTB.txt"
POSTAGS_PTB_UNIV  = "../data/POSTags_PTB_Universal.txt"
POSTAGS_UNIV_FILE = "../data/pos_reference.txt.univ"
POSTAG_NLTK_FILE  = "../data/pos_test.txt.pos.nltk"
POSTAG_STANFORD_FILE = "../data/pos_test.txt.pos.stanford"
POSTAG_UNIV_NLTK_FILE = "../data/pos_test.txt.pos.nltk.univ"
POSTAG_UNIV_STANFORD_FILE = "../data/pos_test.txt.pos.stanford.univ"

########## FICHIERS RECONNAISSANCE ENTITES NOMMEES ##########
TEXT_REF_FILE_NE   = "../data/ne_reference.txt.conll"
TEXT_RES_FILE_NE   = "../data/ne_test.txt"
NE_NLTK_FILE       = "../data/ne_test.txt.ne.nltk"
NE_CONLL_NLTK_FILE = "../data/ne_test.txt.ne.nltk.conll"
NE_STANFORD_FILE       = "../data/ne_test.txt.ne.stanford"
NE_CONLL_STANFORD_FILE = "../data/ne_test.txt.ne.stanford.conll"





########## FONCTIONS GLOBALES ##########
def open_file(text_file):
    """
    Args:
        text_file (str): Chemin du fichier texte à ouvrir.

    Returns:
        str or None: Contenu du fichier texte si le fichier est trouvé et lu avec succès,
                     sinon retourne None.
    """
    # Tente d'ouvrir le fichier spécifié en mode lecture ('r') sinon retourne une erreur
    try:
        with open(text_file, 'r') as file:
            text = file.read()
        print(f"Text file \"{text_file}\" loaded.")
        return text
    except FileNotFoundError:
        print(f"File \"{text_file}\" not found.")
        return None

def extract_sentences_file(file, sentences):
    """
    Args:
        file      (str)  : Le chemin vers le fichier contenant les tokens.
        sentences (array): Le tableau contenant les phrases crées à partir du fichier.

    Returns:
        None: La fonction ne retourne rien, elle remplit le tableau avec des phrases à partir du fichier contenant les tokens.
    """
    
    s = ""
    
    # Récupère le contenu du fichier et s'il n'est pas None, on procède ligne par ligne (split("\n"))
    text = open_file(file)
    if text is not None:
        for line in text.split("\n"):
            # Divise la ligne avec le séparateur tabulation
            l = line.split("\t")
            
            if file == TEXT_REF_FILE_TAG:
                # Si le fichier passé en paramètre est TEXT_REF_FILE_TAG, le changement de ligne se produit quand l a une taille de 1
                if len(l) == 1:
                    sentences.append(s)
                    s = ""
                # Si nous ne sommes pas au changement de ligne, nous remplissons la variable sentence avec un format différent selon ce que contient l[0]
                else:
                    if l[0] == ".":
                        s = s[:-1] + l[0]
                    elif l[0] == ",":
                        s = s[:-1] + l[0] + " "
                    else:
                        s += l[0] + " "            
            else:
                # Si le fichier passé en paramètre est différent, le changement de ligne se produit quand l[0] est égal à ""
                if l[0] == "":
                    sentences.append(s)
                    s = ""
                # Si nous ne sommes pas au changement de ligne, nous remplissons la variable sentence avec un format différent selon ce que contient l[0]
                else:
                    if l[0] == ".":
                        s = s[:-1] + l[0]
                    elif l[0] == ",":
                        s = s[:-1] + l[0] + " "
                    else:
                        s += l[0] + " "             

def extract_tag_stanford_conll(text_res_file):
    try:
        with open(text_res_file, 'r') as file:
            index = 0
            for line in file:
                if index ==30:
                    break
                sentence = st_pos.tag(line.split())
                for s in sentence:
                    TAG_STANFORD.append((s[0], s[1]))
                TAG_STANFORD.append("\n")
                index += 1                
    except FileNotFoundError:
        print(f"File \"{text_res_file}\" not found.")

def extract_postags_ref_ptb_files(file, type):
    """
    Args:
        file (str): Le chemin vers le fichier de référence.
        type (str): Le type de la table de correspondance des POS tags.

    Returns:
        None: La fonction ne retourne rien, elle remplit simplement un dictionnaire soit REF_PTB soit PTB_UNIV.

    Remarque:
        Le format du fichier de référence doit être sous la forme de lignes où la première colonne
        est une étiquette et la deuxième colonne est son étiquette équivalente dans une autre convention.
    """
    # Récupère le contenu du fichier et s'il n'est pas None, on procède ligne par ligne (split("\n"))
    text = open_file(file)
    if text is not None:
        for line in text.split("\n"):
            # Divise la ligne avec le séparateur espace
            l = line.split()
            # Vérifie le nom du fichier fourni en argument
            if type == "REF_PTB":
                # Si le fichier est POSTAGS_REF_PTB, stocke les informations dans le dictionnaire REF_PTB
                REF_PTB[l[0]] = l[1]
            else:
                # Sinon, stocke les informations dans le dictionnaire PTB_UNIV
                PTB_UNIV[l[0]] = l[1]

def create_dic_ref_universal(ref_ptb_file, ptb_univ_file):
    """
    Args:
        ref_ptb_file  (str): Le chemin vers le fichier des étiquettes PTB de référence.
        ptb_univ_file (str): Le chemin vers le fichier UNIV de référence.

    Returns:
        None: La fonction ne retourne rien, elle appelle "extract_postags_ref_ptb_files" pour créer les dictionnaire globaux REF_PTB et PTB_UNIV.
    """
    extract_postags_ref_ptb_files(ref_ptb_file,  "REF_PTB")
    extract_postags_ref_ptb_files(ptb_univ_file, "PTB_UNIV")
    print("Create dictionnary REF/PTB and PTB/UNIV.")
                        
                        

def write_sentences_file(text, sentences):
    """
    Args:
        text      (str)  : Le chemin du fichier où l'on veut écrire les phrases.
        sentences (array): Le tableau contenant les phrases qui ont été construite grâce à la fonction "extract_sentences_file".

    Returns:
        None: La fonction ne retourne rien, elle remplit le fichier passé par référence à partir des phrases contenant dans le tableau.
    """
    with open(text, 'w') as file:
        # Ecriture de chaque phrase (exceptée la dernière) contenu dans le tableau + saut de ligne
        for sentence in sentences[:-1]:
            file.write(sentence + "\n")
        # Ecriture de la dernière phrase du tableau sans ajout de saut de ligne
        file.write(sentences[-1])
    print(f"Constructing sentences for text : File \"{text}\" created.")

def write_postags_universal_file(postags_reference_file, postags_universal_file, tag):
    """
    Args:
        postags_reference_file (str): Le chemin du fichier où sont contenus les tokens de référence.
        postags_universal_file (str): Le chemin du fichier où l'on veut écrire les tokens universels.
        tag (str): Type du tag que l'on veut associé au token

    Returns:
        None: La fonction ne retourne rien, elle remplit le fichier output en remplaçant les tokens du fichier input par les tokens universels.
        
    Remarque:
        Le format du fichier de sortie est un token de référence suivi d'une tabulation et du token universel associé à ce mot.
    """
    try:
        with open(postags_reference_file, 'r') as input_file, open(postags_universal_file, 'w') as output_file:
            for line in input_file:
                # Si la ligne à une longueur de 1 (équivalent à vide), on saute de ligne
                if len(line) == 1:
                    output_file.write("\n")
                else:
                    # Divise la ligne avec le séparateur tabulation
                    l = line.split("\t")
                    
                    # Ecriture du premier token de la ligne suivi d'une tabulation suivi du token universel (récupéré grâce à différents dictionnaires selon le tag)
                    if tag == "REF":
                        output_file.write(l[0] + "\t" + PTB_UNIV[REF_PTB[l[1][:-1]]] + "\n")
                    elif tag == "PTB":
                        output_file.write(l[0] + "\t" + PTB_UNIV[l[1][:-1]] + "\n")
            print(f"Universal Postags : File \"{postags_universal_file}\" created.")
    except FileNotFoundError:
        print(f"File \"{postags_reference_file}\" not found.")

def write_tag_file(token_tag_file, text, postagger):
    """
    Args:
        token_tag_file (str): Le chemin du fichier où l'on veut écrire les tokens.
        text (array): Tableau contenant les phrases à tokeniser.
        postagger (str): Nom du postager.

    Returns:
        None: La fonction ne retourne rien, elle remplit le fichier des tokens des phrases contenues dans le tableau passé en paramètre.
        
    Remarque:
        Le format du fichier de sortie est un mot suivi d'une tabulation et du token associé à ce mot. 
        Une ligne vide signifie le changement de phrase.
    """
    if(postagger=="NLTK"):
        with open(token_tag_file, 'w') as file:
            for line in text:
                # On tokenise chaque ligne du texte grâce au postagger de NLTK
                tokens_TAG   = word_tokenize(line)
                tag_nltk_TAG = nltk.pos_tag(tokens_TAG)

                # On écrit sur une ligne chaque mot du texte suivi d'une tabulation suivi du token NLTK
                for token in tag_nltk_TAG:
                    file.write(token[0] + "\t" + token[1] + '\n')

                # On ajoute une ligne vide pour signifier la fin d'une phrase
                file.write("\n")
        print(f"Tokenisation {postagger} achieved : File \"{token_tag_file}\" created.")

    if (postagger == "Stanford"):
        with open(token_tag_file, 'w') as file:
            extract_tag_stanford_conll(TEXT_RES_FILE_TAG)
            # On écrit sur une ligne chaque mot du texte suivi d'une tabulation suivi du token
            for token in TAG_STANFORD:
                if (token == "\n"):
                    file.write("\n")
                else:
                    file.write(token[0] + "\t" + token[1] + '\n')
                # On ajoute une ligne vide pour signifier la fin d'une phrase

        print(f"Tokenisation {postagger} achieved : File \"{token_tag_file}\" created.")
def write_named_entities_nltk_file(named_entities_file, text):
    """
    Args:
        named_entities_file (str): Le chemin du fichier où l'on veut écrire les entités nommées.
        text (array): Tableau contenant les phrases à tokeniser.

    Returns:
        None: La fonction ne retourne rien, elle remplit le fichier des entités nommées des phrases contenues dans le tableau passé en paramètre.
        
    Remarque:
        Le format du fichier de sortie est un mot suivi d'une tabulation et de l'entité nommée associé à ce mot. 
        Une ligne vide signifie le changement de phrase.
    """
    with open(named_entities_file, 'w') as file:
        for line in text:
            # On tokenise avec les entités nommées chaque ligne du texte grâce au chunk de NLTK
            tokens_NE   = word_tokenize(line)
            tag_nltk_NE = nltk.pos_tag(tokens_NE)
            entities = nltk.ne_chunk(tag_nltk_NE, binary=False)
            
            # On écrit sur une ligne chaque mot du texte suivi d'une tabulation suivi de l'entité nommée NLTK
            for token in entities:
                # Si token[0] est un string alors ce mot n'est pas une entité nommée donc on on l'associe à l'entité nommée "O"
                if isinstance(token[0], str):
                    file.write(token[0] + "\t" + "O" + '\n')
                # Sinon, nous avons une entité nommée et associons chaque mot de l'entité nommée au label de celle-ci
                else:
                    for item in token:
                        file.write(item[0] + "\t" + token.label() + '\n')
                        
            # On ajoute une ligne vide pour signifier la fin d'une phrase
            file.write("\n")
    print(f"Tokenisation NLTK achieved : File \"{named_entities_file}\" created.")


def write_named_entities_stanford_file(named_entities_file, text):
    """
    Args:
        named_entities_file (str): Le chemin du fichier où l'on veut écrire les entités nommées.
        text (array): Tableau contenant les phrases à tokeniser.

    Returns:
        None: La fonction ne retourne rien, elle remplit le fichier des entités nommées des phrases contenues dans le tableau passé en paramètre.

    Remarque:
        Le format du fichier de sortie est un mot suivi d'une tabulation et de l'entité nommée associé à ce mot.
        Une ligne vide signifie le changement de phrase.
    """
    with open(named_entities_file, 'w') as file:
        index = 0
        for line in text:
            # On tokenise avec les entités nommées chaque ligne du texte grâce au tagger de stanford
            tokenized_text = word_tokenize(line)
            entities = st_ner.tag(tokenized_text)
            # On écrit sur une ligne chaque mot du texte suivi d'une tabulation suivi de l'entité nommée Stanford
            for token in entities:
                #Si l'entité est O, alors pas une entité nommé
                if(token[1]=="O"):
                    file.write(token[0] + "\t" + "O" + '\n')
                else:
                    file.write(token[0] + "\t" + token[1] + '\n')

            # On ajoute une ligne vide pour signifier la fin d'une phrase
            file.write("\n")
            index+=1
            if(index==30):
                break
    print(f"Tokenisation Stanford achieved : File \"{named_entities_file}\" created.")
    
def write_named_entities_conll2003_nltk_file(named_entities_file, text):
    """
    Args:
        named_entities_file (str): Le chemin du fichier où l'on veut écrire les entités nommées converties en étiquette CoNLL-2003.
        text (array): Tableau contenant les phrases à tokeniser.

    Returns:
        None: La fonction ne retourne rien, elle remplit le fichier des entités nommées (format CoNLL-2003) des phrases contenues dans le tableau passé en paramètre.
        
    Remarque:
        Le format du fichier de sortie est un mot suivi d'une tabulation et de l'entité nommée (format CoNLL-2003) associé à ce mot. 
        Une ligne vide signifie le changement de phrase.
    """
    with open(named_entities_file, 'w') as file:
        for line in text:
            # On tokenise avec les entités nommées chaque ligne du texte grâce au chunk de NLTK
            tokens_NE   = word_tokenize(line)
            tag_nltk_NE = nltk.pos_tag(tokens_NE)
            entities = nltk.ne_chunk(tag_nltk_NE, binary=False)
            previousTag = ""
            # On écrit sur une ligne chaque mot du texte suivi d'une tabulation suivi de l'entité nommée NLTK convertie au format CoNLL-2003
            i=0
            for token in entities:
                # Si token[0] est un string alors ce mot n'est pas une entité nommée donc on on l'associe à l'entité nommée "O"
                if isinstance(token[0], str):
                    file.write(token[0] + "\t" + "O" + '\n')
                # Sinon, nous avons une entité nommée et associons chaque mot de l'entité nommée au label de celle-ci convertie au format CoNLL-2003
                else:
                    # Si le label est présent dans le dictionnaire CONNL2023, alors on l'associe au mot sinon on associe "MISC"
                    if token.label() in C0NLL2003:
                        ne = "-" + C0NLL2003[token.label()]
                    else:
                        ne = "-MISC"
                    # On ajoute un spécificateur à l'entité nommée avec un B si c'est le premier token de celle-ci et un I sinon
                    for index, item in enumerate(token):
                        if index == 0 and ne!=previousTag:
                            ne_token = "B" + ne
                        else:
                            ne_token = "I" + ne
                        file.write(item[0] + "\t" + ne_token + '\n')
                        previousTag = ne

                i += 1
                # On ajoute une ligne vide pour signifier la fin d'une phrase
            file.write("\n")
    print(f"Tokenisation NLTK achieved : File \"{named_entities_file}\" created.")


def write_named_entities_conll2003_stanford_file(named_entities_file, text):
    """
    Args:
        named_entities_file (str): Le chemin du fichier où l'on veut écrire les entités nommées converties en étiquette CoNLL-2003.
        text (array): Tableau contenant les phrases à tokeniser.

    Returns:
        None: La fonction ne retourne rien, elle remplit le fichier des entités nommées (format CoNLL-2003) des phrases contenues dans le tableau passé en paramètre.

    Remarque:
        Le format du fichier de sortie est un mot suivi d'une tabulation et de l'entité nommée (format CoNLL-2003) associé à ce mot.
        Une ligne vide signifie le changement de phrase.
    """
    with open(named_entities_file, 'w') as file:
        index=0
        for line in text:
            # On tokenise avec les entités nommées chaque ligne du texte grâce au tagger de stanford
            tokenized_text = word_tokenize(line)
            entities = st_ner.tag(tokenized_text)

            # On écrit sur une ligne chaque mot du texte suivi d'une tabulation suivi de l'entité nommée Stanford convertie au format CoNLL-2003
            i=0
            for token in entities:
                # Si token[0] est un string alors ce mot n'est pas une entité nommée donc on on l'associe à l'entité nommée "O"
                if(token[1]=="O"):
                    file.write(token[0] + "\t" + "O" + '\n')
                # Sinon, nous avons une entité nommée et associons chaque mot de l'entité nommée au label de celle-ci convertie au format CoNLL-2003
                else:
                    # Si le label est présent dans le dictionnaire CONNL2023, alors on l'associe au mot sinon on associe "MISC"
                    if token[1] in C0NLL2003:
                        ne = "-" + C0NLL2003[token[1]]
                    else:
                        ne = "-MISC"

                    # On ajoute un spécificateur à l'entité nommée avec un B si c'est le premier token de celle-ci et un I sinon
                    if(i>0):
                        if(entities[i][1]==entities[i-1][1]):
                            ne_token = "I" + ne
                        else:
                            ne_token = "B" + ne
                    else:
                        ne_token = "B" + ne
                    file.write(token[0] + "\t" + ne_token + '\n')
                i+=1
            # On ajoute une ligne vide pour signifier la fin d'une phrase
            file.write("\n")

            index+=1
            if(index==30):
                break
    print(f"Tokenisation Stanford achieved : File \"{named_entities_file}\" created.")


##### MAIN #####
# CREATION DICTIONNAIRE ETIQUETTE
create_dic_ref_universal(POSTAGS_REF_PTB, POSTAGS_PTB_UNIV)
write_postags_universal_file(TEXT_REF_FILE_TAG, POSTAGS_UNIV_FILE, "REF")

# EXTRACTION ET ECRITURE DES PHRASES A PARTIR DES TOKENS MORPHO-SYNTAXIQUE
extract_sentences_file(TEXT_REF_FILE_TAG, SENTENCES_TAG)
write_sentences_file(TEXT_RES_FILE_TAG, SENTENCES_TAG)

# EXTRACTION ET ECRITURE DES PHRASES A PARTIR DES TOKENS ENTITES NOMMEES
extract_sentences_file(TEXT_REF_FILE_NE, SENTENCES_NE)
write_sentences_file(TEXT_RES_FILE_NE, SENTENCES_NE)

# ANALYSE MORPHO-SYNTAXIQUE
text_TAG = open_file(TEXT_RES_FILE_TAG)
write_tag_file(POSTAG_NLTK_FILE, text_TAG.split("\n"), "NLTK")
write_postags_universal_file(POSTAG_NLTK_FILE, POSTAG_UNIV_NLTK_FILE, "PTB")

write_tag_file(POSTAG_STANFORD_FILE, TEXT_RES_FILE_TAG, "Stanford")
write_postags_universal_file(POSTAG_STANFORD_FILE, POSTAG_UNIV_STANFORD_FILE, "PTB")

# EXTRACTION ENTITES NOMMEES
text_NE = open_file(TEXT_RES_FILE_NE) 
write_named_entities_nltk_file(NE_NLTK_FILE, text_NE.split("\n"))
write_named_entities_conll2003_nltk_file(NE_CONLL_NLTK_FILE, text_NE.split("\n"))

write_named_entities_stanford_file(NE_STANFORD_FILE, text_NE.split("\n"))
write_named_entities_conll2003_stanford_file(NE_CONLL_STANFORD_FILE, text_NE.split("\n"))