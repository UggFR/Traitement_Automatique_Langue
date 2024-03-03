# Traitement_Automatique_Langue
TP et Projet du module de Traitement Automatique des Langues dans le cadre des cours ET5 de Polytech Paris-Saclay

# Lignes de commandes :
## TP1
python3 tp1.py wsj_0010_sample.txt structures_rules.txt Determinant-Adjectif-Nom 

python3 [script python] [Fichier contenant le texte] [Fichier contenant toutes les structures] [Structure à matcher]

## TP2
python3 tp2.py wsj_0010_sample.txt structures_rules.txt

python3 [script python] [Fichier contenant le texte] [Fichier contenant toutes les structures]

## TP3
python3 tp3.py wsj_0010_sample.txt structures_rules.txt

python3 [script python] [Fichier contenant le texte] [Fichier contenant toutes les structures]

## Projet

(main.py n'analyse que les 30 premières lignes du texte avec Stanford, par souci de temps d'exécution)

python3 main.py

python3 evaluate.py ../data/pos_test.txt.pos.stanford.univ ../data/pos_reference.txt.univ
python3 evaluate.py ../data/pos_test.txt.pos.nltk.univ ../data/pos_reference.txt.univ

python3 evaluate.py ../data/ne_test.txt.ne.stanford.conll ../data/ne_reference.txt.conll
python3 evaluate.py ../data/ne_test.txt.ne.nltk.conll ../data/ne_reference.txt.conll

python3 [script python]
