
import numpy as np
import random
from collections import Counter

# Paramètres par défaut
L = 2
n = 100
k = 10
eta = 0.01
epochs = 5
min_count = 5

with open(r"C:\Users\thoma\Desktop\Fac\Traitement du langage\TP\data\alexandre_dumas\Le_comte_de_Monte_Cristo.tok", "r", encoding="utf-8") as file:
    corpus = file.read()
corpus = corpus.split()  

# Comptage des mots
words_counts = Counter(corpus)

vocab = {word for word in words_counts if words_counts[word] > min_count}
words_id = {word: i for i, word in enumerate(vocab)}
id_words = {i: word for i, word in enumerate(vocab)}

vocab_size = len(vocab)

data = []

words = {}
pos = []
neg = []
eval_dict = {}

for i, word in enumerate(corpus):
    if word in words_id:
        target_word_id = words_id[word]
        context_ids = []
        
        # Définir la fenêtre contextuelle
        start = max(0, i - (2 * L) + 1)
        end = min(len(corpus), i + (2 * L) + 1)
        for j in range(start, end):
            if i != j and corpus[j] in words_id:
                context_id = words_id[corpus[j]]
                context_ids.append(context_id)
                pos.append([context_id, target_word_id])
        
        # Sélection des échantillons négatifs
        negative_ids = random.choices(
            [w for w in words_id.values() if w not in context_ids],
            k=k
        )
        
        for neg_id in negative_ids:
            neg.append([neg_id, target_word_id])
        
        for context in context_ids:
            for negative in negative_ids:
                eval_dict[target_word_id] = {
                    "context": context,
                    "negative": negative
                }
        
        if word not in words:
            words[word] = {
                "word": word,
                "negative_ids": negative_ids,
                "pos_ids": context_ids,
                "target": target_word_id
            }
        else:
            words[word]["negative_ids"].extend(negative_ids)
            words[word]["pos_ids"].extend(context_ids)


# Fichier d'évaluation
with open("eval.txt", "w", encoding="utf-8") as eval_file:
    for word_id, value in eval_dict.items():
        context_word = value["context"]
        negative_word = value["negative"]
        eval_file.write(f"{word_id} {context_word} {negative_word}\n")    

# Fichier des échantillons négatifs
with open("neg.txt", "w", encoding="utf-8") as neg_file:
    for pair in neg:
        neg_file.write(f"{pair[0]} {pair[1]}\n")

# Fichier des échantillons positifs
with open("pos.txt", "w", encoding="utf-8") as pos_file:
    for pair in pos:
        pos_file.write(f"{pair[0]} {pair[1]}\n")    

# Fichier du corpus final
with open("corpus.txt", "w", encoding="utf-8") as corpus_file:
    corpus_file.write(" ".join(corpus))
