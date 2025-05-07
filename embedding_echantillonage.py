import numpy as np
import random
from collections import Counter
import re

with open(r"C:\Users\thoma\Desktop\Fac\Traitement du langage\TP\data\alexandre_dumas\Le_comte_de_Monte_Cristo.tok", "r", encoding="utf-8") as file:
    corpus = file.read()

corpus = corpus.split()  

# Paramètres par défaut
L = 2
n = 100
k = 10
eta = 0.01
epochs = 10
min_count = 5

# Comptage des mots
words_counts = Counter(corpus)
total_word_count = sum(words_counts.values())

filtered_corpus = []
for word in corpus:
    if word in words_counts:
        # Calculer la probabilité de sous-échantillonnage
        frequency = words_counts[word] / total_word_count
        sampling_prob = 1 - np.sqrt(1e-4 / frequency)
        
        # Si on décide de garder le mot
        if random.random() >= sampling_prob:
            filtered_corpus.append(word)

filtered_words_counts = Counter(filtered_corpus)
vocab = {word for word in filtered_words_counts if filtered_words_counts[word] > min_count}
words_id = {word: i for i, word in enumerate(vocab)}  # Nouveau vocabulaire basé sur les mots restants
id_words = {i: word for i, word in enumerate(vocab)}

vocab_size = len(vocab)

pos = []
neg = []
eval = {}

for i, word in enumerate(filtered_corpus):
    if word in words_id:
        target_word_id = words_id[word]
        context_ids = []
        
        # Définir la fenêtre contextuelle
        start = max(0, i - (2 * L) + 1)
        end = min(len(filtered_corpus), i + (2 * L) + 1)
        for j in range(start, end):
            if i != j and filtered_corpus[j] in words_id:
                context_ids.append(words_id[filtered_corpus[j]])
                pos.append([words_id[filtered_corpus[j]], words_id[word]])

        # Sélection des échantillons négatifs
        negative_ids = random.choices(
            [w_id for w_id in words_id.values() if w_id not in context_ids],
            k=k
        )
        for x in negative_ids:
            neg.append([x, words_id[word]])

        for context in context_ids:
            for negative in negative_ids:
                eval[words_id[word]] = {
                    "context": context,
                    "negative": negative
                }

#  Fichier d'évaluation
with open("eval.txt", "w", encoding="utf-8") as eval_file:
    for word, value in eval.items():
        context_word = value["context"]
        negative_word = value["negative"]
        eval_file.write(f"{word} {context_word} {negative_word}\n")

with open("neg.txt", "w", encoding="utf-8") as neg_file:
    for pair in neg:
        neg_file.write(f"{pair[0]} {pair[1]}\n")

with open("pos.txt", "w", encoding="utf-8") as pos_file:
    for pair in pos:
        pos_file.write(f"{pair[0]} {pair[1]}\n")
