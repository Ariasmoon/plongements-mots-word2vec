import numpy as np
import math

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def cosinus_similarity(x, y):
    dot = np.dot(x, y)
    norm1 = np.linalg.norm(x)
    norm2 = np.linalg.norm(y)
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot / (norm1 * norm2)

pos_file = "pos.txt"
neg_file = "neg.txt"
eval_file = "eval.txt"

# Chargement des données positives
pos_pairs = []
with open(pos_file, "r", encoding="utf-8") as f:
    for line in f:
        pos = [int(x) for x in line.split()]
        pos_pairs.append(pos)

# Chargement des données négatives
neg_pairs = []
with open(neg_file, "r", encoding="utf-8") as f:
    for line in f:
        neg = [int(x) for x in line.split()]
        neg_pairs.append(neg)

# Calcul de la taille du vocabulaire
vocab_size = len(set([pair[0] for pair in pos_pairs]))
print(vocab_size)
n = 100

# Initialisation des matrices M et C
M = np.random.randn(vocab_size + 15, n)
C = np.random.randn(vocab_size + 15, n)
print(M.shape)

eta = 0.01
epochs = 100

# Entraînement
for epoch in range(epochs):
    i = 0
    
    for pos_word, context_word in pos_pairs:
        m_pos = M[pos_word]
        c_pos = M[context_word]
        
        pos_score = sigmoid(np.dot(m_pos, c_pos))
        grad_pos = (pos_score - 1) * c_pos
        grad_c_pos = (pos_score - 1) * m_pos
        
        M[pos_word] -= eta * grad_pos
        C[context_word] -= eta * grad_c_pos
        i += 1

    for neg_word, context_word in neg_pairs:
        m_neg = M[neg_word]
        c_neg = M[context_word]
        
        neg_score = sigmoid(np.dot(m_neg, c_neg))
        grad_neg = neg_score * c_neg
        grad_c_neg = neg_score * m_neg
        
        M[neg_word] -= eta * grad_neg
        C[context_word] -= eta * grad_c_neg

embeddings = M

# Évaluation
correct = 0
nb_ligne = 0

with open(eval_file, "r", encoding="utf-8") as f:
    for line in f:
        nb_ligne += 1
        m, m_pos, m_neg = line.strip().split()
        vec_m = embeddings[int(m)]
        
        vec_m_pos = embeddings[int(m_pos)]
        vec_m_neg = embeddings[int(m_neg)]
        
        sim_pos = cosinus_similarity(vec_m, vec_m_pos)
        sim_neg = cosinus_similarity(vec_m, vec_m_neg)
        if sim_pos > sim_neg:
            correct += 1

accuracy = correct / nb_ligne
print(f"Précision : {accuracy * 100:.2f}%")
