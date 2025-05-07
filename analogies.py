import numpy as np 
from scipy import spatial
import re

vecteurs = {}
fichier_plongements = 'model.txt'  

with open(fichier_plongements, 'r', encoding='utf-8', errors='ignore') as f:
    premiere_ligne = f.readline()  
    for ligne in f:
        valeurs = ligne.strip().split()
        if len(valeurs) < 101:
            continue  
        mot = valeurs[0]
        if not re.match(r'^[a-zA-Zéèêàçù]+$', mot): 
            continue
        try:
            vecteur = np.array(valeurs[1:], dtype=float)
            if vecteur.shape[0] != 100:
                continue
            vecteurs[mot] = vecteur
        except ValueError:
            continue


# Construction de l'arbre k-d
mots = list(vecteurs.keys())
embeddings = np.array([vecteurs[mot] for mot in mots])

arbre = spatial.KDTree(embeddings)

analogies = []

with open("analogietxt.txt", "r", encoding='utf-8') as f:
    for ligne in f:
        valeurs = ligne.strip().split()
        analogies.append(valeurs)


    # évaluation
correct = 0
total = 0
for a, b, c, d in analogies:
    if a not in vecteurs or b not in vecteurs or c not in vecteurs:
        print(f"L'un des mots '{a}', '{b}' ou '{c}' n'est pas dans les plongements.")
        continue

    vecteur_a = vecteurs[a]
    vecteur_b = vecteurs[b]
    vecteur_c = vecteurs[c]
    vecteur_resultat = vecteur_a - vecteur_b + vecteur_c

    distance, indice = arbre.query(vecteur_resultat)
    mot_proche = mots[indice]

    if mot_proche in [a, b, c]:
        distances, indices = arbre.query(vecteur_resultat, k=len(mots))
        mot_proche = None
        for idx in indices:
            candidat = mots[idx]
            if candidat not in [a, b, c]:
                mot_proche = candidat
                break

    if mot_proche is not None:
        total += 1
        print(f"{a} - {b} + {c} = {mot_proche} (attendu: {d})")
        if mot_proche == d:
            correct += 1
    else:
        print(f"Aucun mot trouvé pour l'analogie {a} - {b} + {c}.")


precision = (correct / total) * 100
print(f"Précision: {correct}/{total} ({precision:.2f}%)") 