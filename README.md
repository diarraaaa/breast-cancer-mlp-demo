# Testeur du MLP : cancer du sein (TP1 Deep Learning)

Petite app web pour tester le réseau de neurones du TP1 (PyTorch, `30 → 16 → 8 → 1`) sur le dataset
Breast Cancer Wisconsin (Diagnostic).

**Démo :** https://diarraaaa.github.io/breast-cancer-mlp-demo/

- Chargez un patient du jeu de test (bénin, malin, aléatoire ou une erreur du modèle) ou saisissez les 30 caractéristiques à la main.
- Le modèle s'exécute dans le navigateur (JavaScript pur), sans serveur.
- La matrice de confusion et les métriques sur les 114 patients de test sont recalculées en direct.

| Métrique (test) | Valeur |
|---|---|
| Accuracy | 95,6 % |
| Précision (maligne) | 97,4 % |
| Rappel (maligne) | 90,5 % |
| F1-score | 93,8 % |
| AUC-ROC | 0,994 |

## Réentraîner

```bash
pip install torch scikit-learn numpy
python train.py   # régénère docs/model.json
```

`train.py` reproduit le pipeline du notebook : split stratifié 60/20/20 (`random_state=42`),
`StandardScaler` ajusté sur le train, SGD `lr=0.01`, batch 32, 75 époques.

> ⚠️ Projet pédagogique : ne constitue pas un outil de diagnostic médical.
