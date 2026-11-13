# Science2Voice · Expérience de recommandation de carrière

Le laboratoire reprend les champs de l’assessment Streamlit et les 15 familles de parcours du pilote. Tu entraînes un modèle d’embeddings à classer les parcours, puis tu mesures son apport sur les mêmes profils avec trois méthodes. L’expérience fonctionne sans API de génération.

## 1. Préparer le GPU

Dans un terminal Linux ou WSL, place-toi dans le dossier extrait `career_matching_lab`. Utilise Python 3.11 ou 3.12. Sur Windows natif, remplace `source .venv/bin/activate` par `.\.venv\Scripts\Activate.ps1` et `python3` par `py -3.12`.

```bash
nvidia-smi
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Installe PyTorch 2.6 ou plus récent en utilisant le sélecteur officiel, avec la commande CUDA recommandée par l’équipe du bootcamp pour le pilote installé. Si l’environnement fourni par le bootcamp dispose déjà d’un PyTorch CUDA fonctionnel, tu peux l’utiliser à la place de créer un venv.

- [Sélecteur PyTorch pour Linux ou Windows et CUDA](https://pytorch.org/get-started/locally/)

```bash
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

La dernière commande doit afficher le nom du GPU. Pour un essai sur CPU, utilise un PyTorch CPU et remplace `--device cuda` par `--device cpu`. Le premier lancement télécharge les poids depuis Hugging Face. Après mise en cache, tu peux définir `HF_HUB_OFFLINE=1` pour travailler sans réseau. En cas de mémoire GPU insuffisante, essaie `--batch-size 4`, puis `--batch-size 2`. Vérifie aussi que les autres utilisateurs du serveur laissent de la mémoire disponible.

## 2. Lire les données fictives et leurs cibles

```bash
python prepare_data.py
```

| Lot | Scénarios indépendants | Réponses au formulaire |
|---|---:|---:|
| Entraînement | 24 | 72 |
| Validation | 8 | 16 |
| Test final | 8 | 16 |

Le fichier `data/scenarios.tsv` contient les 40 scénarios écrits pour l’exercice. `data/reference_examples.csv` permet d’ouvrir dans Excel un exemple par scénario et ses trois libellés de référence. Les trois fichiers JSONL contiennent les réponses complètes, l’identifiant de groupe et la cible ordonnée. Les variantes changent l’ordre du récit et retirent certaines informations facultatives. Toutes les variantes d’un scénario restent dans le même lot. Les réponses combinent les choix en anglais du formulaire et des récits fictifs en français. Aucun dossier personnel n’est utilisé.

La cible contient deux options du domaine d’origine, dans l’ordre de préférence, et une transition choisie parmi des options exploratoires prédéfinies. Par exemple, un profil de laboratoire qui a documenté des doublons et veut informatiser un registre a pour cible l’appui à la gestion qualité, les opérations de laboratoire et l’appui à l’informatique de laboratoire. Les cibles sont des hypothèses pédagogiques, pas des avis d’experts ou des vérifications d’éligibilité.

## 3. Évaluer le modèle initial

```bash
python lab.py evaluate --device cuda --split dev --report runs/e5-before-dev.json
```

Le modèle par défaut, [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small), est un encodeur multilingue MIT. Il permet une première exécution rapide. Le script applique les préfixes `query:` et `passage:` prévus pour ce modèle. Le rapport donne aussi un classement de référence par activités et intérêts, avec les mêmes groupes de parcours autorisés.

## 4. Tester puis entraîner

```bash
python lab.py train --device cuda --epochs 1 --max-steps 2 --batch-size 4 --output runs/smoke
python lab.py train --device cuda --epochs 3 --batch-size 8 --learning-rate 0.00002 --output runs/e5
```

Chaque exemple d’entraînement compare un profil, un parcours préféré et une option moins pertinente du même groupe. Trois types de comparaisons sont construits. La première place le parcours principal devant le deuxième, la deuxième place ce deuxième devant les autres options du domaine et la troisième place la reconversion de référence devant les autres reconversions proposées. Le modèle apprend via une perte triplet sur la similarité cosinus. Le premier essai utilise une marge de 0,10 et des textes tronqués à 256 tokens. Le script utilise bfloat16 sur les GPU compatibles et float32 ailleurs.

Le meilleur checkpoint est choisi avec le taux de top 3 exact sur le lot de validation, puis le taux de pivot exact en cas d’égalité. Le dossier `runs/e5/model` contient le modèle exporté. Le fichier `runs/e5/run.json` enregistre les paramètres et les scores de l’époque retenue. Chaque entraînement doit avoir un nouveau dossier de sortie.

## 5. Comparer et générer une recommandation

```bash
python lab.py evaluate --device cuda --split dev --tuned runs/e5/model --report runs/e5-after-dev.json
python lab.py evaluate --device cuda --split test --tuned runs/e5/model --report runs/e5-test.json
python lab.py recommend --device cuda --model runs/e5/model --input data/example_profile.json
```

Lis les métriques suivantes.

- `primary_accuracy` mesure la proportion de premiers parcours conformes à la référence.
- `aligned_pair_accuracy` mesure la conformité des deux parcours du domaine, quel que soit leur ordre.
- `pivot_accuracy` mesure la proportion de reconversions conformes à la référence.
- `ordered_top3_accuracy` mesure la conformité des trois choix dans le bon ordre.

Les scores moyennent d’abord les variantes de chaque scénario. `comparisons` permet de relire les erreurs profil par profil. Garde le test pour la comparaison finale après avoir choisi les paramètres sur la validation. Les 8 scénarios de test donnent une indication pour l’exercice. Il faudra des profils réels anonymisés ou des scénarios indépendants validés par des professionnels au Sénégal pour mesurer la qualité terrain. Deux pistes peuvent être raisonnables même si une seule figure dans la référence.

`recommend` accepte directement un objet JSON reprenant les champs du formulaire Streamlit, comme `data/example_profile.json`. Le moteur affiche les premiers pas et les vérifications du catalogue. Les règles du pilote imposent deux options du domaine et une transition approuvée pour l’exploration. L’éligibilité à un poste, une formation ou un exercice réglementé reste à vérifier auprès de la structure concernée.

## 6. Comparer à BGE-M3 si le GPU est disponible

[BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) est un modèle multilingue MIT. Le script utilise sa représentation dense. L’option `--family bge` retire les préfixes E5. Fais cette comparaison sur la validation avant de choisir quel modèle évaluer sur le test.

```bash
python lab.py evaluate --device cuda --model BAAI/bge-m3 --family bge --batch-size 4 --split dev
python lab.py train --device cuda --model BAAI/bge-m3 --family bge --epochs 3 --batch-size 4 --output runs/bge
python lab.py evaluate --device cuda --model BAAI/bge-m3 --family bge --batch-size 4 --split dev --tuned runs/bge/model --report runs/bge-dev.json
```

Les réglages sont des points de départ. Avec 24 scénarios d’entraînement, un modèle plus grand peut aussi surapprendre. Retenir le modèle initial reste une conclusion valide si le fine-tuning ne fait pas progresser les résultats de validation.
