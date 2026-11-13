"""Train and evaluate a dense embedding model for Science2Voice career matching."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random

import torch
from huggingface_hub import snapshot_download
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer

from assessment import LABELS
from pathways import PATHWAYS, candidates
from prepare_data import DATA

DEFAULT_MODEL = "intfloat/multilingual-e5-small"
MODEL_FILES = [
    "config.json", "config_sentence_transformers.json", "modules.json", "sentence_bert_config.json",
    "model.safetensors", "1_Pooling/config.json", "tokenizer.json", "tokenizer_config.json",
    "sentencepiece.bpe.model", "special_tokens_map.json",
]
DESCRIPTIONS = {
    "primary_care": "Soins primaires, consultation de proximité, suivi des patients, prévention, orientation vers les services du district.",
    "clinical_research": "Appui à la recherche clinique, extraction de données, documentation des études, suivi des visites et revue de dossiers.",
    "care_quality": "Amélioration de la qualité clinique, audit de processus, organisation du parcours patient, analyse des retards et coordination d’équipe.",
    "programme_monitoring": "Suivi-évaluation de programmes, comparaison des résultats aux objectifs, construction d’indicateurs, tableaux de bord et rapport d’activité.",
    "field_surveillance": "Appui à la surveillance épidémiologique, analyse de notifications, synthèse par lieu et par semaine, contrôle des dates et examen des signaux.",
    "community_programmes": "Appui aux programmes communautaires, mobilisation des relais, organisation d’activités de quartier, coordination des actions de proximité.",
    "lab_quality": "Appui à la gestion qualité en laboratoire, audit documentaire, suivi des écarts, traçabilité et amélioration des procédures.",
    "diagnostic_workflow": "Opérations de laboratoire diagnostique, réception et cheminement des échantillons, suivi des délais, coordination des flux quotidiens.",
    "lab_research": "Appui à la recherche en laboratoire, documentation des projets, inventaire, métadonnées et rapprochement des fiches d’étude.",
    "frontline_care": "Pratique infirmière ou maïeutique de première ligne, soins directs, visites de proximité et accompagnement dans le champ de pratique.",
    "patient_education": "Éducation des patients, animation de séances, adaptation des messages, accompagnement des familles et coordination du suivi.",
    "ward_quality": "Qualité des services infirmiers ou de maternité, amélioration des transmissions, organisation du service et suivi documentaire.",
    "health_data_quality": "Systèmes d’information sanitaire, nettoyage et validation de données agrégées, détection d’incohérences et contrôle des rapports.",
    "digital_implementation": "Appui au déploiement d’outils numériques en santé, cartographie des processus, tests utilisateurs et accompagnement des équipes.",
    "lab_informatics": "Informatique de laboratoire, registres électroniques, identifiants et traçabilité des échantillons, contrôle des données de laboratoire.",
}


def read_records(split):
    with (DATA / f"{split}.jsonl").open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def serialize_profile(profile):
    parts = []
    for key, label in LABELS.items():
        value = profile.get(key)
        if value:
            value = ", ".join(value) if isinstance(value, list) else value
            parts.append(f"{label}: {value}")
    return "\n".join(parts)


def query(profile, family):
    prefix = "query: " if family == "e5" else ""
    return prefix + serialize_profile(profile)


def passage(code, family):
    prefix = "passage: " if family == "e5" else ""
    path = PATHWAYS[code]
    return f"{prefix}{path.title}. {DESCRIPTIONS[code]}"


def make_triplets(records, family):
    triplets = []
    for record in records:
        profile = record["profile"]
        anchor = query(profile, family)
        primary, alternative, pivot = record["reference"]
        aligned, pivots = candidates(profile["field"])
        pairs = [(primary, alternative)]
        pairs.extend((alternative, code) for code in aligned if code not in {primary, alternative})
        pairs.extend((pivot, code) for code in pivots if code != pivot)
        triplets.extend((anchor, passage(pos, family), passage(neg, family)) for pos, neg in pairs)
    return triplets


def load_model(args, name):
    location = Path(name)
    if not location.is_dir():
        weights = ["pytorch_model.bin"] if name == "BAAI/bge-m3" else []
        location = Path(snapshot_download(
            repo_id=name,
            allow_patterns=MODEL_FILES + weights,
        ))
    model = SentenceTransformer(str(location.resolve()), device=args.device)
    model.max_seq_length = args.max_length
    return model


def rule_scores(profile):
    activities = set(profile.get("activities", []))
    interests = set(profile.get("interests", []))
    return {
        code: 2 * len(activities.intersection(path.activities)) + int(path.domain in interests)
        for code, path in PATHWAYS.items()
    }


def top_three(profile, scores):
    aligned, pivots = candidates(profile["field"])
    return [*sorted(aligned, key=scores.__getitem__, reverse=True)[:2], max(pivots, key=scores.__getitem__)]


def predict(records, model=None, family="e5", batch_size=8):
    codes = list(PATHWAYS)
    if model is None:
        return [top_three(record["profile"], rule_scores(record["profile"])) for record in records]
    paths = model.encode(
        [passage(code, family) for code in codes], normalize_embeddings=True,
        batch_size=batch_size, show_progress_bar=False,
    )
    profiles = model.encode(
        [query(record["profile"], family) for record in records], normalize_embeddings=True,
        batch_size=batch_size, show_progress_bar=False,
    )
    scores = profiles @ paths.T
    return [top_three(record["profile"], dict(zip(codes, row))) for record, row in zip(records, scores)]


def summarize(records, predictions):
    groups = {}
    for record, actual in zip(records, predictions):
        reference = record["reference"]
        groups.setdefault(record["scenario_id"], []).append([
            int(actual[0] == reference[0]),
            int(set(actual[:2]) == set(reference[:2])),
            int(actual[2] == reference[2]),
            int(actual == reference),
        ])
    means = [[sum(column) / len(group) for column in zip(*group)] for group in groups.values()]
    values = [sum(column) / len(means) for column in zip(*means)]
    keys = ("primary_accuracy", "aligned_pair_accuracy", "pivot_accuracy", "ordered_top3_accuracy")
    return {"profiles": len(records), "scenarios": len(groups), **{key: round(val, 4) for key, val in zip(keys, values)}}


def collate(rows):
    return [list(column) for column in zip(*rows)]


def train(args):
    if args.epochs < 1 or (args.max_steps is not None and args.max_steps < 1):
        raise SystemExit("Use at least one epoch and one training step.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = load_model(args, args.model)
    examples = make_triplets(read_records("train"), args.family)
    development = read_records("dev")
    loader = DataLoader(
        examples, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
        generator=torch.Generator().manual_seed(args.seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    use_bf16 = args.device.startswith("cuda") and torch.cuda.is_bf16_supported()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    print(f"Training on {model.device} | {len(examples)} triplets | {'bf16 autocast' if use_bf16 else 'float32'}", flush=True)
    best = (-1.0, -1.0)
    steps = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for anchors, positives, negatives in loader:
            features = model.tokenize(anchors + positives + negatives)
            features = {key: val.to(model.device) for key, val in features.items()}
            optimizer.zero_grad(set_to_none=True)
            amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()
            with amp:
                embeddings = model(features)["sentence_embedding"].float()
            anchor, positive, negative = F.normalize(embeddings, dim=1).chunk(3)
            loss = F.relu(args.margin - (anchor * positive).sum(1) + (anchor * negative).sum(1)).mean()
            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            batches += 1
            steps += 1
            if args.max_steps is not None and steps >= args.max_steps:
                break

        dev_scores = summarize(development, predict(development, model, args.family, args.batch_size))
        print(f"epoch={epoch} steps={steps} loss={total_loss / batches:.5f} dev={json.dumps(dev_scores)}", flush=True)
        selection = (dev_scores["ordered_top3_accuracy"], dev_scores["pivot_accuracy"])
        if selection > best:
            best = selection
            model.save_pretrained(str(output / "model"))
            settings = {
                "initial_model": args.model, "family": args.family, "max_length": args.max_length,
                "learning_rate": args.learning_rate, "batch_size": args.batch_size,
                "margin": args.margin, "seed": args.seed, "best_epoch": epoch, "dev": dev_scores,
            }
            (output / "run.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
        if args.max_steps is not None and steps >= args.max_steps:
            break


def evaluate(args):
    records = read_records(args.split)
    variants = {"rules": predict(records)}
    base = load_model(args, args.model)
    variants["base"] = predict(records, base, args.family, args.batch_size)
    del base
    if args.tuned:
        tuned = load_model(args, args.tuned)
        variants["fine_tuned"] = predict(records, tuned, args.family, args.batch_size)
    metrics = {key: summarize(records, predictions) for key, predictions in variants.items()}
    print(json.dumps(metrics, indent=2))
    if args.report:
        comparisons = []
        for index, record in enumerate(records):
            comparisons.append({
                "profile_id": record["profile_id"], "scenario_id": record["scenario_id"],
                "reference": record["reference"], **{key: value[index] for key, value in variants.items()},
            })
        destination = Path(args.report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps({"split": args.split, "metrics": metrics, "comparisons": comparisons}, indent=2), encoding="utf-8")


def recommend(args):
    with Path(args.input).open(encoding="utf-8") as source:
        profile = json.load(source)
    model = load_model(args, args.model)
    prediction = predict([{"profile": profile}], model, args.family, args.batch_size)[0]
    for category, code in zip(("Primary", "Alternative", "Career pivot"), prediction):
        path = PATHWAYS[code]
        print(f"{category} | {path.title}\n  {path.first_step}\n  Entry check: {path.requirement}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--model", default=DEFAULT_MODEL)
    shared.add_argument("--family", choices=("e5", "bge"), default="e5")
    shared.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    shared.add_argument("--batch-size", type=int, default=8)
    shared.add_argument("--max-length", type=int, default=256)
    commands = parser.add_subparsers(required=True)
    training = commands.add_parser("train", parents=[shared])
    training.add_argument("--output", required=True)
    training.add_argument("--epochs", type=int, default=3)
    training.add_argument("--learning-rate", type=float, default=2e-5)
    training.add_argument("--margin", type=float, default=0.10)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--max-steps", type=int)
    training.set_defaults(run=train)
    evaluation = commands.add_parser("evaluate", parents=[shared])
    evaluation.add_argument("--split", choices=("dev", "test"), default="dev")
    evaluation.add_argument("--tuned")
    evaluation.add_argument("--report")
    evaluation.set_defaults(run=evaluate)
    inference = commands.add_parser("recommend", parents=[shared])
    inference.add_argument("--input", required=True)
    inference.set_defaults(run=recommend)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("PyTorch cannot see CUDA. Check nvidia-smi and your PyTorch installation.")
    if args.device == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    args.run(args)


if __name__ == "__main__":
    main()
