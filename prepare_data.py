"""Build fictional assessment records with scenario-level splits."""
import csv
import json
from collections import Counter
from pathlib import Path

from assessment import ACTIVITIES, DIGITAL, LAB, MEDICINE, NURSING, PUBLIC_HEALTH
from pathways import PATHWAYS, candidates

DATA = Path(__file__).resolve().parent / "data"
TRACKS = {"med": MEDICINE, "ph": PUBLIC_HEALTH, "lab": LAB, "nur": NURSING}
DIPLOMAS = {
    "med": ("Doctorate", "Doctorat d’État en médecine", "Études de médecine"),
    "ph": ("Master’s degree", "Master en santé publique", "Master de santé publique"),
    "lab": ("Bachelor’s degree", "Licence en analyses biomédicales", "Licence en analyses biomédicales"),
    "nur": ("Professional training", "Diplôme d’État de sage-femme", "Formation de sage-femme"),
}
STAGES = {
    "student": ("Student", "In progress", "None", ""),
    "graduate": ("Looking for a job", "Completed", "Less than 1 year", ""),
    "worker": ("Employed", "Completed", "3 to under 5 years", "Professionnel de santé"),
}
ROLES = {"med": "Médecin généraliste", "ph": "Chargé de projet santé", "lab": "Technicien de laboratoire", "nur": "Sage-femme"}
MOTIVATION = {"med": "Impact on patients", "ph": "Community impact", "lab": "Scientific discovery", "nur": "Impact on patients"}


def profile_for(row, variant):
    track = row["track"]
    level, diploma, studies = DIPLOMAS[track]
    situation, status, years, role = STAGES[row["stage"]]
    experience = row["experience"]
    goal = row["goal"]
    skills = row["skills"]
    activities = [ACTIVITIES[int(index)] for index in row["activities"].split(",")]
    interests = [TRACKS[track]]
    if variant == 0:
        interests.append(DIGITAL if track in {"ph", "lab"} else PUBLIC_HEALTH)
        interests = list(dict.fromkeys(interests))
        narrative = f"{experience} Compétences déclarées : {skills}. {goal}"
    elif variant == 1:
        narrative = f"Objectif : {goal} Exemple de réalisation : {experience} Outils et compétences : {skills}."
    else:
        narrative = f"{experience} {goal}"
        activities = activities[:2]
    exposures = []
    if "Community health" in activities:
        exposures.append("Volunteering or community activities")
    if "Research" in activities:
        exposures.append("Contributed to a research project")
    return {
        "situation": situation, "field": TRACKS[track], "education": level, "education_status": status,
        "qualification": diploma if status == "Completed" and variant != 2 else "",
        "studies": studies if status == "In progress" else "", "experience": years,
        "role": ROLES[track] if role else "", "aspirations": narrative,
        "interests": interests, "activities": activities, "motivation": MOTIVATION[track],
        "country": "Senegal", "exposures": exposures,
    }


def main():
    groups = {split: [] for split in ("train", "dev", "test")}
    seen = set()
    with (DATA / "scenarios.tsv").open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source, delimiter="\t"):
            split = row["split"]
            scenario_id = row["id"]
            assert scenario_id not in seen
            seen.add(scenario_id)
            reference = [row["primary"], row["alternative"], row["pivot"]]
            aligned, pivots = candidates(TRACKS[row["track"]])
            assert len(set(reference)) == 3 and set(reference[:2]) <= set(aligned) and reference[2] in pivots
            for variant in range(3 if split == "train" else 2):
                groups[split].append({
                    "profile_id": f"{scenario_id}-{variant + 1}",
                    "scenario_id": scenario_id, "synthetic": True,
                    "profile": profile_for(row, variant), "reference": reference,
                })
    for split, records in groups.items():
        with (DATA / f"{split}.jsonl").open("w", encoding="utf-8") as target:
            for record in records:
                target.write(json.dumps(record, ensure_ascii=False) + "\n")
        counts = Counter(record["profile"]["field"] for record in records)
        print(f"{split}: {len(records)} profiles, {len({r['scenario_id'] for r in records})} scenarios, {dict(counts)}")
    with (DATA / "reference_examples.csv").open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(["profile_id", "field", "education_status", "qualification_or_studies", "assessment_narrative", "primary", "alternative", "pivot"])
        for records in groups.values():
            for record in records:
                if not record["profile_id"].endswith("-1"):
                    continue
                profile = record["profile"]
                writer.writerow([
                    record["profile_id"], profile["field"], profile["education_status"],
                    profile["qualification"] or profile["studies"], profile["aspirations"],
                    *(PATHWAYS[code].title for code in record["reference"]),
                ])


if __name__ == "__main__":
    main()
