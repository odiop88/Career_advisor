"""Pilot role families, example settings and practical entry steps."""
from dataclasses import dataclass

from assessment import DIGITAL, LAB, MEDICINE, NURSING, PUBLIC_HEALTH

SOURCES = {
    "primary": (
        "WHO | Senegal primary health care and health system transformation",
        "https://extranet.who.int/uhcpartnership/country-profile/senegal",
        "WHO identifies primary health care as a focus in Senegal's health system transformation work.",
    ),
    "surveillance": (
        "WHO | Coordinated surveillance in Senegal and Guinea (2024)",
        "https://extranet.who.int/sph/surveillance-and-information-sharing-operational-tool-senegal-and-guinea",
        "A 2024 Senegal workshop produced a roadmap for coordinated zoonotic disease surveillance.",
    ),
    "digital": (
        "Senegal Ministry of Health | Health and social action sector policy (2025–2029)",
        "https://www.sante.gouv.sn/sites/default/files/Lettre%20de%20Politique%20Sectorielle%20du%20MSAS%202025-2029%20Signe%CC%81.pdf",
        "Senegal's 2025–2029 sector policy identifies digital health information systems and interoperability, and cites DHIS2 for routine health reporting.",
    ),
}

@dataclass(frozen=True)
class Pathway:
    title: str
    domain: str
    activities: tuple[str, str]
    setting: str
    first_step: str
    requirement: str
    source: str = ""

CLINICAL = "Complete the relevant training and verify the role’s supervision and credential requirements locally."
SUPPORT = "Check the organization’s entry criteria and supervision arrangements for internships or assistant roles."
LAB_SCOPE = "Confirm your permitted scope with the lab. Practical tasks require local authorization and appropriate supervision."
DATA = "Build basic spreadsheet and data quality skills. Use synthetic or public aggregate data for practice."

PATHWAYS = {
    "primary_care": Pathway(
        "Primary care practice", MEDICINE, ("Patient care", "Community health"),
        "Health centres and district care teams",
        "Ask a clinical supervisor about an observation or supervised primary care placement.", CLINICAL, "primary"),
    "clinical_research": Pathway(
        "Clinical research support", MEDICINE, ("Research", "Patient care"),
        "Hospital research units and research institutes",
        "Review introductory Good Clinical Practice material and ask a study team about supervised support.", SUPPORT),
    "care_quality": Pathway(
        "Clinical quality improvement", MEDICINE, ("Management / coordination", "Patient care"),
        "Hospital or district quality teams",
        "Draft a short audit plan for a care process using a fictional example.", SUPPORT, "primary"),
    "programme_monitoring": Pathway(
        "Health programme monitoring and evaluation", PUBLIC_HEALTH, ("Data analysis", "Management / coordination"),
        "District health programmes and NGOs",
        "Build a one-page indicator dashboard using synthetic programme data.", SUPPORT, "primary"),
    "field_surveillance": Pathway(
        "Disease surveillance support", PUBLIC_HEALTH, ("Data analysis", "Research"),
        "District surveillance teams and public health projects",
        "Practice summarising a synthetic line list by week and location.", SUPPORT, "surveillance"),
    "community_programmes": Pathway(
        "Community health programme support", PUBLIC_HEALTH, ("Community health", "Management / coordination"),
        "District outreach teams and health NGOs",
        "Draft an outreach activity plan with two simple monitoring indicators.", SUPPORT, "primary"),
    "lab_quality": Pathway(
        "Laboratory quality management support", LAB, ("Laboratory work", "Management / coordination"),
        "Clinical or public health laboratory quality teams",
        "Create a sample checklist for documenting specimen reception and deviations.", LAB_SCOPE),
    "diagnostic_workflow": Pathway(
        "Diagnostic laboratory operations", LAB, ("Laboratory work", "Patient care"),
        "Diagnostic laboratory teams",
        "Ask a lab supervisor to review a sketch of the specimen workflow.", LAB_SCOPE),
    "lab_research": Pathway(
        "Laboratory research support", LAB, ("Laboratory work", "Research"),
        "Biomedical research laboratories",
        "Ask a research lab about supervised work in sample tracking or documentation.", LAB_SCOPE),
    "frontline_care": Pathway(
        "Frontline nursing or midwifery practice", NURSING, ("Patient care", "Community health"),
        "Health centres and hospital units, within your own professional scope",
        "Discuss a supervised assignment in your own discipline with a clinical mentor.", CLINICAL, "primary"),
    "patient_education": Pathway(
        "Patient education and care coordination", NURSING, ("Teaching / training", "Patient care"),
        "Outpatient and community care teams",
        "Draft a patient-education session outline for review by a qualified colleague.", CLINICAL, "primary"),
    "ward_quality": Pathway(
        "Nursing or midwifery service quality", NURSING, ("Management / coordination", "Patient care"),
        "Ward or maternity unit quality teams",
        "Choose a documentation process and write a fictional improvement checklist.", SUPPORT, "primary"),
    "health_data_quality": Pathway(
        "Health information and data quality", DIGITAL, ("Data analysis", "Management / coordination"),
        "District reporting teams and health programme information units",
        "Validate a synthetic monthly report and record missing or inconsistent values.", DATA, "digital"),
    "digital_implementation": Pathway(
        "Digital health implementation support", DIGITAL, ("Management / coordination", "Teaching / training"),
        "Health information projects and clinical system implementation teams",
        "Map a fictional health workflow and write three user acceptance checks.", DATA, "digital"),
    "lab_informatics": Pathway(
        "Laboratory informatics support", DIGITAL, ("Laboratory work", "Data analysis"),
        "Laboratory information system and reporting teams",
        "Create a synthetic specimen register and check identifiers, dates and data completeness.", DATA, "digital"),
}

PIVOTS = {
    MEDICINE: ("field_surveillance", "digital_implementation", "health_data_quality"),
    PUBLIC_HEALTH: ("health_data_quality", "digital_implementation"),
    LAB: ("lab_informatics", "field_surveillance"),
    NURSING: ("community_programmes", "digital_implementation", "health_data_quality"),
}

def candidates(field: str) -> tuple[list[str], list[str]]:
    aligned = [code for code, path in PATHWAYS.items() if path.domain == field]
    return aligned, list(PIVOTS[field])
