"""The English pilot questionnaire and synthetic profiles."""
MEDICINE = "Medicine"
PUBLIC_HEALTH = "Public health"
LAB = "Laboratory / medical biology"
NURSING = "Nursing and midwifery"
DIGITAL = "Health data / digital"
FIELDS = (MEDICINE, NURSING, PUBLIC_HEALTH, LAB)
INTERESTS = (*FIELDS, DIGITAL, "Pharmacy", "Biomedical sciences", "Dentistry", "Animal health", "Other")
SITUATIONS = ("Student", "Employed", "Self-employed", "Looking for a job", "Career change", "Other")
EDUCATION = ("Secondary school", "Bachelor’s degree", "Master’s degree", "Doctorate", "Professional training", "Other")
EDUCATION_STATUS = ("Completed", "In progress")
EXPERIENCE = ("None", "Less than 1 year", "1 to under 3 years", "3 to under 5 years", "5 to 10 years", "More than 10 years")
ACTIVITIES = (
    "Patient care", "Research", "Laboratory work", "Data analysis", "Community health",
    "Teaching / training", "Management / coordination", "Policy / advocacy", "Entrepreneurship",
)
MOTIVATIONS = (
    "Impact on patients", "Community impact", "Scientific discovery", "Career growth",
    "Income and stability", "Independence", "Recognition",
)
EXPOSURES = (
    "Participated in a scientific publication", "Conducted surveys or data collection",
    "Volunteering or community activities", "Contributed to a research project",
    "Developed a personal or digital project", "None of these",
)
LABELS = {
    "situation": "Current situation", "field": "Academic or professional field",
    "education": "Education level", "education_status": "Education status",
    "qualification": "Completed qualification", "studies": "Current studies or specialty",
    "experience": "Professional experience", "role": "Recent professional role",
    "aspirations": "Experience, skills and aspirations", "interests": "Fields of interest",
    "activities": "Preferred activities", "motivation": "Main motivation",
    "country": "Preferred country", "exposures": "Other experiences",
}
DEMO_PROFILES = {
    "Medical student": {
        "situation": "Student", "field": MEDICINE, "education": "Doctorate",
        "education_status": "In progress", "studies": "Medical studies, with supervised clinical placements",
        "experience": "None", "interests": [MEDICINE, PUBLIC_HEALTH],
        "activities": ["Patient care", "Research"], "motivation": "Impact on patients",
        "country": "Senegal", "exposures": ["Volunteering or community activities"],
    },
    "Public health graduate": {
        "situation": "Looking for a job", "field": PUBLIC_HEALTH, "education": "Master’s degree",
        "education_status": "Completed", "qualification": "Master en santé publique",
        "experience": "Less than 1 year", "interests": [PUBLIC_HEALTH, DIGITAL],
        "activities": ["Data analysis", "Community health"], "motivation": "Community impact",
        "country": "Senegal", "exposures": ["Conducted surveys or data collection"],
    },
    "Medical laboratory professional": {
        "situation": "Employed", "field": LAB, "education": "Bachelor’s degree",
        "education_status": "Completed", "experience": "3 to under 5 years",
        "interests": [LAB, DIGITAL], "activities": ["Laboratory work", "Data analysis"],
        "motivation": "Career growth", "country": "Senegal", "exposures": ["None of these"],
    },
    "Early-career midwife": {
        "situation": "Employed", "field": NURSING, "education": "Professional training",
        "education_status": "Completed", "experience": "1 to under 3 years",
        "interests": [NURSING, PUBLIC_HEALTH],
        "activities": ["Patient care", "Community health"], "motivation": "Impact on patients",
        "country": "Senegal", "exposures": ["Volunteering or community activities"],
    },
}
