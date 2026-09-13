"""Central configuration for the data-preparation stage (Role 1).

Every tunable choice lives here so the report can point to a single place
and so Roles 2 and 3 inherit the same encoding and inclusion rules.
"""
from pathlib import Path

# ----------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw" / "Survey_Results_UC.csv"
OUT = ROOT / "outputs"
FIG_DIR = OUT / "figures"
TAB_DIR = OUT / "tables"
PREP_JSON = OUT / "preparation_metrics.json"

# -------------------------------------------------------------- encoding
LIKERT = {
    "Strongly Disagree": -2,
    "Disagree": -1,
    "Neutral": 0,
    "Agree": 1,
    "Strongly Agree": 2,
}
# Responses that are not a position on the agreement scale -> treated as missing
NON_SUBSTANTIVE = {"No Comments"}

DOMAINS = {
    "T": "Technology",
    "E": "Education",
    "S": "Ethics & Society",
    "V": "Environment",
}
DOMAIN_COLORS = {"T": "#3B6FB6", "E": "#E08A2E", "S": "#8E5EA2", "V": "#3E9651"}

# ------------------------------------------------------- inclusion rules
MIN_ANSWERED_TOTAL = 45   # >= 75% of 60 items for the aggregate network
MIN_ANSWERED_LAYER = 12   # >= 80% of 15 items for a topic layer

# A domain answered <= this many of its 15 items counts as a skipped block rather
# than scattered non-response (audit classification only; no effect on inclusion).
MAX_ANSWERED_SKIPPED_BLOCK = 5

# ------------------------------------------------- network construction
# k for the symmetric k-nearest-neighbour backbone; None -> round(sqrt(n))
# Role 1 recommends this default; Role 2 owns the final choice and Role 3 the
# sensitivity scan over k.
K_NEIGHBOURS = None

# ------------------------------------------------------------ inference
SEED = 42
N_PARALLEL = 200        # replicates for parallel analysis (Horn, 1965)

# ------------------------------------------------- short item labels
SHORT = {
    "T01": "AI benefits > harms", "T02": "GenAI as learning aid",
    "T03": "Disclose AI use", "T04": "AI education for engineers",
    "T05": "AI accelerates science", "T06": "Automation replaces jobs",
    "T07": "Robots in hazardous jobs", "T08": "Routine AI diagnosis",
    "T09": "AVs make roads safer", "T10": "Privacy > convenience",
    "T11": "Explain recommenders", "T12": "Stricter AI regulation",
    "T13": "Cybersecurity > new tech", "T14": "Algorithms polarise",
    "T15": "Reliance on AI decisions",
    "E01": "Projects > exams", "E02": "Exams measure knowledge",
    "E03": "Compulsory attendance", "E04": "Online complements class",
    "E05": "Courses outside major", "E06": "Research for every UG",
    "E07": "Publishing optional", "E08": "Honesty > grades",
    "E09": "Collaborative > individual", "E10": "Innovation > rote",
    "E11": "Update curricula faster", "E12": "AI in teaching",
    "E13": "Research > infrastructure", "E14": "Industry collaboration",
    "E15": "Lifelong learning",
    "S01": "Duty to society", "S02": "Community service",
    "S03": "Assess tech impact first", "S04": "Platforms accountable",
    "S05": "Control over own data", "S06": "Shared misinfo duty",
    "S07": "Equal opportunity", "S08": "Diverse teams better",
    "S09": "Inclusive discourse", "S10": "Evidence-based policy",
    "S11": "Free expression w/o harm", "S12": "Youth civic action",
    "S13": "Teach critical evaluation", "S14": "Ethical leadership",
    "S15": "Public trust in tech",
    "V01": "Urgent climate action", "V02": "Renewables > fossil",
    "V03": "Cut personal energy use", "V04": "Promote public transport",
    "V05": "Restrict single-use plastic", "V06": "Water conservation",
    "V07": "Biodiversity essential", "V08": "Green campus despite cost",
    "V09": "Design for reuse/repair", "V10": "Eco-conscious buying",
    "V11": "Tech solves environment", "V12": "Sustainability in curricula",
    "V13": "Corporate eco-accountability", "V14": "International cooperation",
    "V15": "Duty to future generations",
}
