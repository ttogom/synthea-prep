"""Screen generated guideline questions for diagnosis names.

The question step must describe the patient's findings, not guess a diagnosis:
its reply is the GraphRAG query, so a named diagnosis steers retrieval toward
it. This check never reads a patient's label. It looks for the name of any
study condition, or an item specific to one, that the patient's S/O does not
already contain; a condition already in the record may be named.
"""

import re

# Each group lists names for one condition or condition-specific item. A group
# counts as recorded, and so allowed, when any of its names appears in the S/O
# ("high blood pressure" is fine when "essential hypertension" is recorded).
# Covers the study conditions only, never a patient's own label.
DIAGNOSIS_TERM_GROUPS = (
    ("strep", "streptococcal", "streptococcus"),     # strep_throat
    ("pharyngitis",),                                # strep_throat, viral_pharyngitis
    ("tonsillitis",),
    ("sinusitis", "rhinosinusitis"),                 # bacterial_sinusitis
    ("cystitis",),                                   # cystitis
    ("urinary tract infection", "uti"),
    ("pyelonephritis",),
    ("heart failure", "chf", "cardiomyopathy"),      # heart_failure
    ("nyha", "new york heart association"),          # heart failure classification
    ("kccq", "kansas city cardiomyopathy questionnaire"),
    ("hypertension", "hypertensive", "antihypertensive", "high blood pressure"),  # hypertension
    ("copd", "chronic obstructive", "emphysema", "chronic bronchitis"),           # copd
)


def _has_phrase(text, phrase):
    return re.search(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", text) is not None


def find_diagnosis_terms(reply, s_o):
    """Diagnosis names in `reply` whose condition the S/O does not record, in list order."""
    reply, s_o = (reply or "").lower(), (s_o or "").lower()
    found = []
    for group in DIAGNOSIS_TERM_GROUPS:
        if any(_has_phrase(s_o, term) for term in group):
            continue
        found.extend(term for term in group if _has_phrase(reply, term))
    return found


def remove_diagnosis_terms(reply, terms):
    """Replace each term (and a following "symptoms") with "these findings"."""
    for term in sorted(terms, key=len, reverse=True):
        reply = re.sub(r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])(\s+symptoms)?",
                       "these findings", reply, flags=re.IGNORECASE)
    return reply
