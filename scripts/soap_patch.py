#!/usr/bin/env python3
"""Hard-coded fixes to SOAP outputs where the model changed or invented content.

  1. PATCHES: exact (old -> new) replacements for values the model changed, each checked
     against the source record. `old` must occur exactly once in the field.
  2. Wording rules (subjective/objective only), for things the source never says:
       - "(severity 38/100)"  -> "(severity 38)"     source severities have no scale (some exceed 100)
       - "Immunizations up to date / current [per record][, including] X"
                              -> "Immunizations: X"  (sentence dropped when there is no list)

Usage:
  python3 soap_patch.py [--split dev corpus ...] [--dir soap]    # dry run: print every change
  python3 soap_patch.py --apply [...]                             # write; originals kept as *.prepatch

Idempotent: a patch whose `new` is already present and `old` absent is skipped, and the
wording rules find nothing on a second pass.
"""
import argparse, json, os, re, sys
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(REPO_ROOT, "data", "release-v2")
SPLITS = ["dev", "corpus", "dev_nota", "corpus_nota", "test", "test_nota"]
TEXT_FIELDS = ["subjective", "objective"]  # corpus assessment/plan are left alone

# (split, id, field, old, new)
PATCHES = [
    # changed values
    ("dev", "strep_throat_154197d8-b190-f562-ded5-cbaa6c3a2366", "objective",
     "blood pressure 139/78 mmHg", "blood pressure 139/79 mmHg"),
    ("dev", "viral_pharyngitis_03e54fc2-5682-15ec-dfac-1dd652004930", "objective",
     "blood pressure 92/74 mm[Hg] (systolic 92, diastolic 85)", "blood pressure 92/85 mm[Hg]"),
    ("dev", "viral_pharyngitis_76d5d263-befc-7982-42c3-f2f1bbb8b34b", "objective",
     "blood pressure 92/68 mmHg", "blood pressure 92/69 mmHg"),
    ("dev", "viral_pharyngitis_4465b27a-8f60-b22d-d4ec-7fb291909832", "objective",
     "blood pressure 90/98 mm Hg", "blood pressure 90/86 mm Hg"),
    ("dev", "copd_de21bf07-f1b1-b75c-1a5f-0df233bee76f", "objective",
     "BP 126/121 mmHg systolic range", "BP 126/76 mmHg"),
    ("corpus", "cystitis_9f41ad76-5e44-5ac4-3b2d-e75b669e8070", "objective",
     "BP 130/118 mmHg", "BP 130/77 mmHg"),
    ("dev", "viral_pharyngitis_28e961a9-9981-528b-a921-b598d28c3b2d", "objective",
     "Magnesium 1.8 mg/dL", "Magnesium 1.9 mg/dL"),
    ("dev", "viral_pharyngitis_c981c989-19b2-4beb-a4ab-a3e6f111e54e", "objective",
     "height 168.9 cm", "height 166.9 cm"),
    ("corpus", "copd_2f92f659-cc3f-69fc-1eea-7d72d6c628bf", "objective",
     "onset 32 years prior to age 90", "since 58 years earlier"),
    ("dev", "strep_throat_5c4a5938-6e31-2270-3b9f-f4199367d204", "subjective",
     "since approximately 10 years ago", "since approximately 12 years ago"),
    ("dev", "strep_throat_93eed6b0-4253-715e-1ad2-8127cdfe7090", "objective",
     "BMI-for-age 67th percentile", "BMI-for-age 67.1th percentile"),
    ("dev_nota", "nota_viral_sinusitis_07a4d117-29c7-8156-13d5-a79b7d4b99f9", "objective",
     "BMI-for-age 81st percentile", "BMI-for-age 81.2th percentile"),
    # BP with systolic and diastolic from different readings
    ("dev", "copd_4db2bfa5-839e-7c49-561c-8516c32821ef", "objective",
     "BP 109/65 mmHg", "BP 109/64 mmHg"),
    ("dev", "copd_9263eeea-78af-23a4-fea6-bbe3a158a986", "objective",
     "BP 158/90 mmHg", "BP 158/89 mmHg"),
    ("dev", "hypertension_6e298476-a3f1-9a90-e295-0f3fb589216d", "objective",
     "118/72 mmHg (4 years earlier)", "115/72 mmHg (4 years earlier)"),
    ("dev", "hypertension_8b06d801-cbc2-5f7a-e1eb-8a256287133e", "objective",
     "BP 114/75 mmHg", "BP 114/81 mmHg"),
    ("dev", "strep_throat_374411a9-e2ff-497b-f7a3-72b6719a24b9", "objective",
     "blood pressure 106/68 mmHg", "blood pressure 106/66 mmHg"),
    ("dev", "viral_pharyngitis_ac4dab32-3306-8d5b-1c29-ab82f975cead", "objective",
     "body temperature 37.3°C, heart rate 81/min, respiratory rate 16/min, blood pressure 99/62 mmHg "
     "(systolic 99 mm[Hg], diastolic 59 mm[Hg] from visit 15 days earlier), body weight",
     "body temperature 37.3°C. Vitals 15 days earlier: heart rate 81/min, respiratory rate 16/min, "
     "blood pressure 99/59 mmHg, body weight"),
    ("dev_nota", "nota_viral_sinusitis_194d679d-0fd8-6bfc-963d-481a82dcf7af", "objective",
     "blood pressure 109/73 mm[Hg]", "blood pressure 109/75 mm[Hg]"),
    ("corpus", "viral_pharyngitis_dde15a44-d16c-6a70-f468-06d6b14cf1ad", "objective",
     "blood pressure 101/74 mmHg", "blood pressure 101/71 mmHg"),
    ("corpus", "viral_pharyngitis_62719a80-56da-c740-8445-1fec268fb10b", "objective",
     "blood pressure 131/70 mm[Hg] (systolic 14 days earlier, diastolic 1 year earlier)",
     "blood pressure 131/75 mm[Hg] (14 days earlier)"),
    ("corpus_nota", "nota_viral_sinusitis_c03d4df0-0052-c64a-31e9-e3f93d3d36ee", "objective",
     "blood pressure 116/82 mmHg (diastolic from 10 months earlier: 82 mmHg)", "blood pressure 116/77 mmHg"),
    # values given the wrong date
    ("dev", "viral_pharyngitis_a8584e96-f522-c166-e4eb-8f7efde95e79", "objective",
     "Most recent CBC (20 days earlier)", "Most recent CBC (1 day earlier)"),
    ("corpus", "strep_throat_585f7a16-c854-c8c1-1869-53d2343c5eaf", "objective",
     "body weight 87.2 kg, body height", "body weight 87.2 kg (1 day earlier), body height"),
    ("corpus", "viral_pharyngitis_d06cb73d-3bb4-bad0-07f5-45da98d4dc7e", "objective",
     "creatinine 1.9 mg/dL (serum, 7 days earlier)", "creatinine 1.9 mg/dL (serum, 14 days earlier)"),
    ("corpus", "viral_pharyngitis_d06cb73d-3bb4-bad0-07f5-45da98d4dc7e", "objective",
     "sodium 139.8 mmol/L, potassium 5.1 mmol/L,",
     "sodium 139.8 mmol/L (serum, 7 days earlier), potassium 5.1 mmol/L (serum, 7 days earlier),"),
    # diagnosis-day temperature labelled as an earlier visit's vital sign
    ("dev", "strep_throat_846d13cd-f7aa-b236-3192-6cf1f3f2f0ca", "objective",
     "Vital signs (7 days prior): temperature 37.8°C, blood pressure",
     "Body temperature at this visit: 37.8°C. Vital signs (7 days prior): blood pressure"),
    ("dev", "strep_throat_af1d5990-d17c-3cdf-7a65-971fd4fbfd3d", "objective",
     "Vitals (most recent visit, 7 days ago): temperature 39.1°C, blood pressure",
     "Body temperature at this visit: 39.1°C. Vitals (7 days ago): blood pressure"),
    ("dev", "strep_throat_fc02999a-156a-f59d-8dc5-c06181b34b3f", "objective",
     "Vital signs at urgent care visit (7 days prior): body temperature 37.2°C, heart rate",
     "Body temperature at this visit: 37.2°C. Vital signs at urgent care visit (7 days prior): heart rate"),
    ("dev", "viral_pharyngitis_045ad4d3-5b59-f942-7f2b-e2fb5497652b", "objective",
     "Vitals (1 month prior): Temperature 37.4°C, HR",
     "Body temperature at this visit: 37.4°C. Vitals (1 month prior): HR"),
    ("dev", "viral_pharyngitis_13bcf79f-25fa-4369-b277-adf7f8f8f82b", "objective",
     "Vital signs at urgent care visit (7 days ago): temperature 37.1°C, blood pressure",
     "Body temperature at this visit: 37.1°C. Vital signs at urgent care visit (7 days ago): blood pressure"),
    ("dev", "viral_pharyngitis_1b658199-982f-7307-938e-63a24747dde8", "objective",
     "Vitals at urgent care visit (7 days prior): temperature 37.4°C, heart rate",
     "Body temperature at this visit: 37.4°C. Vitals at urgent care visit (7 days prior): heart rate"),
    ("dev", "viral_pharyngitis_8207aeaf-ece7-d24a-e2a9-8a0788ac28ef", "objective",
     "Vital signs at urgent care visit (14 days prior): temperature 37.8°C, heart rate",
     "Body temperature at this visit: 37.8°C. Vital signs at urgent care visit (14 days prior): heart rate"),
    ("dev", "viral_pharyngitis_95add792-debc-8c4c-07de-f07356e25eb9", "objective",
     "Vital signs (14 days prior): temperature 37.4°C at current visit; blood pressure",
     "Body temperature at this visit: 37.4°C. Vital signs (14 days prior): blood pressure"),
    ("dev", "viral_pharyngitis_ac9b55d9-6e54-b839-51f8-737e3fca2c31", "objective",
     "Vital signs (8 days ago): temperature 37.6°C, BP",
     "Body temperature at this visit: 37.6°C. Vital signs (8 days ago): BP"),
    ("dev", "viral_pharyngitis_ad58a9da-3cb0-cdc5-311f-000ff0ab0dbf", "objective",
     "Vital signs (7 days prior): temperature 37.8°C, blood pressure",
     "Body temperature at this visit: 37.8°C. Vital signs (7 days prior): blood pressure"),
    ("dev", "viral_pharyngitis_c4706e08-0fa9-75b6-3e64-c3ce9b81603e", "objective",
     "Vitals (7 days prior): temperature 37.6°C, heart rate",
     "Body temperature at this visit: 37.6°C. Vitals (7 days prior): heart rate"),
    ("dev", "viral_pharyngitis_d7ac0fb8-5d6c-9383-3cda-1edecbdbf15a", "objective",
     "Vital signs at urgent care visit (7 days earlier): temperature 37.4°C, heart rate",
     "Body temperature at this visit: 37.4°C. Vital signs at urgent care visit (7 days earlier): heart rate"),
    ("dev", "viral_pharyngitis_dafdbebd-e31e-b397-7221-f6986691c9d0", "objective",
     "Vital signs (7 days prior): temperature 37.3°C, heart rate",
     "Body temperature at this visit: 37.3°C. Vital signs (7 days prior): heart rate"),
    ("dev", "viral_pharyngitis_dfac38ef-e96c-0a13-08fb-6639c9743546", "objective",
     "Vitals (most recent, 8 days prior): temperature 37.5°C, blood pressure",
     "Body temperature at this visit: 37.5°C. Vitals (8 days prior): blood pressure"),
    ("dev", "viral_pharyngitis_e33c40e0-711a-ed72-22b2-4584fde357cc", "objective",
     "Vitals at most recent visit (7 days prior): body temperature 37.7°C, heart rate",
     "Body temperature at this visit: 37.7°C. Vitals (7 days prior): heart rate"),
    # same-day value reported as the visit value: visit-line value first, other reading labelled
    ("corpus", "viral_pharyngitis_d91ebed7-034d-b36b-f135-7995dacaf428", "objective",
     "Body temperature at this visit: 39.4°C.",
     "Body temperature at this visit: 37.6°C; also recorded same day: 39.4°C."),
    ("dev", "copd_9b64a78b-143d-f52d-a09d-374100e0718b", "objective",
     "BP 132/80 mmHg (systolic per presenting complaint record), heart rate 100/min, respiratory rate 16/min.",
     "BP 132/80 mmHg, heart rate 74/min, respiratory rate 12/min. "
     "Also recorded same day: BP 125/79 mmHg, heart rate 100/min, respiratory rate 16/min."),
    ("dev", "cystitis_d7b322a5-7a10-ff62-4e75-e3cff460b5c7", "objective",
     "BP 128/90 mmHg (systolic/diastolic per presenting complaint record); "
     "repeat labs show BP 123/93 mmHg, HR 88/min, RR 15/min.",
     "BP 128/90 mmHg, HR 68/min, RR 15/min; also recorded same day: BP 123/93 mmHg, HR 88/min, RR 15/min."),
    # immunization claims the rule below can't rewrite
    ("dev", "hypertension_d5a42573-f20f-cf34-8897-35516adf366e", "objective",
     "Zoster and influenza vaccinations up to date.",
     "Immunizations: zoster vaccine (1 year earlier), influenza (most recent 1 year earlier)."),
    ("dev", "bacterial_sinusitis_4c522f16-07ed-25d2-754a-0a35c2861c0c", "objective",
     "Influenza and hepatitis A immunizations current.",
     "Immunizations: influenza (most recent 12 months earlier), hepatitis A (12 months earlier)."),
    ("dev_nota", "nota_viral_sinusitis_0e6eca2b-5b43-a6d4-14c2-f8296a90257c", "objective",
     "Most recent influenza immunization current.", "Most recent influenza immunization at this visit."),
    # earlier-dated values presented as current-visit values: visit-day values keep the
    # "current visit" header, the rest get their own sentence with the source date
    ("dev", "strep_throat_01f84063-c556-13ef-aa6c-275862b4e495", "objective",
     "Vital signs at current visit: temperature 38.3°C, heart rate 95/min, respiratory rate 13/min, "
     "systolic blood pressure 137 mmHg, diastolic blood pressure 88 mmHg.",
     "Vital signs at current visit: temperature 38.3°C. Vital signs 7 months earlier: heart rate 95/min, "
     "respiratory rate 13/min, systolic blood pressure 137 mmHg, diastolic blood pressure 88 mmHg."),
    ("dev", "strep_throat_07a094aa-6055-dc7a-ddff-ace7f7a74791", "objective",
     "Vitals at this visit: body temperature 38.0°C, heart rate 77/min, respiratory rate 12/min, blood "
     "pressure 113/70 mmHg, body weight 16.3 kg, body height 97.6 cm, BMI 17.1 kg/m², BMI-for-age 87.6th "
     "percentile, weight-for-length 93.3rd percentile, head circumference 48.1 cm (17.3rd percentile).",
     "Vitals at this visit: body temperature 38.0°C. Vitals 26 days earlier: heart rate 77/min, "
     "respiratory rate 12/min, blood pressure 113/70 mmHg, body weight 16.3 kg, body height 97.6 cm, BMI "
     "17.1 kg/m², BMI-for-age 87.6th percentile, weight-for-length 93.3rd percentile, head circumference "
     "48.1 cm (17.3rd percentile)."),
    ("dev", "strep_throat_07a4d117-29c7-8156-13d5-a79b7d4b99f9", "objective",
     "Vital signs at current visit: temperature 37.9°C, heart rate 86/min, respiratory rate 14/min, blood "
     "pressure 113/84 mmHg.",
     "Vital signs at current visit: temperature 37.9°C. Vital signs 7 days earlier: heart rate 86/min, "
     "respiratory rate 14/min, blood pressure 113/84 mmHg."),
    ("dev", "strep_throat_0b22ff4b-8a3c-ac55-281a-b7c227a37e4e", "objective",
     "Vital signs at current visit: body temperature 37.6°C, blood pressure 105/77 mmHg, heart rate "
     "81/min, respiratory rate 15/min.",
     "Vital signs at current visit: body temperature 37.6°C. Vital signs 7 days earlier: blood pressure "
     "105/77 mmHg, heart rate 81/min, respiratory rate 15/min."),
    ("dev", "strep_throat_0e146c85-d39a-17f3-f82b-47e0456a2c87", "objective",
     "Vital signs at this visit: body temperature 38.4°C, heart rate 90/min, respiratory rate 14/min, "
     "blood pressure 135/92 mmHg.",
     "Vital signs at this visit: body temperature 38.4°C. Vital signs 1 month earlier: heart rate 90/min, "
     "respiratory rate 14/min, blood pressure 135/92 mmHg."),
    ("dev", "strep_throat_13f7cda5-2a12-1df6-32c1-d89ccbf28d06", "objective",
     "Vital signs at current visit: body temperature 37.6°C, heart rate 81/min, respiratory rate 14/min, "
     "blood pressure 138/80 mmHg.",
     "Vital signs at current visit: body temperature 37.6°C. Vital signs 14 days earlier: heart rate "
     "81/min, respiratory rate 14/min, blood pressure 138/80 mmHg."),
    ("dev", "strep_throat_150cb6e8-d527-af70-50e7-71d4e0e64d8b", "objective",
     "Vital signs at current visit: temperature 38.1°C, heart rate 85/min, respiratory rate 14/min, "
     "systolic BP 129 mmHg, diastolic BP 78 mmHg.",
     "Vital signs at current visit: temperature 38.1°C. Vital signs 23 days earlier: heart rate 85/min, "
     "respiratory rate 14/min, systolic BP 129 mmHg, diastolic BP 78 mmHg."),
    ("dev", "strep_throat_1691247e-5686-82fc-ceeb-4944bdeccfad", "objective",
     "Vital signs at current visit: body temperature 39.1°C, heart rate 94/min, respiratory rate 14/min, "
     "blood pressure 133/81 mm[Hg].",
     "Vital signs at current visit: body temperature 39.1°C. Vital signs 7 months earlier: heart rate "
     "94/min, respiratory rate 14/min, blood pressure 133/81 mm[Hg]."),
    ("dev", "strep_throat_17ed8063-3c38-7f12-36df-408911470d13", "objective",
     "Vital signs at current visit: body temperature 37.6°C, heart rate 60/min, respiratory rate 15/min, "
     "blood pressure 120/72 mmHg.",
     "Vital signs at current visit: body temperature 37.6°C. Vital signs 21 days earlier: heart rate "
     "60/min, respiratory rate 15/min, blood pressure 120/72 mmHg."),
    ("dev", "strep_throat_182c58fa-0f2e-655a-1e77-9c8c7bf976c2", "objective",
     "Vital signs at current visit: temperature 38.3°C, heart rate 94/min, respiratory rate 14/min, blood "
     "pressure 136/74 mmHg.",
     "Vital signs at current visit: temperature 38.3°C. Vital signs 14 days earlier: heart rate 94/min, "
     "respiratory rate 14/min, blood pressure 136/74 mmHg."),
    ("dev", "strep_throat_1afb5812-f8a0-aedd-30ab-5afe0c32400b", "objective",
     "Vital signs at current visit: temperature 38.0°C, heart rate 96/min, blood pressure 123/75 mmHg, "
     "respiratory rate 13/min.",
     "Vital signs at current visit: temperature 38.0°C. Vital signs 14 days earlier: heart rate 96/min, "
     "blood pressure 123/75 mmHg, respiratory rate 13/min."),
    ("dev", "strep_throat_27c5008c-158b-7649-28b4-4db0649305b2", "objective",
     "Vital signs at current visit: body temperature 38.0°C, heart rate 63/min, respiratory rate 13/min, "
     "blood pressure 126/88 mmHg.",
     "Vital signs at current visit: body temperature 38.0°C. Vital signs 9 days earlier: heart rate "
     "63/min, respiratory rate 13/min, blood pressure 126/88 mmHg."),
    ("dev", "strep_throat_28e961a9-9981-528b-a921-b598d28c3b2d", "objective",
     "Vital signs at current visit: temperature 38.4°C, heart rate 87/min, respiratory rate 13/min, blood "
     "pressure 101/83 mmHg.",
     "Vital signs at current visit: temperature 38.4°C. Vital signs 14 days earlier: heart rate 87/min, "
     "respiratory rate 13/min, blood pressure 101/83 mmHg."),
    ("dev", "strep_throat_2f122251-30bd-8fbd-37f6-6b512467f2f1", "objective",
     "Vital signs at current visit: body temperature 37.6°C, heart rate 98/min, respiratory rate 15/min, "
     "systolic blood pressure 143 mmHg, diastolic blood pressure 86 mmHg.",
     "Vital signs at current visit: body temperature 37.6°C. Vital signs 7 days earlier: heart rate "
     "98/min, respiratory rate 15/min, systolic blood pressure 143 mmHg, diastolic blood pressure 86 mmHg."),
    ("dev", "strep_throat_2fe8b8d9-611e-dcce-cb29-e3e13329928c", "objective",
     "Vital signs at this visit: body temperature 38.3°C, heart rate 83/min, respiratory rate 15/min, "
     "systolic blood pressure 115 mmHg, diastolic blood pressure 77 mmHg.",
     "Vital signs at this visit: body temperature 38.3°C. Vital signs 6 days earlier: heart rate 83/min, "
     "respiratory rate 15/min, systolic blood pressure 115 mmHg, diastolic blood pressure 77 mmHg."),
    ("dev", "strep_throat_34f61770-0b28-7963-9247-964f206310d0", "objective",
     "Vital signs at current visit: temperature 38.8°C, heart rate 94/min, respiratory rate 16/min, blood "
     "pressure 98/78 mmHg, body weight 67.3 kg, height 154.6 cm, BMI 28.1 kg/m².",
     "Vital signs at current visit: temperature 38.8°C. Vital signs 14 days earlier: heart rate 94/min, "
     "respiratory rate 16/min, blood pressure 98/78 mmHg, body weight 67.3 kg, height 154.6 cm, BMI 28.1 "
     "kg/m²."),
    ("dev", "strep_throat_36c5c383-64f7-4f19-2258-3c91129e143c", "objective",
     "Vital signs at current visit: temperature 38.3°C, heart rate 92/min, blood pressure 113/88 mmHg, "
     "respiratory rate 15/min.",
     "Vital signs at current visit: temperature 38.3°C. Vital signs 21 days earlier: heart rate 92/min, "
     "blood pressure 113/88 mmHg, respiratory rate 15/min."),
    ("dev", "strep_throat_37e0119a-8425-c083-d94f-0ae7819fee14", "objective",
     "Vital signs at this visit: body temperature 38.8°C, heart rate 87/min, respiratory rate 15/min, "
     "blood pressure 132/85 mmHg.",
     "Vital signs at this visit: body temperature 38.8°C. Vital signs 11 months earlier: heart rate "
     "87/min, respiratory rate 15/min, blood pressure 132/85 mmHg."),
    ("dev", "strep_throat_400e8288-d355-3beb-6208-074e29bbf3c3", "objective",
     "Vital signs at current visit: temperature 39.0°C, heart rate 89/min, respiratory rate 13/min, blood "
     "pressure 119/78 mmHg.",
     "Vital signs at current visit: temperature 39.0°C. Vital signs 7 days earlier: heart rate 89/min, "
     "respiratory rate 13/min, blood pressure 119/78 mmHg."),
    ("dev", "strep_throat_4b18aff0-57fd-144e-7153-9755e365fab6", "objective",
     "Vital signs at current visit: body temperature 39.0°C, heart rate 92/min, respiratory rate 15/min, "
     "systolic blood pressure 142 mmHg, diastolic blood pressure 88 mmHg.",
     "Vital signs at current visit: body temperature 39.0°C. Vital signs 3 days earlier: heart rate "
     "92/min, respiratory rate 15/min, systolic blood pressure 142 mmHg, diastolic blood pressure 88 mmHg."),
    ("dev", "strep_throat_4d01e7ca-2654-a3b2-b660-2173402feeb5", "objective",
     "Vital signs at current visit: body temperature 38.9°C, heart rate 92/min, respiratory rate 12/min, "
     "blood pressure 110/71 mmHg.",
     "Vital signs at current visit: body temperature 38.9°C. Vital signs 14 days earlier: heart rate "
     "92/min, respiratory rate 12/min, blood pressure 110/71 mmHg."),
    ("dev", "strep_throat_50c04e74-04b6-db66-1965-ad0720cf6929", "objective",
     "Vital signs at current visit: temperature 38.2°C, heart rate 74/min, respiratory rate 14/min, blood "
     "pressure 121/80 mmHg.",
     "Vital signs at current visit: temperature 38.2°C. Vital signs 25 days earlier: heart rate 74/min, "
     "respiratory rate 14/min, blood pressure 121/80 mmHg."),
    ("dev", "strep_throat_52dfb8b2-0bb4-7780-041f-a98faa0f1a3f", "objective",
     "Vital signs at this visit: body temperature 38.6°C, heart rate 100/min, respiratory rate 16/min, "
     "systolic blood pressure 97 mmHg, diastolic blood pressure 64 mmHg.",
     "Vital signs at this visit: body temperature 38.6°C. Vital signs 21 days earlier: heart rate 100/min, "
     "respiratory rate 16/min, systolic blood pressure 97 mmHg, diastolic blood pressure 64 mmHg."),
    ("dev", "strep_throat_60177dfc-96c6-fdc1-d0d3-33f4243c3e0d", "objective",
     "Vital signs at this visit: body temperature 39.1°C, heart rate 87/min, respiratory rate 13/min, "
     "systolic blood pressure 120 mmHg, diastolic blood pressure 100 mmHg.",
     "Vital signs at this visit: body temperature 39.1°C. Vital signs 14 days earlier: heart rate 87/min, "
     "respiratory rate 13/min, systolic blood pressure 120 mmHg, diastolic blood pressure 100 mmHg."),
    ("dev", "strep_throat_6c1a2ac5-e6cf-5b4e-f0fe-6340f6bc5a0e", "objective",
     "Vital signs at current visit: body temperature 37.6°C, blood pressure 133/75 mmHg, heart rate "
     "63/min, respiratory rate 15/min.",
     "Vital signs at current visit: body temperature 37.6°C. Vital signs 1 month earlier: blood pressure "
     "133/75 mmHg, heart rate 63/min, respiratory rate 15/min."),
    ("dev", "strep_throat_6eaeef11-3e0b-09f4-42d6-ea3d6fecea6b", "objective",
     "Vital signs at current visit: body temperature 37.9°C, heart rate 92/min, respiratory rate 13/min, "
     "blood pressure 113/72 mmHg.",
     "Vital signs at current visit: body temperature 37.9°C. Vital signs 5 days earlier: heart rate "
     "92/min, respiratory rate 13/min, blood pressure 113/72 mmHg."),
    ("dev", "strep_throat_6f5d97f5-ed42-0c70-c5b0-76f023ed1aed", "objective",
     "Vital signs at current visit: body temperature 39.3°C, heart rate 96/min, respiratory rate 15/min, "
     "blood pressure 128/90 mmHg, body weight 27.3 kg, body height 116.3 cm, BMI 20.2 kg/m², "
     "BMI-for-age/sex percentile 96.3%.",
     "Vital signs at current visit: body temperature 39.3°C. Vital signs 9 days earlier: heart rate "
     "96/min, respiratory rate 15/min, blood pressure 128/90 mmHg, body weight 27.3 kg, body height 116.3 "
     "cm, BMI 20.2 kg/m², BMI-for-age/sex percentile 96.3%."),
    ("dev", "strep_throat_71a4c9d8-7f77-3957-3854-9409afcfa2c7", "objective",
     "Vital signs at current visit: body temperature 38.5°C, heart rate 68/min, respiratory rate 15/min, "
     "blood pressure 114/79 mmHg.",
     "Vital signs at current visit: body temperature 38.5°C. Vital signs 15 days earlier: heart rate "
     "68/min, respiratory rate 15/min, blood pressure 114/79 mmHg."),
    ("dev", "strep_throat_73c93186-fe15-c5c6-3c91-2928ff347bf5", "objective",
     "Vital signs at current visit: temperature 38.5°C, heart rate 74/min, respiratory rate 13/min, blood "
     "pressure 102/81 mmHg (recorded 7 days prior).",
     "Vital signs at current visit: temperature 38.5°C. Vital signs 7 days earlier: heart rate 74/min, "
     "respiratory rate 13/min, blood pressure 102/81 mmHg (recorded 7 days prior)."),
    ("dev", "strep_throat_74668831-b343-793b-3618-f54e67a4e4c6", "objective",
     "Vital signs at current visit: body temperature 38.1°C, heart rate 93/min, respiratory rate 13/min, "
     "systolic blood pressure 141 mmHg, diastolic blood pressure 75 mmHg.",
     "Vital signs at current visit: body temperature 38.1°C. Vital signs 23 days earlier: heart rate "
     "93/min, respiratory rate 13/min, systolic blood pressure 141 mmHg, diastolic blood pressure 75 mmHg."),
    ("dev", "strep_throat_779bcba0-69d1-6318-2e01-80afbf5c62b0", "objective",
     "Current visit vital signs: body temperature 39.2°C, heart rate 90/min, respiratory rate 14/min, "
     "blood pressure 124/82 mmHg, body weight 10.8 kg, body height 82.5 cm, head circumference 46.4 cm "
     "(23.5th percentile), weight-for-length 25.9th percentile, pain score 1/10.",
     "Current visit vital signs: body temperature 39.2°C. Vital signs 24 days earlier: heart rate 90/min, "
     "respiratory rate 14/min, blood pressure 124/82 mmHg, body weight 10.8 kg, body height 82.5 cm, head "
     "circumference 46.4 cm (23.5th percentile), weight-for-length 25.9th percentile, pain score 1/10."),
    ("dev", "strep_throat_78c058e0-af9b-a008-d50e-3d95f0f4a901", "objective",
     "Vital signs at current visit: body temperature 38.0°C, heart rate 81/min, respiratory rate 14/min, "
     "systolic blood pressure 136 mmHg, diastolic blood pressure 88 mmHg.",
     "Vital signs at current visit: body temperature 38.0°C. Vital signs 14 days earlier: heart rate "
     "81/min, respiratory rate 14/min, systolic blood pressure 136 mmHg, diastolic blood pressure 88 mmHg."),
    ("dev", "strep_throat_7a2b2039-ad00-1b7c-3799-999531215b95", "objective",
     "Vitals at this visit: temperature 38.3°C, heart rate 97/min, respiratory rate 13/min, blood pressure "
     "113/90 mmHg, height 171.8 cm, weight 67.6 kg, BMI 22.9 kg/m².",
     "Vitals at this visit: temperature 38.3°C. Vitals 14 days earlier: heart rate 97/min, respiratory "
     "rate 13/min, blood pressure 113/90 mmHg, height 171.8 cm, weight 67.6 kg, BMI 22.9 kg/m²."),
    ("dev", "strep_throat_7b01f808-7240-6232-f5fe-35e2df4f7f3d", "objective",
     "Vital signs at current visit: temperature 38.0°C, heart rate 61/min, respiratory rate 14/min, blood "
     "pressure 114/99 mmHg.",
     "Vital signs at current visit: temperature 38.0°C. Vital signs 1 month earlier: heart rate 61/min, "
     "respiratory rate 14/min, blood pressure 114/99 mmHg."),
    ("dev", "strep_throat_89538db3-6aa0-eb1f-a551-cc87ce51b84e", "objective",
     "Vitals at current visit: temperature 38.0°C, heart rate 66/min, respiratory rate 14/min, systolic BP "
     "123 mmHg, diastolic BP 78 mmHg, body weight 15.7 kg, body height 101.7 cm, BMI 15.2 kg/m², "
     "BMI-for-age 32.8th percentile, weight-for-length 37.4th percentile, head circumference 49.7 cm "
     "(50.8th percentile).",
     "Vitals at current visit: temperature 38.0°C. Vitals 26 days earlier: heart rate 66/min, respiratory "
     "rate 14/min, systolic BP 123 mmHg, diastolic BP 78 mmHg, body weight 15.7 kg, body height 101.7 cm, "
     "BMI 15.2 kg/m², BMI-for-age 32.8th percentile, weight-for-length 37.4th percentile, head "
     "circumference 49.7 cm (50.8th percentile)."),
    ("dev", "strep_throat_8aa7d902-b2de-c752-e023-059672c50846", "objective",
     "Vital signs at current visit: temperature 38.3°C, heart rate 99/min, blood pressure 119/87 mmHg, "
     "respiratory rate 12/min.",
     "Vital signs at current visit: temperature 38.3°C. Vital signs 8 days earlier: heart rate 99/min, "
     "blood pressure 119/87 mmHg, respiratory rate 12/min."),
    ("dev", "strep_throat_8b878844-1613-8307-1eea-645e595be45b", "objective",
     "Vital signs at current visit: temperature 38.2°C, heart rate 72/min, respiratory rate 13/min, "
     "systolic blood pressure 100 mmHg, diastolic blood pressure 60 mmHg.",
     "Vital signs at current visit: temperature 38.2°C. Vital signs 15 days earlier: heart rate 72/min, "
     "respiratory rate 13/min, systolic blood pressure 100 mmHg, diastolic blood pressure 60 mmHg."),
    ("dev", "strep_throat_8ede6b48-d752-1f9f-ca54-8a9bd0652f75", "objective",
     "Vital signs at current visit: body temperature 38.4°C, blood pressure 126/82 mmHg, heart rate "
     "64/min, respiratory rate 14/min.",
     "Vital signs at current visit: body temperature 38.4°C. Vital signs 5 days earlier: blood pressure "
     "126/82 mmHg, heart rate 64/min, respiratory rate 14/min."),
    ("dev", "strep_throat_8ede6b48-d752-1f9f-ca54-8a9bd0652f75", "objective",
     "Labs at current visit: calcium 9.8 mg/dL, carbon dioxide total 24 mmol/L, chloride 109.9 mmol/L, "
     "creatinine 1 mg/dL, glucose 76.9 mg/dL, hemoglobin A1c 6.2%, potassium 4.5 mmol/L, sodium 137.7 "
     "mmol/L, urea nitrogen 11.8 mg/dL.",
     "Labs 5 days earlier: calcium 9.8 mg/dL, carbon dioxide total 24 mmol/L, chloride 109.9 mmol/L, "
     "creatinine 1 mg/dL, glucose 76.9 mg/dL, hemoglobin A1c 6.2%, potassium 4.5 mmol/L, sodium 137.7 "
     "mmol/L, urea nitrogen 11.8 mg/dL."),
    ("dev", "strep_throat_907c6d2e-5e7c-0022-078e-28f6f3d08a07", "objective",
     "Vital signs at current visit: body temperature 38.8°C, heart rate 71/min, respiratory rate 15/min, "
     "blood pressure 107/68 mmHg, BMI 27.9 kg/m², weight 102.1 kg, height 191.4 cm.",
     "Vital signs at current visit: body temperature 38.8°C. Vital signs 14 days earlier: heart rate "
     "71/min, respiratory rate 15/min, blood pressure 107/68 mmHg, BMI 27.9 kg/m², weight 102.1 kg, height "
     "191.4 cm."),
    ("dev", "strep_throat_94835e3b-bb6a-19da-b219-a114013aa086", "objective",
     "Vital signs at current visit: body temperature 37.7°C, heart rate 81/min, respiratory rate 15/min, "
     "blood pressure 106/88 mmHg.",
     "Vital signs at current visit: body temperature 37.7°C. Vital signs 19 days earlier: heart rate "
     "81/min, respiratory rate 15/min, blood pressure 106/88 mmHg."),
    ("dev", "strep_throat_a601124d-857a-1522-999d-e808287638bf", "objective",
     "Vital signs at current visit: body temperature 38.8°C, systolic blood pressure 93 mmHg, diastolic "
     "blood pressure 73 mmHg, heart rate 62/min, respiratory rate 13/min.",
     "Vital signs at current visit: body temperature 38.8°C. Vital signs 7 days earlier: systolic blood "
     "pressure 93 mmHg, diastolic blood pressure 73 mmHg, heart rate 62/min, respiratory rate 13/min."),
    ("dev", "strep_throat_ab803a58-6e97-054d-e1ca-1ebf5c6ba923", "objective",
     "Vital signs at current visit: temperature 38.2°C, blood pressure 119/90 mmHg, heart rate 71/min, "
     "respiratory rate 15/min.",
     "Vital signs at current visit: temperature 38.2°C. Vital signs 7 days earlier: blood pressure 119/90 "
     "mmHg, heart rate 71/min, respiratory rate 15/min."),
    ("dev", "strep_throat_ad58a9da-3cb0-cdc5-311f-000ff0ab0dbf", "objective",
     "Vital signs at current visit: body temperature 38.3°C, heart rate 94/min, blood pressure 119/76 "
     "mmHg, respiratory rate 13/min.",
     "Vital signs at current visit: body temperature 38.3°C. Vital signs 14 days earlier: heart rate "
     "94/min, blood pressure 119/76 mmHg, respiratory rate 13/min."),
    ("dev", "strep_throat_ae5c20cf-6926-6997-9441-583f7b177b53", "objective",
     "Vital signs at current visit: body temperature 38.9°C, heart rate 93/min, respiratory rate 13/min, "
     "blood pressure 107/80 mmHg.",
     "Vital signs at current visit: body temperature 38.9°C. Vital signs 7 days earlier: heart rate "
     "93/min, respiratory rate 13/min, blood pressure 107/80 mmHg."),
    ("dev", "strep_throat_b9677390-68ff-6028-dd16-e7df4117b7cf", "objective",
     "Vital signs at current visit: temperature 38.5°C, heart rate 99/min, respiratory rate 13/min, "
     "systolic BP 104 mmHg, diastolic BP 69 mmHg.",
     "Vital signs at current visit: temperature 38.5°C. Vital signs 25 days earlier: heart rate 99/min, "
     "respiratory rate 13/min, systolic BP 104 mmHg, diastolic BP 69 mmHg."),
    ("dev", "strep_throat_c2831bc0-6f38-f2e3-ad22-9052758e733e", "objective",
     "Vital signs at current visit: body temperature 37.4°C, heart rate 69/min, respiratory rate 16/min, "
     "systolic blood pressure 102 mmHg, diastolic blood pressure 70 mmHg.",
     "Vital signs at current visit: body temperature 37.4°C. Vital signs 7 months earlier: heart rate "
     "69/min, respiratory rate 16/min, systolic blood pressure 102 mmHg, diastolic blood pressure 70 mmHg."),
    ("dev", "strep_throat_cb0cc5d0-e8ab-d956-b10f-64e8af178dcd", "objective",
     "Vitals at current visit: temperature 38.1°C, heart rate 99/min, respiratory rate 15/min, systolic "
     "blood pressure 133 mmHg, diastolic blood pressure 85 mmHg, height 104.9 cm, weight 18.2 kg, BMI 16.5 "
     "kg/m2, BMI-for-age 80.1th percentile.",
     "Vitals at current visit: temperature 38.1°C. Vitals 20 days earlier: heart rate 99/min, respiratory "
     "rate 15/min, systolic blood pressure 133 mmHg, diastolic blood pressure 85 mmHg, height 104.9 cm, "
     "weight 18.2 kg, BMI 16.5 kg/m2, BMI-for-age 80.1th percentile."),
    ("dev", "strep_throat_cb9f07f0-5436-45c9-38dc-63facb306e91", "objective",
     "Vital signs at current visit: temperature 37.2°C, heart rate 99/min, respiratory rate 12/min, "
     "systolic blood pressure 93 mmHg, diastolic blood pressure 82 mmHg.",
     "Vital signs at current visit: temperature 37.2°C. Vital signs 6 days earlier: heart rate 99/min, "
     "respiratory rate 12/min, systolic blood pressure 93 mmHg, diastolic blood pressure 82 mmHg."),
    ("dev", "strep_throat_d13e8394-8a01-1e25-c596-41bd0ee034e9", "objective",
     "Vital signs at current visit: temperature 37.6°C, heart rate 72/min, respiratory rate 15/min, "
     "systolic BP 105 mmHg, diastolic BP 90 mmHg.",
     "Vital signs at current visit: temperature 37.6°C. Vital signs 7 days earlier: heart rate 72/min, "
     "respiratory rate 15/min, systolic BP 105 mmHg, diastolic BP 90 mmHg."),
    ("dev", "strep_throat_d95f0b6c-1b07-ee7e-410f-06e2709f4e96", "objective",
     "Vital signs at current visit: body temperature 37.4°C, heart rate 64/min, respiratory rate 15/min, "
     "blood pressure 115/79 mmHg, body weight 20.3 kg, body height 108.5 cm, BMI 17.3 kg/m2 (89.9th "
     "percentile for age and sex).",
     "Vital signs at current visit: body temperature 37.4°C. Vital signs 20 days earlier: heart rate "
     "64/min, respiratory rate 15/min, blood pressure 115/79 mmHg, body weight 20.3 kg, body height 108.5 "
     "cm, BMI 17.3 kg/m2 (89.9th percentile for age and sex)."),
    ("dev", "strep_throat_dc622d90-707f-6d94-46a0-f42a3feeec52", "objective",
     "Vital signs at current visit: body temperature 39.2°C, heart rate 67/min, respiratory rate 14/min, "
     "blood pressure 106/87 mmHg.",
     "Vital signs at current visit: body temperature 39.2°C. Vital signs 14 days earlier: heart rate "
     "67/min, respiratory rate 14/min, blood pressure 106/87 mmHg."),
    ("dev", "strep_throat_dd67eccd-c361-8976-d6bc-013f7dd25223", "objective",
     "Vital signs at current visit: temperature 37.8°C, heart rate 94/min, respiratory rate 16/min, blood "
     "pressure 115/77 mmHg, body weight 100.3 kg, height 188.5 cm, BMI 28.2 kg/m².",
     "Vital signs at current visit: temperature 37.8°C. Vital signs 14 days earlier: heart rate 94/min, "
     "respiratory rate 16/min, blood pressure 115/77 mmHg, body weight 100.3 kg, height 188.5 cm, BMI 28.2 "
     "kg/m²."),
    ("dev", "strep_throat_dd67eccd-c361-8976-d6bc-013f7dd25223", "objective",
     "PHQ-2 total score: 1 (current visit)",
     "PHQ-2 total score: 1 (14 days earlier)"),
    ("dev", "strep_throat_dd67eccd-c361-8976-d6bc-013f7dd25223", "objective",
     "AUDIT-C total score: 1 (current visit)",
     "AUDIT-C total score: 1 (14 days earlier)"),
    ("dev", "strep_throat_e9270627-4b25-9b72-6d84-0ee6387811e4", "objective",
     "Vital signs at current visit: temperature 37.9°C, heart rate 91/min, respiratory rate 13/min, "
     "systolic blood pressure 123 mmHg, diastolic blood pressure 86 mmHg.",
     "Vital signs at current visit: temperature 37.9°C. Vital signs 15 days earlier: heart rate 91/min, "
     "respiratory rate 13/min, systolic blood pressure 123 mmHg, diastolic blood pressure 86 mmHg."),
    ("dev", "strep_throat_e93acce0-80b7-d79b-391d-54ae78296fbf", "objective",
     "Current vitals: temperature 38.3°C, heart rate 61/min, respiratory rate 15/min, systolic BP 131 "
     "mmHg, diastolic BP 78 mmHg.",
     "Current vitals: temperature 38.3°C. Vitals 25 days earlier: heart rate 61/min, respiratory rate "
     "15/min, systolic BP 131 mmHg, diastolic BP 78 mmHg."),
    ("dev", "strep_throat_eb0b05ff-8d9b-c908-eaf7-adc57086d04c", "objective",
     "Vitals at current visit: temperature 38.9°C, heart rate 96/min, respiratory rate 15/min, blood "
     "pressure 122/85 mmHg, body weight 91.3 kg, height 181.6 cm, BMI 27.7 kg/m².",
     "Vitals at current visit: temperature 38.9°C. Vitals 7 days earlier: heart rate 96/min, respiratory "
     "rate 15/min, blood pressure 122/85 mmHg, body weight 91.3 kg, height 181.6 cm, BMI 27.7 kg/m²."),
    ("dev", "strep_throat_eeafe28b-a9cb-d8de-c764-152049451a7f", "objective",
     "Vital signs at current visit: temperature 37.5°C, heart rate 97/min, respiratory rate 14/min, blood "
     "pressure 117/85 mmHg, pain score 1/10.",
     "Vital signs at current visit: temperature 37.5°C. Vital signs 7 days earlier: heart rate 97/min, "
     "respiratory rate 14/min, blood pressure 117/85 mmHg, pain score 1/10."),
    ("dev", "strep_throat_f34473ec-45df-1b8a-386a-429e6d4e8faa", "objective",
     "Vital signs at this visit: body temperature 37.8°C, heart rate 63/min (2 months prior: 63/min), "
     "respiratory rate 15/min, blood pressure 132/75 mm[Hg].",
     "Vital signs at this visit: body temperature 37.8°C. Vital signs 2 months earlier: heart rate 63/min "
     "(2 months prior: 63/min), respiratory rate 15/min, blood pressure 132/75 mm[Hg]."),
    ("dev", "strep_throat_f484f5a6-d76e-805a-4987-b0d39424b2ed", "objective",
     "Vital signs at current visit: temperature 38.0°C, systolic blood pressure 103 mmHg, diastolic blood "
     "pressure 86 mmHg, heart rate 76/min, respiratory rate 15/min.",
     "Vital signs at current visit: temperature 38.0°C. Vital signs 8 days earlier: systolic blood "
     "pressure 103 mmHg, diastolic blood pressure 86 mmHg, heart rate 76/min, respiratory rate 15/min."),
    ("dev", "strep_throat_f861cd46-6f28-426c-b7b8-79cd0fb46dff", "objective",
     "Vital signs at current visit: temperature 37.5°C, heart rate 97/min, respiratory rate 16/min, "
     "systolic BP 134 mmHg, diastolic BP 86 mmHg.",
     "Vital signs at current visit: temperature 37.5°C. Vital signs 1 month earlier: heart rate 97/min, "
     "respiratory rate 16/min, systolic BP 134 mmHg, diastolic BP 86 mmHg."),
    ("dev", "strep_throat_fad4919d-30a4-ce22-7304-c3dc246ddfba", "objective",
     "Vital signs at current visit: body temperature 39.4°C, heart rate 73/min, respiratory rate 15/min, "
     "blood pressure 110/72 mmHg.",
     "Vital signs at current visit: body temperature 39.4°C. Vital signs 15 days earlier: heart rate "
     "73/min, respiratory rate 15/min, blood pressure 110/72 mmHg."),
    ("dev", "viral_pharyngitis_014eb5b0-954b-d6a5-b176-8af04b194845", "objective",
     "Vitals at current visit: temperature 37.0°C, heart rate 70/min, respiratory rate 15/min, blood "
     "pressure 111/79 mmHg, body weight 29.9 kg, height 139.9 cm, BMI 15.3 kg/m², BMI-for-age 28.8th "
     "percentile.",
     "Vitals at current visit: temperature 37.0°C. Vitals 7 days earlier: heart rate 70/min, respiratory "
     "rate 15/min, blood pressure 111/79 mmHg, body weight 29.9 kg, height 139.9 cm, BMI 15.3 kg/m², "
     "BMI-for-age 28.8th percentile."),
    ("dev", "viral_pharyngitis_01ed5317-338a-007d-13e5-c98d4ee86901", "objective",
     "Vital signs at this visit: body temperature 37.9°C, heart rate 90/min, respiratory rate 16/min, "
     "blood pressure 107/76 mmHg.",
     "Vital signs at this visit: body temperature 37.9°C. Vital signs 23 days earlier: heart rate 90/min, "
     "respiratory rate 16/min, blood pressure 107/76 mmHg."),
    ("dev", "viral_pharyngitis_02613e7a-48c9-3a26-d741-bace2666a286", "objective",
     "Vital signs at current visit: body temperature 37.0°C, heart rate 62/min, respiratory rate 15/min, "
     "systolic blood pressure 112 mmHg, diastolic blood pressure 73 mmHg.",
     "Vital signs at current visit: body temperature 37.0°C. Vital signs 14 days earlier: heart rate "
     "62/min, respiratory rate 15/min, systolic blood pressure 112 mmHg, diastolic blood pressure 73 mmHg."),
    ("dev", "viral_pharyngitis_02613e7a-48c9-3a26-d741-bace2666a286", "objective",
     "PHQ-2 total score 2 (current visit)",
     "PHQ-2 total score 2 (14 days earlier)"),
    ("dev", "viral_pharyngitis_0361b700-8737-5414-43c7-cce67fadc06a", "objective",
     "Vital signs at current visit: body temperature 37.0°C, blood pressure 115/82 mmHg, heart rate "
     "62/min, respiratory rate 15/min.",
     "Vital signs at current visit: body temperature 37.0°C. Vital signs 15 days earlier: blood pressure "
     "115/82 mmHg, heart rate 62/min, respiratory rate 15/min."),
    ("dev", "viral_pharyngitis_03e54fc2-5682-15ec-dfac-1dd652004930", "objective",
     "Vital signs at current visit: body temperature 37.1°C, heart rate 73/min, respiratory rate 15/min, "
     "blood pressure 92/85 mm[Hg], body weight 63.5 kg, body height 172 cm, BMI 21.5 kg/m2.",
     "Vital signs at current visit: body temperature 37.1°C. Vital signs 15 days earlier: heart rate "
     "73/min, respiratory rate 15/min, blood pressure 92/85 mm[Hg], body weight 63.5 kg, body height 172 "
     "cm, BMI 21.5 kg/m2."),
    ("dev", "viral_pharyngitis_08370a81-ba03-3bc0-ff33-0d61f18c15c0", "objective",
     "Vitals at current visit: temperature 37.9°C, heart rate 96/min, respiratory rate 13/min, systolic "
     "blood pressure 159 mmHg, diastolic blood pressure 86 mmHg, height 97 cm, weight 16.6 kg, BMI 17.6 "
     "kg/m², BMI-for-age/sex 89.8th percentile, head circumference 50 cm (82.4th percentile), "
     "weight-for-length 91.6th percentile.",
     "Vitals at current visit: temperature 37.9°C. Vitals 25 days earlier: heart rate 96/min, respiratory "
     "rate 13/min, systolic blood pressure 159 mmHg, diastolic blood pressure 86 mmHg, height 97 cm, "
     "weight 16.6 kg, BMI 17.6 kg/m², BMI-for-age/sex 89.8th percentile, head circumference 50 cm (82.4th "
     "percentile), weight-for-length 91.6th percentile."),
    ("dev", "viral_pharyngitis_08ad0cac-36c2-dfc9-f15b-d6c058a30571", "objective",
     "Vital signs at this visit: temperature 37.5°C, heart rate 91/min, respiratory rate 13/min, systolic "
     "BP 111 mmHg, diastolic BP 67 mmHg.",
     "Vital signs at this visit: temperature 37.5°C. Vital signs 21 days earlier: heart rate 91/min, "
     "respiratory rate 13/min, systolic BP 111 mmHg, diastolic BP 67 mmHg."),
    ("dev", "viral_pharyngitis_0da5c85d-10e1-c940-aba3-b5276ed9da95", "objective",
     "Vitals at current visit: temperature 37.7°C, heart rate 94/min, respiratory rate 15/min, systolic "
     "blood pressure 151 mmHg, diastolic blood pressure 90 mmHg, height 136.3 cm, weight 32.6 kg, BMI 17.5 "
     "kg/m², BMI-for-age 78.3th percentile.",
     "Vitals at current visit: temperature 37.7°C. Vitals 7 days earlier: heart rate 94/min, respiratory "
     "rate 15/min, systolic blood pressure 151 mmHg, diastolic blood pressure 90 mmHg, height 136.3 cm, "
     "weight 32.6 kg, BMI 17.5 kg/m², BMI-for-age 78.3th percentile."),
    ("dev", "viral_pharyngitis_0e042350-6de1-008c-fd31-c1d91959d45a", "objective",
     "Vital signs at current visit: body temperature 38.0°C, blood pressure 136/86 mmHg, heart rate "
     "66/min, respiratory rate 15/min.",
     "Vital signs at current visit: body temperature 38.0°C. Vital signs 14 days earlier: blood pressure "
     "136/86 mmHg, heart rate 66/min, respiratory rate 15/min."),
    ("dev", "viral_pharyngitis_0e6eca2b-5b43-a6d4-14c2-f8296a90257c", "objective",
     "Vital signs at current visit: temperature 37.1°C, heart rate 82/min, respiratory rate 13/min, blood "
     "pressure 109/78 mmHg.",
     "Vital signs at current visit: temperature 37.1°C. Vital signs 14 days earlier: heart rate 82/min, "
     "respiratory rate 13/min, blood pressure 109/78 mmHg."),
    ("dev", "viral_pharyngitis_0f3369e1-882b-da1d-d89c-fe56f513bfa0", "objective",
     "Vital signs at current visit: body temperature 37.3°C, heart rate 91/min, respiratory rate 15/min, "
     "blood pressure 109/74 mmHg, body weight 27.9 kg, body height 133.5 cm, BMI 15.7 kg/m², BMI "
     "percentile 27.7% per age and sex.",
     "Vital signs at current visit: body temperature 37.3°C. Vital signs 7 days earlier: heart rate "
     "91/min, respiratory rate 15/min, blood pressure 109/74 mmHg, body weight 27.9 kg, body height 133.5 "
     "cm, BMI 15.7 kg/m², BMI percentile 27.7% per age and sex."),
    ("dev", "viral_pharyngitis_12210652-6ab3-7a07-5306-7ddb57e02675", "objective",
     "Vitals at this visit: body temperature 37.8°C, heart rate 73/min, respiratory rate 12/min, blood "
     "pressure 118/78 mmHg, body weight 30.4 kg, body height 128.2 cm, BMI 18.5 kg/m², BMI-for-age 89.3rd "
     "percentile.",
     "Vitals at this visit: body temperature 37.8°C. Vitals 3 days earlier: heart rate 73/min, respiratory "
     "rate 12/min, blood pressure 118/78 mmHg, body weight 30.4 kg, body height 128.2 cm, BMI 18.5 kg/m², "
     "BMI-for-age 89.3rd percentile."),
    ("dev", "viral_pharyngitis_138d4d90-2f3e-f623-967d-725dea0a2167", "objective",
     "Vital signs at current visit: temperature 37.4°C, heart rate 75/min, respiratory rate 13/min, blood "
     "pressure 123/85 mmHg.",
     "Vital signs at current visit: temperature 37.4°C. Vital signs 14 days earlier: heart rate 75/min, "
     "respiratory rate 13/min, blood pressure 123/85 mmHg."),
    ("dev", "viral_pharyngitis_13f0476c-0299-d0c6-a182-080d63ff5625", "objective",
     "Vital signs at current visit: temperature 37.6°C, heart rate 98/min, respiratory rate 15/min, blood "
     "pressure 121/77 mmHg.",
     "Vital signs at current visit: temperature 37.6°C. Vital signs 14 days earlier: heart rate 98/min, "
     "respiratory rate 15/min, blood pressure 121/77 mmHg."),
    ("dev", "viral_pharyngitis_13f81129-bcd9-1146-00c9-61f0492da872", "objective",
     "Vital signs at current visit: body temperature 37.1°C, heart rate 90/min, respiratory rate 14/min, "
     "systolic blood pressure 142 mmHg, diastolic blood pressure 85 mmHg.",
     "Vital signs at current visit: body temperature 37.1°C. Vital signs 7 days earlier: heart rate "
     "90/min, respiratory rate 14/min, systolic blood pressure 142 mmHg, diastolic blood pressure 85 mmHg."),
    ("dev", "viral_pharyngitis_1581eb11-ce2f-f0e0-2a6f-7980fbcd0864", "objective",
     "Vital signs at current visit: body temperature 37.3°C, blood pressure 127/73 mm[Hg], heart rate "
     "99/min, respiratory rate 14/min.",
     "Vital signs at current visit: body temperature 37.3°C. Vital signs 1 month earlier: blood pressure "
     "127/73 mm[Hg], heart rate 99/min, respiratory rate 14/min."),
    ("dev", "viral_pharyngitis_16096637-27f6-517d-5d6b-f7e7429a783c", "objective",
     "Vital signs at current visit: temperature 37.4°C, heart rate 82/min, respiratory rate 13/min, blood "
     "pressure 128/83 mmHg.",
     "Vital signs at current visit: temperature 37.4°C. Vital signs 14 days earlier: heart rate 82/min, "
     "respiratory rate 13/min, blood pressure 128/83 mmHg."),
    ("dev", "viral_pharyngitis_16bc5857-531f-7958-5e94-6511595d1277", "objective",
     "Vital signs at current visit: temperature 38.0°C, blood pressure 150/87 mmHg, heart rate 83/min, "
     "respiratory rate 13/min.",
     "Vital signs at current visit: temperature 38.0°C. Vital signs 14 days earlier: blood pressure 150/87 "
     "mmHg, heart rate 83/min, respiratory rate 13/min."),
    ("dev", "viral_pharyngitis_194d679d-0fd8-6bfc-963d-481a82dcf7af", "objective",
     "Vital signs at current visit: body temperature 38.0°C, heart rate 78/min, respiratory rate 15/min, "
     "blood pressure 106/70 mmHg, body weight 11.5 kg, body height 78.6 cm, head circumference 48 cm "
     "(91.8th percentile), weight-for-length 82.9th percentile.",
     "Vital signs at current visit: body temperature 38.0°C. Vital signs 22 days earlier: heart rate "
     "78/min, respiratory rate 15/min, blood pressure 106/70 mmHg, body weight 11.5 kg, body height 78.6 "
     "cm, head circumference 48 cm (91.8th percentile), weight-for-length 82.9th percentile."),
    ("dev", "viral_pharyngitis_1ac86040-f2ed-2604-40af-96cff9437185", "objective",
     "Vital signs at current visit: body temperature 37.8°C, heart rate 89/min, respiratory rate 14/min, "
     "blood pressure 132/85 mm[Hg].",
     "Vital signs at current visit: body temperature 37.8°C. Vital signs 5 months earlier: heart rate "
     "89/min, respiratory rate 14/min, blood pressure 132/85 mm[Hg]."),
    ("dev", "viral_pharyngitis_1ae210bf-3532-730b-85e2-78b631c28730", "objective",
     "Vital signs at current visit: temperature 37.6°C, blood pressure 90/62 mmHg, heart rate 70/min, "
     "respiratory rate 14/min.",
     "Vital signs at current visit: temperature 37.6°C. Vital signs 14 days earlier: blood pressure 90/62 "
     "mmHg, heart rate 70/min, respiratory rate 14/min."),
    ("dev", "viral_pharyngitis_1ca75867-5599-b263-2bbb-adf9939102b3", "objective",
     "Vital signs at current visit: temperature 37.6°C, blood pressure 129/92 mmHg, heart rate 80/min, "
     "respiratory rate 13/min.",
     "Vital signs at current visit: temperature 37.6°C. Vital signs 15 days earlier: blood pressure 129/92 "
     "mmHg, heart rate 80/min, respiratory rate 13/min."),
    ("dev", "viral_pharyngitis_1dee2a42-2972-b48a-8c59-47aa464f3d14", "objective",
     "Vital signs at current visit: body temperature 37.5°C, blood pressure 111/73 mmHg, heart rate "
     "67/min, respiratory rate 13/min.",
     "Vital signs at current visit: body temperature 37.5°C. Vital signs 26 days earlier: blood pressure "
     "111/73 mmHg, heart rate 67/min, respiratory rate 13/min."),
    ("dev", "viral_pharyngitis_1eeb6660-599a-7396-a612-90ba67028859", "objective",
     "Vital signs at current visit: temperature 37.9°C, blood pressure 123/66 mmHg, heart rate 77/min, "
     "respiratory rate 14/min.",
     "Vital signs at current visit: temperature 37.9°C. Vital signs 14 days earlier: blood pressure 123/66 "
     "mmHg, heart rate 77/min, respiratory rate 14/min."),
    ("dev", "viral_pharyngitis_1fb70c0f-085d-ec93-1a0a-e4cb233c7244", "objective",
     "Vital signs at current visit: body temperature 37.3°C, heart rate 82/min, respiratory rate 16/min, "
     "blood pressure 111/79 mmHg, body weight 12.6 kg, body height 85.9 cm, head circumference 48.3 cm "
     "(41.6th percentile), weight-for-length 64.7th percentile.",
     "Vital signs at current visit: body temperature 37.3°C. Vital signs 23 days earlier: heart rate "
     "82/min, respiratory rate 16/min, blood pressure 111/79 mmHg, body weight 12.6 kg, body height 85.9 "
     "cm, head circumference 48.3 cm (41.6th percentile), weight-for-length 64.7th percentile."),
    ("dev", "viral_pharyngitis_2255f0c0-a37f-3809-59c8-b8ec7a933dc1", "objective",
     "Vital signs at current visit: temperature 37.0°C, heart rate 66/min, respiratory rate 14/min, blood "
     "pressure 122/80 mmHg.",
     "Vital signs at current visit: temperature 37.0°C. Vital signs 1 month earlier: heart rate 66/min, "
     "respiratory rate 14/min, blood pressure 122/80 mmHg."),
    ("dev", "viral_pharyngitis_23ef44b2-0199-32ec-14c1-0d25216fc051", "objective",
     "Vital signs at current visit: temperature 37.0°C, heart rate 78/min, respiratory rate 15/min, blood "
     "pressure 107/67 mmHg.",
     "Vital signs at current visit: temperature 37.0°C. Vital signs 7 days earlier: heart rate 78/min, "
     "respiratory rate 15/min, blood pressure 107/67 mmHg."),
    ("dev", "viral_pharyngitis_25a26258-0e40-85e9-dee0-d606c63b309d", "objective",
     "Vital signs at current visit: temperature 37.5°C, blood pressure 124/63 mmHg, heart rate 61/min, "
     "respiratory rate 13/min.",
     "Vital signs at current visit: temperature 37.5°C. Vital signs 23 days earlier: blood pressure 124/63 "
     "mmHg, heart rate 61/min, respiratory rate 13/min."),
    ("dev", "viral_pharyngitis_2a47350f-e35a-51e8-e36c-4f2598130c10", "objective",
     "Vital signs at current visit: body temperature 37.9°C, blood pressure 104/81 mmHg, heart rate "
     "83/min, respiratory rate 14/min (all from 4 months prior baseline).",
     "Vital signs at current visit: body temperature 37.9°C. Vital signs 4 months earlier: blood pressure "
     "104/81 mmHg, heart rate 83/min, respiratory rate 14/min (all from 4 months prior baseline)."),
    ("dev", "viral_pharyngitis_2acea727-b7d8-0607-d7d6-5b427720ea87", "objective",
     "Vital signs at current visit: body temperature 37.2°C, blood pressure 129/86 mmHg, heart rate "
     "69/min, respiratory rate 13/min.",
     "Vital signs at current visit: body temperature 37.2°C. Vital signs 7 days earlier: blood pressure "
     "129/86 mmHg, heart rate 69/min, respiratory rate 13/min."),
    ("dev", "viral_pharyngitis_2c3566e1-9bc1-945f-c06a-e54a8a199f4e", "objective",
     "Vital signs at current visit: body temperature 37.4°C, heart rate 67/min, respiratory rate 15/min, "
     "blood pressure 134/91 mmHg.",
     "Vital signs at current visit: body temperature 37.4°C. Vital signs 14 days earlier: heart rate "
     "67/min, respiratory rate 15/min, blood pressure 134/91 mmHg."),
    ("dev", "viral_pharyngitis_2cb1e76b-4f40-a9bf-7c2b-15d4ac74d2a8", "objective",
     "Vital signs at current visit: temperature 37.3°C, heart rate 97/min, respiratory rate 15/min, blood "
     "pressure 132/107 mmHg, body weight 102.4 kg, height 193.3 cm, BMI 27.4 kg/m².",
     "Vital signs at current visit: temperature 37.3°C. Vital signs 14 days earlier: heart rate 97/min, "
     "respiratory rate 15/min, blood pressure 132/107 mmHg, body weight 102.4 kg, height 193.3 cm, BMI "
     "27.4 kg/m²."),
    ("dev", "viral_pharyngitis_320dbc5f-28bd-245b-1312-eb1d2da16924", "objective",
     "Vital signs at this visit: body temperature 37.7°C, heart rate 88/min, respiratory rate 13/min, "
     "blood pressure 124/83 mmHg.",
     "Vital signs at this visit: body temperature 37.7°C. Vital signs 1 year earlier: heart rate 88/min, "
     "respiratory rate 13/min, blood pressure 124/83 mmHg."),
    ("dev", "viral_pharyngitis_3243ce0f-bea4-3572-ae38-97b25f081a24", "objective",
     "Vital signs at current visit: temperature 37.2°C, heart rate 70/min, respiratory rate 15/min, blood "
     "pressure 122/84 mmHg.",
     "Vital signs at current visit: temperature 37.2°C. Vital signs 3 months earlier: heart rate 70/min, "
     "respiratory rate 15/min, blood pressure 122/84 mmHg."),
    ("dev", "viral_pharyngitis_327cc684-5e6e-9f9d-c425-5f14717a526f", "objective",
     "Vitals at this visit: temperature 37.2°C, heart rate 94/min, respiratory rate 13/min, blood pressure "
     "107/76 mmHg.",
     "Vitals at this visit: temperature 37.2°C. Vitals 3 months earlier: heart rate 94/min, respiratory "
     "rate 13/min, blood pressure 107/76 mmHg."),
    ("dev", "viral_pharyngitis_32f05fa3-07db-1c0c-ff7e-7ce6835c9640", "objective",
     "Vital signs at current visit: temperature 37.9°C, heart rate 90/min, respiratory rate 14/min, blood "
     "pressure 113/92 mmHg (measured 14 days prior).",
     "Vital signs at current visit: temperature 37.9°C. Vital signs 14 days earlier: heart rate 90/min, "
     "respiratory rate 14/min, blood pressure 113/92 mmHg (measured 14 days prior)."),
    ("dev", "viral_pharyngitis_338d6680-15ff-9b77-9a2b-75db9ee44bf2", "objective",
     "Vitals at current visit: body temperature 37.6°C, heart rate 70/min, respiratory rate 13/min, blood "
     "pressure 108/81 mm[Hg], body weight 28.3 kg, body height 133.9 cm, BMI 15.8 kg/m², BMI-for-age/sex "
     "percentile 29.6%.",
     "Vitals at current visit: body temperature 37.6°C. Vitals 7 days earlier: heart rate 70/min, "
     "respiratory rate 13/min, blood pressure 108/81 mm[Hg], body weight 28.3 kg, body height 133.9 cm, "
     "BMI 15.8 kg/m², BMI-for-age/sex percentile 29.6%."),
    ("dev", "viral_pharyngitis_3590acf7-1954-b6fc-796f-32ad74dadd58", "objective",
     "Vital signs at current visit: body temperature 37.2°C, heart rate 83/min, respiratory rate 16/min, "
     "systolic blood pressure 92 mmHg, diastolic blood pressure 65 mmHg.",
     "Vital signs at current visit: body temperature 37.2°C. Vital signs 7 months earlier: heart rate "
     "83/min, respiratory rate 16/min, systolic blood pressure 92 mmHg, diastolic blood pressure 65 mmHg."),
    ("dev", "viral_pharyngitis_36e0a716-ce4f-34f8-3114-9e4da6ad7cef", "objective",
     "Vital signs at current visit: temperature 37.5°C, heart rate 90/min, respiratory rate 15/min, blood "
     "pressure 116/90 mmHg.",
     "Vital signs at current visit: temperature 37.5°C. Vital signs 7 days earlier: heart rate 90/min, "
     "respiratory rate 15/min, blood pressure 116/90 mmHg."),
    ("dev", "viral_pharyngitis_3844d2cf-4f3a-32ad-0f22-53d30e9c8add", "objective",
     "Vital signs at current visit: body temperature 37.7°C, blood pressure 114/83 mmHg, heart rate "
     "73/min, respiratory rate 15/min.",
     "Vital signs at current visit: body temperature 37.7°C. Vital signs 2 months earlier: blood pressure "
     "114/83 mmHg, heart rate 73/min, respiratory rate 15/min."),
    ("dev", "viral_pharyngitis_391d869c-e11f-577c-db73-e706f7813b1d", "objective",
     "Vital signs at this visit: temperature 37.5°C, heart rate 62/min, respiratory rate 13/min, blood "
     "pressure 139/89 mmHg, body weight 25.4 kg, height 131.2 cm, BMI 14.8 kg/m², BMI-for-age 16.9th "
     "percentile.",
     "Vital signs at this visit: temperature 37.5°C. Vital signs 7 days earlier: heart rate 62/min, "
     "respiratory rate 13/min, blood pressure 139/89 mmHg, body weight 25.4 kg, height 131.2 cm, BMI 14.8 "
     "kg/m², BMI-for-age 16.9th percentile."),
    ("dev", "viral_pharyngitis_3948670c-14a7-f50d-8359-5ddb5d357eff", "objective",
     "Vital signs at current visit: temperature 37.5°C, heart rate 93/min, respiratory rate 14/min, blood "
     "pressure 119/80 mmHg.",
     "Vital signs at current visit: temperature 37.5°C. Vital signs 9 months earlier: heart rate 93/min, "
     "respiratory rate 14/min, blood pressure 119/80 mmHg."),
    ("dev", "viral_pharyngitis_3d2efadd-78e0-b585-6500-e6aead9e712b", "objective",
     "Vital signs at current visit: temperature 37.5°C, heart rate 90/min, blood pressure 116/63 mmHg, "
     "respiratory rate 13/min.",
     "Vital signs at current visit: temperature 37.5°C. Vital signs 14 days earlier: heart rate 90/min, "
     "blood pressure 116/63 mmHg, respiratory rate 13/min."),
    ("dev", "viral_pharyngitis_3d349388-d757-1404-f5fc-284de1d71ac9", "objective",
     "Vital signs at current visit: temperature 37.0°C, heart rate 79/min, respiratory rate 15/min, blood "
     "pressure 110/76 mmHg, body weight 17 kg, height 103.6 cm, BMI 15.8 kg/m², BMI-for-age 68.6th "
     "percentile, pain score 0/10.",
     "Vital signs at current visit: temperature 37.0°C. Vital signs 20 days earlier: heart rate 79/min, "
     "respiratory rate 15/min, blood pressure 110/76 mmHg, body weight 17 kg, height 103.6 cm, BMI 15.8 "
     "kg/m², BMI-for-age 68.6th percentile, pain score 0/10."),
    ("dev", "viral_pharyngitis_3d84d770-781a-f48f-f4c3-d7b097a9ffc4", "objective",
     "Vital signs at current visit: temperature 37.7°C, heart rate 84/min, respiratory rate 13/min, "
     "systolic BP 134 mmHg, diastolic BP 86 mmHg.",
     "Vital signs at current visit: temperature 37.7°C. Vital signs 9 days earlier: heart rate 84/min, "
     "respiratory rate 13/min, systolic BP 134 mmHg, diastolic BP 86 mmHg."),
    ("dev", "viral_pharyngitis_3fecb9a9-64b0-019a-b0a9-ebbc46df782b", "objective",
     "Vital signs at current visit: body temperature 37.2°C, blood pressure 126/71 mmHg, heart rate "
     "69/min, respiratory rate 14/min.",
     "Vital signs at current visit: body temperature 37.2°C. Vital signs 10 days earlier: blood pressure "
     "126/71 mmHg, heart rate 69/min, respiratory rate 14/min."),
    ("dev", "viral_pharyngitis_4032f10d-a8fa-98e9-b01a-3b239468f7dc", "objective",
     "Vital signs at current visit: temperature 37.3°C, blood pressure 110/63 mmHg, heart rate 87/min, "
     "respiratory rate 15/min.",
     "Vital signs at current visit: temperature 37.3°C. Vital signs 18 days earlier: blood pressure 110/63 "
     "mmHg, heart rate 87/min, respiratory rate 15/min."),
    ("dev", "viral_pharyngitis_4607dace-cd9b-0de9-e6d3-2eca9bb01812", "objective",
     "Vital signs at current visit: temperature 37.8°C, heart rate 99/min, respiratory rate 15/min, blood "
     "pressure 108/73 mmHg.",
     "Vital signs at current visit: temperature 37.8°C. Vital signs 3 months earlier: heart rate 99/min, "
     "respiratory rate 15/min, blood pressure 108/73 mmHg."),
    ("dev", "viral_pharyngitis_49f7afb0-3294-0443-56c0-ab13c4b78edb", "objective",
     "Vital signs at current visit: temperature 37.2°C, heart rate 96/min, respiratory rate 13/min, blood "
     "pressure 132/75 mmHg.",
     "Vital signs at current visit: temperature 37.2°C. Vital signs 14 days earlier: heart rate 96/min, "
     "respiratory rate 13/min, blood pressure 132/75 mmHg."),
    ("dev", "viral_pharyngitis_49f7afb0-3294-0443-56c0-ab13c4b78edb", "objective",
     "AUDIT-C score: 2 (current visit)",
     "AUDIT-C score: 2 (14 days earlier)"),
    ("dev", "viral_pharyngitis_49f7afb0-3294-0443-56c0-ab13c4b78edb", "objective",
     "PHQ-2 score: 0 (current visit)",
     "PHQ-2 score: 0 (14 days earlier)"),
    ("dev", "viral_pharyngitis_4b0886c6-aca6-167e-606e-3029f891f972", "objective",
     "Vital signs at current visit: body temperature 37.1°C, heart rate 87/min, respiratory rate 13/min, "
     "systolic blood pressure 138 mmHg, diastolic blood pressure 76 mmHg.",
     "Vital signs at current visit: body temperature 37.1°C. Vital signs 7 days earlier: heart rate "
     "87/min, respiratory rate 13/min, systolic blood pressure 138 mmHg, diastolic blood pressure 76 mmHg."),
    ("dev", "viral_pharyngitis_4e677afd-459c-6576-a075-cf1b816f4ad9", "objective",
     "Vitals at this visit: temperature 37.4°C, heart rate 94/min, respiratory rate 12/min, systolic BP "
     "102 mmHg, diastolic BP 60 mmHg.",
     "Vitals at this visit: temperature 37.4°C. Vitals 7 days earlier: heart rate 94/min, respiratory rate "
     "12/min, systolic BP 102 mmHg, diastolic BP 60 mmHg."),
    ("dev", "viral_pharyngitis_4fe56046-fbc8-0825-7b25-b6413feb874e", "objective",
     "Vital signs at current visit: temperature 38.0°C, heart rate 92/min, respiratory rate 13/min, blood "
     "pressure 128/79 mmHg.",
     "Vital signs at current visit: temperature 38.0°C. Vital signs 15 days earlier: heart rate 92/min, "
     "respiratory rate 13/min, blood pressure 128/79 mmHg."),
    ("dev", "viral_pharyngitis_51d69a38-57ae-47c4-5529-6191f80947f5", "objective",
     "Vital signs at current visit: body temperature 38.0°C, heart rate 64/min, respiratory rate 13/min, "
     "systolic blood pressure 145 mmHg, diastolic blood pressure 72 mmHg.",
     "Vital signs at current visit: body temperature 38.0°C. Vital signs 3 days earlier: heart rate "
     "64/min, respiratory rate 13/min, systolic blood pressure 145 mmHg, diastolic blood pressure 72 mmHg."),
    ("dev", "viral_pharyngitis_5300249f-f9ca-8ab6-5ed4-274cc6cbf897", "objective",
     "Vital signs at current visit: body temperature 37.6°C, heart rate 89/min, respiratory rate 12/min, "
     "systolic blood pressure 100 mmHg, diastolic blood pressure 70 mmHg.",
     "Vital signs at current visit: body temperature 37.6°C. Vital signs 4 months earlier: heart rate "
     "89/min, respiratory rate 12/min, systolic blood pressure 100 mmHg, diastolic blood pressure 70 mmHg."),
    ("dev", "viral_pharyngitis_5341bfc3-c3d3-3d47-5899-1ad78700265d", "objective",
     "Vital signs at current visit: body temperature 37.6°C, heart rate 96/min, respiratory rate 13/min, "
     "blood pressure 118/83 mmHg.",
     "Vital signs at current visit: body temperature 37.6°C. Vital signs 14 days earlier: heart rate "
     "96/min, respiratory rate 13/min, blood pressure 118/83 mmHg."),
    ("dev", "viral_pharyngitis_534c7a92-d592-8e02-c457-6377aa329236", "objective",
     "Vital signs at current visit: body temperature 37.7°C, heart rate 90/min, respiratory rate 16/min, "
     "systolic blood pressure 135 mmHg, diastolic blood pressure 85 mmHg.",
     "Vital signs at current visit: body temperature 37.7°C. Vital signs 14 days earlier: heart rate "
     "90/min, respiratory rate 16/min, systolic blood pressure 135 mmHg, diastolic blood pressure 85 mmHg."),
    ("dev", "viral_pharyngitis_54002027-8593-cfd2-9133-aabbd21148ec", "objective",
     "Vital signs at current visit: body temperature 37.3°C, heart rate 71/min, respiratory rate 14/min, "
     "systolic blood pressure 101 mmHg, diastolic blood pressure 79 mmHg.",
     "Vital signs at current visit: body temperature 37.3°C. Vital signs 25 days earlier: heart rate "
     "71/min, respiratory rate 14/min, systolic blood pressure 101 mmHg, diastolic blood pressure 79 mmHg."),
    ("dev", "viral_pharyngitis_57a68784-da2f-4788-1a15-78b67321e8f6", "objective",
     "Vital signs at current visit: temperature 37.1°C, blood pressure 120/82 mmHg, heart rate 61/min, "
     "respiratory rate 14/min.",
     "Vital signs at current visit: temperature 37.1°C. Vital signs 15 days earlier: blood pressure 120/82 "
     "mmHg, heart rate 61/min, respiratory rate 14/min."),
    ("dev", "viral_pharyngitis_57ad65a5-0daf-ab7d-b826-9b65699303e1", "objective",
     "Vital signs at current visit: body temperature 37.6°C, heart rate 88/min, respiratory rate 13/min, "
     "blood pressure 107/80 mmHg.",
     "Vital signs at current visit: body temperature 37.6°C. Vital signs 14 days earlier: heart rate "
     "88/min, respiratory rate 13/min, blood pressure 107/80 mmHg."),
    ("dev", "viral_pharyngitis_59810f7c-b8a0-6430-b80b-5705111724cb", "objective",
     "Vital signs at this visit: body temperature 37.1°C, blood pressure 113/86 mmHg, heart rate 85/min, "
     "respiratory rate 15/min.",
     "Vital signs at this visit: body temperature 37.1°C. Vital signs 1 month earlier: blood pressure "
     "113/86 mmHg, heart rate 85/min, respiratory rate 15/min."),
    ("dev", "viral_pharyngitis_59ed01d8-0c53-2804-9892-02be8398d746", "objective",
     "Vital signs at current visit: body temperature 37.9°C, heart rate 96/min, respiratory rate 15/min, "
     "systolic blood pressure 120 mmHg, diastolic blood pressure 72 mmHg.",
     "Vital signs at current visit: body temperature 37.9°C. Vital signs 7 days earlier: heart rate "
     "96/min, respiratory rate 15/min, systolic blood pressure 120 mmHg, diastolic blood pressure 72 mmHg."),
    ("dev", "viral_pharyngitis_5abc2f11-ff42-efa3-aca5-4d5fbac6993a", "objective",
     "Vital signs at current visit: temperature 37.2°C, heart rate 87/min, respiratory rate 14/min, blood "
     "pressure 138/66 mmHg.",
     "Vital signs at current visit: temperature 37.2°C. Vital signs 7 days earlier: heart rate 87/min, "
     "respiratory rate 14/min, blood pressure 138/66 mmHg."),
    ("dev", "viral_pharyngitis_5c806a38-1619-639b-6aa4-76e3b0d59b89", "objective",
     "Vital signs at current visit: temperature 37.7°C, heart rate 86/min, respiratory rate 14/min, blood "
     "pressure 125/87 mmHg.",
     "Vital signs at current visit: temperature 37.7°C. Vital signs 7 months earlier: heart rate 86/min, "
     "respiratory rate 14/min, blood pressure 125/87 mmHg."),
    ("dev", "viral_pharyngitis_5dab19a8-ba62-c4f8-19d3-34fb57276dc8", "objective",
     "Vital signs at current visit: body temperature 37.0°C, heart rate 95/min, respiratory rate 15/min, "
     "systolic blood pressure 121 mmHg, diastolic blood pressure 72 mmHg.",
     "Vital signs at current visit: body temperature 37.0°C. Vital signs 3 months earlier: heart rate "
     "95/min, respiratory rate 15/min, systolic blood pressure 121 mmHg, diastolic blood pressure 72 mmHg."),
    ("dev", "viral_pharyngitis_5ea7dc74-2259-6089-019f-936c181bc81a", "objective",
     "Vital signs at current visit: temperature 37.3°C, heart rate 98/min, respiratory rate 14/min, "
     "systolic blood pressure 96 mmHg, diastolic blood pressure 85 mmHg.",
     "Vital signs at current visit: temperature 37.3°C. Vital signs 14 days earlier: heart rate 98/min, "
     "respiratory rate 14/min, systolic blood pressure 96 mmHg, diastolic blood pressure 85 mmHg."),
    ("dev", "viral_pharyngitis_5ea7dc74-2259-6089-019f-936c181bc81a", "objective",
     "PHQ-9: 3 (current visit)",
     "PHQ-9: 3 (14 days earlier)"),
    ("dev", "viral_pharyngitis_60a2eaec-c5c2-2773-cc02-a464fa44894e", "objective",
     "Vital signs at current visit: body temperature 37.1°C, heart rate 84/min, respiratory rate 13/min, "
     "blood pressure 138/106 mmHg (from 23 days prior), body weight 89.9 kg, height 172 cm, BMI 30.4 kg/m².",
     "Vital signs at current visit: body temperature 37.1°C. Vital signs 23 days earlier: heart rate "
     "84/min, respiratory rate 13/min, blood pressure 138/106 mmHg (from 23 days prior), body weight 89.9 "
     "kg, height 172 cm, BMI 30.4 kg/m²."),
    ("dev", "viral_pharyngitis_6462ad59-9497-7de5-8f01-6c0863edffaa", "objective",
     "Vital signs at current visit: body temperature 37.0°C, heart rate 94/min, respiratory rate 16/min, "
     "systolic blood pressure 140 mmHg, diastolic blood pressure 76 mmHg.",
     "Vital signs at current visit: body temperature 37.0°C. Vital signs 7 days earlier: heart rate "
     "94/min, respiratory rate 16/min, systolic blood pressure 140 mmHg, diastolic blood pressure 76 mmHg."),
    ("dev", "viral_pharyngitis_65705095-c976-63c6-6e0c-4e8a71d04079", "objective",
     "Vital signs at current visit: body temperature 37.2°C, heart rate 72/min, respiratory rate 14/min, "
     "systolic blood pressure 103 mmHg, diastolic blood pressure 73 mmHg.",
     "Vital signs at current visit: body temperature 37.2°C. Vital signs 15 days earlier: heart rate "
     "72/min, respiratory rate 14/min, systolic blood pressure 103 mmHg, diastolic blood pressure 73 mmHg."),
    ("dev", "viral_pharyngitis_662c83ec-f09b-5291-6978-11cbc98968f9", "objective",
     "Vital signs at current visit: body temperature 37.4°C, heart rate 67/min, respiratory rate 12/min, "
     "systolic blood pressure 151 mmHg, diastolic blood pressure 86 mmHg.",
     "Vital signs at current visit: body temperature 37.4°C. Vital signs 7 days earlier: heart rate "
     "67/min, respiratory rate 12/min, systolic blood pressure 151 mmHg, diastolic blood pressure 86 mmHg."),
    ("dev", "viral_pharyngitis_665fc81d-303a-4a55-f585-3bac02659927", "objective",
     "Vitals at current visit: temperature 37.2°C, heart rate 67/min, respiratory rate 12/min, blood "
     "pressure 125/71 mmHg, height 129.5 cm, weight 28.5 kg, BMI 17 kg/m², BMI-for-age 72.2th percentile.",
     "Vitals at current visit: temperature 37.2°C. Vitals 3 days earlier: heart rate 67/min, respiratory "
     "rate 12/min, blood pressure 125/71 mmHg, height 129.5 cm, weight 28.5 kg, BMI 17 kg/m², BMI-for-age "
     "72.2th percentile."),
    ("dev", "viral_pharyngitis_6ad1fda3-102a-9ccf-c44a-12221cb4a4f3", "objective",
     "Vital signs at current visit: body temperature 37.2°C, blood pressure 119/83 mmHg, heart rate "
     "72/min, respiratory rate 13/min.",
     "Vital signs at current visit: body temperature 37.2°C. Vital signs 14 days earlier: blood pressure "
     "119/83 mmHg, heart rate 72/min, respiratory rate 13/min."),
    ("dev", "viral_pharyngitis_6d4ed3c3-1e37-7d63-74e2-8bd8ef09dcf6", "objective",
     "Vital signs at current visit: temperature 37.6°C, heart rate 85/min, respiratory rate 15/min, blood "
     "pressure 115/77 mmHg.",
     "Vital signs at current visit: temperature 37.6°C. Vital signs 7 days earlier: heart rate 85/min, "
     "respiratory rate 15/min, blood pressure 115/77 mmHg."),
    ("dev", "viral_pharyngitis_6d5ac655-7c9a-b912-55a3-5064556b7bea", "objective",
     "Vital signs at this visit: temperature 37.2°C, heart rate 62/min, respiratory rate 14/min, blood "
     "pressure 106/79 mmHg (from most recent prior visit).",
     "Vital signs at this visit: temperature 37.2°C. Vital signs 6 months earlier: heart rate 62/min, "
     "respiratory rate 14/min, blood pressure 106/79 mmHg (from most recent prior visit)."),
    ("dev", "viral_pharyngitis_6ddd7177-0639-e136-b51c-bdc642a1a48b", "objective",
     "Vital signs at current visit: temperature 37.4°C, heart rate 61/min, respiratory rate 14/min, blood "
     "pressure 120/71 mmHg.",
     "Vital signs at current visit: temperature 37.4°C. Vital signs 30 days earlier: heart rate 61/min, "
     "respiratory rate 14/min, blood pressure 120/71 mmHg."),
    ("dev", "viral_pharyngitis_6ddd7177-0639-e136-b51c-bdc642a1a48b", "objective",
     "FEV1/FVC 11.2% (current visit)",
     "FEV1/FVC 11.2% (30 days earlier)"),
    ("dev", "viral_pharyngitis_6e7ae980-4ebe-6361-dcd7-c7deeb1af068", "objective",
     "Vital signs at current visit: temperature 37.5°C, heart rate 71/min, respiratory rate 14/min, blood "
     "pressure 119/76 mmHg.",
     "Vital signs at current visit: temperature 37.5°C. Vital signs 15 days earlier: heart rate 71/min, "
     "respiratory rate 14/min, blood pressure 119/76 mmHg."),
    ("dev", "viral_pharyngitis_6f5d97f5-ed42-0c70-c5b0-76f023ed1aed", "objective",
     "Vital signs at current visit: body temperature 37.5°C, heart rate 66/min, respiratory rate 13/min, "
     "blood pressure 128/90 mmHg.",
     "Vital signs at current visit: body temperature 37.5°C. Vital signs 20 days earlier: heart rate "
     "66/min, respiratory rate 13/min, blood pressure 128/90 mmHg."),
    ("dev", "viral_pharyngitis_738ce114-e880-dc8b-e058-3bbed40a16c4", "objective",
     "Vital signs at current visit: temperature 37.6°C, blood pressure 101/70 mmHg, heart rate 74/min, "
     "respiratory rate 15/min.",
     "Vital signs at current visit: temperature 37.6°C. Vital signs 15 days earlier: blood pressure 101/70 "
     "mmHg, heart rate 74/min, respiratory rate 15/min."),
    ("dev", "viral_pharyngitis_76d5d263-befc-7982-42c3-f2f1bbb8b34b", "objective",
     "Vital signs at current visit: body temperature 37.3°C, heart rate 95/min, respiratory rate 13/min, "
     "blood pressure 92/69 mmHg, height 168.8 cm, weight 79.4 kg, BMI 27.9 kg/m².",
     "Vital signs at current visit: body temperature 37.3°C. Vital signs 7 days earlier: heart rate "
     "95/min, respiratory rate 13/min, blood pressure 92/69 mmHg, height 168.8 cm, weight 79.4 kg, BMI "
     "27.9 kg/m²."),
    ("dev", "viral_pharyngitis_7c5e8e1e-10d3-165d-cf62-7671c77d01c6", "objective",
     "Labs (current visit): glucose 116.2 mg/dL, HbA1c 6.9%, total cholesterol 210.6 mg/dL, HDL 58.9 "
     "mg/dL, LDL 114.6 mg/dL, triglycerides 185.2 mg/dL, sodium 139.6 mmol/L, potassium 3.7 mmol/L, "
     "chloride 108 mmol/L, CO2 20.4 mmol/L, creatinine 0.8 mg/dL, BUN 14.6 mg/dL, calcium 9 mg/dL, GFR "
     "93.8 mL/min/1.73m², microalbumin/creatinine 4.4 mg/g.",
     "Labs 20 days earlier: glucose 116.2 mg/dL, HbA1c 6.9%, total cholesterol 210.6 mg/dL, HDL 58.9 "
     "mg/dL, LDL 114.6 mg/dL, triglycerides 185.2 mg/dL, sodium 139.6 mmol/L, potassium 3.7 mmol/L, "
     "chloride 108 mmol/L, CO2 20.4 mmol/L, creatinine 0.8 mg/dL, BUN 14.6 mg/dL, calcium 9 mg/dL, GFR "
     "93.8 mL/min/1.73m², microalbumin/creatinine 4.4 mg/g."),
    ("dev", "viral_pharyngitis_7c6e3c74-ad1f-8a28-ec5f-6f19e8b74a8a", "objective",
     "Current vitals: temperature 37.6°C, heart rate 97/min, respiratory rate 13/min, blood pressure "
     "130/83 mmHg.",
     "Current vitals: temperature 37.6°C. Vitals 8 months earlier: heart rate 97/min, respiratory rate "
     "13/min, blood pressure 130/83 mmHg."),
    ("dev", "viral_pharyngitis_7c74395b-4e20-9653-7fd6-66337e9d79c6", "objective",
     "Vital signs at current visit: body temperature 37.3°C, heart rate 61/min, respiratory rate 14/min, "
     "systolic blood pressure 110 mmHg, diastolic blood pressure 85 mmHg.",
     "Vital signs at current visit: body temperature 37.3°C. Vital signs 26 days earlier: heart rate "
     "61/min, respiratory rate 14/min, systolic blood pressure 110 mmHg, diastolic blood pressure 85 mmHg."),
    ("dev", "viral_pharyngitis_7e3c36cc-8b65-7bd8-4cd7-18b57ebed0f5", "objective",
     "Vital signs at current visit: body temperature 37.5°C, heart rate 92/min, respiratory rate 14/min, "
     "systolic blood pressure 123 mmHg, diastolic blood pressure 72 mmHg.",
     "Vital signs at current visit: body temperature 37.5°C. Vital signs 7 days earlier: heart rate "
     "92/min, respiratory rate 14/min, systolic blood pressure 123 mmHg, diastolic blood pressure 72 mmHg."),
    ("dev", "viral_pharyngitis_7f6a16f8-fd37-0182-ac55-ef2087e117b4", "objective",
     "Vital signs at current visit: body temperature 37.9°C, heart rate 91/min, respiratory rate 15/min, "
     "blood pressure 97/74 mmHg.",
     "Vital signs at current visit: body temperature 37.9°C. Vital signs 26 days earlier: heart rate "
     "91/min, respiratory rate 15/min, blood pressure 97/74 mmHg."),
    ("dev", "viral_pharyngitis_80067502-0b32-fcec-91fe-d3e7b711f326", "objective",
     "Vital signs at current visit: body temperature 37.2°C, heart rate 76/min, respiratory rate 13/min, "
     "systolic blood pressure 135 mmHg, diastolic blood pressure 102 mmHg.",
     "Vital signs at current visit: body temperature 37.2°C. Vital signs 14 days earlier: heart rate "
     "76/min, respiratory rate 13/min, systolic blood pressure 135 mmHg, diastolic blood pressure 102 mmHg."),
    ("dev", "viral_pharyngitis_80303d71-c98d-4d74-1e45-6bad1a0e4410", "objective",
     "Vital signs at this visit: temperature 37.5°C, heart rate 66/min, respiratory rate 15/min, systolic "
     "BP 132 mmHg, diastolic BP 115 mmHg.",
     "Vital signs at this visit: temperature 37.5°C. Vital signs 7 days earlier: heart rate 66/min, "
     "respiratory rate 15/min, systolic BP 132 mmHg, diastolic BP 115 mmHg."),
    ("dev", "viral_pharyngitis_803b4f9f-6606-5468-5f14-f72689ffa88a", "objective",
     "Vital signs at current visit: temperature 37.6°C, heart rate 83/min, respiratory rate 13/min, blood "
     "pressure 103/76 mm[Hg], height 154.2 cm, weight 71.2 kg, BMI 30 kg/m².",
     "Vital signs at current visit: temperature 37.6°C. Vital signs 7 days earlier: heart rate 83/min, "
     "respiratory rate 13/min, blood pressure 103/76 mm[Hg], height 154.2 cm, weight 71.2 kg, BMI 30 kg/m²."),
    ("dev", "viral_pharyngitis_803b4f9f-6606-5468-5f14-f72689ffa88a", "objective",
     "GAD-7 score: 1 (current visit)",
     "GAD-7 score: 1 (7 days earlier)"),
    ("dev", "viral_pharyngitis_803b4f9f-6606-5468-5f14-f72689ffa88a", "objective",
     "PHQ-2 score: 1 (current visit)",
     "PHQ-2 score: 1 (7 days earlier)"),
    ("dev", "viral_pharyngitis_81e9ecbf-b479-4ecc-b3fd-33c723d3a4c2", "objective",
     "Vital signs at current visit: temperature 38.0°C, heart rate 91/min, systolic blood pressure 113 "
     "mmHg, diastolic blood pressure 71 mmHg, respiratory rate 15/min.",
     "Vital signs at current visit: temperature 38.0°C. Vital signs 2 months earlier: heart rate 91/min, "
     "systolic blood pressure 113 mmHg, diastolic blood pressure 71 mmHg, respiratory rate 15/min."),
    ("dev", "viral_pharyngitis_82ac9f0c-bb99-5764-5b3b-a96a9b004739", "objective",
     "Vital signs at current visit: temperature 37.1°C, blood pressure 114/86 mmHg, heart rate 67/min, "
     "respiratory rate 13/min.",
     "Vital signs at current visit: temperature 37.1°C. Vital signs 23 days earlier: blood pressure 114/86 "
     "mmHg, heart rate 67/min, respiratory rate 13/min."),
    ("dev", "viral_pharyngitis_839ea010-a89f-8e46-b2f4-d7bcbe652c04", "objective",
     "Vital signs at current visit: temperature 37.3°C, heart rate 100/min, respiratory rate 15/min, "
     "systolic blood pressure 138 mmHg, diastolic blood pressure 93 mmHg.",
     "Vital signs at current visit: temperature 37.3°C. Vital signs 7 days earlier: heart rate 100/min, "
     "respiratory rate 15/min, systolic blood pressure 138 mmHg, diastolic blood pressure 93 mmHg."),
    ("dev", "viral_pharyngitis_8672d635-f67a-1e69-6e51-b1c562b9192b", "objective",
     "Vital signs at this visit: temperature 37.1°C, heart rate 62/min, respiratory rate 13/min, blood "
     "pressure 136/77 mmHg.",
     "Vital signs at this visit: temperature 37.1°C. Vital signs 14 days earlier: heart rate 62/min, "
     "respiratory rate 13/min, blood pressure 136/77 mmHg."),
    ("dev", "viral_pharyngitis_8ab431a2-a3a8-2e5a-3aae-e82b3f023ac1", "objective",
     "Vital signs at this visit: body temperature 37.0°C, blood pressure 100/73 mmHg, heart rate 64/min, "
     "respiratory rate 13/min.",
     "Vital signs at this visit: body temperature 37.0°C. Vital signs 14 days earlier: blood pressure "
     "100/73 mmHg, heart rate 64/min, respiratory rate 13/min."),
    ("dev", "viral_pharyngitis_8b539cb8-0198-34f5-172f-a6dcb6f58b9a", "objective",
     "Vital signs at current visit: body temperature 37.1°C, systolic blood pressure 125 mm[Hg], diastolic "
     "blood pressure 82 mm[Hg], heart rate 65/min, respiratory rate 15/min.",
     "Vital signs at current visit: body temperature 37.1°C. Vital signs 8 months earlier: systolic blood "
     "pressure 125 mm[Hg], diastolic blood pressure 82 mm[Hg], heart rate 65/min, respiratory rate 15/min."),
    ("dev", "viral_pharyngitis_8c792490-87ec-f222-7584-1375c14f5cae", "objective",
     "Vital signs at current visit: temperature 37.9°C, heart rate 85/min, respiratory rate 13/min, blood "
     "pressure 132/94 mmHg.",
     "Vital signs at current visit: temperature 37.9°C. Vital signs 14 days earlier: heart rate 85/min, "
     "respiratory rate 13/min, blood pressure 132/94 mmHg."),
    ("dev", "viral_pharyngitis_8d876255-30ca-0c56-e66a-586e8280e28d", "objective",
     "Vital signs at current visit: temperature 37.9°C, heart rate 85/min, respiratory rate 14/min, blood "
     "pressure 117/75 mmHg, body weight 52.7 kg, height 155.4 cm, BMI 21.8 kg/m², BMI-for-age 75.7th "
     "percentile.",
     "Vital signs at current visit: temperature 37.9°C. Vital signs 14 days earlier: heart rate 85/min, "
     "respiratory rate 14/min, blood pressure 117/75 mmHg, body weight 52.7 kg, height 155.4 cm, BMI 21.8 "
     "kg/m², BMI-for-age 75.7th percentile."),
    ("dev", "viral_pharyngitis_8f5f0f25-c3fb-b1c4-3bd7-cbf735dc378b", "objective",
     "Vital signs at current visit: temperature 37.0°C, blood pressure 128/78 mmHg, heart rate 69/min, "
     "respiratory rate 16/min.",
     "Vital signs at current visit: temperature 37.0°C. Vital signs 7 days earlier: blood pressure 128/78 "
     "mmHg, heart rate 69/min, respiratory rate 16/min."),
    ("dev", "viral_pharyngitis_903b21b2-78e6-c1b5-43f8-1c626a6e8973", "objective",
     "Vital signs at current visit: temperature 37.6°C, heart rate 90/min, respiratory rate 12/min, blood "
     "pressure 117/89 mmHg.",
     "Vital signs at current visit: temperature 37.6°C. Vital signs 7 days earlier: heart rate 90/min, "
     "respiratory rate 12/min, blood pressure 117/89 mmHg."),
    ("dev", "viral_pharyngitis_903b21b2-78e6-c1b5-43f8-1c626a6e8973", "objective",
     "Pain severity score 1/10 (current visit)",
     "Pain severity score 1/10 (7 days earlier)"),
    ("dev", "viral_pharyngitis_9061d70a-1820-7809-1b1e-ed4458b35d9e", "objective",
     "Vital signs at current visit: temperature 37.9°C, blood pressure 119/65 mmHg, heart rate 78/min, "
     "respiratory rate 16/min.",
     "Vital signs at current visit: temperature 37.9°C. Vital signs 14 days earlier: blood pressure 119/65 "
     "mmHg, heart rate 78/min, respiratory rate 16/min."),
    ("dev", "viral_pharyngitis_9263eeea-78af-23a4-fea6-bbe3a158a986", "objective",
     "Vital signs at current visit: temperature 37.3°C, blood pressure 144/86 mmHg, heart rate 91/min, "
     "respiratory rate 16/min.",
     "Vital signs at current visit: temperature 37.3°C. Vital signs 14 days earlier: blood pressure 144/86 "
     "mmHg, heart rate 91/min, respiratory rate 16/min."),
    ("dev", "viral_pharyngitis_94835e3b-bb6a-19da-b219-a114013aa086", "objective",
     "Vital signs at current visit: body temperature 37.4°C, heart rate 87/min, respiratory rate 16/min, "
     "blood pressure 107/83 mmHg.",
     "Vital signs at current visit: body temperature 37.4°C. Vital signs 21 days earlier: heart rate "
     "87/min, respiratory rate 16/min, blood pressure 107/83 mmHg."),
    ("dev", "viral_pharyngitis_94f9dbd9-7230-c137-e37f-bf29f41bcbaf", "objective",
     "Vital signs at current visit: temperature 37.4°C, heart rate 76/min, respiratory rate 14/min, "
     "systolic blood pressure 98 mmHg, diastolic blood pressure 58 mmHg.",
     "Vital signs at current visit: temperature 37.4°C. Vital signs 7 days earlier: heart rate 76/min, "
     "respiratory rate 14/min, systolic blood pressure 98 mmHg, diastolic blood pressure 58 mmHg."),
    ("dev", "viral_pharyngitis_95505054-8376-57ba-01a8-2262ff321405", "objective",
     "Vital signs at current visit: body temperature 37.6°C, blood pressure 110/94 mmHg, heart rate "
     "63/min, respiratory rate 14/min.",
     "Vital signs at current visit: body temperature 37.6°C. Vital signs 14 days earlier: blood pressure "
     "110/94 mmHg, heart rate 63/min, respiratory rate 14/min."),
    ("dev", "viral_pharyngitis_967b1f2a-5b63-c050-5df6-2be533461eb2", "objective",
     "Vital signs at current visit: body temperature 37.8°C, heart rate 67/min, respiratory rate 14/min, "
     "blood pressure 116/73 mmHg.",
     "Vital signs at current visit: body temperature 37.8°C. Vital signs 14 days earlier: heart rate "
     "67/min, respiratory rate 14/min, blood pressure 116/73 mmHg."),
    ("dev", "viral_pharyngitis_9a528f02-f58c-3da6-12a6-0d0278660669", "objective",
     "Vital signs at this visit: temperature 37.2°C, heart rate 69/min, respiratory rate 12/min, systolic "
     "BP 105 mmHg, diastolic BP 71 mmHg.",
     "Vital signs at this visit: temperature 37.2°C. Vital signs 7 days earlier: heart rate 69/min, "
     "respiratory rate 12/min, systolic BP 105 mmHg, diastolic BP 71 mmHg."),
    ("dev", "viral_pharyngitis_9b12a68c-178a-b1a5-82db-2b922049318c", "objective",
     "Vital signs at current visit: temperature 37.5°C, blood pressure 111/73 mmHg, heart rate 86/min, "
     "respiratory rate 14/min.",
     "Vital signs at current visit: temperature 37.5°C. Vital signs 14 days earlier: blood pressure 111/73 "
     "mmHg, heart rate 86/min, respiratory rate 14/min."),
    ("dev", "viral_pharyngitis_9bd57f5e-52d9-0434-cddd-cedeabca2393", "objective",
     "Vital signs at current visit: body temperature 37.2°C, heart rate 90/min, respiratory rate 12/min, "
     "systolic blood pressure 124 mmHg, diastolic blood pressure 71 mmHg.",
     "Vital signs at current visit: body temperature 37.2°C. Vital signs 20 days earlier: heart rate "
     "90/min, respiratory rate 12/min, systolic blood pressure 124 mmHg, diastolic blood pressure 71 mmHg."),
    ("dev", "viral_pharyngitis_9ceddcb0-07b0-800f-baf4-3c3f8c8d0d32", "objective",
     "Vital signs at current visit: body temperature 37.7°C, heart rate 77/min, respiratory rate 13/min, "
     "systolic blood pressure 121 mmHg, diastolic blood pressure 70 mmHg.",
     "Vital signs at current visit: body temperature 37.7°C. Vital signs 24 days earlier: heart rate "
     "77/min, respiratory rate 13/min, systolic blood pressure 121 mmHg, diastolic blood pressure 70 mmHg."),
    ("dev", "viral_pharyngitis_9d05f023-42cc-dc27-ed5f-03ea1df28b9a", "objective",
     "Vital signs at this visit: temperature 37.2°C, heart rate 96/min, respiratory rate 15/min, systolic "
     "blood pressure 156 mmHg, diastolic blood pressure 114 mmHg.",
     "Vital signs at this visit: temperature 37.2°C. Vital signs 14 days earlier: heart rate 96/min, "
     "respiratory rate 15/min, systolic blood pressure 156 mmHg, diastolic blood pressure 114 mmHg."),
    ("dev", "viral_pharyngitis_9dae7c9e-89c0-7f68-a57f-83b274a606d8", "objective",
     "Vital signs at current visit: temperature 37.2°C, heart rate 63/min, respiratory rate 15/min, blood "
     "pressure 132/75 mmHg.",
     "Vital signs at current visit: temperature 37.2°C. Vital signs 14 days earlier: heart rate 63/min, "
     "respiratory rate 15/min, blood pressure 132/75 mmHg."),
    ("dev", "viral_pharyngitis_a1735a09-2a23-5251-5853-326f459fb535", "objective",
     "Vital signs at current visit: temperature 37.2°C, heart rate 99/min, respiratory rate 13/min, blood "
     "pressure 122/85 mmHg.",
     "Vital signs at current visit: temperature 37.2°C. Vital signs 9 months earlier: heart rate 99/min, "
     "respiratory rate 13/min, blood pressure 122/85 mmHg."),
    ("dev", "viral_pharyngitis_a1967b2e-49b0-bec1-d8ef-32cff78b903b", "objective",
     "Vital signs at current visit: body temperature 37.8°C, heart rate 72/min, respiratory rate 16/min, "
     "systolic blood pressure 135 mmHg, diastolic blood pressure 81 mmHg.",
     "Vital signs at current visit: body temperature 37.8°C. Vital signs 9 days earlier: heart rate "
     "72/min, respiratory rate 16/min, systolic blood pressure 135 mmHg, diastolic blood pressure 81 mmHg."),
    ("dev", "viral_pharyngitis_a1f047e0-1478-b314-4946-3045f6450031", "objective",
     "Vital signs at current visit: body temperature 37.0°C, heart rate 87/min, respiratory rate 13/min, "
     "systolic blood pressure 111 mmHg, diastolic blood pressure 84 mmHg.",
     "Vital signs at current visit: body temperature 37.0°C. Vital signs 15 days earlier: heart rate "
     "87/min, respiratory rate 13/min, systolic blood pressure 111 mmHg, diastolic blood pressure 84 mmHg."),
    ("dev", "viral_pharyngitis_a2efd40f-a6a8-5516-dad9-a3faa4c68bc5", "objective",
     "Vital signs at this visit: body temperature 37.7°C, heart rate 98/min, respiratory rate 14/min, "
     "systolic blood pressure 104 mmHg, diastolic blood pressure 71 mmHg.",
     "Vital signs at this visit: body temperature 37.7°C. Vital signs 7 days earlier: heart rate 98/min, "
     "respiratory rate 14/min, systolic blood pressure 104 mmHg, diastolic blood pressure 71 mmHg."),
    ("dev", "viral_pharyngitis_a3b9a5d7-ba21-9dd9-1704-503979ae3919", "objective",
     "Vital signs at current visit: body temperature 37.0°C, heart rate 92/min, respiratory rate 15/min, "
     "blood pressure 105/78 mmHg.",
     "Vital signs at current visit: body temperature 37.0°C. Vital signs 11 months earlier: heart rate "
     "92/min, respiratory rate 15/min, blood pressure 105/78 mmHg."),
    ("dev", "viral_pharyngitis_ab0c7f22-5ab8-64a5-45ca-59e7efdce412", "objective",
     "Vital signs at this visit: body temperature 37.2°C, heart rate 100/min, respiratory rate 14/min, "
     "blood pressure 114/83 mmHg.",
     "Vital signs at this visit: body temperature 37.2°C. Vital signs 7 days earlier: heart rate 100/min, "
     "respiratory rate 14/min, blood pressure 114/83 mmHg."),
    ("dev", "viral_pharyngitis_ac5c312a-c008-df6c-5570-3a870bee921a", "objective",
     "Vital signs at current visit: temperature 37.3°C, heart rate 97/min, respiratory rate 14/min, blood "
     "pressure 151/87 mmHg.",
     "Vital signs at current visit: temperature 37.3°C. Vital signs 14 days earlier: heart rate 97/min, "
     "respiratory rate 14/min, blood pressure 151/87 mmHg."),
    ("dev", "viral_pharyngitis_ae5c20cf-6926-6997-9441-583f7b177b53", "objective",
     "Vital signs at current visit: temperature 37.1°C, blood pressure 110/78 mmHg, heart rate 82/min, "
     "respiratory rate 12/min.",
     "Vital signs at current visit: temperature 37.1°C. Vital signs 3 months earlier: blood pressure "
     "110/78 mmHg, heart rate 82/min, respiratory rate 12/min."),
    ("dev", "viral_pharyngitis_af28212b-2b55-dee5-ed59-c6eb56352e05", "objective",
     "Vital signs at current visit: temperature 37.8°C, heart rate 99/min, respiratory rate 15/min, blood "
     "pressure 125/65 mmHg.",
     "Vital signs at current visit: temperature 37.8°C. Vital signs 14 days earlier: heart rate 99/min, "
     "respiratory rate 15/min, blood pressure 125/65 mmHg."),
    ("dev", "viral_pharyngitis_b140acc8-e6ae-7ca0-0473-0521e0282141", "objective",
     "Vital signs at this visit: temperature 37.1°C, heart rate 91/min, respiratory rate 15/min, blood "
     "pressure 101/79 mmHg.",
     "Vital signs at this visit: temperature 37.1°C. Vital signs 7 days earlier: heart rate 91/min, "
     "respiratory rate 15/min, blood pressure 101/79 mmHg."),
    ("dev", "viral_pharyngitis_b2d3adfa-1cb5-ea8d-f10d-5d77a777e864", "objective",
     "Vital signs at current visit: body temperature 37.4°C, heart rate 84/min, respiratory rate 14/min, "
     "blood pressure 107/80 mmHg, body weight 9.1 kg, body height 73.5 cm, head occipital-frontal "
     "circumference 45.6 cm (33.2nd percentile), weight-for-length 28.2th percentile.",
     "Vital signs at current visit: body temperature 37.4°C. Vital signs 22 days earlier: heart rate "
     "84/min, respiratory rate 14/min, blood pressure 107/80 mmHg, body weight 9.1 kg, body height 73.5 "
     "cm, head occipital-frontal circumference 45.6 cm (33.2nd percentile), weight-for-length 28.2th "
     "percentile."),
    ("dev", "viral_pharyngitis_b4002aba-2e2c-df54-e048-c0ff1d75c9a4", "objective",
     "Vital signs at current visit: temperature 37.1°C, heart rate 86/min, respiratory rate 14/min, "
     "systolic blood pressure 123 mmHg, diastolic blood pressure 98 mmHg.",
     "Vital signs at current visit: temperature 37.1°C. Vital signs 7 days earlier: heart rate 86/min, "
     "respiratory rate 14/min, systolic blood pressure 123 mmHg, diastolic blood pressure 98 mmHg."),
    ("dev", "viral_pharyngitis_b4e693be-b2b4-aadc-805f-afcc6caa83d1", "objective",
     "Vital signs at current visit: body temperature 37.6°C, heart rate 71/min, respiratory rate 15/min, "
     "blood pressure 96/55 mmHg, body weight 77.1 kg, height 169.5 cm, BMI 26.8 kg/m².",
     "Vital signs at current visit: body temperature 37.6°C. Vital signs 23 days earlier: heart rate "
     "71/min, respiratory rate 15/min, blood pressure 96/55 mmHg, body weight 77.1 kg, height 169.5 cm, "
     "BMI 26.8 kg/m²."),
    ("dev", "viral_pharyngitis_b5862dde-44d1-bffd-fc90-8e15d5ecffe5", "objective",
     "Vital signs at current visit: temperature 37.6°C, heart rate 92/min, respiratory rate 15/min, blood "
     "pressure 102/76 mmHg.",
     "Vital signs at current visit: temperature 37.6°C. Vital signs 24 days earlier: heart rate 92/min, "
     "respiratory rate 15/min, blood pressure 102/76 mmHg."),
    ("dev", "viral_pharyngitis_b5c80e84-e3be-de07-11e0-bb10be4a9ba5", "objective",
     "Vital signs at current visit: body temperature 37.5°C, heart rate 91/min, respiratory rate 14/min, "
     "systolic blood pressure 91 mmHg, diastolic blood pressure 77 mmHg.",
     "Vital signs at current visit: body temperature 37.5°C. Vital signs 15 days earlier: heart rate "
     "91/min, respiratory rate 14/min, systolic blood pressure 91 mmHg, diastolic blood pressure 77 mmHg."),
    ("dev", "viral_pharyngitis_b7d612ec-b77a-59b7-886b-d7125f10e34e", "objective",
     "Vital signs at current visit: temperature 37.3°C, heart rate 63/min, respiratory rate 12/min, "
     "systolic BP 139 mmHg, diastolic BP 83 mmHg.",
     "Vital signs at current visit: temperature 37.3°C. Vital signs 15 days earlier: heart rate 63/min, "
     "respiratory rate 12/min, systolic BP 139 mmHg, diastolic BP 83 mmHg."),
    ("dev", "viral_pharyngitis_b80d58ef-336b-cee3-f66e-443fd5bcc0c4", "objective",
     "Vitals at current visit: temperature 37.8°C, blood pressure 116/94 mmHg, heart rate 86/min, "
     "respiratory rate 12/min.",
     "Vitals at current visit: temperature 37.8°C. Vitals 8 days earlier: blood pressure 116/94 mmHg, "
     "heart rate 86/min, respiratory rate 12/min."),
    ("dev", "viral_pharyngitis_b8c80493-3a58-eb07-880b-114a406dba6f", "objective",
     "Vital signs at this visit: body temperature 37.3°C, heart rate 88/min, respiratory rate 16/min, "
     "blood pressure 117/77 mmHg.",
     "Vital signs at this visit: body temperature 37.3°C. Vital signs 26 days earlier: heart rate 88/min, "
     "respiratory rate 16/min, blood pressure 117/77 mmHg."),
    ("dev", "viral_pharyngitis_b8d0e098-d211-c7cf-4024-0770f4df7d19", "objective",
     "Vital signs at current visit: body temperature 37.1°C, blood pressure 103/61 mmHg, heart rate "
     "82/min, respiratory rate 16/min.",
     "Vital signs at current visit: body temperature 37.1°C. Vital signs 2 months earlier: blood pressure "
     "103/61 mmHg, heart rate 82/min, respiratory rate 16/min."),
    ("dev", "viral_pharyngitis_b955c3c8-0af4-7148-114f-0b9b0986457d", "objective",
     "Vital signs at current visit: body temperature 37.2°C, heart rate 68/min, blood pressure 129/83 "
     "mmHg, respiratory rate 14/min.",
     "Vital signs at current visit: body temperature 37.2°C. Vital signs 2 months earlier: heart rate "
     "68/min, blood pressure 129/83 mmHg, respiratory rate 14/min."),
    ("dev", "viral_pharyngitis_b955c3c8-0af4-7148-114f-0b9b0986457d", "objective",
     "PHQ-2 score 2 (current visit)",
     "PHQ-2 score 2 (2 months earlier)"),
    ("dev", "viral_pharyngitis_bb5422eb-f999-9c40-b503-f6ca314d1c7b", "objective",
     "Vital signs at this visit: body temperature 37.2°C, heart rate 64/min, respiratory rate 13/min, "
     "blood pressure 133/74 mmHg.",
     "Vital signs at this visit: body temperature 37.2°C. Vital signs 8 months earlier: heart rate 64/min, "
     "respiratory rate 13/min, blood pressure 133/74 mmHg."),
    ("dev", "viral_pharyngitis_bc27a7ab-93ff-b06f-6027-c0533e0f40e7", "objective",
     "Vital signs at this visit: temperature 37.8°C, heart rate 88/min, respiratory rate 14/min, systolic "
     "blood pressure 78 mmHg, diastolic blood pressure 55 mmHg.",
     "Vital signs at this visit: temperature 37.8°C. Vital signs 14 days earlier: heart rate 88/min, "
     "respiratory rate 14/min, systolic blood pressure 78 mmHg, diastolic blood pressure 55 mmHg."),
    ("dev", "viral_pharyngitis_bdaf8e41-a7f4-2e10-979b-da14fcc20dde", "objective",
     "Vitals at current visit: temperature 37.7°C, blood pressure 117/75 mmHg, heart rate 64/min, "
     "respiratory rate 15/min.",
     "Vitals at current visit: temperature 37.7°C. Vitals 14 days earlier: blood pressure 117/75 mmHg, "
     "heart rate 64/min, respiratory rate 15/min."),
    ("dev", "viral_pharyngitis_bdd36742-bad4-0257-8556-3c09b69e8ee5", "objective",
     "Vital signs at current visit: temperature 37.6°C, heart rate 89/min, respiratory rate 16/min, blood "
     "pressure 109/70 mmHg.",
     "Vital signs at current visit: temperature 37.6°C. Vital signs 7 days earlier: heart rate 89/min, "
     "respiratory rate 16/min, blood pressure 109/70 mmHg."),
    ("dev", "viral_pharyngitis_be6493a1-341d-9101-46ca-a31b9f0672c2", "objective",
     "Vital signs at this visit: temperature 37.3°C, heart rate 83/min, respiratory rate 13/min, blood "
     "pressure 106/80 mmHg.",
     "Vital signs at this visit: temperature 37.3°C. Vital signs 15 days earlier: heart rate 83/min, "
     "respiratory rate 13/min, blood pressure 106/80 mmHg."),
    ("dev", "viral_pharyngitis_c026a95d-e0e7-3ddb-5d6a-7fa0c4acec7b", "objective",
     "Vital signs at current visit: body temperature 37.9°C, blood pressure 116/61 mmHg, heart rate "
     "88/min, respiratory rate 15/min.",
     "Vital signs at current visit: body temperature 37.9°C. Vital signs 14 days earlier: blood pressure "
     "116/61 mmHg, heart rate 88/min, respiratory rate 15/min."),
    ("dev", "viral_pharyngitis_c1818685-1952-385e-0115-40d50a309d1c", "objective",
     "Vital signs at current visit: temperature 37.1°C, heart rate 83/min, respiratory rate 14/min, blood "
     "pressure 120/80 mmHg.",
     "Vital signs at current visit: temperature 37.1°C. Vital signs 21 days earlier: heart rate 83/min, "
     "respiratory rate 14/min, blood pressure 120/80 mmHg."),
    ("dev", "viral_pharyngitis_c2132c4e-a4f7-b57e-e7c0-1ceccd3270c5", "objective",
     "Vitals at current visit: body temperature 37.2°C, heart rate 84/min, respiratory rate 13/min, "
     "systolic BP 140 mmHg, diastolic BP 95 mmHg, body weight 29.2 kg, height 118.6 cm, BMI 20.8 kg/m², "
     "BMI-for-age/sex 95.3th percentile.",
     "Vitals at current visit: body temperature 37.2°C. Vitals 3 days earlier: heart rate 84/min, "
     "respiratory rate 13/min, systolic BP 140 mmHg, diastolic BP 95 mmHg, body weight 29.2 kg, height "
     "118.6 cm, BMI 20.8 kg/m², BMI-for-age/sex 95.3th percentile."),
    ("dev", "viral_pharyngitis_c27fdc62-df88-f4c3-8388-e0af47aa4347", "objective",
     "Vital signs at current visit: temperature 37.9°C, heart rate 85/min, respiratory rate 12/min, "
     "systolic blood pressure 116 mmHg, diastolic blood pressure 80 mmHg.",
     "Vital signs at current visit: temperature 37.9°C. Vital signs 18 days earlier: heart rate 85/min, "
     "respiratory rate 12/min, systolic blood pressure 116 mmHg, diastolic blood pressure 80 mmHg."),
    ("dev", "viral_pharyngitis_c2831bc0-6f38-f2e3-ad22-9052758e733e", "objective",
     "Vital signs at current visit: body temperature 37.6°C, heart rate 85/min, respiratory rate 16/min, "
     "blood pressure 142/98 mmHg.",
     "Vital signs at current visit: body temperature 37.6°C. Vital signs 7 days earlier: heart rate "
     "85/min, respiratory rate 16/min, blood pressure 142/98 mmHg."),
    ("dev", "viral_pharyngitis_c37319dd-562d-5aa7-894e-cc79c7249c8c", "objective",
     "Vital signs at current visit: temperature 37.5°C, heart rate 85/min, blood pressure 114/92 mmHg, "
     "respiratory rate 16/min.",
     "Vital signs at current visit: temperature 37.5°C. Vital signs 14 days earlier: heart rate 85/min, "
     "blood pressure 114/92 mmHg, respiratory rate 16/min."),
    ("dev", "viral_pharyngitis_c4283966-cf9c-e7b1-34d1-dab33de969f5", "objective",
     "Vital signs at current visit: body temperature 37.0°C, heart rate 67/min, respiratory rate 13/min, "
     "blood pressure 142/78 mmHg, body weight 32.5 kg, height 141 cm, BMI 16.3 kg/m², BMI-for-age 40.7th "
     "percentile.",
     "Vital signs at current visit: body temperature 37.0°C. Vital signs 7 days earlier: heart rate "
     "67/min, respiratory rate 13/min, blood pressure 142/78 mmHg, body weight 32.5 kg, height 141 cm, BMI "
     "16.3 kg/m², BMI-for-age 40.7th percentile."),
    ("dev", "viral_pharyngitis_c4ec06ae-58d0-1515-4c18-91bd406d439e", "objective",
     "Vital signs at current visit: temperature 37.0°C, heart rate 79/min, respiratory rate 14/min, blood "
     "pressure 117/73 mmHg.",
     "Vital signs at current visit: temperature 37.0°C. Vital signs 7 days earlier: heart rate 79/min, "
     "respiratory rate 14/min, blood pressure 117/73 mmHg."),
    ("dev", "viral_pharyngitis_c728bcb5-aa8c-dd3a-9fe4-ce537b30038c", "objective",
     "Vital signs at current visit: body temperature 37.9°C, blood pressure 112/72 mmHg, heart rate "
     "76/min, respiratory rate 15/min.",
     "Vital signs at current visit: body temperature 37.9°C. Vital signs 14 days earlier: blood pressure "
     "112/72 mmHg, heart rate 76/min, respiratory rate 15/min."),
    ("dev", "viral_pharyngitis_c899f6e7-bb7c-294c-c9a2-f4f25ec32a1d", "objective",
     "Vital signs at current visit: temperature 37.0°C, heart rate 90/min, respiratory rate 14/min, "
     "systolic BP 138 mmHg, diastolic BP 69 mmHg.",
     "Vital signs at current visit: temperature 37.0°C. Vital signs 3 days earlier: heart rate 90/min, "
     "respiratory rate 14/min, systolic BP 138 mmHg, diastolic BP 69 mmHg."),
    ("dev", "viral_pharyngitis_c9ca077c-8972-5384-e314-6bce57b90608", "objective",
     "Vital signs at current visit: temperature 37.8°C, heart rate 84/min, respiratory rate 13/min, "
     "systolic blood pressure 116 mmHg, diastolic blood pressure 110 mmHg.",
     "Vital signs at current visit: temperature 37.8°C. Vital signs 7 days earlier: heart rate 84/min, "
     "respiratory rate 13/min, systolic blood pressure 116 mmHg, diastolic blood pressure 110 mmHg."),
    ("dev", "viral_pharyngitis_ca8b098f-a9b0-29e0-2b86-67a1d1ec0a21", "objective",
     "Vital signs at this visit: body temperature 37.3°C, heart rate 80/min, blood pressure 116/73 mmHg, "
     "respiratory rate 13/min.",
     "Vital signs at this visit: body temperature 37.3°C. Vital signs 7 months earlier: heart rate 80/min, "
     "blood pressure 116/73 mmHg, respiratory rate 13/min."),
    ("dev", "viral_pharyngitis_cb0cc5d0-e8ab-d956-b10f-64e8af178dcd", "objective",
     "Vitals at current visit: body temperature 37.5°C, heart rate 88/min, respiratory rate 13/min, "
     "systolic BP 138 mmHg, diastolic BP 88 mmHg, height 117.2 cm, weight 24.4 kg, BMI 17.8 kg/m², "
     "BMI-for-age 88.4th percentile, pain score 1/10.",
     "Vitals at current visit: body temperature 37.5°C. Vitals 9 days earlier: heart rate 88/min, "
     "respiratory rate 13/min, systolic BP 138 mmHg, diastolic BP 88 mmHg, height 117.2 cm, weight 24.4 "
     "kg, BMI 17.8 kg/m², BMI-for-age 88.4th percentile, pain score 1/10."),
    ("dev", "viral_pharyngitis_cb9f07f0-5436-45c9-38dc-63facb306e91", "objective",
     "Vital signs at current visit: body temperature 37.2°C, heart rate 82/min, respiratory rate 13/min, "
     "blood pressure 90/79 mm[Hg], body weight 17.6 kg, body height 107.5 cm, BMI 15.3 kg/m², "
     "BMI-for-age/sex 52.1st percentile.",
     "Vital signs at current visit: body temperature 37.2°C. Vital signs 14 days earlier: heart rate "
     "82/min, respiratory rate 13/min, blood pressure 90/79 mm[Hg], body weight 17.6 kg, body height 107.5 "
     "cm, BMI 15.3 kg/m², BMI-for-age/sex 52.1st percentile."),
    ("dev", "viral_pharyngitis_cca125e5-1a50-536d-4271-f398518e1d11", "objective",
     "Vital signs at current visit: body temperature 38.0°C, blood pressure 116/75 mmHg, heart rate "
     "84/min, respiratory rate 14/min.",
     "Vital signs at current visit: body temperature 38.0°C. Vital signs 3 days earlier: blood pressure "
     "116/75 mmHg, heart rate 84/min, respiratory rate 14/min."),
    ("dev", "viral_pharyngitis_cd4a7601-8ce7-156c-4b2e-a093abae53d5", "objective",
     "Vital signs at current visit: temperature 37.6°C, heart rate 87/min, respiratory rate 13/min, blood "
     "pressure 133/80 mmHg.",
     "Vital signs at current visit: temperature 37.6°C. Vital signs 7 days earlier: heart rate 87/min, "
     "respiratory rate 13/min, blood pressure 133/80 mmHg."),
    ("dev", "viral_pharyngitis_cdba2e5c-b5e2-0dc2-0317-f9f6bf5efd84", "objective",
     "Vital signs at current visit: temperature 37.1°C, heart rate 74/min, respiratory rate 14/min, "
     "systolic blood pressure 97 mmHg, diastolic blood pressure 68 mmHg.",
     "Vital signs at current visit: temperature 37.1°C. Vital signs 14 days earlier: heart rate 74/min, "
     "respiratory rate 14/min, systolic blood pressure 97 mmHg, diastolic blood pressure 68 mmHg."),
    ("dev", "viral_pharyngitis_ce05d199-cbd0-89fd-f6cc-7fcbae509c99", "objective",
     "Vital signs at current visit: temperature 37.4°C, blood pressure 118/97 mmHg, heart rate 80/min, "
     "respiratory rate 14/min, body weight 80.1 kg, height 173.9 cm, BMI 26.5 kg/m².",
     "Vital signs at current visit: temperature 37.4°C. Vital signs 25 days earlier: blood pressure 118/97 "
     "mmHg, heart rate 80/min, respiratory rate 14/min, body weight 80.1 kg, height 173.9 cm, BMI 26.5 "
     "kg/m²."),
    ("dev", "viral_pharyngitis_ce290b6a-8bda-e3e3-3faf-c83d2feb04cd", "objective",
     "Vitals at current visit: temperature 37.1°C, heart rate 83/min, respiratory rate 15/min, blood "
     "pressure 109/78 mmHg, height 173.9 cm, weight 76.7 kg, BMI 25.4 kg/m².",
     "Vitals at current visit: temperature 37.1°C. Vitals 14 days earlier: heart rate 83/min, respiratory "
     "rate 15/min, blood pressure 109/78 mmHg, height 173.9 cm, weight 76.7 kg, BMI 25.4 kg/m²."),
    ("dev", "viral_pharyngitis_cfc8f8a7-45cc-abf8-7b4f-48c55dcf0ab2", "objective",
     "Vital signs at current visit: body temperature 37.7°C, heart rate 69/min, respiratory rate 14/min, "
     "systolic blood pressure 133 mmHg, diastolic blood pressure 70 mmHg.",
     "Vital signs at current visit: body temperature 37.7°C. Vital signs 24 days earlier: heart rate "
     "69/min, respiratory rate 14/min, systolic blood pressure 133 mmHg, diastolic blood pressure 70 mmHg."),
    ("dev", "viral_pharyngitis_d00ccac7-a78f-976f-9610-d3ce045aba2d", "objective",
     "PHQ-9 score 16 (current visit)",
     "PHQ-9 score 16 (14 days earlier)"),
    ("dev", "viral_pharyngitis_d00ccac7-a78f-976f-9610-d3ce045aba2d", "objective",
     "PHQ-2 score 3 (current visit)",
     "PHQ-2 score 3 (14 days earlier)"),
    ("dev", "viral_pharyngitis_d026b5e2-027f-2d56-3d38-709c39263434", "objective",
     "Vitals at this visit: temperature 37.6°C, blood pressure 101/80 mmHg, heart rate 89/min, respiratory "
     "rate 13/min.",
     "Vitals at this visit: temperature 37.6°C. Vitals 1 month earlier: blood pressure 101/80 mmHg, heart "
     "rate 89/min, respiratory rate 13/min."),
    ("dev", "viral_pharyngitis_d0577f12-faef-d88f-9846-2d17efaf4309", "objective",
     "Vital signs at current visit: body temperature 37.1°C, heart rate 66/min, respiratory rate 14/min, "
     "blood pressure 130/73 mmHg.",
     "Vital signs at current visit: body temperature 37.1°C. Vital signs 14 days earlier: heart rate "
     "66/min, respiratory rate 14/min, blood pressure 130/73 mmHg."),
    ("dev", "viral_pharyngitis_d0a77214-6eac-5cea-a0dd-0cbb99a1335c", "objective",
     "Vital signs at current visit: body temperature 37.7°C, blood pressure 102/76 mmHg, heart rate "
     "78/min, respiratory rate 12/min.",
     "Vital signs at current visit: body temperature 37.7°C. Vital signs 3 months earlier: blood pressure "
     "102/76 mmHg, heart rate 78/min, respiratory rate 12/min."),
    ("dev", "viral_pharyngitis_d2c15b52-0078-f3bd-94e3-627c6563bb73", "objective",
     "Vital signs at current visit: body temperature 37.3°C, heart rate 100/min, respiratory rate 13/min, "
     "systolic blood pressure 135 mmHg, diastolic blood pressure 94 mmHg.",
     "Vital signs at current visit: body temperature 37.3°C. Vital signs 7 days earlier: heart rate "
     "100/min, respiratory rate 13/min, systolic blood pressure 135 mmHg, diastolic blood pressure 94 mmHg."),
    ("dev", "viral_pharyngitis_d342b6ed-fb8e-a369-0988-36d394a682a5", "objective",
     "Vital signs at current visit: body temperature 38.0°C, heart rate 68/min, respiratory rate 16/min, "
     "blood pressure 133/81 mmHg.",
     "Vital signs at current visit: body temperature 38.0°C. Vital signs 14 days earlier: heart rate "
     "68/min, respiratory rate 16/min, blood pressure 133/81 mmHg."),
    ("dev", "viral_pharyngitis_d47186b8-3580-5cfb-d782-4e2b9073921c", "objective",
     "Vital signs at current visit: temperature 37.8°C, heart rate 77/min, respiratory rate 14/min, blood "
     "pressure 100/90 mmHg.",
     "Vital signs at current visit: temperature 37.8°C. Vital signs 1 month earlier: heart rate 77/min, "
     "respiratory rate 14/min, blood pressure 100/90 mmHg."),
    ("dev", "viral_pharyngitis_d488eee3-9f33-c938-b2a3-cac004606ea7", "objective",
     "Vital signs at current visit: temperature 37.9°C, heart rate 99/min, respiratory rate 13/min, blood "
     "pressure 98/80 mmHg, body weight 71.7 kg, body height 167.3 cm, BMI 25.6 kg/m², BMI-for-age/sex "
     "percentile 91%.",
     "Vital signs at current visit: temperature 37.9°C. Vital signs 3 days earlier: heart rate 99/min, "
     "respiratory rate 13/min, blood pressure 98/80 mmHg, body weight 71.7 kg, body height 167.3 cm, BMI "
     "25.6 kg/m², BMI-for-age/sex percentile 91%."),
    ("dev", "viral_pharyngitis_d5a42573-f20f-cf34-8897-35516adf366e", "objective",
     "Vital signs at current visit: body temperature 37.3°C, heart rate 96/min, respiratory rate 14/min, "
     "blood pressure 138/95 mmHg.",
     "Vital signs at current visit: body temperature 37.3°C. Vital signs 14 days earlier: heart rate "
     "96/min, respiratory rate 14/min, blood pressure 138/95 mmHg."),
    ("dev", "viral_pharyngitis_d727162b-e7f0-fcdf-1238-15dbec8414e4", "objective",
     "Vital signs at current visit: temperature 37.1°C, heart rate 93/min, respiratory rate 13/min, "
     "systolic blood pressure 82 mmHg, diastolic blood pressure 43 mmHg.",
     "Vital signs at current visit: temperature 37.1°C. Vital signs 15 days earlier: heart rate 93/min, "
     "respiratory rate 13/min, systolic blood pressure 82 mmHg, diastolic blood pressure 43 mmHg."),
    ("dev", "viral_pharyngitis_d75b6e57-2a21-725e-d984-677b290f00cd", "objective",
     "Vital signs at current visit: body temperature 37.2°C, heart rate 64/min, respiratory rate 13/min, "
     "systolic blood pressure 120 mmHg, diastolic blood pressure 84 mmHg.",
     "Vital signs at current visit: body temperature 37.2°C. Vital signs 20 days earlier: heart rate "
     "64/min, respiratory rate 13/min, systolic blood pressure 120 mmHg, diastolic blood pressure 84 mmHg."),
    ("dev", "viral_pharyngitis_d7cba0f2-cd91-e049-c7a4-a615429111ae", "objective",
     "Vital signs at current visit: body temperature 37.8°C, heart rate 100/min, blood pressure 126/86 "
     "mmHg, respiratory rate 14/min.",
     "Vital signs at current visit: body temperature 37.8°C. Vital signs 1 month earlier: heart rate "
     "100/min, blood pressure 126/86 mmHg, respiratory rate 14/min."),
    ("dev", "viral_pharyngitis_d8edace8-8d09-c0ab-f37a-345533aa9390", "objective",
     "Vital signs at this visit: body temperature 37.3°C, heart rate 99/min, respiratory rate 15/min, "
     "blood pressure 124/66 mmHg (from 7 months prior).",
     "Vital signs at this visit: body temperature 37.3°C. Vital signs 7 months earlier: heart rate 99/min, "
     "respiratory rate 15/min, blood pressure 124/66 mmHg (from 7 months prior)."),
    ("dev", "viral_pharyngitis_da2ac22e-fd38-f73d-d486-213ea31afd0b", "objective",
     "Vitals at current visit: body temperature 37.7°C, heart rate 99/min, respiratory rate 14/min, blood "
     "pressure 101/79 mmHg, body weight 40.7 kg, height 153.2 cm, BMI 17.4 kg/m², BMI-for-age 38.3th "
     "percentile.",
     "Vitals at current visit: body temperature 37.7°C. Vitals 21 days earlier: heart rate 99/min, "
     "respiratory rate 14/min, blood pressure 101/79 mmHg, body weight 40.7 kg, height 153.2 cm, BMI 17.4 "
     "kg/m², BMI-for-age 38.3th percentile."),
    ("dev", "viral_pharyngitis_da7d55c8-6b5a-ce4a-ee55-da834438a2e8", "objective",
     "Vital signs at current visit: body temperature 37.4°C, heart rate 85/min, respiratory rate 13/min, "
     "blood pressure 117/76 mmHg, body weight 76.5 kg, height 161.4 cm, BMI 29.4 kg/m².",
     "Vital signs at current visit: body temperature 37.4°C. Vital signs 6 days earlier: heart rate "
     "85/min, respiratory rate 13/min, blood pressure 117/76 mmHg, body weight 76.5 kg, height 161.4 cm, "
     "BMI 29.4 kg/m²."),
    ("dev", "viral_pharyngitis_daa2378f-7faf-4938-32bb-321f535458e4", "objective",
     "Vital signs at current visit: temperature 37.6°C, heart rate 83/min, respiratory rate 13/min, blood "
     "pressure 96/75 mmHg.",
     "Vital signs at current visit: temperature 37.6°C. Vital signs 7 days earlier: heart rate 83/min, "
     "respiratory rate 13/min, blood pressure 96/75 mmHg."),
    ("dev", "viral_pharyngitis_dc622d90-707f-6d94-46a0-f42a3feeec52", "objective",
     "Vital signs at current visit: body temperature 37.6°C, heart rate 64/min, respiratory rate 14/min, "
     "blood pressure 101/85 mmHg, body weight 84.7 kg, height 170.6 cm, BMI 29.1 kg/m².",
     "Vital signs at current visit: body temperature 37.6°C. Vital signs 7 days earlier: heart rate "
     "64/min, respiratory rate 14/min, blood pressure 101/85 mmHg, body weight 84.7 kg, height 170.6 cm, "
     "BMI 29.1 kg/m²."),
    ("dev", "viral_pharyngitis_dc622d90-707f-6d94-46a0-f42a3feeec52", "objective",
     "PHQ-2 score: 1 (current visit)",
     "PHQ-2 score: 1 (7 days earlier)"),
    ("dev", "viral_pharyngitis_dc622d90-707f-6d94-46a0-f42a3feeec52", "objective",
     "AUDIT-C score: 1 (current visit)",
     "AUDIT-C score: 1 (7 days earlier)"),
    ("dev", "viral_pharyngitis_dd35834c-52cb-a531-2b13-ea33ff262041", "objective",
     "Vital signs at current visit: temperature 37.3°C, heart rate 93/min, respiratory rate 15/min, blood "
     "pressure 107/81 mmHg.",
     "Vital signs at current visit: temperature 37.3°C. Vital signs 14 days earlier: heart rate 93/min, "
     "respiratory rate 15/min, blood pressure 107/81 mmHg."),
    ("dev", "viral_pharyngitis_df0aee1b-82f2-78d6-d096-ad48e9d7ad9f", "objective",
     "Vitals at current visit: body temperature 37.1°C, heart rate 75/min, respiratory rate 13/min, "
     "systolic blood pressure 133 mmHg, diastolic blood pressure 83 mmHg.",
     "Vitals at current visit: body temperature 37.1°C. Vitals 2 months earlier: heart rate 75/min, "
     "respiratory rate 13/min, systolic blood pressure 133 mmHg, diastolic blood pressure 83 mmHg."),
    ("dev", "viral_pharyngitis_df0aee1b-82f2-78d6-d096-ad48e9d7ad9f", "objective",
     "PHQ-2 total score 1 (current visit)",
     "PHQ-2 total score 1 (2 months earlier)"),
    ("dev", "viral_pharyngitis_df84cebe-bf77-5268-476d-51783f78340c", "objective",
     "Vital signs at current visit: temperature 37.3°C, heart rate 91/min, respiratory rate 14/min, blood "
     "pressure 122/71 mmHg.",
     "Vital signs at current visit: temperature 37.3°C. Vital signs 20 days earlier: heart rate 91/min, "
     "respiratory rate 14/min, blood pressure 122/71 mmHg."),
    ("dev", "viral_pharyngitis_e0c34fca-a799-5ffd-b054-460e244d87b0", "objective",
     "Vital signs at current visit: temperature 37.5°C, heart rate 98/min, respiratory rate 13/min, blood "
     "pressure 123/95 mmHg, body weight 64.4 kg, height 169.7 cm, BMI 22.4 kg/m2 (59th percentile for age "
     "and sex).",
     "Vital signs at current visit: temperature 37.5°C. Vital signs 14 days earlier: heart rate 98/min, "
     "respiratory rate 13/min, blood pressure 123/95 mmHg, body weight 64.4 kg, height 169.7 cm, BMI 22.4 "
     "kg/m2 (59th percentile for age and sex)."),
    ("dev", "viral_pharyngitis_e2e1a8fa-c9f4-b5b5-cc05-3c1effb050ea", "objective",
     "Vital signs at current visit: body temperature 37.6°C, blood pressure 132/78 mmHg, heart rate "
     "71/min, respiratory rate 14/min.",
     "Vital signs at current visit: body temperature 37.6°C. Vital signs 1 month earlier: blood pressure "
     "132/78 mmHg, heart rate 71/min, respiratory rate 14/min."),
    ("dev", "viral_pharyngitis_e2f0523c-a58d-f429-b965-1d2d44b9e52e", "objective",
     "Vital signs at current visit: temperature 37.6°C, blood pressure 103/75 mmHg, heart rate 74/min, "
     "respiratory rate 13/min.",
     "Vital signs at current visit: temperature 37.6°C. Vital signs 7 days earlier: blood pressure 103/75 "
     "mmHg, heart rate 74/min, respiratory rate 13/min."),
    ("dev", "viral_pharyngitis_e4bba3d1-3000-3477-7714-8cc3d64d559c", "objective",
     "Vital signs at current visit: temperature 37.1°C, heart rate 75/min, respiratory rate 16/min, blood "
     "pressure 102/87 mmHg, weight 24.2 kg, height 112 cm, BMI 19.3 kg/m², BMI-for-age 98.8th percentile.",
     "Vital signs at current visit: temperature 37.1°C. Vital signs 20 days earlier: heart rate 75/min, "
     "respiratory rate 16/min, blood pressure 102/87 mmHg, weight 24.2 kg, height 112 cm, BMI 19.3 kg/m², "
     "BMI-for-age 98.8th percentile."),
    ("dev", "viral_pharyngitis_e587f7a2-76b3-4bfa-ece2-8280be9f8245", "objective",
     "Vital signs at current visit: temperature 37.0°C, heart rate 86/min, respiratory rate 13/min, blood "
     "pressure 82/57 mmHg.",
     "Vital signs at current visit: temperature 37.0°C. Vital signs 15 days earlier: heart rate 86/min, "
     "respiratory rate 13/min, blood pressure 82/57 mmHg."),
    ("dev", "viral_pharyngitis_e58f3ccd-dbed-3a90-b685-d053db15b223", "objective",
     "Vital signs at this visit: body temperature 37.0°C, heart rate 91/min, respiratory rate 14/min, "
     "blood pressure 124/66 mmHg (from prior visit 11 months ago).",
     "Vital signs at this visit: body temperature 37.0°C. Vital signs 11 months earlier: heart rate "
     "91/min, respiratory rate 14/min, blood pressure 124/66 mmHg (from prior visit 11 months ago)."),
    ("dev", "viral_pharyngitis_e63d4130-9b35-5f2d-8c0e-1183a3e95513", "objective",
     "Vital signs at current visit: body temperature 37.5°C, heart rate 74/min, respiratory rate 14/min, "
     "blood pressure 110/75 mmHg.",
     "Vital signs at current visit: body temperature 37.5°C. Vital signs 14 days earlier: heart rate "
     "74/min, respiratory rate 14/min, blood pressure 110/75 mmHg."),
    ("dev", "viral_pharyngitis_e6c94fcc-2027-3ed2-2130-fa8c48e420ca", "objective",
     "Vital signs at current visit: body temperature 37.2°C, heart rate 90/min, respiratory rate 15/min, "
     "blood pressure 102/69 mmHg (measured 21 days prior).",
     "Vital signs at current visit: body temperature 37.2°C. Vital signs 21 days earlier: heart rate "
     "90/min, respiratory rate 15/min, blood pressure 102/69 mmHg (measured 21 days prior)."),
    ("dev", "viral_pharyngitis_eadee610-636c-3c9b-7a54-84e4a0ef3744", "objective",
     "Vital signs at this visit: temperature 37.5°C, heart rate 73/min, respiratory rate 14/min, blood "
     "pressure 114/76 mmHg.",
     "Vital signs at this visit: temperature 37.5°C. Vital signs 14 days earlier: heart rate 73/min, "
     "respiratory rate 14/min, blood pressure 114/76 mmHg."),
    ("dev", "viral_pharyngitis_ec990c7d-d8e7-ec8a-69a2-6812ad429563", "objective",
     "Vital signs at current visit: body temperature 37.1°C, heart rate 93/min, respiratory rate 14/min, "
     "blood pressure 114/83 mmHg, body weight 83.6 kg, height 172.4 cm, BMI 28.1 kg/m2.",
     "Vital signs at current visit: body temperature 37.1°C. Vital signs 14 days earlier: heart rate "
     "93/min, respiratory rate 14/min, blood pressure 114/83 mmHg, body weight 83.6 kg, height 172.4 cm, "
     "BMI 28.1 kg/m2."),
    ("dev", "viral_pharyngitis_ede73cfb-5795-1c75-ac80-fb1fd13eeaaf", "objective",
     "Labs (current visit): glucose 80 mg/dL, HbA1c 5.5%, calcium 9.9 mg/dL, sodium 137.3 mmol/L, "
     "potassium 3.8 mmol/L, chloride 103 mmol/L, CO2 27.3 mmol/L, creatinine 1.2 mg/dL, BUN 19.3 mg/dL, "
     "GFR 63.6 mL/min/1.73m², total cholesterol 187.3 mg/dL, LDL 108.4 mg/dL, HDL 56.9 mg/dL, "
     "triglycerides 110.5 mg/dL, microalbumin/creatinine ratio 14.8 mg/g, AUDIT-C score 0.",
     "Labs 15 days earlier: glucose 80 mg/dL, HbA1c 5.5%, calcium 9.9 mg/dL, sodium 137.3 mmol/L, "
     "potassium 3.8 mmol/L, chloride 103 mmol/L, CO2 27.3 mmol/L, creatinine 1.2 mg/dL, BUN 19.3 mg/dL, "
     "GFR 63.6 mL/min/1.73m², total cholesterol 187.3 mg/dL, LDL 108.4 mg/dL, HDL 56.9 mg/dL, "
     "triglycerides 110.5 mg/dL, microalbumin/creatinine ratio 14.8 mg/g, AUDIT-C score 0."),
    ("dev", "viral_pharyngitis_ee209904-5eea-d477-06f7-5a70c446b977", "objective",
     "Vital signs at current visit: body temperature 37.6°C, heart rate 80/min, respiratory rate 13/min, "
     "blood pressure 134/89 mmHg.",
     "Vital signs at current visit: body temperature 37.6°C. Vital signs 14 days earlier: heart rate "
     "80/min, respiratory rate 13/min, blood pressure 134/89 mmHg."),
    ("dev", "viral_pharyngitis_f028e4b6-bd20-c9b0-b1b0-769a9b0a5b33", "objective",
     "Vital signs at this visit: body temperature 37.1°C, heart rate 86/min, respiratory rate 15/min, "
     "blood pressure 129/70 mmHg.",
     "Vital signs at this visit: body temperature 37.1°C. Vital signs 10 months earlier: heart rate "
     "86/min, respiratory rate 15/min, blood pressure 129/70 mmHg."),
    ("dev", "viral_pharyngitis_f1ffd6e1-0583-a330-ce43-46619a1776e2", "objective",
     "Vital signs at current visit: temperature 37.7°C, blood pressure 94/69 mmHg, heart rate 71/min, "
     "respiratory rate 15/min.",
     "Vital signs at current visit: temperature 37.7°C. Vital signs 7 days earlier: blood pressure 94/69 "
     "mmHg, heart rate 71/min, respiratory rate 15/min."),
    ("dev", "viral_pharyngitis_f304a5fa-2b0a-157b-1b5e-6c7429b33d69", "objective",
     "Vital signs at current visit: temperature 37.2°C, heart rate 81/min, respiratory rate 13/min, "
     "systolic blood pressure 87 mmHg, diastolic blood pressure 76 mmHg.",
     "Vital signs at current visit: temperature 37.2°C. Vital signs 7 days earlier: heart rate 81/min, "
     "respiratory rate 13/min, systolic blood pressure 87 mmHg, diastolic blood pressure 76 mmHg."),
    ("dev", "viral_pharyngitis_f4467ccd-5993-2756-7696-8138be74733e", "objective",
     "Vital signs at current visit: body temperature 37.4°C, heart rate 94/min, respiratory rate 16/min, "
     "blood pressure 123/76 mmHg (from 1 month prior).",
     "Vital signs at current visit: body temperature 37.4°C. Vital signs 1 month earlier: heart rate "
     "94/min, respiratory rate 16/min, blood pressure 123/76 mmHg (from 1 month prior)."),
    ("dev", "viral_pharyngitis_f77609f0-70dc-aeea-5b47-b51a660e8196", "objective",
     "Vital signs at current visit: body temperature 37.6°C, heart rate 90/min, respiratory rate 14/min, "
     "blood pressure 111/82 mmHg.",
     "Vital signs at current visit: body temperature 37.6°C. Vital signs 14 days earlier: heart rate "
     "90/min, respiratory rate 14/min, blood pressure 111/82 mmHg."),
    ("dev", "viral_pharyngitis_f8242f61-cf3f-9aea-3b38-648370378e13", "objective",
     "Vital signs at current visit: body temperature 37.3°C, heart rate 79/min, respiratory rate 15/min, "
     "blood pressure 99/73 mmHg, pain severity score 2/10.",
     "Vital signs at current visit: body temperature 37.3°C. Vital signs 7 days earlier: heart rate "
     "79/min, respiratory rate 15/min, blood pressure 99/73 mmHg, pain severity score 2/10."),
    ("dev", "viral_pharyngitis_f8917d08-e945-4af8-233c-c9da843b662d", "objective",
     "Vital signs at current visit: body temperature 38.0°C, blood pressure 141/75 mmHg, heart rate "
     "89/min, respiratory rate 14/min.",
     "Vital signs at current visit: body temperature 38.0°C. Vital signs 14 days earlier: blood pressure "
     "141/75 mmHg, heart rate 89/min, respiratory rate 14/min."),
    ("dev", "viral_pharyngitis_faac6aaf-6e8d-152d-36af-7d9b6c6e1846", "objective",
     "Vital signs at current visit: body temperature 37.3°C, blood pressure 104/85 mmHg, heart rate "
     "70/min, respiratory rate 15/min.",
     "Vital signs at current visit: body temperature 37.3°C. Vital signs 14 days earlier: blood pressure "
     "104/85 mmHg, heart rate 70/min, respiratory rate 15/min."),
    ("dev", "viral_pharyngitis_fad4919d-30a4-ce22-7304-c3dc246ddfba", "objective",
     "Vital signs at current visit: body temperature 37.1°C, heart rate 69/min, respiratory rate 13/min, "
     "blood pressure 112/82 mmHg.",
     "Vital signs at current visit: body temperature 37.1°C. Vital signs 15 days earlier: heart rate "
     "69/min, respiratory rate 13/min, blood pressure 112/82 mmHg."),
    ("dev", "viral_pharyngitis_fb6300fb-4251-947e-0683-48bafdde92fd", "objective",
     "Vital signs at current visit: temperature 37.0°C, blood pressure 111/80 mmHg, heart rate 81/min, "
     "respiratory rate 14/min.",
     "Vital signs at current visit: temperature 37.0°C. Vital signs 14 days earlier: blood pressure 111/80 "
     "mmHg, heart rate 81/min, respiratory rate 14/min."),
    ("dev", "viral_pharyngitis_fbc9c15a-d3dd-0d30-7b03-a86762cd7404", "objective",
     "Vital signs at current visit: temperature 38.0°C, blood pressure 115/82 mmHg, heart rate 78/min, "
     "respiratory rate 15/min.",
     "Vital signs at current visit: temperature 38.0°C. Vital signs 14 days earlier: blood pressure 115/82 "
     "mmHg, heart rate 78/min, respiratory rate 15/min."),
    ("dev", "viral_pharyngitis_fe0c2fe9-339e-47f6-6e0e-196723f76209", "objective",
     "Vitals at current visit: temperature 37.1°C, heart rate 90/min, respiratory rate 15/min, blood "
     "pressure 133/94 mmHg, height 167.7 cm, weight 79.6 kg, BMI 28.3 kg/m².",
     "Vitals at current visit: temperature 37.1°C. Vitals 7 days earlier: heart rate 90/min, respiratory "
     "rate 15/min, blood pressure 133/94 mmHg, height 167.7 cm, weight 79.6 kg, BMI 28.3 kg/m²."),
    ("dev", "viral_pharyngitis_ff447a0b-0b6e-3f72-437c-b499e922ff1f", "objective",
     "Vital signs at current visit: temperature 37.7°C, heart rate 86/min, respiratory rate 14/min, blood "
     "pressure 120/88 mmHg.",
     "Vital signs at current visit: temperature 37.7°C. Vital signs 14 days earlier: heart rate 86/min, "
     "respiratory rate 14/min, blood pressure 120/88 mmHg."),
    ("dev", "cystitis_f07a0e6f-1d16-c9f3-6789-f36861e74e06", "subjective",
     "HARK score 0 (current visit)",
     "HARK score 0 (8 months earlier)"),
    ("dev", "cystitis_f07a0e6f-1d16-c9f3-6789-f36861e74e06", "subjective",
     "PHQ-2 total score 1 (current visit)",
     "PHQ-2 total score 1 (8 months earlier)"),
    ("corpus", "strep_throat_006068d8-da63-6ba2-0fc2-d611826259dd", "objective",
     "Vital signs at current visit: body temperature 39.0 °C, heart rate 64/min, respiratory rate 15/min, "
     "systolic blood pressure 124 mmHg, diastolic blood pressure 91 mmHg.",
     "Vital signs at current visit: body temperature 39.0 °C. Vital signs 14 days earlier: heart rate "
     "64/min, respiratory rate 15/min, systolic blood pressure 124 mmHg, diastolic blood pressure 91 mmHg."),
    ("corpus", "strep_throat_08495cf9-85f6-1f60-389c-6dde8870373f", "objective",
     "Vital signs at this visit: body temperature 38.5 °C, heart rate 61 /min, respiratory rate 14 /min, "
     "blood pressure 124/75 mm[Hg], pain severity 4/10.",
     "Vital signs at this visit: body temperature 38.5 °C. Vital signs 7 days earlier: heart rate 61 /min, "
     "respiratory rate 14 /min, blood pressure 124/75 mm[Hg], pain severity 4/10."),
    ("corpus", "strep_throat_08e897af-d631-d2a3-a4ed-734ba3079431", "objective",
     "Vital signs at this visit: body temperature 38.1°C, heart rate 86/min, respiratory rate 14/min, "
     "systolic blood pressure 112 mmHg, diastolic blood pressure 73 mmHg.",
     "Vital signs at this visit: body temperature 38.1°C. Vital signs 7 days earlier: heart rate 86/min, "
     "respiratory rate 14/min, systolic blood pressure 112 mmHg, diastolic blood pressure 73 mmHg."),
    ("corpus", "strep_throat_1954c107-604d-86cd-e8b9-844283294848", "objective",
     "Vital signs at this visit: body temperature 37.5°C, heart rate 65/min, respiratory rate 16/min, "
     "systolic blood pressure 136 mmHg, diastolic blood pressure 75 mmHg.",
     "Vital signs at this visit: body temperature 37.5°C. Vital signs 21 days earlier: heart rate 65/min, "
     "respiratory rate 16/min, systolic blood pressure 136 mmHg, diastolic blood pressure 75 mmHg."),
    ("corpus", "strep_throat_1afc86de-be87-26f6-ea2e-3c190e4c94e8", "objective",
     "Vital signs at this visit: body temperature 39.1°C, heart rate 68/min, respiratory rate 14/min, "
     "systolic blood pressure 139 mmHg, diastolic blood pressure 85 mmHg.",
     "Vital signs at this visit: body temperature 39.1°C. Vital signs 7 days earlier: heart rate 68/min, "
     "respiratory rate 14/min, systolic blood pressure 139 mmHg, diastolic blood pressure 85 mmHg."),
    ("corpus", "strep_throat_1e62e6b7-ea81-4a0c-5905-3d66c1539c27", "objective",
     "Vital signs at this visit: body temperature 38.7 °C, heart rate 92/min, respiratory rate 14/min, "
     "systolic blood pressure 103 mmHg, diastolic blood pressure 81 mmHg.",
     "Vital signs at this visit: body temperature 38.7 °C. Vital signs 7 days earlier: heart rate 92/min, "
     "respiratory rate 14/min, systolic blood pressure 103 mmHg, diastolic blood pressure 81 mmHg."),
    ("corpus", "strep_throat_22890e70-c995-17fc-ead6-287ce92cac37", "objective",
     "Vital signs at this visit: body temperature 38.2 °C, heart rate 97 /min, blood pressure 128/84 "
     "mm[Hg], respiratory rate 15 /min.",
     "Vital signs at this visit: body temperature 38.2 °C. Vital signs 15 days earlier: heart rate 97 "
     "/min, blood pressure 128/84 mm[Hg], respiratory rate 15 /min."),
    ("corpus", "strep_throat_2a688cc3-8bb0-d62f-bfd7-77509d11c62d", "objective",
     "Vitals at this visit: body temperature 37.8°C, heart rate 73/min, respiratory rate 13/min, blood "
     "pressure 93/65 mm[Hg], body weight 32 kg, body height 137.4 cm, BMI 17 kg/m2 (71.3th percentile for "
     "age and sex).",
     "Vitals at this visit: body temperature 37.8°C. Vitals 3 days earlier: heart rate 73/min, respiratory "
     "rate 13/min, blood pressure 93/65 mm[Hg], body weight 32 kg, body height 137.4 cm, BMI 17 kg/m2 "
     "(71.3th percentile for age and sex)."),
    ("corpus", "strep_throat_2dbc1828-98c4-7441-ec5c-0517d624880a", "objective",
     "Vital signs at this visit: body temperature 38.6°C, heart rate 96/min, respiratory rate 12/min, "
     "blood pressure 95/68 mmHg, body weight 61 kg, body height 159.7 cm, BMI 23.9 kg/m², BMI percentile "
     "92.5% per age and sex.",
     "Vital signs at this visit: body temperature 38.6°C. Vital signs 15 days earlier: heart rate 96/min, "
     "respiratory rate 12/min, blood pressure 95/68 mmHg, body weight 61 kg, body height 159.7 cm, BMI "
     "23.9 kg/m², BMI percentile 92.5% per age and sex."),
    ("corpus", "strep_throat_32a44228-9ec1-0b28-6423-364cac9d228d", "objective",
     "Vital signs at current visit: body temperature 38.4°C, heart rate 98/min, respiratory rate 15/min, "
     "blood pressure 132/75 mmHg, pain severity 4/10.",
     "Vital signs at current visit: body temperature 38.4°C. Vital signs 14 days earlier: heart rate "
     "98/min, respiratory rate 15/min, blood pressure 132/75 mmHg, pain severity 4/10."),
    ("corpus", "strep_throat_34103e3c-77be-bb59-d149-6ac08583a1c0", "objective",
     "Vital signs at this visit: body temperature 37.2°C, heart rate 93/min, respiratory rate 15/min, "
     "systolic blood pressure 94 mmHg, diastolic blood pressure 71 mmHg.",
     "Vital signs at this visit: body temperature 37.2°C. Vital signs 28 days earlier: heart rate 93/min, "
     "respiratory rate 15/min, systolic blood pressure 94 mmHg, diastolic blood pressure 71 mmHg."),
    ("corpus", "strep_throat_38c75fb3-2544-eaa5-4948-c560f4bccb2f", "objective",
     "Vital signs at this visit: body temperature 38.9°C, heart rate 87/min, respiratory rate 15/min, "
     "systolic blood pressure 130 mmHg, diastolic blood pressure 104 mmHg.",
     "Vital signs at this visit: body temperature 38.9°C. Vital signs 2 months earlier: heart rate 87/min, "
     "respiratory rate 15/min, systolic blood pressure 130 mmHg, diastolic blood pressure 104 mmHg."),
    ("corpus", "strep_throat_437ffed2-035d-6245-5a0c-91d9c45edcfd", "objective",
     "Vital signs at current visit: body temperature 39.3°C, heart rate 65/min, respiratory rate 13/min, "
     "blood pressure 98/77 mm[Hg], body weight 29.8 kg, body height 126 cm, BMI 18.8 kg/m2 (88.4th "
     "percentile for age and sex), pain severity 3/10.",
     "Vital signs at current visit: body temperature 39.3°C. Vital signs 3 days earlier: heart rate "
     "65/min, respiratory rate 13/min, blood pressure 98/77 mm[Hg], body weight 29.8 kg, body height 126 "
     "cm, BMI 18.8 kg/m2 (88.4th percentile for age and sex), pain severity 3/10."),
    ("corpus", "strep_throat_47cf273a-f3c5-e630-00b1-77aed350eb5f", "objective",
     "Vital signs at this visit: body temperature 38.5 °C, heart rate 97/min, respiratory rate 13/min, "
     "systolic blood pressure 149 mmHg, diastolic blood pressure 107 mmHg.",
     "Vital signs at this visit: body temperature 38.5 °C. Vital signs 14 days earlier: heart rate 97/min, "
     "respiratory rate 13/min, systolic blood pressure 149 mmHg, diastolic blood pressure 107 mmHg."),
    ("corpus", "strep_throat_48ff19ee-135e-f45e-a4f1-b4cd6715bae9", "objective",
     "Vital signs at this visit: body temperature 37.3 °C, heart rate 75/min, respiratory rate 15/min, "
     "blood pressure 101/67 mmHg, body weight 15.6 kg, body height 101.2 cm, BMI 15.2 kg/m², BMI "
     "percentile 51% for age and sex.",
     "Vital signs at this visit: body temperature 37.3 °C. Vital signs 20 days earlier: heart rate 75/min, "
     "respiratory rate 15/min, blood pressure 101/67 mmHg, body weight 15.6 kg, body height 101.2 cm, BMI "
     "15.2 kg/m², BMI percentile 51% for age and sex."),
    ("corpus", "strep_throat_4af07ad2-e809-6037-e2e8-1bc1150c9caa", "objective",
     "Vital signs at this visit: body temperature 38.5°C, heart rate 73/min, respiratory rate 15/min, "
     "blood pressure 116/83 mm[Hg].",
     "Vital signs at this visit: body temperature 38.5°C. Vital signs 14 days earlier: heart rate 73/min, "
     "respiratory rate 15/min, blood pressure 116/83 mm[Hg]."),
    ("corpus", "strep_throat_4c7b8f30-0c42-facb-ce69-9439d0cce8cd", "objective",
     "Vital signs at this visit: body temperature 39.2 °C, heart rate 94 /min, systolic blood pressure 119 "
     "mm[Hg], diastolic blood pressure 67 mm[Hg], respiratory rate 12 /min.",
     "Vital signs at this visit: body temperature 39.2 °C. Vital signs 14 days earlier: heart rate 94 "
     "/min, systolic blood pressure 119 mm[Hg], diastolic blood pressure 67 mm[Hg], respiratory rate 12 "
     "/min."),
    ("corpus", "strep_throat_5e882027-227d-3e0e-1b71-1ae317495f45", "objective",
     "Vital signs at this visit: body temperature 38.5°C, heart rate 100/min, respiratory rate 13/min, "
     "blood pressure 119/79 mmHg, body weight 31.1 kg, body height 136.3 cm, BMI 16.7 kg/m², BMI "
     "percentile 43.1% per age and sex, pain severity 4/10.",
     "Vital signs at this visit: body temperature 38.5°C. Vital signs 7 days earlier: heart rate 100/min, "
     "respiratory rate 13/min, blood pressure 119/79 mmHg, body weight 31.1 kg, body height 136.3 cm, BMI "
     "16.7 kg/m², BMI percentile 43.1% per age and sex, pain severity 4/10."),
    ("corpus", "strep_throat_638d74c8-4987-69c6-dd98-9caf2159bf9c", "objective",
     "Vital signs at this visit: body temperature 37.5°C, heart rate 98/min, respiratory rate 13/min, "
     "systolic blood pressure 132 mmHg, diastolic blood pressure 79 mmHg, BMI 29.1 kg/m².",
     "Vital signs at this visit: body temperature 37.5°C. Vital signs 14 days earlier: heart rate 98/min, "
     "respiratory rate 13/min, systolic blood pressure 132 mmHg, diastolic blood pressure 79 mmHg, BMI "
     "29.1 kg/m²."),
    ("corpus", "strep_throat_73ea9f5d-27f1-38f7-5d56-5ab85651df27", "objective",
     "Vital signs at this visit: body temperature 38.6°C, heart rate 93/min, respiratory rate 13/min, "
     "blood pressure 110/74 mmHg.",
     "Vital signs at this visit: body temperature 38.6°C. Vital signs 14 days earlier: heart rate 93/min, "
     "respiratory rate 13/min, blood pressure 110/74 mmHg."),
    ("corpus", "strep_throat_863b8eea-f111-d763-35c1-3c03ac3b3c44", "objective",
     "Vital signs at this visit: body temperature 39.1°C, heart rate 95/min, respiratory rate 14/min, "
     "blood pressure 108/71 mmHg.",
     "Vital signs at this visit: body temperature 39.1°C. Vital signs 14 days earlier: heart rate 95/min, "
     "respiratory rate 14/min, blood pressure 108/71 mmHg."),
    ("corpus", "strep_throat_88d88570-5917-1cde-b3ac-5529fcd54bf7", "objective",
     "Vital signs at this visit: body temperature 37.7 °C, heart rate 95 /min, blood pressure 116/61 "
     "mm[Hg], respiratory rate 13 /min.",
     "Vital signs at this visit: body temperature 37.7 °C. Vital signs 3 months earlier: heart rate 95 "
     "/min, blood pressure 116/61 mm[Hg], respiratory rate 13 /min."),
    ("corpus", "strep_throat_89672161-b8c4-74a3-9bf4-4805988d3bfe", "objective",
     "Vital signs at this visit: body temperature 37.2°C, heart rate 83/min, respiratory rate 13/min, "
     "systolic blood pressure 105 mmHg, diastolic blood pressure 77 mmHg.",
     "Vital signs at this visit: body temperature 37.2°C. Vital signs 3 days earlier: heart rate 83/min, "
     "respiratory rate 13/min, systolic blood pressure 105 mmHg, diastolic blood pressure 77 mmHg."),
    ("corpus", "strep_throat_8bc51b71-8144-2d29-bdfe-5a13fea25538", "objective",
     "Vital signs at this visit: body temperature 38.6°C, heart rate 83/min, respiratory rate 14/min, "
     "blood pressure 106/72 mmHg, body weight 32.3 kg, body height 142.8 cm, BMI 15.8 kg/m2 (22.3rd "
     "percentile for age and sex).",
     "Vital signs at this visit: body temperature 38.6°C. Vital signs 7 days earlier: heart rate 83/min, "
     "respiratory rate 14/min, blood pressure 106/72 mmHg, body weight 32.3 kg, body height 142.8 cm, BMI "
     "15.8 kg/m2 (22.3rd percentile for age and sex)."),
    ("corpus", "strep_throat_8faf9578-fe07-3d04-4e51-616cdd157045", "objective",
     "Vital signs at this visit: body temperature 39.0 Cel, heart rate 75 /min, respiratory rate 13 /min, "
     "systolic blood pressure 104 mm[Hg], diastolic blood pressure 71 mm[Hg].",
     "Vital signs at this visit: body temperature 39.0 Cel. Vital signs 7 days earlier: heart rate 75 "
     "/min, respiratory rate 13 /min, systolic blood pressure 104 mm[Hg], diastolic blood pressure 71 "
     "mm[Hg]."),
    ("corpus", "strep_throat_944d7287-b952-3bc3-1e3a-fdd63a570798", "objective",
     "Vital signs at this visit: body temperature 39.3°C, blood pressure 121/89 mmHg, heart rate 86/min, "
     "respiratory rate 13/min.",
     "Vital signs at this visit: body temperature 39.3°C. Vital signs 7 days earlier: blood pressure "
     "121/89 mmHg, heart rate 86/min, respiratory rate 13/min."),
    ("corpus", "strep_throat_98e0457f-5aa6-ec5f-a6d3-3759e7a42f47", "objective",
     "Vital signs at this visit: body temperature 38.9 °C, heart rate 97 /min, respiratory rate 14 /min, "
     "systolic blood pressure 132 mm[Hg], diastolic blood pressure 67 mm[Hg].",
     "Vital signs at this visit: body temperature 38.9 °C. Vital signs 14 days earlier: heart rate 97 "
     "/min, respiratory rate 14 /min, systolic blood pressure 132 mm[Hg], diastolic blood pressure 67 "
     "mm[Hg]."),
    ("corpus", "strep_throat_9af4b769-ea96-1637-2a8c-0e597b79bc63", "objective",
     "Vital signs at this visit: body temperature 37.5°C, heart rate 89/min, respiratory rate 13/min, "
     "blood pressure 112/72 mmHg, body weight 14.1 kg, body height 97.7 cm, BMI 14.7 kg/m², BMI percentile "
     "24.7%, pain severity score 1/10.",
     "Vital signs at this visit: body temperature 37.5°C. Vital signs 7 days earlier: heart rate 89/min, "
     "respiratory rate 13/min, blood pressure 112/72 mmHg, body weight 14.1 kg, body height 97.7 cm, BMI "
     "14.7 kg/m², BMI percentile 24.7%, pain severity score 1/10."),
    ("corpus", "strep_throat_9b9126c7-bfd1-53ef-760c-5c4241726e9c", "objective",
     "Vital signs at this visit: body temperature 37.7 °C, heart rate 60/min, respiratory rate 15/min, "
     "systolic blood pressure 117 mm[Hg], diastolic blood pressure 64 mm[Hg].",
     "Vital signs at this visit: body temperature 37.7 °C. Vital signs 1 month earlier: heart rate 60/min, "
     "respiratory rate 15/min, systolic blood pressure 117 mm[Hg], diastolic blood pressure 64 mm[Hg]."),
    ("corpus", "strep_throat_a1943e0e-0f1b-08e7-87ce-07f3502e8a57", "objective",
     "Vital signs at this visit: body temperature 38.2 Cel, heart rate 69 /min, respiratory rate 15 /min, "
     "systolic blood pressure 127 mm[Hg], diastolic blood pressure 79 mm[Hg].",
     "Vital signs at this visit: body temperature 38.2 Cel. Vital signs 7 days earlier: heart rate 69 "
     "/min, respiratory rate 15 /min, systolic blood pressure 127 mm[Hg], diastolic blood pressure 79 "
     "mm[Hg]."),
    ("corpus", "strep_throat_a4e35b91-26e5-fc8e-b181-22eb330df321", "objective",
     "Vital signs at this visit: body temperature 38.0 °C, heart rate 93/min, respiratory rate 16/min, "
     "systolic blood pressure 120 mmHg, diastolic blood pressure 87 mmHg, body weight 14.2 kg, body height "
     "96.5 cm, BMI 15.3 kg/m2 (29.6th percentile for age and sex), pain severity score 3/10.",
     "Vital signs at this visit: body temperature 38.0 °C. Vital signs 7 days earlier: heart rate 93/min, "
     "respiratory rate 16/min, systolic blood pressure 120 mmHg, diastolic blood pressure 87 mmHg, body "
     "weight 14.2 kg, body height 96.5 cm, BMI 15.3 kg/m2 (29.6th percentile for age and sex), pain "
     "severity score 3/10."),
    ("corpus", "strep_throat_a76ce079-7b16-ca0f-9548-83c5b9adb844", "objective",
     "Vital signs at this visit: body temperature 39.0 Cel, heart rate 94 /min, systolic blood pressure 93 "
     "mm[Hg], diastolic blood pressure 67 mm[Hg], respiratory rate 13 /min.",
     "Vital signs at this visit: body temperature 39.0 Cel. Vital signs 15 days earlier: heart rate 94 "
     "/min, systolic blood pressure 93 mm[Hg], diastolic blood pressure 67 mm[Hg], respiratory rate 13 "
     "/min."),
    ("corpus", "strep_throat_a92eadb8-3bff-d0a6-d540-0f9be72c637d", "objective",
     "Vital signs at this visit: body temperature 37.3 °C, heart rate 97 /min, blood pressure 136/89 mmHg, "
     "respiratory rate 12 /min.",
     "Vital signs at this visit: body temperature 37.3 °C. Vital signs 2 months earlier: heart rate 97 "
     "/min, blood pressure 136/89 mmHg, respiratory rate 12 /min."),
    ("corpus", "strep_throat_a99c8a6e-cb8e-4f37-f59b-ec5f9427870e", "objective",
     "Vital signs at this visit: body temperature 38.9°C, systolic blood pressure 113 mmHg, diastolic "
     "blood pressure 80 mmHg, heart rate 62/min, respiratory rate 13/min.",
     "Vital signs at this visit: body temperature 38.9°C. Vital signs 7 days earlier: systolic blood "
     "pressure 113 mmHg, diastolic blood pressure 80 mmHg, heart rate 62/min, respiratory rate 13/min."),
    ("corpus", "strep_throat_ae575a01-5a3f-89bd-6c8a-45217d0eb6b4", "objective",
     "Vital signs at this visit: body temperature 37.2 °C, heart rate 79/min, respiratory rate 13/min, "
     "systolic blood pressure 104 mmHg, diastolic blood pressure 62 mmHg.",
     "Vital signs at this visit: body temperature 37.2 °C. Vital signs 8 days earlier: heart rate 79/min, "
     "respiratory rate 13/min, systolic blood pressure 104 mmHg, diastolic blood pressure 62 mmHg."),
    ("corpus", "strep_throat_b2a8fdba-c2af-df32-33db-eaeacd7fe625", "objective",
     "Vital signs at this visit: body temperature 38.9°C, heart rate 85/min, respiratory rate 14/min, "
     "systolic blood pressure 107 mmHg, diastolic blood pressure 84 mmHg, body weight 72.8 kg, body height "
     "161 cm, BMI 28.1 kg/m².",
     "Vital signs at this visit: body temperature 38.9°C. Vital signs 15 days earlier: heart rate 85/min, "
     "respiratory rate 14/min, systolic blood pressure 107 mmHg, diastolic blood pressure 84 mmHg, body "
     "weight 72.8 kg, body height 161 cm, BMI 28.1 kg/m²."),
    ("corpus", "strep_throat_bb7e3731-e8ac-d4c7-30b1-0fa2349a5a86", "objective",
     "Vital signs at this visit: body temperature 38.2°C, blood pressure 110/81 mmHg, heart rate 74/min, "
     "respiratory rate 14/min.",
     "Vital signs at this visit: body temperature 38.2°C. Vital signs 15 days earlier: blood pressure "
     "110/81 mmHg, heart rate 74/min, respiratory rate 14/min."),
    ("corpus", "strep_throat_cfba649e-1c2c-7431-2a14-66358b106bc9", "objective",
     "Vital signs at this visit: body temperature 37.8 °C, heart rate 86 /min, blood pressure 128/72 "
     "mm[Hg], respiratory rate 14 /min.",
     "Vital signs at this visit: body temperature 37.8 °C. Vital signs 1 month earlier: heart rate 86 "
     "/min, blood pressure 128/72 mm[Hg], respiratory rate 14 /min."),
    ("corpus", "strep_throat_d5f29ddd-b2ba-0beb-378f-6bd46de42cd3", "objective",
     "Vital signs at current visit: body temperature 37.5°C, heart rate 92/min, respiratory rate 14/min, "
     "systolic blood pressure 115 mm[Hg], diastolic blood pressure 80 mm[Hg], body weight 13.5 kg, body "
     "height 84.7 cm, head circumference 47.8 cm (29.6th percentile), weight-for-length 92.8th percentile.",
     "Vital signs at current visit: body temperature 37.5°C. Vital signs 24 days earlier: heart rate "
     "92/min, respiratory rate 14/min, systolic blood pressure 115 mm[Hg], diastolic blood pressure 80 "
     "mm[Hg], body weight 13.5 kg, body height 84.7 cm, head circumference 47.8 cm (29.6th percentile), "
     "weight-for-length 92.8th percentile."),
    ("corpus", "strep_throat_d60dafd2-3ce9-d8b2-3f74-1897929098c1", "objective",
     "Vital signs at this visit: body temperature 37.7°C, heart rate 63/min, respiratory rate 14/min, "
     "systolic blood pressure 109 mmHg, diastolic blood pressure 73 mmHg.",
     "Vital signs at this visit: body temperature 37.7°C. Vital signs 14 days earlier: heart rate 63/min, "
     "respiratory rate 14/min, systolic blood pressure 109 mmHg, diastolic blood pressure 73 mmHg."),
    ("corpus", "strep_throat_e17605d6-bc4b-09aa-b664-222f8fe1da61", "objective",
     "Vital signs at this visit: body temperature 38.1 °C, heart rate 76/min, respiratory rate 15/min, "
     "blood pressure 92/66 mm[Hg], pain severity 2/10.",
     "Vital signs at this visit: body temperature 38.1 °C. Vital signs 7 days earlier: heart rate 76/min, "
     "respiratory rate 15/min, blood pressure 92/66 mm[Hg], pain severity 2/10."),
    ("corpus", "strep_throat_e71b08e2-814d-5a40-9993-4887e3c34294", "objective",
     "Vital signs at this visit: body temperature 39.4°C, heart rate 91/min, systolic blood pressure 113 "
     "mmHg, diastolic blood pressure 65 mmHg, respiratory rate 13/min.",
     "Vital signs at this visit: body temperature 39.4°C. Vital signs 2 days earlier: heart rate 91/min, "
     "systolic blood pressure 113 mmHg, diastolic blood pressure 65 mmHg, respiratory rate 13/min."),
    ("corpus", "strep_throat_f635cd26-1074-4350-64fc-0c466c3b98b4", "objective",
     "Vital signs at this visit: body temperature 37.3 °C, heart rate 79/min, respiratory rate 15/min, "
     "systolic blood pressure 96 mmHg, diastolic blood pressure 81 mmHg.",
     "Vital signs at this visit: body temperature 37.3 °C. Vital signs 1 month earlier: heart rate 79/min, "
     "respiratory rate 15/min, systolic blood pressure 96 mmHg, diastolic blood pressure 81 mmHg."),
    ("corpus", "strep_throat_fc6769c1-7d84-90d9-aec0-5fd05ab908b3", "objective",
     "Vital signs at this visit: body temperature 38.9°C, heart rate 66/min, respiratory rate 14/min, "
     "systolic blood pressure 129 mm[Hg], diastolic blood pressure 82 mm[Hg].",
     "Vital signs at this visit: body temperature 38.9°C. Vital signs 7 days earlier: heart rate 66/min, "
     "respiratory rate 14/min, systolic blood pressure 129 mm[Hg], diastolic blood pressure 82 mm[Hg]."),
    ("corpus", "viral_pharyngitis_00c69f4a-12c2-9e3e-cf09-b2186b0bdab3", "objective",
     "Vital signs at this visit: body temperature 37.9°C, heart rate 99/min, respiratory rate 15/min, "
     "systolic blood pressure 97 mmHg, diastolic blood pressure 55 mmHg.",
     "Vital signs at this visit: body temperature 37.9°C. Vital signs 21 days earlier: heart rate 99/min, "
     "respiratory rate 15/min, systolic blood pressure 97 mmHg, diastolic blood pressure 55 mmHg."),
    ("corpus", "viral_pharyngitis_01091d90-7dea-6a1f-f99f-96aee288b930", "objective",
     "Vital signs at this visit: body temperature 37.0 °C, heart rate 87 /min, respiratory rate 14 /min, "
     "systolic blood pressure 92 mm[Hg], diastolic blood pressure 73 mm[Hg].",
     "Vital signs at this visit: body temperature 37.0 °C. Vital signs 3 days earlier: heart rate 87 /min, "
     "respiratory rate 14 /min, systolic blood pressure 92 mm[Hg], diastolic blood pressure 73 mm[Hg]."),
    ("corpus", "viral_pharyngitis_02e389d8-d844-aebe-7bd9-afcd0dd8a16c", "objective",
     "Vital signs at this visit: body temperature 37.3°C, heart rate 64/min, respiratory rate 16/min, "
     "blood pressure 115/74 mmHg, BMI 27.8 kg/m².",
     "Vital signs at this visit: body temperature 37.3°C. Vital signs 14 days earlier: heart rate 64/min, "
     "respiratory rate 16/min, blood pressure 115/74 mmHg, BMI 27.8 kg/m²."),
    ("corpus", "viral_pharyngitis_07af40aa-5823-659d-51b2-3c638c724f4e", "objective",
     "Vital signs at this visit: body temperature 37.1°C, heart rate 93/min, respiratory rate 14/min, "
     "systolic blood pressure 114 mmHg, diastolic blood pressure 91 mmHg.",
     "Vital signs at this visit: body temperature 37.1°C. Vital signs 14 days earlier: heart rate 93/min, "
     "respiratory rate 14/min, systolic blood pressure 114 mmHg, diastolic blood pressure 91 mmHg."),
    ("corpus", "viral_pharyngitis_09afb9e3-0677-3562-a856-41b386630e7a", "objective",
     "Vital signs at this visit: body temperature 37.7 Cel, heart rate 68 /min, respiratory rate 14 /min, "
     "blood pressure 114/86 mm[Hg].",
     "Vital signs at this visit: body temperature 37.7 Cel. Vital signs 14 days earlier: heart rate 68 "
     "/min, respiratory rate 14 /min, blood pressure 114/86 mm[Hg]."),
    ("corpus", "viral_pharyngitis_09afb9e3-0677-3562-a856-41b386630e7a", "objective",
     "Pain severity score 0 (current visit)",
     "Pain severity score 0 (14 days earlier)"),
    ("corpus", "viral_pharyngitis_0be48113-d501-a5e6-7acc-05deb219c795", "objective",
     "Vital signs at this visit: body temperature 37.2 °C, heart rate 74 /min, respiratory rate 13 /min, "
     "systolic blood pressure 105 mm[Hg], diastolic blood pressure 98 mm[Hg].",
     "Vital signs at this visit: body temperature 37.2 °C. Vital signs 3 months earlier: heart rate 74 "
     "/min, respiratory rate 13 /min, systolic blood pressure 105 mm[Hg], diastolic blood pressure 98 "
     "mm[Hg]."),
    ("corpus", "viral_pharyngitis_0c7ffc9c-d9e2-7611-fbdd-eda866307431", "objective",
     "Vital signs at this visit: body temperature 37.7 °C, blood pressure 123/75 mmHg, heart rate 89/min, "
     "respiratory rate 16/min.",
     "Vital signs at this visit: body temperature 37.7 °C. Vital signs 14 days earlier: blood pressure "
     "123/75 mmHg, heart rate 89/min, respiratory rate 16/min."),
    ("corpus", "viral_pharyngitis_0d90e0ce-dba9-de12-cf2d-12e7efd7edd1", "objective",
     "Vital signs at this visit: body temperature 37.3 °C, heart rate 78 /min, respiratory rate 15 /min, "
     "blood pressure 115/82 mm[Hg].",
     "Vital signs at this visit: body temperature 37.3 °C. Vital signs 14 days earlier: heart rate 78 "
     "/min, respiratory rate 15 /min, blood pressure 115/82 mm[Hg]."),
    ("corpus", "viral_pharyngitis_0f3bbe0d-76ff-7fca-7ff0-ad533bb1c702", "objective",
     "Vital signs at this visit: body temperature 37.2 °C, heart rate 63/min, respiratory rate 16/min, "
     "systolic blood pressure 108 mmHg, diastolic blood pressure 95 mmHg.",
     "Vital signs at this visit: body temperature 37.2 °C. Vital signs 7 days earlier: heart rate 63/min, "
     "respiratory rate 16/min, systolic blood pressure 108 mmHg, diastolic blood pressure 95 mmHg."),
    ("corpus", "viral_pharyngitis_0fdad854-e9a3-abf1-c207-d46d32aea93f", "objective",
     "Vital signs at this visit: body temperature 37.8 °C, heart rate 92 /min, respiratory rate 15 /min, "
     "systolic blood pressure 137 mm[Hg], diastolic blood pressure 73 mm[Hg].",
     "Vital signs at this visit: body temperature 37.8 °C. Vital signs 14 days earlier: heart rate 92 "
     "/min, respiratory rate 15 /min, systolic blood pressure 137 mm[Hg], diastolic blood pressure 73 "
     "mm[Hg]."),
    ("corpus", "viral_pharyngitis_122aad1d-d970-e249-40de-3cd3519089fc", "objective",
     "Vital signs at this visit: body temperature 37.9°C, heart rate 67/min, respiratory rate 14/min, "
     "blood pressure 111/74 mmHg, body weight 60.4 kg, height 162.2 cm, BMI 23 kg/m2.",
     "Vital signs at this visit: body temperature 37.9°C. Vital signs 14 days earlier: heart rate 67/min, "
     "respiratory rate 14/min, blood pressure 111/74 mmHg, body weight 60.4 kg, height 162.2 cm, BMI 23 "
     "kg/m2."),
    ("corpus", "viral_pharyngitis_13180cf2-4f28-4143-ec9d-bf9abc40da6c", "objective",
     "Vitals at current visit: body temperature 37.1°C, heart rate 68/min, respiratory rate 14/min, blood "
     "pressure 95/75 mmHg, body weight 26.3 kg, body height 125.8 cm, BMI 16.7 kg/m2 (74.3th percentile "
     "for age and sex), pain severity 0/10.",
     "Vitals at current visit: body temperature 37.1°C. Vitals 9 days earlier: heart rate 68/min, "
     "respiratory rate 14/min, blood pressure 95/75 mmHg, body weight 26.3 kg, body height 125.8 cm, BMI "
     "16.7 kg/m2 (74.3th percentile for age and sex), pain severity 0/10."),
    ("corpus", "viral_pharyngitis_14ebd9fe-e9e5-e0b6-a62c-60a63e89d017", "objective",
     "Vital signs at this visit: body temperature 37.1°C, heart rate 80/min, respiratory rate 15/min, "
     "blood pressure 116/86 mm[Hg], body weight 18 kg, body height 109.1 cm, BMI 15.1 kg/m2 (48.2nd "
     "percentile for age and sex).",
     "Vital signs at this visit: body temperature 37.1°C. Vital signs 20 days earlier: heart rate 80/min, "
     "respiratory rate 15/min, blood pressure 116/86 mm[Hg], body weight 18 kg, body height 109.1 cm, BMI "
     "15.1 kg/m2 (48.2nd percentile for age and sex)."),
    ("corpus", "viral_pharyngitis_17bfe795-074e-d655-7e52-7956c883c55b", "objective",
     "Vital signs at this visit: body temperature 38.0°C, heart rate 76/min, respiratory rate 15/min, "
     "systolic blood pressure 134 mm[Hg], diastolic blood pressure 83 mm[Hg].",
     "Vital signs at this visit: body temperature 38.0°C. Vital signs 9 days earlier: heart rate 76/min, "
     "respiratory rate 15/min, systolic blood pressure 134 mm[Hg], diastolic blood pressure 83 mm[Hg]."),
    ("corpus", "viral_pharyngitis_1b4d3dfa-becd-4b26-657d-c70f82c69d89", "objective",
     "Vital signs at this visit: body temperature 37.3 Cel, heart rate 95 /min, respiratory rate 13 /min, "
     "blood pressure 129/87 mm[Hg] (from 3 months earlier).",
     "Vital signs at this visit: body temperature 37.3 Cel. Vital signs 3 months earlier: heart rate 95 "
     "/min, respiratory rate 13 /min, blood pressure 129/87 mm[Hg] (from 3 months earlier)."),
    ("corpus", "viral_pharyngitis_1b60b4d5-900a-c084-e285-ffecc2a11e85", "objective",
     "Vital signs at this visit: body temperature 37.5 °C, heart rate 100 /min, blood pressure 124/77 "
     "mm[Hg], respiratory rate 13 /min.",
     "Vital signs at this visit: body temperature 37.5 °C. Vital signs 14 days earlier: heart rate 100 "
     "/min, blood pressure 124/77 mm[Hg], respiratory rate 13 /min."),
    ("corpus", "viral_pharyngitis_1b718ebf-39f4-3515-e346-c9fe932a73b4", "objective",
     "Vital signs at this visit: body temperature 37.7 °C, heart rate 94 /min, respiratory rate 12 /min, "
     "blood pressure 114/79 mmHg.",
     "Vital signs at this visit: body temperature 37.7 °C. Vital signs 14 days earlier: heart rate 94 "
     "/min, respiratory rate 12 /min, blood pressure 114/79 mmHg."),
    ("corpus", "viral_pharyngitis_1e2c77ff-69d6-340e-5b7a-765a52e414bd", "objective",
     "Vital signs at this visit: body temperature 37.0 °C, heart rate 94/min, respiratory rate 12/min, "
     "systolic blood pressure 134 mmHg, diastolic blood pressure 71 mmHg.",
     "Vital signs at this visit: body temperature 37.0 °C. Vital signs 14 days earlier: heart rate 94/min, "
     "respiratory rate 12/min, systolic blood pressure 134 mmHg, diastolic blood pressure 71 mmHg."),
    ("corpus", "viral_pharyngitis_1e35965c-3aaf-44b3-5bb2-10ef4d2f5fb9", "objective",
     "Vital signs at this visit: body temperature 38.0°C, heart rate 97/min, respiratory rate 13/min, "
     "systolic blood pressure 115 mmHg, diastolic blood pressure 75 mmHg.",
     "Vital signs at this visit: body temperature 38.0°C. Vital signs 7 days earlier: heart rate 97/min, "
     "respiratory rate 13/min, systolic blood pressure 115 mmHg, diastolic blood pressure 75 mmHg."),
    ("corpus", "viral_pharyngitis_2086fea9-96eb-5017-5566-b4541c42324f", "objective",
     "Vital signs at this visit: body temperature 37.1°C, heart rate 75/min, respiratory rate 13/min, "
     "blood pressure 143/88 mmHg.",
     "Vital signs at this visit: body temperature 37.1°C. Vital signs 14 days earlier: heart rate 75/min, "
     "respiratory rate 13/min, blood pressure 143/88 mmHg."),
    ("corpus", "viral_pharyngitis_22f1e5ad-126f-bb64-cfe5-13fcfbdbbc96", "objective",
     "Vital signs at this visit: body temperature 37.6 °C, heart rate 95 /min, respiratory rate 15 /min, "
     "blood pressure 127/78 mm[Hg], body weight 72.8 kg, height 170.6 cm, BMI 25 kg/m2.",
     "Vital signs at this visit: body temperature 37.6 °C. Vital signs 14 days earlier: heart rate 95 "
     "/min, respiratory rate 15 /min, blood pressure 127/78 mm[Hg], body weight 72.8 kg, height 170.6 cm, "
     "BMI 25 kg/m2."),
    ("corpus", "viral_pharyngitis_23455487-a46b-63e4-0c4b-aeb138ae484a", "objective",
     "Vital signs at this visit: body temperature 37.4 °C, heart rate 75/min, respiratory rate 15/min, "
     "systolic blood pressure 143 mmHg, diastolic blood pressure 111 mmHg.",
     "Vital signs at this visit: body temperature 37.4 °C. Vital signs 29 days earlier: heart rate 75/min, "
     "respiratory rate 15/min, systolic blood pressure 143 mmHg, diastolic blood pressure 111 mmHg."),
    ("corpus", "viral_pharyngitis_2346ff58-74c9-ea54-a4e7-f2ab267e5960", "objective",
     "Vital signs at this visit: body temperature 37.5 °C, blood pressure 121/72 mmHg, heart rate 69/min, "
     "respiratory rate 14/min.",
     "Vital signs at this visit: body temperature 37.5 °C. Vital signs 15 days earlier: blood pressure "
     "121/72 mmHg, heart rate 69/min, respiratory rate 14/min."),
    ("corpus", "viral_pharyngitis_25473ecb-3832-d7d4-0a61-c3256f359b48", "objective",
     "Vital signs at this visit: body temperature 37.1 °C, heart rate 66/min, respiratory rate 14/min, "
     "systolic blood pressure 126 mmHg, diastolic blood pressure 105 mmHg.",
     "Vital signs at this visit: body temperature 37.1 °C. Vital signs 14 days earlier: heart rate 66/min, "
     "respiratory rate 14/min, systolic blood pressure 126 mmHg, diastolic blood pressure 105 mmHg."),
    ("corpus", "viral_pharyngitis_2559ea51-8205-a710-2a1b-b3e2b0f88315", "objective",
     "Vital signs at this visit: body temperature 37.4°C, heart rate 86/min, respiratory rate 16/min, "
     "systolic blood pressure 97 mmHg, diastolic blood pressure 91 mmHg.",
     "Vital signs at this visit: body temperature 37.4°C. Vital signs 15 days earlier: heart rate 86/min, "
     "respiratory rate 16/min, systolic blood pressure 97 mmHg, diastolic blood pressure 91 mmHg."),
    ("corpus", "viral_pharyngitis_268efd1b-4950-47be-75c3-9064cf3166a1", "objective",
     "Vital signs at this visit: body temperature 37.8°C, heart rate 63/min, respiratory rate 15/min, "
     "blood pressure 130/76 mmHg.",
     "Vital signs at this visit: body temperature 37.8°C. Vital signs 7 days earlier: heart rate 63/min, "
     "respiratory rate 15/min, blood pressure 130/76 mmHg."),
    ("corpus", "viral_pharyngitis_275d5c69-f012-f2bd-5c1c-58e8469506ed", "objective",
     "Vitals at this visit: body temperature 37.6°C, heart rate 74/min, respiratory rate 12/min, blood "
     "pressure 111/88 mmHg, body weight 76.2 kg, height 178.8 cm, BMI 23.8 kg/m2.",
     "Vitals at this visit: body temperature 37.6°C. Vitals 17 days earlier: heart rate 74/min, "
     "respiratory rate 12/min, blood pressure 111/88 mmHg, body weight 76.2 kg, height 178.8 cm, BMI 23.8 "
     "kg/m2."),
    ("corpus", "viral_pharyngitis_29b30e9d-45a8-6411-8e7f-ccbd31449724", "objective",
     "Vital signs at this visit: body temperature 37.4°C, heart rate 96/min, respiratory rate 13/min, "
     "systolic blood pressure 136 mm[Hg], diastolic blood pressure 69 mm[Hg].",
     "Vital signs at this visit: body temperature 37.4°C. Vital signs 8 days earlier: heart rate 96/min, "
     "respiratory rate 13/min, systolic blood pressure 136 mm[Hg], diastolic blood pressure 69 mm[Hg]."),
    ("corpus", "viral_pharyngitis_2a1014ee-5def-1e1e-d2b7-7f1458a258aa", "objective",
     "Vital signs at this visit: body temperature 38.0 °C, blood pressure 119/93 mmHg, heart rate 65/min, "
     "respiratory rate 14/min.",
     "Vital signs at this visit: body temperature 38.0 °C. Vital signs 14 days earlier: blood pressure "
     "119/93 mmHg, heart rate 65/min, respiratory rate 14/min."),
    ("corpus", "viral_pharyngitis_2d48aa65-0dfe-c565-159d-16d061581692", "objective",
     "Vital signs at this visit: body temperature 37.6 °C, heart rate 88/min, respiratory rate 13/min, "
     "blood pressure 123/69 mmHg.",
     "Vital signs at this visit: body temperature 37.6 °C. Vital signs 14 days earlier: heart rate 88/min, "
     "respiratory rate 13/min, blood pressure 123/69 mmHg."),
    ("corpus", "viral_pharyngitis_2f753f9e-05d9-5656-72ec-cb7b7cff9341", "objective",
     "Vital signs at current visit: body temperature 37.9°C, heart rate 94/min, respiratory rate 14/min, "
     "blood pressure 121/75 mmHg, body height 165.1 cm, body weight 55.5 kg, BMI 20.4 kg/m2 (61.2nd "
     "percentile for age and sex).",
     "Vital signs at current visit: body temperature 37.9°C. Vital signs 15 days earlier: heart rate "
     "94/min, respiratory rate 14/min, blood pressure 121/75 mmHg, body height 165.1 cm, body weight 55.5 "
     "kg, BMI 20.4 kg/m2 (61.2nd percentile for age and sex)."),
    ("corpus", "viral_pharyngitis_30e883eb-7502-5590-cfb1-f34137f31c28", "objective",
     "Vital signs at current visit: body temperature 37.2°C, heart rate 75/min, respiratory rate 14/min, "
     "blood pressure 104/72 mmHg, body weight 21.9 kg, body height 125.3 cm, BMI 13.9 kg/m2 (10th "
     "percentile for age and sex).",
     "Vital signs at current visit: body temperature 37.2°C. Vital signs 3 days earlier: heart rate "
     "75/min, respiratory rate 14/min, blood pressure 104/72 mmHg, body weight 21.9 kg, body height 125.3 "
     "cm, BMI 13.9 kg/m2 (10th percentile for age and sex)."),
    ("corpus", "viral_pharyngitis_3108a353-100b-c0f9-362e-55e869b9c3d9", "objective",
     "Vital signs at this visit: body temperature 37.1°C, heart rate 85/min, respiratory rate 13/min, "
     "systolic blood pressure 95 mmHg, diastolic blood pressure 63 mmHg.",
     "Vital signs at this visit: body temperature 37.1°C. Vital signs 15 days earlier: heart rate 85/min, "
     "respiratory rate 13/min, systolic blood pressure 95 mmHg, diastolic blood pressure 63 mmHg."),
    ("corpus", "viral_pharyngitis_31532f55-2dbe-dc37-ab9d-1f1138995fb9", "objective",
     "Vital signs at this visit: body temperature 37.7 °C, heart rate 80/min, respiratory rate 14/min, "
     "blood pressure 116/73 mm[Hg], body weight 88.3 kg, height 181.2 cm, BMI 26.9 kg/m2.",
     "Vital signs at this visit: body temperature 37.7 °C. Vital signs 14 days earlier: heart rate 80/min, "
     "respiratory rate 14/min, blood pressure 116/73 mm[Hg], body weight 88.3 kg, height 181.2 cm, BMI "
     "26.9 kg/m2."),
    ("corpus", "viral_pharyngitis_3236adf0-461c-ee73-2667-ea7b83a52cfc", "objective",
     "Vital signs at this visit: body temperature 37.5°C, heart rate 61/min, respiratory rate 14/min, "
     "blood pressure 124/72 mmHg, body weight 16.4 kg, body height 99.7 cm, BMI 16.5 kg/m², BMI percentile "
     "79.9% per age and sex, pain severity 1/10.",
     "Vital signs at this visit: body temperature 37.5°C. Vital signs 26 days earlier: heart rate 61/min, "
     "respiratory rate 14/min, blood pressure 124/72 mmHg, body weight 16.4 kg, body height 99.7 cm, BMI "
     "16.5 kg/m², BMI percentile 79.9% per age and sex, pain severity 1/10."),
    ("corpus", "viral_pharyngitis_34676331-9446-ce74-5fa5-5b6bbcb65280", "objective",
     "Vital signs at this visit: body temperature 37.2°C, blood pressure 134/78 mmHg, heart rate 65/min, "
     "respiratory rate 14/min.",
     "Vital signs at this visit: body temperature 37.2°C. Vital signs 7 days earlier: blood pressure "
     "134/78 mmHg, heart rate 65/min, respiratory rate 14/min."),
    ("corpus", "viral_pharyngitis_3a90c73e-1ccb-c706-b18e-e1b15d1235ef", "objective",
     "Vital signs at this visit: body temperature 37.4 °C, heart rate 96/min, respiratory rate 14/min, "
     "systolic blood pressure 128 mmHg, diastolic blood pressure 87 mmHg.",
     "Vital signs at this visit: body temperature 37.4 °C. Vital signs 7 days earlier: heart rate 96/min, "
     "respiratory rate 14/min, systolic blood pressure 128 mmHg, diastolic blood pressure 87 mmHg."),
    ("corpus", "viral_pharyngitis_3aa5ff17-1f99-be89-918a-338950a33b72", "objective",
     "Vital signs at this visit: body temperature 37.6°C, heart rate 65/min, respiratory rate 12/min, "
     "blood pressure 123/76 mmHg.",
     "Vital signs at this visit: body temperature 37.6°C. Vital signs 3 days earlier: heart rate 65/min, "
     "respiratory rate 12/min, blood pressure 123/76 mmHg."),
    ("corpus", "viral_pharyngitis_3b384833-3cc5-43ca-96e3-a3ed152f6efb", "objective",
     "Vital signs at this visit: body temperature 37.6 °C, heart rate 90 /min, blood pressure 128/76 "
     "mm[Hg], respiratory rate 13 /min.",
     "Vital signs at this visit: body temperature 37.6 °C. Vital signs 15 days earlier: heart rate 90 "
     "/min, blood pressure 128/76 mm[Hg], respiratory rate 13 /min."),
    ("corpus", "viral_pharyngitis_3fddf3ae-70a0-d803-826b-8d9b8da79144", "objective",
     "Vital signs at this visit: body temperature 37.4°C, heart rate 97/min, respiratory rate 15/min, "
     "systolic blood pressure 127 mmHg, diastolic blood pressure 78 mmHg, body height 113.8 cm, body "
     "weight 24.3 kg, BMI 18.8 kg/m², BMI percentile for age and sex 94.1%.",
     "Vital signs at this visit: body temperature 37.4°C. Vital signs 7 days earlier: heart rate 97/min, "
     "respiratory rate 15/min, systolic blood pressure 127 mmHg, diastolic blood pressure 78 mmHg, body "
     "height 113.8 cm, body weight 24.3 kg, BMI 18.8 kg/m², BMI percentile for age and sex 94.1%."),
    ("corpus", "viral_pharyngitis_406961f3-adcc-8888-121d-3c140a6c474a", "objective",
     "Vital signs at this visit: body temperature 37.7°C, heart rate 85/min, respiratory rate 13/min, "
     "blood pressure 116/85 mmHg.",
     "Vital signs at this visit: body temperature 37.7°C. Vital signs 9 months earlier: heart rate 85/min, "
     "respiratory rate 13/min, blood pressure 116/85 mmHg."),
    ("corpus", "viral_pharyngitis_408d920c-3128-47e1-3259-b6ce6f302867", "objective",
     "Vital signs at this visit: body temperature 37.2 °C, heart rate 82/min, respiratory rate 15/min, "
     "systolic blood pressure 152 mmHg, diastolic blood pressure 94 mmHg.",
     "Vital signs at this visit: body temperature 37.2 °C. Vital signs 7 days earlier: heart rate 82/min, "
     "respiratory rate 15/min, systolic blood pressure 152 mmHg, diastolic blood pressure 94 mmHg."),
    ("corpus", "viral_pharyngitis_4124f6eb-110d-edac-9ddc-4379d7d4f7ba", "objective",
     "Vital signs at this visit: body temperature 37.7°C, heart rate 92/min, respiratory rate 13/min, "
     "systolic blood pressure 125 mm[Hg], diastolic blood pressure 107 mm[Hg].",
     "Vital signs at this visit: body temperature 37.7°C. Vital signs 8 days earlier: heart rate 92/min, "
     "respiratory rate 13/min, systolic blood pressure 125 mm[Hg], diastolic blood pressure 107 mm[Hg]."),
    ("corpus", "viral_pharyngitis_454e2bb8-d658-b9f8-7a81-20daf33a42aa", "objective",
     "Vital signs at this visit: body temperature 37.3°C, heart rate 92/min, respiratory rate 15/min, "
     "blood pressure 118/75 mm[Hg], body weight 17.2 kg, body height 105.7 cm, BMI 15.4 kg/m2 (56.7th "
     "percentile for age and sex).",
     "Vital signs at this visit: body temperature 37.3°C. Vital signs 21 days earlier: heart rate 92/min, "
     "respiratory rate 15/min, blood pressure 118/75 mm[Hg], body weight 17.2 kg, body height 105.7 cm, "
     "BMI 15.4 kg/m2 (56.7th percentile for age and sex)."),
    ("corpus", "viral_pharyngitis_462438db-abb6-5206-8de4-ed825e40923e", "objective",
     "Vital signs at this visit: body temperature 37.1°C, blood pressure 122/93 mmHg (3 months earlier), "
     "heart rate 74/min, respiratory rate 14/min.",
     "Vital signs at this visit: body temperature 37.1°C, blood pressure 122/93 mmHg (3 months earlier). "
     "Vital signs 3 months earlier: heart rate 74/min, respiratory rate 14/min."),
    ("corpus", "viral_pharyngitis_48ff19ee-135e-f45e-a4f1-b4cd6715bae9", "objective",
     "Vital signs at this visit: body temperature 37.9°C, heart rate 85/min, respiratory rate 13/min, "
     "systolic blood pressure 102 mmHg, diastolic blood pressure 64 mmHg.",
     "Vital signs at this visit: body temperature 37.9°C. Vital signs 9 days earlier: heart rate 85/min, "
     "respiratory rate 13/min, systolic blood pressure 102 mmHg, diastolic blood pressure 64 mmHg."),
    ("corpus", "viral_pharyngitis_4984f5b9-6c92-c236-16b2-a759dcf40d79", "objective",
     "Vital signs at this visit: body temperature 37.8 °C, heart rate 84/min, respiratory rate 15/min, "
     "systolic blood pressure 117 mmHg, diastolic blood pressure 74 mmHg.",
     "Vital signs at this visit: body temperature 37.8 °C. Vital signs 7 days earlier: heart rate 84/min, "
     "respiratory rate 15/min, systolic blood pressure 117 mmHg, diastolic blood pressure 74 mmHg."),
    ("corpus", "viral_pharyngitis_49ab405f-8701-db16-3791-1bab9a77dd27", "objective",
     "Vital signs at this visit: body temperature 37.8°C, heart rate 65/min, respiratory rate 14/min, "
     "blood pressure 134/79 mmHg.",
     "Vital signs at this visit: body temperature 37.8°C. Vital signs 14 days earlier: heart rate 65/min, "
     "respiratory rate 14/min, blood pressure 134/79 mmHg."),
    ("corpus", "viral_pharyngitis_4a79d024-fc64-4b1a-168e-d1055e922d04", "objective",
     "Vital signs at this visit: body temperature 37.1°C, heart rate 68/min, respiratory rate 14/min, "
     "systolic blood pressure 122 mmHg, diastolic blood pressure 80 mmHg.",
     "Vital signs at this visit: body temperature 37.1°C. Vital signs 7 days earlier: heart rate 68/min, "
     "respiratory rate 14/min, systolic blood pressure 122 mmHg, diastolic blood pressure 80 mmHg."),
    ("corpus", "viral_pharyngitis_4b4e4da4-2846-b8b0-9e99-dab9e6eb45ea", "objective",
     "Vital signs at this visit: body temperature 37.7 °C, heart rate 92/min, respiratory rate 13/min, "
     "systolic blood pressure 127 mmHg, diastolic blood pressure 71 mmHg.",
     "Vital signs at this visit: body temperature 37.7 °C. Vital signs 14 days earlier: heart rate 92/min, "
     "respiratory rate 13/min, systolic blood pressure 127 mmHg, diastolic blood pressure 71 mmHg."),
    ("corpus", "viral_pharyngitis_4c364d4a-ff6f-848a-4634-db17d253deec", "objective",
     "Vital signs at this visit: body temperature 37.1°C, heart rate 100/min, blood pressure 111/69 mmHg, "
     "respiratory rate 14/min.",
     "Vital signs at this visit: body temperature 37.1°C. Vital signs 14 days earlier: heart rate 100/min, "
     "blood pressure 111/69 mmHg, respiratory rate 14/min."),
    ("corpus", "viral_pharyngitis_4e02c32e-c757-efea-6713-7371bd3b0bf5", "objective",
     "Vital signs at this visit: body temperature 37.3°C, systolic blood pressure 140 mmHg, diastolic "
     "blood pressure 101 mmHg, heart rate 74/min, respiratory rate 16/min.",
     "Vital signs at this visit: body temperature 37.3°C. Vital signs 6 months earlier: systolic blood "
     "pressure 140 mmHg, diastolic blood pressure 101 mmHg, heart rate 74/min, respiratory rate 16/min."),
    ("corpus", "viral_pharyngitis_5000c36d-ae9d-26b0-a46c-efb9ade804b4", "objective",
     "Vital signs at this visit: body temperature 37.2 °C, heart rate 82/min, respiratory rate 14/min, "
     "blood pressure 126/99 mmHg, body weight 85.7 kg, height 175.6 cm, BMI 27.8 kg/m2.",
     "Vital signs at this visit: body temperature 37.2 °C. Vital signs 14 days earlier: heart rate 82/min, "
     "respiratory rate 14/min, blood pressure 126/99 mmHg, body weight 85.7 kg, height 175.6 cm, BMI 27.8 "
     "kg/m2."),
    ("corpus", "viral_pharyngitis_51b175f1-303c-9092-a833-bc7aacc07040", "objective",
     "Vital signs at this visit: body temperature 37.4 °C, heart rate 70/min, respiratory rate 14/min, "
     "systolic blood pressure 110 mm[Hg], diastolic blood pressure 79 mm[Hg].",
     "Vital signs at this visit: body temperature 37.4 °C. Vital signs 3 months earlier: heart rate "
     "70/min, respiratory rate 14/min, systolic blood pressure 110 mm[Hg], diastolic blood pressure 79 "
     "mm[Hg]."),
    ("corpus", "viral_pharyngitis_51f028df-9e69-f38f-686e-987e9ca03e93", "objective",
     "Vital signs at this visit: body temperature 37.6°C, blood pressure 108/70 mmHg, heart rate 74/min, "
     "respiratory rate 13/min.",
     "Vital signs at this visit: body temperature 37.6°C. Vital signs 14 days earlier: blood pressure "
     "108/70 mmHg, heart rate 74/min, respiratory rate 13/min."),
    ("corpus", "viral_pharyngitis_5410cf75-9ed6-aa76-570b-0f918e55c0c9", "objective",
     "Vital signs at this visit: body temperature 37.8°C, blood pressure 121/85 mmHg, heart rate 81/min, "
     "respiratory rate 13/min.",
     "Vital signs at this visit: body temperature 37.8°C. Vital signs 14 days earlier: blood pressure "
     "121/85 mmHg, heart rate 81/min, respiratory rate 13/min."),
    ("corpus", "viral_pharyngitis_55c03c08-9b59-a60a-c80b-24f84f7dc46f", "objective",
     "Vital signs at this visit: temperature 37.6°C, heart rate 84/min, respiratory rate 15/min, blood "
     "pressure 111/74 mmHg.",
     "Vital signs at this visit: temperature 37.6°C. Vital signs 14 days earlier: heart rate 84/min, "
     "respiratory rate 15/min, blood pressure 111/74 mmHg."),
    ("corpus", "viral_pharyngitis_55e84570-094a-6b9a-10d0-eb941c963a24", "objective",
     "Vital signs at this visit: body temperature 37.4 °C, heart rate 97 /min, respiratory rate 12 /min, "
     "systolic blood pressure 97 mm[Hg], diastolic blood pressure 69 mm[Hg].",
     "Vital signs at this visit: body temperature 37.4 °C. Vital signs 14 days earlier: heart rate 97 "
     "/min, respiratory rate 12 /min, systolic blood pressure 97 mm[Hg], diastolic blood pressure 69 "
     "mm[Hg]."),
    ("corpus", "viral_pharyngitis_5618ff7a-7760-3a14-ea43-793487c82373", "objective",
     "Vital signs at this visit: body temperature 37.8 °C, blood pressure 122/76 mmHg, heart rate 73/min, "
     "respiratory rate 12/min.",
     "Vital signs at this visit: body temperature 37.8 °C. Vital signs 2 months earlier: blood pressure "
     "122/76 mmHg, heart rate 73/min, respiratory rate 12/min."),
    ("corpus", "viral_pharyngitis_5622efc9-c3dd-4bab-57f6-43280e22d0e5", "objective",
     "Vital signs at this visit: body temperature 37.2°C, heart rate 74/min, respiratory rate 13/min, "
     "blood pressure 104/73 mmHg.",
     "Vital signs at this visit: body temperature 37.2°C. Vital signs 7 days earlier: heart rate 74/min, "
     "respiratory rate 13/min, blood pressure 104/73 mmHg."),
    ("corpus", "viral_pharyngitis_563f139d-d860-add3-d8f3-26abef44e9d7", "objective",
     "Vital signs at this visit: body temperature 37.3°C, heart rate 71/min, respiratory rate 15/min, "
     "blood pressure 124/81 mmHg, pain severity 2/10, BMI 27.9 kg/m², weight 75.2 kg, height 164.3 cm.",
     "Vital signs at this visit: body temperature 37.3°C. Vital signs 14 days earlier: heart rate 71/min, "
     "respiratory rate 15/min, blood pressure 124/81 mmHg, pain severity 2/10, BMI 27.9 kg/m², weight 75.2 "
     "kg, height 164.3 cm."),
    ("corpus", "viral_pharyngitis_5708eb9e-af86-edc1-3b16-d9cf2446964a", "objective",
     "Vital signs at this visit: body temperature 37.8°C, heart rate 99/min, respiratory rate 14/min, "
     "blood pressure 111/80 mm[Hg], body weight 25.3 kg, body height 125.9 cm, BMI 16 kg/m2 (57.2th "
     "percentile for age/sex), pain severity 3/10.",
     "Vital signs at this visit: body temperature 37.8°C. Vital signs 14 days earlier: heart rate 99/min, "
     "respiratory rate 14/min, blood pressure 111/80 mm[Hg], body weight 25.3 kg, body height 125.9 cm, "
     "BMI 16 kg/m2 (57.2th percentile for age/sex), pain severity 3/10."),
    ("corpus", "viral_pharyngitis_579ccfce-5a60-6d26-5592-163fe860fa5e", "objective",
     "Vital signs at this visit: body temperature 37.1°C, heart rate 81/min, respiratory rate 16/min, "
     "systolic blood pressure 112 mmHg, diastolic blood pressure 70 mmHg.",
     "Vital signs at this visit: body temperature 37.1°C. Vital signs 7 days earlier: heart rate 81/min, "
     "respiratory rate 16/min, systolic blood pressure 112 mmHg, diastolic blood pressure 70 mmHg."),
    ("corpus", "viral_pharyngitis_57ce14c8-7a78-3a10-4203-a1c007dc90de", "objective",
     "Vital signs at current visit: body temperature 37.9°C, heart rate 97/min, respiratory rate 16/min, "
     "blood pressure 105/61 mmHg.",
     "Vital signs at current visit: body temperature 37.9°C. Vital signs 6 months earlier: heart rate "
     "97/min, respiratory rate 16/min, blood pressure 105/61 mmHg."),
    ("corpus", "viral_pharyngitis_5a3de6fc-25bf-61be-70a1-a762c66c6163", "objective",
     "Vital signs at this visit: body temperature 37.8°C, blood pressure 124/60 mmHg, heart rate 62/min, "
     "respiratory rate 14/min.",
     "Vital signs at this visit: body temperature 37.8°C. Vital signs 3 months earlier: blood pressure "
     "124/60 mmHg, heart rate 62/min, respiratory rate 14/min."),
    ("corpus", "viral_pharyngitis_5d874c66-d7ce-4072-1235-c936c0ff8f45", "objective",
     "Vital signs at this visit: body temperature 37.7 °C, blood pressure 101/60 mmHg, heart rate 87/min, "
     "respiratory rate 12/min.",
     "Vital signs at this visit: body temperature 37.7 °C. Vital signs 14 days earlier: blood pressure "
     "101/60 mmHg, heart rate 87/min, respiratory rate 12/min."),
    ("corpus", "viral_pharyngitis_5e483465-59cb-aa6d-b219-79c5c251f628", "objective",
     "Vital signs at this visit: body temperature 37.1°C, systolic blood pressure 138 mm[Hg], diastolic "
     "blood pressure 98 mm[Hg], heart rate 74 /min, respiratory rate 16 /min.",
     "Vital signs at this visit: body temperature 37.1°C, systolic blood pressure 138 mm[Hg], diastolic "
     "blood pressure 98 mm[Hg]. Vital signs 5 months earlier: heart rate 74 /min, respiratory rate 16 /min."),
    ("corpus", "viral_pharyngitis_5e8b74e8-bf6a-e6fe-bfb9-a98a6f805d28", "objective",
     "Vital signs at this visit: body temperature 37.4°C, blood pressure 111/82 mmHg, heart rate 64/min, "
     "respiratory rate 15/min, body weight 75.4 kg, height 161.7 cm, BMI 28.9 kg/m2.",
     "Vital signs at this visit: body temperature 37.4°C. Vital signs 14 days earlier: blood pressure "
     "111/82 mmHg, heart rate 64/min, respiratory rate 15/min, body weight 75.4 kg, height 161.7 cm, BMI "
     "28.9 kg/m2."),
    ("corpus", "viral_pharyngitis_5edf14ad-48eb-d13a-acb8-41ac60ad4143", "objective",
     "Vital signs at this visit: body temperature 37.1 °C, heart rate 96 /min, respiratory rate 16 /min, "
     "blood pressure 123/92 mm[Hg].",
     "Vital signs at this visit: body temperature 37.1 °C. Vital signs 14 days earlier: heart rate 96 "
     "/min, respiratory rate 16 /min, blood pressure 123/92 mm[Hg]."),
    ("corpus", "viral_pharyngitis_617880e7-e7cf-5dca-42a6-c1fa482b16b9", "objective",
     "Vital signs at this visit: body temperature 37.8°C, heart rate 79/min, respiratory rate 14/min, "
     "blood pressure 120/60 mmHg, body weight 76.1 kg, height 159 cm, BMI 30.1 kg/m².",
     "Vital signs at this visit: body temperature 37.8°C. Vital signs 9 days earlier: heart rate 79/min, "
     "respiratory rate 14/min, blood pressure 120/60 mmHg, body weight 76.1 kg, height 159 cm, BMI 30.1 "
     "kg/m²."),
    ("corpus", "viral_pharyngitis_617a48bf-14f3-0b21-d7ed-85b3bdfeb36c", "objective",
     "Vital signs at this visit: body temperature 37.0 °C, heart rate 96 /min, respiratory rate 13 /min, "
     "blood pressure 112/67 mm[Hg].",
     "Vital signs at this visit: body temperature 37.0 °C. Vital signs 19 days earlier: heart rate 96 "
     "/min, respiratory rate 13 /min, blood pressure 112/67 mm[Hg]."),
    ("corpus", "viral_pharyngitis_62719a80-56da-c740-8445-1fec268fb10b", "objective",
     "Vital signs at this visit: body temperature 37.5 °C, heart rate 98/min, blood pressure 131/75 mm[Hg] "
     "(14 days earlier), respiratory rate 14/min.",
     "Vital signs at this visit: body temperature 37.5 °C. Vital signs 14 days earlier: heart rate 98/min, "
     "blood pressure 131/75 mm[Hg] (14 days earlier), respiratory rate 14/min."),
    ("corpus", "viral_pharyngitis_65153529-335c-4f22-d637-1d550a5ae38b", "objective",
     "Vitals at current visit: temperature 37.1°C, heart rate 71/min, respiratory rate 13/min, blood "
     "pressure 95/76 mmHg, body weight 96.4 kg, height 185.2 cm, BMI 28.1 kg/m2, pain severity 1/10.",
     "Vitals at current visit: temperature 37.1°C. Vitals 7 days earlier: heart rate 71/min, respiratory "
     "rate 13/min, blood pressure 95/76 mmHg, body weight 96.4 kg, height 185.2 cm, BMI 28.1 kg/m2, pain "
     "severity 1/10."),
    ("corpus", "viral_pharyngitis_66c2bbee-66c8-a24c-c028-8466467d5f66", "objective",
     "Vital signs at this visit: body temperature 37.4°C, heart rate 60/min, respiratory rate 16/min, "
     "systolic blood pressure 143 mmHg, diastolic blood pressure 89 mmHg, body weight 87.8 kg, height "
     "178.3 cm, BMI 27.6 kg/m².",
     "Vital signs at this visit: body temperature 37.4°C. Vital signs 14 days earlier: heart rate 60/min, "
     "respiratory rate 16/min, systolic blood pressure 143 mmHg, diastolic blood pressure 89 mmHg, body "
     "weight 87.8 kg, height 178.3 cm, BMI 27.6 kg/m²."),
    ("corpus", "viral_pharyngitis_677f617f-228f-98b3-0cc1-d7ab61370092", "objective",
     "Vital signs at this visit: body temperature 37.5 °C, heart rate 73/min, respiratory rate 12/min, "
     "systolic blood pressure 136 mmHg, diastolic blood pressure 91 mmHg.",
     "Vital signs at this visit: body temperature 37.5 °C. Vital signs 7 days earlier: heart rate 73/min, "
     "respiratory rate 12/min, systolic blood pressure 136 mmHg, diastolic blood pressure 91 mmHg."),
    ("corpus", "viral_pharyngitis_6915e962-1a27-5dd2-1377-ca157e989734", "objective",
     "Vital signs at this visit: body temperature 37.6°C, systolic blood pressure 112 mmHg, diastolic "
     "blood pressure 73 mmHg, heart rate 66/min, respiratory rate 15/min.",
     "Vital signs at this visit: body temperature 37.6°C. Vital signs 7 days earlier: systolic blood "
     "pressure 112 mmHg, diastolic blood pressure 73 mmHg, heart rate 66/min, respiratory rate 15/min."),
    ("corpus", "viral_pharyngitis_693a6d62-b545-0a6e-8d9a-180ea2b57dd3", "objective",
     "Vital signs at this visit: body temperature 37.2°C, heart rate 63/min, respiratory rate 15/min, "
     "blood pressure 110/97 mmHg, body weight 30 kg, height 124.9 cm, BMI 19.2 kg/m², BMI percentile 94.8% "
     "for age and sex.",
     "Vital signs at this visit: body temperature 37.2°C. Vital signs 7 days earlier: heart rate 63/min, "
     "respiratory rate 15/min, blood pressure 110/97 mmHg, body weight 30 kg, height 124.9 cm, BMI 19.2 "
     "kg/m², BMI percentile 94.8% for age and sex."),
    ("corpus", "viral_pharyngitis_6b4e0874-ee7d-7dd8-e8eb-90d07325cbdc", "objective",
     "Vital signs at this visit: body temperature 37.4 Cel, heart rate 88/min, respiratory rate 14/min, "
     "systolic blood pressure 121 mmHg, diastolic blood pressure 86 mmHg.",
     "Vital signs at this visit: body temperature 37.4 Cel. Vital signs 1 month earlier: heart rate "
     "88/min, respiratory rate 14/min, systolic blood pressure 121 mmHg, diastolic blood pressure 86 mmHg."),
    ("corpus", "viral_pharyngitis_6d425f68-c4f0-572e-dbe4-4eba4aca480f", "objective",
     "Vital signs at this visit: body temperature 37.5°C, blood pressure 110/96 mmHg, heart rate 87/min, "
     "respiratory rate 16/min.",
     "Vital signs at this visit: body temperature 37.5°C. Vital signs 2 months earlier: blood pressure "
     "110/96 mmHg, heart rate 87/min, respiratory rate 16/min."),
    ("corpus", "viral_pharyngitis_6e2ba469-8c94-b89f-af5a-ac482d1ed501", "objective",
     "Vital signs at this visit: body temperature 37.2°C, blood pressure 103/71 mmHg, heart rate 82/min, "
     "respiratory rate 15/min.",
     "Vital signs at this visit: body temperature 37.2°C. Vital signs 3 months earlier: blood pressure "
     "103/71 mmHg, heart rate 82/min, respiratory rate 15/min."),
    ("corpus", "viral_pharyngitis_70999e65-93fb-7bd3-7723-445a19558d22", "objective",
     "Vital signs at this visit: body temperature 37.8°C, heart rate 73/min, respiratory rate 13/min, "
     "blood pressure 121/83 mmHg.",
     "Vital signs at this visit: body temperature 37.8°C. Vital signs 14 days earlier: heart rate 73/min, "
     "respiratory rate 13/min, blood pressure 121/83 mmHg."),
    ("corpus", "viral_pharyngitis_71effb58-5923-ac64-fb31-557ffc151eaf", "objective",
     "Vital signs at this visit: body temperature 37.6°C, heart rate 94/min, respiratory rate 14/min, "
     "systolic blood pressure 102 mmHg, diastolic blood pressure 65 mmHg.",
     "Vital signs at this visit: body temperature 37.6°C. Vital signs 1 month earlier: heart rate 94/min, "
     "respiratory rate 14/min, systolic blood pressure 102 mmHg, diastolic blood pressure 65 mmHg."),
    ("corpus", "viral_pharyngitis_7255ac8f-d8bc-ebb5-3f0f-737b6c8611b4", "objective",
     "Vital signs at this visit: body temperature 37.5°C, heart rate 71/min, respiratory rate 15/min, "
     "blood pressure 116/88 mmHg (4 months earlier).",
     "Vital signs at this visit: body temperature 37.5°C. Vital signs 4 months earlier: heart rate 71/min, "
     "respiratory rate 15/min, blood pressure 116/88 mmHg (4 months earlier)."),
    ("corpus", "viral_pharyngitis_73779b9d-700a-818f-29f9-743ab3fc61d4", "objective",
     "Vital signs at this visit: body temperature 37.3°C, heart rate 81/min, respiratory rate 16/min, "
     "systolic blood pressure 97 mmHg, diastolic blood pressure 59 mmHg.",
     "Vital signs at this visit: body temperature 37.3°C. Vital signs 7 days earlier: heart rate 81/min, "
     "respiratory rate 16/min, systolic blood pressure 97 mmHg, diastolic blood pressure 59 mmHg."),
    ("corpus", "viral_pharyngitis_752a4744-f97b-63b5-ca41-0705de7fc1e2", "objective",
     "Vital signs at this visit: body temperature 37.3 °C, blood pressure 110/87 mmHg, heart rate 66/min, "
     "respiratory rate 12/min.",
     "Vital signs at this visit: body temperature 37.3 °C. Vital signs 15 days earlier: blood pressure "
     "110/87 mmHg, heart rate 66/min, respiratory rate 12/min."),
    ("corpus", "viral_pharyngitis_75507415-af32-7820-6d9e-d4a3ffb9aef5", "objective",
     "Vital signs at this visit: body temperature 37.3 °C, heart rate 61 /min, blood pressure 133/81 "
     "mm[Hg], respiratory rate 13 /min.",
     "Vital signs at this visit: body temperature 37.3 °C. Vital signs 1 month earlier: heart rate 61 "
     "/min, blood pressure 133/81 mm[Hg], respiratory rate 13 /min."),
    ("corpus", "viral_pharyngitis_764cfe41-ccba-6564-b281-e27c78cff8e1", "objective",
     "Vital signs at this visit: body temperature 37.6 Cel, heart rate 95 /min, respiratory rate 14 /min, "
     "blood pressure 116/69 mm[Hg], body weight 77.1 kg, body height 175.8 cm, BMI 24.9 kg/m2.",
     "Vital signs at this visit: body temperature 37.6 Cel. Vital signs 14 days earlier: heart rate 95 "
     "/min, respiratory rate 14 /min, blood pressure 116/69 mm[Hg], body weight 77.1 kg, body height 175.8 "
     "cm, BMI 24.9 kg/m2."),
    ("corpus", "viral_pharyngitis_77740c24-e5ed-4a7d-6d93-9313d6de5481", "objective",
     "Vital signs at this visit: body temperature 37.9 °C, heart rate 97/min, respiratory rate 15/min, "
     "blood pressure 128/70 mmHg, body weight 58.2 kg, body height 158.8 cm, BMI 23.1 kg/m².",
     "Vital signs at this visit: body temperature 37.9 °C. Vital signs 28 days earlier: heart rate 97/min, "
     "respiratory rate 15/min, blood pressure 128/70 mmHg, body weight 58.2 kg, body height 158.8 cm, BMI "
     "23.1 kg/m²."),
    ("corpus", "viral_pharyngitis_7b8e8d43-32fd-f9a2-97d9-db037b08e04a", "objective",
     "Vital signs at current visit: body temperature 37.7°C, heart rate 89/min, respiratory rate 13/min, "
     "systolic blood pressure 103 mmHg, diastolic blood pressure 79 mmHg, body weight 14.8 kg, body height "
     "93.3 cm, BMI 17 kg/m², BMI percentile 75.3%, pain severity score 3/10.",
     "Vital signs at current visit: body temperature 37.7°C. Vital signs 7 days earlier: heart rate "
     "89/min, respiratory rate 13/min, systolic blood pressure 103 mmHg, diastolic blood pressure 79 mmHg, "
     "body weight 14.8 kg, body height 93.3 cm, BMI 17 kg/m², BMI percentile 75.3%, pain severity score "
     "3/10."),
    ("corpus", "viral_pharyngitis_7ccf849d-93cf-f338-60cf-5c2fcdc61186", "objective",
     "Vital signs at this visit: body temperature 37.7°C, blood pressure 116/87 mmHg, heart rate 81/min, "
     "respiratory rate 15/min.",
     "Vital signs at this visit: body temperature 37.7°C. Vital signs 14 days earlier: blood pressure "
     "116/87 mmHg, heart rate 81/min, respiratory rate 15/min."),
    ("corpus", "viral_pharyngitis_7e62cfe0-a515-6123-ac7f-5f81da2daba5", "objective",
     "Vital signs at this visit: body temperature 37.5°C, blood pressure 123/70 mmHg, heart rate 74/min, "
     "respiratory rate 13/min.",
     "Vital signs at this visit: body temperature 37.5°C. Vital signs 14 days earlier: blood pressure "
     "123/70 mmHg, heart rate 74/min, respiratory rate 13/min."),
    ("corpus", "viral_pharyngitis_7f58ad3b-7f02-d661-b018-eed0db4b3223", "objective",
     "Vital signs at this visit: body temperature 37.2°C, heart rate 87/min, respiratory rate 14/min, "
     "blood pressure 126/84 mm[Hg], body weight 25.6 kg, body height 121.8 cm, BMI 17.3 kg/m2 (84.1st "
     "percentile for age and sex).",
     "Vital signs at this visit: body temperature 37.2°C. Vital signs 9 days earlier: heart rate 87/min, "
     "respiratory rate 14/min, blood pressure 126/84 mm[Hg], body weight 25.6 kg, body height 121.8 cm, "
     "BMI 17.3 kg/m2 (84.1st percentile for age and sex)."),
    ("corpus", "viral_pharyngitis_828d5054-3649-5794-695f-9593a96d18ac", "objective",
     "Vital signs at this visit: body temperature 37.1°C, heart rate 78/min, respiratory rate 15/min, "
     "blood pressure 115/78 mmHg, BMI 24.2 kg/m².",
     "Vital signs at this visit: body temperature 37.1°C. Vital signs 14 days earlier: heart rate 78/min, "
     "respiratory rate 15/min, blood pressure 115/78 mmHg, BMI 24.2 kg/m²."),
    ("corpus", "viral_pharyngitis_8319e29f-230d-7f5a-be4f-1de1c222fa95", "objective",
     "Vital signs at current visit: body temperature 37.8°C, heart rate 82/min, respiratory rate 13/min, "
     "blood pressure 102/78 mm[Hg], body weight 35.5 kg, body height 118.9 cm, BMI 25.1 kg/m2 (99.2nd "
     "percentile for age and sex), pain severity 1/10.",
     "Vital signs at current visit: body temperature 37.8°C. Vital signs 3 days earlier: heart rate "
     "82/min, respiratory rate 13/min, blood pressure 102/78 mm[Hg], body weight 35.5 kg, body height "
     "118.9 cm, BMI 25.1 kg/m2 (99.2nd percentile for age and sex), pain severity 1/10."),
    ("corpus", "viral_pharyngitis_84fff2c2-9fb9-1b79-d83d-ba9c7864a558", "objective",
     "Vital signs at this visit: body temperature 37.9 Cel, heart rate 98 /min, respiratory rate 15 /min, "
     "systolic blood pressure 122 mm[Hg], diastolic blood pressure 68 mm[Hg].",
     "Vital signs at this visit: body temperature 37.9 Cel. Vital signs 14 days earlier: heart rate 98 "
     "/min, respiratory rate 15 /min, systolic blood pressure 122 mm[Hg], diastolic blood pressure 68 "
     "mm[Hg]."),
    ("corpus", "viral_pharyngitis_89672161-b8c4-74a3-9bf4-4805988d3bfe", "objective",
     "Vital signs at this visit: body temperature 37.3°C, heart rate 99/min, respiratory rate 14/min, "
     "blood pressure 104/81 mmHg, body weight 46.6 kg, body height 139.2 cm, BMI 24 kg/m2 (96.1st "
     "percentile for age and sex), pain severity 1/10.",
     "Vital signs at this visit: body temperature 37.3°C. Vital signs 7 days earlier: heart rate 99/min, "
     "respiratory rate 14/min, blood pressure 104/81 mmHg, body weight 46.6 kg, body height 139.2 cm, BMI "
     "24 kg/m2 (96.1st percentile for age and sex), pain severity 1/10."),
    ("corpus", "viral_pharyngitis_89912a26-321a-fdd2-b416-e6675183b551", "objective",
     "Vital signs at this visit: body temperature 37.8°C, blood pressure 134/80 mmHg, heart rate 81/min, "
     "respiratory rate 13/min.",
     "Vital signs at this visit: body temperature 37.8°C. Vital signs 15 days earlier: blood pressure "
     "134/80 mmHg, heart rate 81/min, respiratory rate 13/min."),
    ("corpus", "viral_pharyngitis_8ccf1fee-7816-b9a2-8182-51f64d742aef", "objective",
     "Vital signs at this visit: body temperature 37.1°C, blood pressure 111/76 mmHg, heart rate 76/min, "
     "respiratory rate 15/min.",
     "Vital signs at this visit: body temperature 37.1°C. Vital signs 7 days earlier: blood pressure "
     "111/76 mmHg, heart rate 76/min, respiratory rate 15/min."),
    ("corpus", "viral_pharyngitis_8d7a04da-b357-930e-fc1c-b52182bf2fcc", "objective",
     "Vital signs at this visit: body temperature 37.9 Cel, heart rate 70 /min, respiratory rate 14 /min, "
     "systolic blood pressure 137 mm[Hg], diastolic blood pressure 85 mm[Hg].",
     "Vital signs at this visit: body temperature 37.9 Cel. Vital signs 14 days earlier: heart rate 70 "
     "/min, respiratory rate 14 /min, systolic blood pressure 137 mm[Hg], diastolic blood pressure 85 "
     "mm[Hg]."),
    ("corpus", "viral_pharyngitis_9066427c-8fba-e097-669a-b99766ce37b5", "objective",
     "Vital signs at this visit: body temperature 37.2 °C, body height 168.4 cm, body weight 77.3 kg, BMI "
     "27.3 kg/m2, systolic blood pressure 105 mm[Hg], diastolic blood pressure 75 mm[Hg], heart rate 84 "
     "/min, respiratory rate 13 /min.",
     "Vital signs at this visit: body temperature 37.2 °C. Vital signs 15 days earlier: body height 168.4 "
     "cm, body weight 77.3 kg, BMI 27.3 kg/m2, systolic blood pressure 105 mm[Hg], diastolic blood "
     "pressure 75 mm[Hg], heart rate 84 /min, respiratory rate 13 /min."),
    ("corpus", "viral_pharyngitis_9081e8ad-1ee1-a605-0347-bbe8c983e780", "objective",
     "Vital signs at this visit: body temperature 37.6 °C, heart rate 92 /min, blood pressure 120/87 mmHg, "
     "respiratory rate 14 /min.",
     "Vital signs at this visit: body temperature 37.6 °C. Vital signs 14 days earlier: heart rate 92 "
     "/min, blood pressure 120/87 mmHg, respiratory rate 14 /min."),
    ("corpus", "viral_pharyngitis_909c187f-eb98-bcd0-34eb-1599b51b8039", "objective",
     "Vital signs at this visit: body temperature 37.3 °C, heart rate 78/min, respiratory rate 12/min, "
     "blood pressure 112/69 mmHg, body weight 32.3 kg, body height 141.4 cm, BMI 16.2 kg/m², BMI "
     "percentile 59.7% per age and sex.",
     "Vital signs at this visit: body temperature 37.3 °C. Vital signs 7 days earlier: heart rate 78/min, "
     "respiratory rate 12/min, blood pressure 112/69 mmHg, body weight 32.3 kg, body height 141.4 cm, BMI "
     "16.2 kg/m², BMI percentile 59.7% per age and sex."),
    ("corpus", "viral_pharyngitis_909d8664-f283-3dcd-0b67-14f32786f8bd", "objective",
     "Vital signs at this visit: body temperature 37.5°C, heart rate 81/min, respiratory rate 13/min, "
     "systolic blood pressure 95 mmHg, diastolic blood pressure 55 mmHg.",
     "Vital signs at this visit: body temperature 37.5°C. Vital signs 7 days earlier: heart rate 81/min, "
     "respiratory rate 13/min, systolic blood pressure 95 mmHg, diastolic blood pressure 55 mmHg."),
    ("corpus", "viral_pharyngitis_943a6e72-cceb-eed3-54a5-bc9b157b2de5", "objective",
     "Vital signs at this visit: body temperature 37.1°C, heart rate 76/min, respiratory rate 14/min, "
     "systolic blood pressure 113 mm[Hg], diastolic blood pressure 95 mm[Hg].",
     "Vital signs at this visit: body temperature 37.1°C. Vital signs 14 days earlier: heart rate 76/min, "
     "respiratory rate 14/min, systolic blood pressure 113 mm[Hg], diastolic blood pressure 95 mm[Hg]."),
    ("corpus", "viral_pharyngitis_95014bf7-b26e-3ac8-8e70-4ca544d0fe4d", "objective",
     "Vital signs at this visit: body temperature 37.5°C, systolic blood pressure 105 mmHg, diastolic "
     "blood pressure 68 mmHg, heart rate 75/min, respiratory rate 13/min.",
     "Vital signs at this visit: body temperature 37.5°C. Vital signs 14 days earlier: systolic blood "
     "pressure 105 mmHg, diastolic blood pressure 68 mmHg, heart rate 75/min, respiratory rate 13/min."),
    ("corpus", "viral_pharyngitis_95a8683e-2c76-e035-c04a-3a3db61ac1c5", "objective",
     "Vital signs at this visit: body temperature 37.1°C, blood pressure 129/74 mmHg, heart rate 69/min, "
     "respiratory rate 13/min.",
     "Vital signs at this visit: body temperature 37.1°C. Vital signs 15 days earlier: blood pressure "
     "129/74 mmHg, heart rate 69/min, respiratory rate 13/min."),
    ("corpus", "viral_pharyngitis_95c4a286-a415-1d62-13b1-bde7cda36300", "objective",
     "Vital signs at this visit: body temperature 37.3°C, heart rate 99/min, respiratory rate 15/min, "
     "blood pressure 125/85 mmHg, body weight 80 kg, height 169.1 cm, BMI 28 kg/m2.",
     "Vital signs at this visit: body temperature 37.3°C. Vital signs 14 days earlier: heart rate 99/min, "
     "respiratory rate 15/min, blood pressure 125/85 mmHg, body weight 80 kg, height 169.1 cm, BMI 28 "
     "kg/m2."),
    ("corpus", "viral_pharyngitis_969de29d-5dbe-0619-0b31-4870149e7c7e", "objective",
     "Vital signs at this visit: body temperature 37.8°C, heart rate 62/min, respiratory rate 13/min, "
     "blood pressure 109/88 mmHg.",
     "Vital signs at this visit: body temperature 37.8°C. Vital signs 14 days earlier: heart rate 62/min, "
     "respiratory rate 13/min, blood pressure 109/88 mmHg."),
    ("corpus", "viral_pharyngitis_98711a32-846b-3fc4-0a3f-a6de7c46d3b4", "objective",
     "Vital signs at this visit: body temperature 37.3°C, heart rate 95/min, respiratory rate 12/min, "
     "blood pressure 110/73 mmHg, body weight 19 kg, body height 106.1 cm, BMI 16.9 kg/m2 (85.8th "
     "percentile for age and sex), pain severity 2/10.",
     "Vital signs at this visit: body temperature 37.3°C. Vital signs 21 days earlier: heart rate 95/min, "
     "respiratory rate 12/min, blood pressure 110/73 mmHg, body weight 19 kg, body height 106.1 cm, BMI "
     "16.9 kg/m2 (85.8th percentile for age and sex), pain severity 2/10."),
    ("corpus", "viral_pharyngitis_9a79df7f-66d7-59cb-3eb7-8052af239559", "objective",
     "Vital signs at this visit: body temperature 37.5°C, heart rate 62/min, respiratory rate 15/min, "
     "blood pressure 108/84 mmHg.",
     "Vital signs at this visit: body temperature 37.5°C. Vital signs 21 days earlier: heart rate 62/min, "
     "respiratory rate 15/min, blood pressure 108/84 mmHg."),
    ("corpus", "viral_pharyngitis_9be5621e-e78a-e06c-e99b-c42f5be5f382", "objective",
     "Vital signs at this visit: body temperature 37.5°C, heart rate 96/min, respiratory rate 13/min, "
     "systolic blood pressure 125 mmHg, diastolic blood pressure 79 mmHg.",
     "Vital signs at this visit: body temperature 37.5°C. Vital signs 8 days earlier: heart rate 96/min, "
     "respiratory rate 13/min, systolic blood pressure 125 mmHg, diastolic blood pressure 79 mmHg."),
    ("corpus", "viral_pharyngitis_9cae1167-8132-7015-4164-007b2943e7ff", "objective",
     "Vital signs at this visit: body temperature 37.9 °C, heart rate 68/min, respiratory rate 13/min, "
     "systolic blood pressure 151 mmHg, diastolic blood pressure 113 mmHg.",
     "Vital signs at this visit: body temperature 37.9 °C. Vital signs 14 days earlier: heart rate 68/min, "
     "respiratory rate 13/min, systolic blood pressure 151 mmHg, diastolic blood pressure 113 mmHg."),
    ("corpus", "viral_pharyngitis_a048f312-f74f-499d-823f-06debe9e7daa", "objective",
     "Vital signs at this visit: body temperature 37.3°C, heart rate 79/min, respiratory rate 15/min, "
     "blood pressure 143/82 mmHg, body weight 31.7 kg, height 131.1 cm, BMI 18.5 kg/m2 (79.8th percentile "
     "for age and sex).",
     "Vital signs at this visit: body temperature 37.3°C. Vital signs 7 days earlier: heart rate 79/min, "
     "respiratory rate 15/min, blood pressure 143/82 mmHg, body weight 31.7 kg, height 131.1 cm, BMI 18.5 "
     "kg/m2 (79.8th percentile for age and sex)."),
    ("corpus", "viral_pharyngitis_a083e08d-a247-7d4a-636b-68f3de794d53", "objective",
     "Vital signs at this visit: body temperature 37.3°C, systolic blood pressure 116 mm[Hg], diastolic "
     "blood pressure 78 mm[Hg], heart rate 99/min, respiratory rate 13/min.",
     "Vital signs at this visit: body temperature 37.3°C. Vital signs 7 days earlier: systolic blood "
     "pressure 116 mm[Hg], diastolic blood pressure 78 mm[Hg], heart rate 99/min, respiratory rate 13/min."),
    ("corpus", "viral_pharyngitis_a2acda3c-b65e-7eae-e825-3b8d592dc2d8", "objective",
     "Vital signs at this visit: body temperature 37.7°C, heart rate 95/min, respiratory rate 13/min, "
     "blood pressure 116/89 mmHg, BMI 27.8 kg/m2, body weight 78.3 kg, body height 167.9 cm.",
     "Vital signs at this visit: body temperature 37.7°C. Vital signs 14 days earlier: heart rate 95/min, "
     "respiratory rate 13/min, blood pressure 116/89 mmHg, BMI 27.8 kg/m2, body weight 78.3 kg, body "
     "height 167.9 cm."),
    ("corpus", "viral_pharyngitis_a2d3d0ca-282d-e0e9-8625-98f2d530e700", "objective",
     "Vital signs at this visit: body temperature 37.5°C, heart rate 61 /min, respiratory rate 16 /min, "
     "systolic blood pressure 125 mm[Hg], diastolic blood pressure 85 mm[Hg].",
     "Vital signs at this visit: body temperature 37.5°C. Vital signs 3 days earlier: heart rate 61 /min, "
     "respiratory rate 16 /min, systolic blood pressure 125 mm[Hg], diastolic blood pressure 85 mm[Hg]."),
    ("corpus", "viral_pharyngitis_a3bc3c25-2c07-6e92-18b0-f964285f280f", "objective",
     "Vital signs at this visit: body temperature 37.8°C, heart rate 84/min, respiratory rate 13/min, "
     "systolic blood pressure 81 mmHg, diastolic blood pressure 42 mmHg.",
     "Vital signs at this visit: body temperature 37.8°C. Vital signs 7 days earlier: heart rate 84/min, "
     "respiratory rate 13/min, systolic blood pressure 81 mmHg, diastolic blood pressure 42 mmHg."),
    ("corpus", "viral_pharyngitis_a548213b-2b49-04c6-1cee-aa3e32865436", "objective",
     "Vital signs at this visit: body temperature 37.4°C, heart rate 98/min, respiratory rate 14/min, "
     "systolic blood pressure 139 mmHg, diastolic blood pressure 75 mmHg.",
     "Vital signs at this visit: body temperature 37.4°C. Vital signs 14 days earlier: heart rate 98/min, "
     "respiratory rate 14/min, systolic blood pressure 139 mmHg, diastolic blood pressure 75 mmHg."),
    ("corpus", "viral_pharyngitis_a7e712e2-9a4b-302a-8db5-446e98b673a4", "objective",
     "Vital signs at this visit: body temperature 37.3°C, heart rate 100/min, respiratory rate 14/min, "
     "blood pressure 111/70 mm[Hg].",
     "Vital signs at this visit: body temperature 37.3°C. Vital signs 7 days earlier: heart rate 100/min, "
     "respiratory rate 14/min, blood pressure 111/70 mm[Hg]."),
    ("corpus", "viral_pharyngitis_a86567fc-da0a-37aa-bb80-13dce41a700a", "objective",
     "Vital signs at this visit: body temperature 37.4 °C, systolic blood pressure 121 mm[Hg], diastolic "
     "blood pressure 68 mm[Hg], heart rate 79 /min, respiratory rate 13 /min (all from 3 months earlier).",
     "Vital signs at this visit: body temperature 37.4 °C. Vital signs 3 months earlier: systolic blood "
     "pressure 121 mm[Hg], diastolic blood pressure 68 mm[Hg], heart rate 79 /min, respiratory rate 13 "
     "/min (all from 3 months earlier)."),
    ("corpus", "viral_pharyngitis_ac846549-4798-9813-4537-3f04e516cb85", "objective",
     "Vital signs at this visit: body temperature 37.1°C, heart rate 85/min, respiratory rate 15/min, "
     "systolic blood pressure 140 mmHg, diastolic blood pressure 82 mmHg.",
     "Vital signs at this visit: body temperature 37.1°C. Vital signs 14 days earlier: heart rate 85/min, "
     "respiratory rate 15/min, systolic blood pressure 140 mmHg, diastolic blood pressure 82 mmHg."),
    ("corpus", "viral_pharyngitis_ada85cf7-7402-b5ba-f84f-ee7466a144a7", "objective",
     "Vital signs at this visit: body temperature 38.0°C, blood pressure 120/85 mmHg, heart rate 81/min, "
     "respiratory rate 13/min.",
     "Vital signs at this visit: body temperature 38.0°C. Vital signs 9 months earlier: blood pressure "
     "120/85 mmHg, heart rate 81/min, respiratory rate 13/min."),
    ("corpus", "viral_pharyngitis_af563f5f-8e20-a518-fbc7-74093a7b5162", "objective",
     "Vital signs at this visit: body temperature 37.5°C, heart rate 68/min, respiratory rate 14/min, "
     "blood pressure 110/84 mmHg.",
     "Vital signs at this visit: body temperature 37.5°C. Vital signs 14 days earlier: heart rate 68/min, "
     "respiratory rate 14/min, blood pressure 110/84 mmHg."),
    ("corpus", "viral_pharyngitis_af5c9b37-3c78-2155-aea8-9aacdbe59e7a", "objective",
     "Vital signs at this visit: body temperature 37.1 °C, systolic blood pressure 113 mm[Hg], diastolic "
     "blood pressure 81 mm[Hg], heart rate 76 /min, respiratory rate 13 /min.",
     "Vital signs at this visit: body temperature 37.1 °C. Vital signs 7 days earlier: systolic blood "
     "pressure 113 mm[Hg], diastolic blood pressure 81 mm[Hg], heart rate 76 /min, respiratory rate 13 "
     "/min."),
    ("corpus", "viral_pharyngitis_b3f6d9dd-a02b-b267-f413-454ff4565482", "objective",
     "Vital signs at this visit: body temperature 37.5 Cel, heart rate 68/min, respiratory rate 15/min, "
     "systolic blood pressure 102 mm[Hg], diastolic blood pressure 65 mm[Hg].",
     "Vital signs at this visit: body temperature 37.5 Cel. Vital signs 14 days earlier: heart rate "
     "68/min, respiratory rate 15/min, systolic blood pressure 102 mm[Hg], diastolic blood pressure 65 "
     "mm[Hg]."),
    ("corpus", "viral_pharyngitis_bb2a593a-8c68-0dc5-9f18-c2187ffaa18c", "objective",
     "Vital signs at this visit: body temperature 38.0°C, blood pressure 128/92 mmHg, heart rate 89/min, "
     "respiratory rate 14/min.",
     "Vital signs at this visit: body temperature 38.0°C. Vital signs 14 days earlier: blood pressure "
     "128/92 mmHg, heart rate 89/min, respiratory rate 14/min."),
    ("corpus", "viral_pharyngitis_bbc5d5de-c91a-a727-1039-b364e675ad41", "objective",
     "Vital signs at this visit: body temperature 37.2 °C, heart rate 73/min (6 months earlier), "
     "respiratory rate 13/min (6 months earlier), systolic blood pressure 85 mmHg, diastolic blood "
     "pressure 69 mmHg.",
     "Vital signs at this visit: body temperature 37.2 °C, heart rate 73/min (6 months earlier), "
     "respiratory rate 13/min (6 months earlier). Vital signs 6 months earlier: systolic blood pressure 85 "
     "mmHg, diastolic blood pressure 69 mmHg."),
    ("corpus", "viral_pharyngitis_bc200bee-307d-cf34-588d-1816a2d6a8bc", "objective",
     "Vital signs at this visit: body temperature 37.2°C, heart rate 95/min, respiratory rate 13/min, "
     "systolic blood pressure 106 mmHg, diastolic blood pressure 57 mmHg, body weight 97.1 kg, height "
     "188.5 cm, BMI 27.3 kg/m².",
     "Vital signs at this visit: body temperature 37.2°C. Vital signs 12 days earlier: heart rate 95/min, "
     "respiratory rate 13/min, systolic blood pressure 106 mmHg, diastolic blood pressure 57 mmHg, body "
     "weight 97.1 kg, height 188.5 cm, BMI 27.3 kg/m²."),
    ("corpus", "viral_pharyngitis_c236289f-856a-e6ab-9026-d55928db6a79", "objective",
     "Vital signs at this visit: body temperature 37.6°C, heart rate 78/min, respiratory rate 13/min, "
     "systolic blood pressure 144 mmHg, diastolic blood pressure 77 mmHg.",
     "Vital signs at this visit: body temperature 37.6°C. Vital signs 8 days earlier: heart rate 78/min, "
     "respiratory rate 13/min, systolic blood pressure 144 mmHg, diastolic blood pressure 77 mmHg."),
    ("corpus", "viral_pharyngitis_c274c4c7-c762-b887-17b9-8ccf9f3a0e71", "objective",
     "Vital signs at this visit: body temperature 38.0°C, heart rate 94/min, respiratory rate 12/min, "
     "blood pressure 109/87 mmHg.",
     "Vital signs at this visit: body temperature 38.0°C. Vital signs 14 days earlier: heart rate 94/min, "
     "respiratory rate 12/min, blood pressure 109/87 mmHg."),
    ("corpus", "viral_pharyngitis_c290a587-aeaf-047d-385c-47d4380b220e", "objective",
     "Vital signs at this visit: body temperature 37.6°C, blood pressure 117/75 mm[Hg], heart rate 60/min, "
     "respiratory rate 15/min.",
     "Vital signs at this visit: body temperature 37.6°C. Vital signs 7 days earlier: blood pressure "
     "117/75 mm[Hg], heart rate 60/min, respiratory rate 15/min."),
    ("corpus", "viral_pharyngitis_c318a0be-b178-6e0c-6648-0f765dc7a157", "objective",
     "Vital signs at this visit: body temperature 37.9°C, heart rate 93/min, blood pressure 125/74 mmHg, "
     "respiratory rate 16/min.",
     "Vital signs at this visit: body temperature 37.9°C. Vital signs 9 months earlier: heart rate 93/min, "
     "blood pressure 125/74 mmHg, respiratory rate 16/min."),
    ("corpus", "viral_pharyngitis_c32559f4-3fa9-3d1b-8d22-9fb164563c61", "objective",
     "Vital signs at this visit: body temperature 37.4 °C, heart rate 86/min, respiratory rate 14/min, "
     "systolic blood pressure 97 mmHg, diastolic blood pressure 67 mmHg.",
     "Vital signs at this visit: body temperature 37.4 °C. Vital signs 1 month earlier: heart rate 86/min, "
     "respiratory rate 14/min, systolic blood pressure 97 mmHg, diastolic blood pressure 67 mmHg."),
    ("corpus", "viral_pharyngitis_c8bcc4ba-4410-9fb4-cef5-b3fbbe8776ae", "objective",
     "Vital signs at this visit: body temperature 37.5 °C, heart rate 78/min, respiratory rate 15/min, "
     "blood pressure 124/72 mmHg, body weight 59.4 kg, height 162.5 cm, BMI 22.5 kg/m2.",
     "Vital signs at this visit: body temperature 37.5 °C. Vital signs 14 days earlier: heart rate 78/min, "
     "respiratory rate 15/min, blood pressure 124/72 mmHg, body weight 59.4 kg, height 162.5 cm, BMI 22.5 "
     "kg/m2."),
    ("corpus", "viral_pharyngitis_c99a9fb9-4d7c-ed3d-67ed-4e051e88cb25", "objective",
     "Vital signs at current visit: body temperature 37.0 °C, heart rate 69 /min, respiratory rate 13 "
     "/min, systolic blood pressure 120 mm[Hg], diastolic blood pressure 83 mm[Hg].",
     "Vital signs at current visit: body temperature 37.0 °C. Vital signs 15 days earlier: heart rate 69 "
     "/min, respiratory rate 13 /min, systolic blood pressure 120 mm[Hg], diastolic blood pressure 83 "
     "mm[Hg]."),
    ("corpus", "viral_pharyngitis_c9c17b3a-fa94-292e-25ec-8e83f91fac10", "objective",
     "Vital signs at this visit: body temperature 37.9°C, blood pressure 121/72 mmHg, heart rate 72/min, "
     "respiratory rate 12/min.",
     "Vital signs at this visit: body temperature 37.9°C. Vital signs 7 days earlier: blood pressure "
     "121/72 mmHg, heart rate 72/min, respiratory rate 12/min."),
    ("corpus", "viral_pharyngitis_cb3a15dc-e40c-847f-2558-683bfebb8274", "objective",
     "Vital signs at this visit: body temperature 37.6°C, blood pressure 114/70 mmHg, heart rate 68/min, "
     "respiratory rate 15/min.",
     "Vital signs at this visit: body temperature 37.6°C. Vital signs 2 months earlier: blood pressure "
     "114/70 mmHg, heart rate 68/min, respiratory rate 15/min."),
    ("corpus", "viral_pharyngitis_cc8265b7-3344-4da3-5d75-50e45f2ab7d3", "objective",
     "Vital signs at this visit: body temperature 37.4°C, heart rate 65/min, respiratory rate 15/min, "
     "blood pressure 116/73 mmHg, body weight 15.3 kg, body height 95.4 cm, BMI 16.8 kg/m², BMI-for-age "
     "73.4th percentile, pain score 1/10.",
     "Vital signs at this visit: body temperature 37.4°C. Vital signs 7 days earlier: heart rate 65/min, "
     "respiratory rate 15/min, blood pressure 116/73 mmHg, body weight 15.3 kg, body height 95.4 cm, BMI "
     "16.8 kg/m², BMI-for-age 73.4th percentile, pain score 1/10."),
    ("corpus", "viral_pharyngitis_cdea2c60-1227-b0e8-0bc8-0bf80824f832", "objective",
     "Vital signs at this visit: body temperature 37.2°C, heart rate 90/min, respiratory rate 13/min, "
     "systolic blood pressure 143 mmHg, diastolic blood pressure 79 mmHg.",
     "Vital signs at this visit: body temperature 37.2°C. Vital signs 7 days earlier: heart rate 90/min, "
     "respiratory rate 13/min, systolic blood pressure 143 mmHg, diastolic blood pressure 79 mmHg."),
    ("corpus", "viral_pharyngitis_ce7b65ac-e5cb-0dfc-6c60-409d2be38ba4", "objective",
     "Vital signs at this visit: body temperature 37.8 °C, heart rate 63/min, respiratory rate 14/min, "
     "systolic blood pressure 114 mmHg, diastolic blood pressure 73 mmHg.",
     "Vital signs at this visit: body temperature 37.8 °C. Vital signs 14 days earlier: heart rate 63/min, "
     "respiratory rate 14/min, systolic blood pressure 114 mmHg, diastolic blood pressure 73 mmHg."),
    ("corpus", "viral_pharyngitis_d20e6572-b44b-0052-f58a-310dcbcf84e1", "objective",
     "Vital signs at this visit: body temperature 37.7 °C, heart rate 77/min, respiratory rate 15/min, "
     "systolic blood pressure 123 mmHg, diastolic blood pressure 79 mmHg.",
     "Vital signs at this visit: body temperature 37.7 °C. Vital signs 10 months earlier: heart rate "
     "77/min, respiratory rate 15/min, systolic blood pressure 123 mmHg, diastolic blood pressure 79 mmHg."),
    ("corpus", "viral_pharyngitis_d3ae6a9b-7445-3201-466d-bb40a146489c", "objective",
     "Vital signs at this visit: body temperature 37.4 °C, blood pressure 120/84 mmHg, heart rate 71/min, "
     "respiratory rate 15/min.",
     "Vital signs at this visit: body temperature 37.4 °C. Vital signs 2 months earlier: blood pressure "
     "120/84 mmHg, heart rate 71/min, respiratory rate 15/min."),
    ("corpus", "viral_pharyngitis_d601f49b-cdb7-eebb-a4b5-ded0850bf44d", "objective",
     "Vital signs at this visit: body temperature 37.6 °C, heart rate 66/min, respiratory rate 15/min, "
     "blood pressure 123/88 mm[Hg].",
     "Vital signs at this visit: body temperature 37.6 °C. Vital signs 13 days earlier: heart rate 66/min, "
     "respiratory rate 15/min, blood pressure 123/88 mm[Hg]."),
    ("corpus", "viral_pharyngitis_d6669295-1870-b861-ad70-e439a5fbccda", "objective",
     "Vital signs at this visit: body temperature 37.7°C, blood pressure 128/71 mmHg, heart rate 78/min, "
     "respiratory rate 15/min.",
     "Vital signs at this visit: body temperature 37.7°C. Vital signs 7 days earlier: blood pressure "
     "128/71 mmHg, heart rate 78/min, respiratory rate 15/min."),
    ("corpus", "viral_pharyngitis_d6c295be-0e5d-bd9e-8f5d-689b4d4fdbb2", "objective",
     "Vital signs at this visit: body temperature 37.7 Cel, heart rate 95 /min, systolic blood pressure "
     "127 mm[Hg], diastolic blood pressure 62 mm[Hg], respiratory rate 16 /min.",
     "Vital signs at this visit: body temperature 37.7 Cel. Vital signs 5 months earlier: heart rate 95 "
     "/min, systolic blood pressure 127 mm[Hg], diastolic blood pressure 62 mm[Hg], respiratory rate 16 "
     "/min."),
    ("corpus", "viral_pharyngitis_d760526e-39e1-91c6-9977-06e163854a19", "objective",
     "Vitals at this visit: body temperature 37.8 °C, heart rate 66/min, respiratory rate 15/min, systolic "
     "blood pressure 139 mmHg, diastolic blood pressure 87 mmHg.",
     "Vitals at this visit: body temperature 37.8 °C. Vitals 14 days earlier: heart rate 66/min, "
     "respiratory rate 15/min, systolic blood pressure 139 mmHg, diastolic blood pressure 87 mmHg."),
    ("corpus", "viral_pharyngitis_d87953e2-5807-a0f8-9422-87be8c738edc", "objective",
     "Vital signs at this visit: body temperature 37.9°C, heart rate 77/min, respiratory rate 14/min, "
     "systolic blood pressure 131 mmHg, diastolic blood pressure 67 mmHg.",
     "Vital signs at this visit: body temperature 37.9°C. Vital signs 9 days earlier: heart rate 77/min, "
     "respiratory rate 14/min, systolic blood pressure 131 mmHg, diastolic blood pressure 67 mmHg."),
    ("corpus", "viral_pharyngitis_d8abfe9b-392d-befe-6c8c-614dffc68604", "objective",
     "Vital signs at this visit: body temperature 37.6 °C, heart rate 66/min, respiratory rate 14/min, "
     "systolic blood pressure 120 mmHg, diastolic blood pressure 64 mmHg.",
     "Vital signs at this visit: body temperature 37.6 °C. Vital signs 9 days earlier: heart rate 66/min, "
     "respiratory rate 14/min, systolic blood pressure 120 mmHg, diastolic blood pressure 64 mmHg."),
    ("corpus", "viral_pharyngitis_d92be68e-9455-4b0a-814d-93bdd2802ad8", "objective",
     "Vital signs at this visit: body temperature 37.7°C, blood pressure 125/103 mmHg, heart rate 63/min, "
     "respiratory rate 13/min.",
     "Vital signs at this visit: body temperature 37.7°C. Vital signs 14 days earlier: blood pressure "
     "125/103 mmHg, heart rate 63/min, respiratory rate 13/min."),
    ("corpus", "viral_pharyngitis_d9aa7a26-5642-60d5-9555-cf5738703900", "objective",
     "Vital signs at this visit: body temperature 37.7°C, heart rate 99/min, respiratory rate 12/min, "
     "blood pressure 120/98 mmHg.",
     "Vital signs at this visit: body temperature 37.7°C. Vital signs 9 months earlier: heart rate 99/min, "
     "respiratory rate 12/min, blood pressure 120/98 mmHg."),
    ("corpus", "viral_pharyngitis_da6247f1-cd8c-8607-f6d5-80b45709d4db", "objective",
     "Vital signs at this visit: body temperature 38.0 °C, systolic BP 105 mmHg, diastolic BP 85 mmHg, "
     "heart rate 79/min, respiratory rate 13/min.",
     "Vital signs at this visit: body temperature 38.0 °C. Vital signs 5 months earlier: systolic BP 105 "
     "mmHg, diastolic BP 85 mmHg, heart rate 79/min, respiratory rate 13/min."),
    ("corpus", "viral_pharyngitis_db0fdbed-a23c-3a15-0d9b-4f0fe0c39b3e", "objective",
     "Vital signs at this visit: body temperature 37.0 °C, heart rate 79/min, respiratory rate 15/min, "
     "blood pressure 123/87 mmHg.",
     "Vital signs at this visit: body temperature 37.0 °C. Vital signs 17 days earlier: heart rate 79/min, "
     "respiratory rate 15/min, blood pressure 123/87 mmHg."),
    ("corpus", "viral_pharyngitis_dcee36f7-ff9a-7274-aa31-68d406ea2761", "objective",
     "Vital signs at current visit: body temperature 37.6°C, heart rate 84/min, respiratory rate 13/min, "
     "blood pressure 116/83 mm[Hg], body weight 21.2 kg, body height 111.2 cm, BMI 17.1 kg/m2 (88.7th "
     "percentile for age and sex).",
     "Vital signs at current visit: body temperature 37.6°C. Vital signs 7 days earlier: heart rate "
     "84/min, respiratory rate 13/min, blood pressure 116/83 mm[Hg], body weight 21.2 kg, body height "
     "111.2 cm, BMI 17.1 kg/m2 (88.7th percentile for age and sex)."),
    ("corpus", "viral_pharyngitis_df1c7cbc-4c71-e034-c2a7-94f6766d0a91", "objective",
     "Vital signs at this visit: body temperature 37.3°C, heart rate 78/min, respiratory rate 12/min, "
     "systolic BP 99 mmHg, diastolic BP 68 mmHg.",
     "Vital signs at this visit: body temperature 37.3°C. Vital signs 7 days earlier: heart rate 78/min, "
     "respiratory rate 12/min, systolic BP 99 mmHg, diastolic BP 68 mmHg."),
    ("corpus", "viral_pharyngitis_e25c8d66-794e-6ccc-6a96-dfda4a143bda", "objective",
     "Vital signs at this visit: body temperature 37.7°C, heart rate 65/min, respiratory rate 12/min, "
     "systolic blood pressure 139 mmHg, diastolic blood pressure 80 mmHg.",
     "Vital signs at this visit: body temperature 37.7°C. Vital signs 7 days earlier: heart rate 65/min, "
     "respiratory rate 12/min, systolic blood pressure 139 mmHg, diastolic blood pressure 80 mmHg."),
    ("corpus", "viral_pharyngitis_e2606a72-b470-0de3-316b-c94c4aad085f", "objective",
     "Vital signs at current visit: body temperature 37.8°C, heart rate 72/min, respiratory rate 15/min, "
     "blood pressure 126/70 mmHg, body weight 22 kg, body height 116.4 cm, BMI 16.2 kg/m2 (68.4th "
     "percentile for age and sex).",
     "Vital signs at current visit: body temperature 37.8°C. Vital signs 7 days earlier: heart rate "
     "72/min, respiratory rate 15/min, blood pressure 126/70 mmHg, body weight 22 kg, body height 116.4 "
     "cm, BMI 16.2 kg/m2 (68.4th percentile for age and sex)."),
    ("corpus", "viral_pharyngitis_e4d65768-124e-2c5b-2a8c-8034a5d80527", "objective",
     "Vital signs at this visit: body temperature 37.2°C, heart rate 97/min, respiratory rate 12/min, "
     "blood pressure 130/61 mmHg.",
     "Vital signs at this visit: body temperature 37.2°C. Vital signs 14 days earlier: heart rate 97/min, "
     "respiratory rate 12/min, blood pressure 130/61 mmHg."),
    ("corpus", "viral_pharyngitis_e537ba76-32ff-a938-b8a8-9ecf372f5ca2", "objective",
     "Vital signs at this visit: body temperature 37.2 °C, heart rate 90/min, respiratory rate 12/min, "
     "blood pressure 141/68 mmHg.",
     "Vital signs at this visit: body temperature 37.2 °C. Vital signs 14 days earlier: heart rate 90/min, "
     "respiratory rate 12/min, blood pressure 141/68 mmHg."),
    ("corpus", "viral_pharyngitis_e7688dfd-bfca-1a0c-464a-eacedc223c9f", "objective",
     "Vital signs at this visit: body temperature 37.5°C, blood pressure 123/78 mmHg, heart rate 68/min, "
     "respiratory rate 14/min.",
     "Vital signs at this visit: body temperature 37.5°C. Vital signs 14 days earlier: blood pressure "
     "123/78 mmHg, heart rate 68/min, respiratory rate 14/min."),
    ("corpus", "viral_pharyngitis_e836f7dc-564f-43d1-50d6-48970f1508fe", "objective",
     "Vital signs at this visit: body temperature 37.3°C, heart rate 77/min, respiratory rate 12/min, "
     "systolic blood pressure 107 mmHg, diastolic blood pressure 77 mmHg.",
     "Vital signs at this visit: body temperature 37.3°C. Vital signs 8 days earlier: heart rate 77/min, "
     "respiratory rate 12/min, systolic blood pressure 107 mmHg, diastolic blood pressure 77 mmHg."),
    ("corpus", "viral_pharyngitis_e8fe4916-8fa7-a389-0996-c2dd9e406bf4", "objective",
     "Vital signs at this visit: body temperature 37.1 Cel, heart rate 89/min, respiratory rate 12/min, "
     "systolic blood pressure 106 mm[Hg], diastolic blood pressure 77 mm[Hg].",
     "Vital signs at this visit: body temperature 37.1 Cel. Vital signs 14 days earlier: heart rate "
     "89/min, respiratory rate 12/min, systolic blood pressure 106 mm[Hg], diastolic blood pressure 77 "
     "mm[Hg]."),
    ("corpus", "viral_pharyngitis_e9e5f364-d7c7-b5df-f462-e9002016be1e", "objective",
     "Vital signs at this visit: body temperature 37.9°C, heart rate 97/min, respiratory rate 14/min, "
     "blood pressure 103/86 mm[Hg], body weight 38.7 kg, body height 160.4 cm, BMI 15 kg/m2 (6.5th "
     "percentile for age and sex), pain severity score 3/10.",
     "Vital signs at this visit: body temperature 37.9°C. Vital signs 7 days earlier: heart rate 97/min, "
     "respiratory rate 14/min, blood pressure 103/86 mm[Hg], body weight 38.7 kg, body height 160.4 cm, "
     "BMI 15 kg/m2 (6.5th percentile for age and sex), pain severity score 3/10."),
    ("corpus", "viral_pharyngitis_ea9ad1f0-0a80-dfec-7805-8d128951ed8a", "objective",
     "Vital signs at this visit: body temperature 37.8°C, heart rate 89/min, respiratory rate 13/min, "
     "blood pressure 129/73 mmHg, BMI 29 kg/m2, body weight 71.1 kg, body height 156.6 cm.",
     "Vital signs at this visit: body temperature 37.8°C. Vital signs 3 months earlier: heart rate 89/min, "
     "respiratory rate 13/min, blood pressure 129/73 mmHg, BMI 29 kg/m2, body weight 71.1 kg, body height "
     "156.6 cm."),
    ("corpus", "viral_pharyngitis_eca391ff-8c80-afdc-c4de-621d115c2334", "objective",
     "Vital signs at this visit: body temperature 37.1°C, heart rate 67/min, respiratory rate 13/min, "
     "systolic blood pressure 132 mmHg, diastolic blood pressure 105 mmHg, body weight 91 kg, BMI 29.1 "
     "kg/m².",
     "Vital signs at this visit: body temperature 37.1°C. Vital signs 14 days earlier: heart rate 67/min, "
     "respiratory rate 13/min, systolic blood pressure 132 mmHg, diastolic blood pressure 105 mmHg, body "
     "weight 91 kg, BMI 29.1 kg/m²."),
    ("corpus", "viral_pharyngitis_ef33138a-8105-c72c-69c8-4ac9a80fcebe", "objective",
     "Vital signs at current visit: body temperature 37.5°C, heart rate 99/min, respiratory rate 14/min, "
     "systolic blood pressure 115 mmHg, diastolic blood pressure 65 mmHg, body weight 14.7 kg, body height "
     "101.7 cm, BMI 14.2 kg/m², BMI percentile 18.3%, pain severity score 4/10.",
     "Vital signs at current visit: body temperature 37.5°C. Vital signs 20 days earlier: heart rate "
     "99/min, respiratory rate 14/min, systolic blood pressure 115 mmHg, diastolic blood pressure 65 mmHg, "
     "body weight 14.7 kg, body height 101.7 cm, BMI 14.2 kg/m², BMI percentile 18.3%, pain severity score "
     "4/10."),
    ("corpus", "viral_pharyngitis_f1eeee4a-60df-744e-f1ea-3d3f98307826", "objective",
     "Vital signs at this visit: body temperature 37.1 °C, heart rate 99 /min, respiratory rate 13 /min, "
     "blood pressure 117/90 mmHg.",
     "Vital signs at this visit: body temperature 37.1 °C. Vital signs 7 days earlier: heart rate 99 /min, "
     "respiratory rate 13 /min, blood pressure 117/90 mmHg."),
    ("corpus", "viral_pharyngitis_f415855e-d98e-fe01-b07b-e906d1d89f21", "objective",
     "Vital signs at this visit: body temperature 37.0 °C, heart rate 95 /min, blood pressure 121/78 "
     "mm[Hg], respiratory rate 15 /min.",
     "Vital signs at this visit: body temperature 37.0 °C. Vital signs 14 days earlier: heart rate 95 "
     "/min, blood pressure 121/78 mm[Hg], respiratory rate 15 /min."),
    ("corpus", "viral_pharyngitis_f5a4c38c-94e9-5674-1ae0-b000fe3b5185", "objective",
     "Vital signs at this visit: body temperature 37.3°C, heart rate 78/min, respiratory rate 15/min, "
     "systolic blood pressure 105 mm[Hg], diastolic blood pressure 97 mm[Hg].",
     "Vital signs at this visit: body temperature 37.3°C. Vital signs 7 days earlier: heart rate 78/min, "
     "respiratory rate 15/min, systolic blood pressure 105 mm[Hg], diastolic blood pressure 97 mm[Hg]."),
    ("corpus", "viral_pharyngitis_f71656c6-251f-5351-2b66-2229668d8be9", "objective",
     "Vital signs at this visit: body temperature 37.2°C, heart rate 80/min, respiratory rate 13/min, "
     "blood pressure 129/76 mm[Hg], body weight 54.6 kg, body height 168.3 cm, BMI 19.3 kg/m2.",
     "Vital signs at this visit: body temperature 37.2°C. Vital signs 8 months earlier: heart rate 80/min, "
     "respiratory rate 13/min, blood pressure 129/76 mm[Hg], body weight 54.6 kg, body height 168.3 cm, "
     "BMI 19.3 kg/m2."),
    ("corpus", "viral_pharyngitis_f7edf79a-af78-a622-0514-76aff44014cb", "objective",
     "Vital signs at this visit: body temperature 37.2°C, heart rate 91/min, respiratory rate 14/min, "
     "blood pressure 117/83 mmHg.",
     "Vital signs at this visit: body temperature 37.2°C. Vital signs 14 days earlier: heart rate 91/min, "
     "respiratory rate 14/min, blood pressure 117/83 mmHg."),
    ("corpus", "viral_pharyngitis_fa16fa7a-ff3a-8d53-647d-aea4c8c8409f", "objective",
     "Vital signs at this visit: body temperature 37.7 °C, heart rate 61/min, respiratory rate 13/min, "
     "systolic blood pressure 135 mm[Hg], diastolic blood pressure 71 mm[Hg].",
     "Vital signs at this visit: body temperature 37.7 °C. Vital signs 14 days earlier: heart rate 61/min, "
     "respiratory rate 13/min, systolic blood pressure 135 mm[Hg], diastolic blood pressure 71 mm[Hg]."),
    ("corpus", "viral_pharyngitis_fab8e6e8-d8ee-1b79-963d-d60cc143da75", "objective",
     "Vital signs at this visit: body temperature 37.1°C, heart rate 75/min, respiratory rate 14/min, "
     "blood pressure 145/91 mmHg, pain severity 3/10, BMI 27.7 kg/m2, body weight 89.8 kg, body height "
     "180.2 cm.",
     "Vital signs at this visit: body temperature 37.1°C. Vital signs 29 days earlier: heart rate 75/min, "
     "respiratory rate 14/min, blood pressure 145/91 mmHg, pain severity 3/10, BMI 27.7 kg/m2, body weight "
     "89.8 kg, body height 180.2 cm."),
    ("corpus", "viral_pharyngitis_fb9dbc4c-0965-afcd-9a86-9f7889b471fb", "objective",
     "Vital signs at this visit: body temperature 37.3°C, heart rate 75/min, respiratory rate 15/min, "
     "blood pressure 123/72 mm[Hg], body weight 52.1 kg, height 167.7 cm, BMI 18.5 kg/m2.",
     "Vital signs at this visit: body temperature 37.3°C. Vital signs 2 months earlier: heart rate 75/min, "
     "respiratory rate 15/min, blood pressure 123/72 mm[Hg], body weight 52.1 kg, height 167.7 cm, BMI "
     "18.5 kg/m2."),
    ("corpus", "viral_pharyngitis_fcb17b88-9193-4fd1-7d14-5abd6711bc5b", "objective",
     "Vital signs at this visit: body temperature 37.5°C, heart rate 86/min, respiratory rate 14/min, "
     "systolic blood pressure 111 mmHg, diastolic blood pressure 74 mmHg.",
     "Vital signs at this visit: body temperature 37.5°C. Vital signs 14 days earlier: heart rate 86/min, "
     "respiratory rate 14/min, systolic blood pressure 111 mmHg, diastolic blood pressure 74 mmHg."),
    ("corpus", "viral_pharyngitis_fcd880d0-405a-3f0b-05df-ff1d365d59d5", "objective",
     "Vital signs at this visit: body temperature 37.0°C, heart rate 73/min, respiratory rate 15/min, "
     "systolic blood pressure 97 mmHg, diastolic blood pressure 79 mmHg.",
     "Vital signs at this visit: body temperature 37.0°C. Vital signs 21 days earlier: heart rate 73/min, "
     "respiratory rate 15/min, systolic blood pressure 97 mmHg, diastolic blood pressure 79 mmHg."),
    ("corpus", "viral_pharyngitis_fdda640f-68f7-1cd6-a1c8-a390b0bee1dd", "objective",
     "Vital signs at this visit: body temperature 37.3°C, heart rate 75/min, respiratory rate 14/min, "
     "systolic blood pressure 109 mmHg, diastolic blood pressure 73 mmHg.",
     "Vital signs at this visit: body temperature 37.3°C. Vital signs 7 days earlier: heart rate 75/min, "
     "respiratory rate 14/min, systolic blood pressure 109 mmHg, diastolic blood pressure 73 mmHg."),
    # dev_nota, second pass
    # earlier vitals under a "current day" header
    ("dev_nota", "nota_covid19_943b2c81-3683-310d-15d3-afa9452f0349", "objective",
     "Most recent vitals (current day): systolic BP 138 mmHg, diastolic BP 94 mmHg, heart rate 94 /min, "
     "respiratory rate 14 /min.",
     "Most recent vitals (current day): systolic BP 138 mmHg, diastolic BP 94 mmHg. Vital signs 30 days "
     "earlier: heart rate 94 /min, respiratory rate 14 /min."),
    # "most recent" on vitals older than the visit-line vitals
    ("dev_nota", "nota_covid19_7c74395b-4e20-9653-7fd6-66337e9d79c6", "objective",
     "Most recent measurements (14 days earlier):", "Measurements 14 days earlier:"),
    ("dev_nota", "nota_covid19_821ed16c-fb74-7131-c87f-e3cd2b08f332", "objective",
     "Most recent labs (14 days prior): systolic BP", "Vital signs 14 days prior: systolic BP"),
    # earlier values called "current"
    ("dev_nota", "nota_covid19_09be4fdf-8531-deb2-5855-48aa73f81637", "subjective",
     "Patient reports current pain severity of 3 out of 10.",
     "Patient reported pain severity of 3 out of 10 (2 months earlier)."),
    ("dev_nota", "nota_covid19_2dd3c425-de21-9fc4-3a83-b5bc23223908", "subjective",
     "Patient reports current pain severity of 2 out of 10.",
     "Patient reported pain severity of 2 out of 10 (1 day earlier)."),
    ("dev_nota", "nota_covid19_5b19cda8-3fec-a742-10f7-bf3a0fdc5a2d", "subjective",
     "Reports a current pain severity score of 2 out of 10.",
     "Reported a pain severity score of 2 out of 10 (14 days earlier)."),
    ("dev_nota", "nota_acute_bronchitis_0bf1c9ba-211d-7eea-76d1-de2a69feb896", "objective",
     "current BMI 27.8", "most recent BMI 27.8 (14 days earlier)"),
    ("dev_nota", "nota_covid19_b8d0e098-d211-c7cf-4024-0770f4df7d19", "objective",
     "current BMI 27.4", "most recent BMI 27.4 (3 months earlier)"),
    # invented or garbled time spans
    ("dev_nota", "nota_covid19_3053033e-ed36-4c69-d3f8-baeab8bc31c0", "objective",
     "Tdap and influenza vaccines administered approximately 3 months into prenatal care.",
     "Pertussis/tetanus/diphtheria vaccine administered 3 months earlier; Td and influenza immunizations "
     "1 year earlier."),
    ("dev_nota", "nota_viral_sinusitis_03e54fc2-5682-15ec-dfac-1dd652004930", "subjective",
     "since approximately age 16 years prior", "since approximately 16 years prior"),
    ("dev_nota", "nota_covid19_38795eda-d583-039d-1bc1-c170bfd271a0", "objective",
     "(5 ambulatory visits in past 14 days)", "(5 ambulatory visits, most recent 14 days earlier)"),
    # dev, second pass
    # scores, pain severity and BMI from an earlier date labelled as the current visit
    ("dev", "bacterial_sinusitis_d7d86f1c-9509-46e9-1d9d-dd363019a021", "subjective",
     "Patient-reported pain score is 3/10 at current visit",
     "Patient-reported pain score is 3/10 (14 days earlier)"),
    ("dev", "copd_8c792490-87ec-f222-7584-1375c14f5cae", "subjective",
     "Reports pain severity of 1/10 at current visit",
     "Reports pain severity of 1/10 (14 days earlier)"),
    ("dev", "copd_8c792490-87ec-f222-7584-1375c14f5cae", "subjective",
     "GAD-7 total score of 3 at current visit",
     "GAD-7 total score of 3 (14 days earlier)"),
    ("dev", "copd_8c792490-87ec-f222-7584-1375c14f5cae", "subjective",
     "DAST-10 score of 1 at current visit",
     "DAST-10 score of 1 (14 days earlier)"),
    ("dev", "copd_8d7dd72d-58e3-6337-7c6a-27ecb5e6aae4", "subjective",
     "Pain severity rated 2/10 at current visit",
     "Pain severity rated 2/10 (3 days earlier)"),
    ("dev", "copd_8d7dd72d-58e3-6337-7c6a-27ecb5e6aae4", "subjective",
     "PHQ-2 total score 2 at current visit",
     "PHQ-2 total score 2 (3 days earlier)"),
    ("dev", "copd_8d7dd72d-58e3-6337-7c6a-27ecb5e6aae4", "subjective",
     "DAST-10 total score 0 at current visit",
     "DAST-10 total score 0 (3 days earlier)"),
    ("dev", "cystitis_52dfb8b2-0bb4-7780-041f-a98faa0f1a3f", "subjective",
     "PHQ-2 total score 2 at current visit",
     "PHQ-2 total score 2 (14 days earlier)"),
    ("dev", "cystitis_587a3ab9-dc43-5bd5-a3d3-ca48710c97eb", "subjective",
     "Patient reports a pain severity score of 4 out of 10 at the current visit",
     "Patient reports a pain severity score of 4 out of 10 (5 months earlier)"),
    ("dev", "hypertension_062d2911-3634-d1f6-9822-f425efe9da5b", "subjective",
     "57-year-old male presenting with a reported pain severity score of 3 out of 10 at the current visit",
     "57-year-old male presenting with a reported pain severity score of 3 out of 10 (9 months earlier)"),
    ("dev", "hypertension_062d2911-3634-d1f6-9822-f425efe9da5b", "subjective",
     "PHQ-2 total score of 1 at current visit",
     "PHQ-2 total score of 1 (9 months earlier)"),
    ("dev", "hypertension_3dc7cd12-84b0-ad27-e6f3-db021f737ae9", "subjective",
     "Reports pain severity of 1 out of 10 at current visit",
     "Reports pain severity of 1 out of 10 (2 months earlier)"),
    ("dev", "hypertension_5217d321-8186-f43a-703e-fce08b87071f", "subjective",
     "Reports pain severity of 2 out of 10 at current visit",
     "Reports pain severity of 2 out of 10 (3 months earlier)"),
    ("dev", "hypertension_839ea010-a89f-8e46-b2f4-d7bcbe652c04", "subjective",
     "Reports pain severity of 1/10 at current visit",
     "Reports pain severity of 1/10 (7 months earlier)"),
    ("dev", "hypertension_9c1e58eb-4f67-6d81-1458-5c205dad02e8", "objective",
     "BMI 28 kg/m² at current visit",
     "BMI 28 kg/m² (5 months earlier)"),
    ("dev", "hypertension_9c1e58eb-4f67-6d81-1458-5c205dad02e8", "subjective",
     "Reports pain severity of 2 out of 10 at current visit",
     "Reports pain severity of 2 out of 10 (5 months earlier)"),
    ("dev", "hypertension_9c1e58eb-4f67-6d81-1458-5c205dad02e8", "subjective",
     "PHQ-2 score of 2 at current visit",
     "PHQ-2 score of 2 (5 months earlier)"),
    ("dev", "hypertension_af1cc7a8-b30d-9ddd-0130-2cf9e1ea550a", "subjective",
     "Patient reports a pain severity score of 4 out of 10 at the current visit",
     "Patient reports a pain severity score of 4 out of 10 (10 months earlier)"),
    ("dev", "hypertension_c2831bc0-6f38-f2e3-ad22-9052758e733e", "subjective",
     "Reports pain severity of 2/10 (scale 0–10) at current visit",
     "Reports pain severity of 2/10 (scale 0–10, 28 days earlier)"),
    ("dev", "hypertension_c2831bc0-6f38-f2e3-ad22-9052758e733e", "subjective",
     "PHQ-2 total score of 1 at current visit",
     "PHQ-2 total score of 1 (28 days earlier)"),
    ("dev", "hypertension_c2831bc0-6f38-f2e3-ad22-9052758e733e", "subjective",
     "DAST-10 total score of 1 at current visit",
     "DAST-10 total score of 1 (28 days earlier)"),
    ("dev", "hypertension_d68a1c13-88f4-f94e-b1d5-c930db741575", "subjective",
     "Reports pain severity of 3 out of 10 at current visit",
     "Reports pain severity of 3 out of 10 (9 months earlier)"),
    ("dev", "hypertension_dcd21f58-37dc-b7f4-03f3-161161e7bfe6", "subjective",
     "PHQ-2 total score of 1 at current visit",
     "PHQ-2 total score of 1 (10 months earlier)"),
    ("dev", "hypertension_eb2d403a-b9a0-062d-a73f-44e9f5bfd3cd", "subjective",
     "Reports a pain severity score of 3 out of 10 at this visit",
     "Reports a pain severity score of 3 out of 10 (2 days earlier)"),
    ("dev", "strep_throat_03172307-0082-3de1-9a48-4e2f3aacd61c", "subjective",
     "Pain severity rated 4/10 at current visit",
     "Pain severity rated 4/10 (15 days earlier)"),
    ("dev", "strep_throat_13f7cda5-2a12-1df6-32c1-d89ccbf28d06", "subjective",
     "Pain severity reported as 3 out of 10 at current visit",
     "Pain severity reported as 3 out of 10 (14 days earlier)"),
    ("dev", "strep_throat_4b18aff0-57fd-144e-7153-9755e365fab6", "objective",
     "Pain severity reported 3/10 at current visit",
     "Pain severity reported 3/10 (3 days earlier)"),
    ("dev", "strep_throat_4b18aff0-57fd-144e-7153-9755e365fab6", "objective",
     "PHQ-2 total score 2 at current visit",
     "PHQ-2 total score 2 (3 days earlier)"),
    ("dev", "strep_throat_4d01e7ca-2654-a3b2-b660-2173402feeb5", "objective",
     "Pain severity reported 3/10 (current visit)",
     "Pain severity reported 3/10 (14 days earlier)"),
    ("dev", "strep_throat_78c058e0-af9b-a008-d50e-3d95f0f4a901", "subjective",
     "Pain severity reported as 4/10 at current visit",
     "Pain severity reported as 4/10 (14 days earlier)"),
    ("dev", "strep_throat_8ede6b48-d752-1f9f-ca54-8a9bd0652f75", "subjective",
     "GAD-7 total score of 4 at this visit",
     "GAD-7 total score of 4 (5 days earlier)"),
    ("dev", "strep_throat_ac91a689-860e-b86b-38fd-2b2395a25aeb", "subjective",
     "Reports pain severity of 3/10 at current visit",
     "Reports pain severity of 3/10 (14 days earlier)"),
    ("dev", "strep_throat_ac91a689-860e-b86b-38fd-2b2395a25aeb", "subjective",
     "PHQ-2 score 1 at current visit",
     "PHQ-2 score 1 (14 days earlier)"),
    ("dev", "strep_throat_af1d5990-d17c-3cdf-7a65-971fd4fbfd3d", "subjective",
     "Pain severity self-reported as 1/10 at current visit",
     "Pain severity self-reported as 1/10 (7 days earlier)"),
    ("dev", "strep_throat_f484f5a6-d76e-805a-4987-b0d39424b2ed", "subjective",
     "PHQ-2 score of 2 at current visit",
     "PHQ-2 score of 2 (8 days earlier)"),
    ("dev", "strep_throat_f484f5a6-d76e-805a-4987-b0d39424b2ed", "subjective",
     "DAST-10 score of 0 at current visit",
     "DAST-10 score of 0 (8 days earlier)"),
    ("dev", "viral_pharyngitis_0361b700-8737-5414-43c7-cce67fadc06a", "subjective",
     "Pain severity self-reported as 2/10 at current visit",
     "Pain severity self-reported as 2/10 (15 days earlier)"),
    ("dev", "viral_pharyngitis_08370a81-ba03-3bc0-ff33-0d61f18c15c0", "subjective",
     "Pain severity reported as 4/10 at current visit",
     "Pain severity reported as 4/10 (25 days earlier)"),
    ("dev", "viral_pharyngitis_18db769d-0497-35af-02fa-27469736ba13", "subjective",
     "Reported pain severity 3/10 at current visit",
     "Reported pain severity 3/10 (1 day earlier)"),
    ("dev", "viral_pharyngitis_21a880c2-d923-c6b9-4900-447beaede194", "subjective",
     "Pain severity reported as 0/10 at this visit",
     "Pain severity reported as 0/10 (1 day earlier)"),
    ("dev", "viral_pharyngitis_2dd368ce-54eb-b9f7-73b0-ca065e498ca9", "subjective",
     "Pain severity rated 2/10 at current visit",
     "Pain severity rated 2/10 (14 days earlier)"),
    ("dev", "viral_pharyngitis_3e5eeccf-2156-0d14-7d92-7e1a3a945674", "subjective",
     "Pain severity rated 4 out of 10 at current visit",
     "Pain severity rated 4 out of 10 (14 days earlier)"),
    ("dev", "viral_pharyngitis_49f7afb0-3294-0443-56c0-ab13c4b78edb", "subjective",
     "Pain severity reported as 3/10 at current visit",
     "Pain severity reported as 3/10 (14 days earlier)"),
    ("dev", "viral_pharyngitis_51d69a38-57ae-47c4-5529-6191f80947f5", "subjective",
     "Pain severity reported as 1/10 at current visit",
     "Pain severity reported as 1/10 (3 days earlier)"),
    ("dev", "viral_pharyngitis_5435c679-29d8-dfe5-efb5-1f8832779f60", "subjective",
     "Pain severity rated 4/10 at this visit",
     "Pain severity rated 4/10 (7 days earlier)"),
    ("dev", "viral_pharyngitis_5435c679-29d8-dfe5-efb5-1f8832779f60", "subjective",
     "PHQ-2 score of 2 at this visit",
     "PHQ-2 score of 2 (7 days earlier)"),
    ("dev", "viral_pharyngitis_5ea7dc74-2259-6089-019f-936c181bc81a", "subjective",
     "Patient reports pain severity of 4/10 at current visit",
     "Patient reports pain severity of 4/10 (14 days earlier)"),
    ("dev", "viral_pharyngitis_5ea7dc74-2259-6089-019f-936c181bc81a", "subjective",
     "PHQ-2 total score of 5 and PHQ-9 total score of 3 recorded at current visit",
     "PHQ-2 total score of 5 and PHQ-9 total score of 3 recorded 14 days earlier"),
    ("dev", "viral_pharyngitis_603ec245-5b3e-761b-2af9-2e675375a660", "subjective",
     "PHQ-2 score of 2 at current visit",
     "PHQ-2 score of 2 (14 days earlier)"),
    ("dev", "viral_pharyngitis_603ec245-5b3e-761b-2af9-2e675375a660", "subjective",
     "DAST-10 score of 2 at current visit",
     "DAST-10 score of 2 (14 days earlier)"),
    ("dev", "viral_pharyngitis_65705095-c976-63c6-6e0c-4e8a71d04079", "subjective",
     "Patient reports pain severity of 3 out of 10 at this visit",
     "Patient reports pain severity of 3 out of 10 (15 days earlier)"),
    ("dev", "viral_pharyngitis_65705095-c976-63c6-6e0c-4e8a71d04079", "subjective",
     "PHQ-2 total score of 1 at this visit",
     "PHQ-2 total score of 1 (15 days earlier)"),
    ("dev", "viral_pharyngitis_8207aeaf-ece7-d24a-e2a9-8a0788ac28ef", "subjective",
     "Patient reports pain severity of 2 out of 10 at current visit",
     "Patient reports pain severity of 2 out of 10 (14 days earlier)"),
    ("dev", "viral_pharyngitis_839ea010-a89f-8e46-b2f4-d7bcbe652c04", "objective",
     "PHQ-2 score 0 at current visit",
     "PHQ-2 score 0 (7 days earlier)"),
    ("dev", "viral_pharyngitis_8f5f0f25-c3fb-b1c4-3bd7-cbf735dc378b", "subjective",
     "Pain severity reported as 0/10 at current visit",
     "Pain severity reported as 0/10 (7 days earlier)"),
    ("dev", "viral_pharyngitis_9128304a-447d-bd98-110c-7f6d53d7f004", "subjective",
     "Pain severity reported as 1/10 at current visit",
     "Pain severity reported as 1/10 (14 days earlier)"),
    ("dev", "viral_pharyngitis_9263eeea-78af-23a4-fea6-bbe3a158a986", "subjective",
     "GAD-7 total score of 14 at current visit",
     "GAD-7 total score of 14 (14 days earlier)"),
    ("dev", "viral_pharyngitis_9263eeea-78af-23a4-fea6-bbe3a158a986", "subjective",
     "PHQ-2 total score of 0 at current visit",
     "PHQ-2 total score of 0 (14 days earlier)"),
    ("dev", "viral_pharyngitis_94581533-73da-a338-81ce-99aea0d88c33", "objective",
     "DAST-10 total score 1 (current)",
     "DAST-10 total score 1 (7 months earlier)"),
    ("dev", "viral_pharyngitis_94835e3b-bb6a-19da-b219-a114013aa086", "objective",
     "PHQ-Teen score 3 (current visit)",
     "PHQ-Teen score 3 (21 days earlier)"),
    ("dev", "viral_pharyngitis_94f9dbd9-7230-c137-e37f-bf29f41bcbaf", "objective",
     "PHQ-2 score 0 (current)",
     "PHQ-2 score 0 (7 days earlier)"),
    ("dev", "viral_pharyngitis_b5862dde-44d1-bffd-fc90-8e15d5ecffe5", "subjective",
     "Pain severity reported as 4/10 at current visit",
     "Pain severity reported as 4/10 (24 days earlier)"),
    ("dev", "viral_pharyngitis_b955c3c8-0af4-7148-114f-0b9b0986457d", "objective",
     "Morse Fall Scale score 4 (current visit)",
     "Morse Fall Scale score 4 (2 months earlier)"),
    ("dev", "viral_pharyngitis_b955c3c8-0af4-7148-114f-0b9b0986457d", "objective",
     "DAST-10 score 0 (current visit)",
     "DAST-10 score 0 (2 months earlier)"),
    ("dev", "viral_pharyngitis_bc27a7ab-93ff-b06f-6027-c0533e0f40e7", "subjective",
     "Pain severity rated 2/10 at this visit",
     "Pain severity rated 2/10 (14 days earlier)"),
    ("dev", "viral_pharyngitis_bc27a7ab-93ff-b06f-6027-c0533e0f40e7", "subjective",
     "PHQ-2 score 0 at this visit",
     "PHQ-2 score 0 (14 days earlier)"),
    ("dev", "viral_pharyngitis_c0d58c55-c55d-174c-076c-7e3deb6d6214", "subjective",
     "Pain severity reported as 3/10 at current visit",
     "Pain severity reported as 3/10 (1 day earlier)"),
    ("dev", "viral_pharyngitis_c2831bc0-6f38-f2e3-ad22-9052758e733e", "subjective",
     "Reports a pain severity score of 3 out of 10 at current visit",
     "Reports a pain severity score of 3 out of 10 (7 days earlier)"),
    ("dev", "viral_pharyngitis_cdba2e5c-b5e2-0dc2-0317-f9f6bf5efd84", "subjective",
     "Reported pain severity score 2/10 at current visit",
     "Reported pain severity score 2/10 (14 days earlier)"),
    ("dev", "viral_pharyngitis_cdba2e5c-b5e2-0dc2-0317-f9f6bf5efd84", "subjective",
     "PHQ-2 score 0 at current visit",
     "PHQ-2 score 0 (14 days earlier)"),
    ("dev", "viral_pharyngitis_d0a77214-6eac-5cea-a0dd-0cbb99a1335c", "subjective",
     "Pain severity rated 1/10 at current visit",
     "Pain severity rated 1/10 (3 months earlier)"),
    ("dev", "viral_pharyngitis_d88c5636-cd71-ce55-f532-157b79d307e5", "objective",
     "Pain severity reported 3/10 (0–10 scale) at current visit",
     "Pain severity reported 3/10 (0–10 scale, 1 day earlier)"),
    ("dev", "viral_pharyngitis_da7d55c8-6b5a-ce4a-ee55-da834438a2e8", "subjective",
     "Pain severity rated 4/10 at current visit",
     "Pain severity rated 4/10 (6 days earlier)"),
    ("dev", "viral_pharyngitis_dc622d90-707f-6d94-46a0-f42a3feeec52", "subjective",
     "Pain severity rated 2/10 at this visit",
     "Pain severity rated 2/10 (7 days earlier)"),
    ("dev", "viral_pharyngitis_e11bb615-89e9-2a1a-d67b-4c8b7a27aa27", "subjective",
     "PHQ-2 score of 2 at current visit",
     "PHQ-2 score of 2 (1 day earlier)"),
    ("dev", "viral_pharyngitis_ede73cfb-5795-1c75-ac80-fb1fd13eeaaf", "subjective",
     "Pain severity self-reported as 3/10 at current visit",
     "Pain severity self-reported as 3/10 (15 days earlier)"),
    ("dev", "viral_pharyngitis_f304a5fa-2b0a-157b-1b5e-6c7429b33d69", "subjective",
     "DAST-10 score of 2 at current visit",
     "DAST-10 score of 2 (7 days earlier)"),
    ("dev", "viral_pharyngitis_f304a5fa-2b0a-157b-1b5e-6c7429b33d69", "subjective",
     "PHQ-2 score of 4 and PHQ-9 score of 25 at current visit",
     "PHQ-2 score of 4 and PHQ-9 score of 25 (7 days earlier)"),
    ("dev", "viral_pharyngitis_f304a5fa-2b0a-157b-1b5e-6c7429b33d69", "subjective",
     "HARK score of 1 at current visit",
     "HARK score of 1 (7 days earlier)"),
    ("dev", "viral_pharyngitis_f8917d08-e945-4af8-233c-c9da843b662d", "subjective",
     "Pain severity reported as 1/10 at current visit",
     "Pain severity reported as 1/10 (14 days earlier)"),
    ("dev", "viral_pharyngitis_f8917d08-e945-4af8-233c-c9da843b662d", "subjective",
     "PHQ-2 score of 5 and PHQ-9 score of 17 reported at current visit",
     "PHQ-2 score of 5 and PHQ-9 score of 17 reported 14 days earlier"),
    ("dev", "viral_pharyngitis_fb6300fb-4251-947e-0683-48bafdde92fd", "subjective",
     "GAD-7 total score of 3 at current visit",
     "GAD-7 total score of 3 (14 days earlier)"),
    ("dev", "viral_pharyngitis_fb6300fb-4251-947e-0683-48bafdde92fd", "subjective",
     "PHQ-2 total score of 2 at current visit",
     "PHQ-2 total score of 2 (14 days earlier)"),
    ("dev", "viral_pharyngitis_fb6300fb-4251-947e-0683-48bafdde92fd", "subjective",
     "AUDIT-C total score of 0 at current visit",
     "AUDIT-C total score of 0 (14 days earlier)"),
    ("dev", "viral_pharyngitis_fe0c2fe9-339e-47f6-6e0e-196723f76209", "subjective",
     "Patient reports pain severity 2/10 at this visit",
     "Patient reports pain severity 2/10 (7 days earlier)"),
    # earlier-dated measurements under a "current" header
    ("dev", "strep_throat_13f7cda5-2a12-1df6-32c1-d89ccbf28d06", "objective",
     "Current measurements: height 121.5 cm",
     "Measurements 14 days earlier: height 121.5 cm"),
    ("dev", "strep_throat_4d01e7ca-2654-a3b2-b660-2173402feeb5", "objective",
     "Current CBC: hemoglobin 12.4 g/dL",
     "CBC 14 days earlier: hemoglobin 12.4 g/dL"),
    ("dev", "strep_throat_6eaeef11-3e0b-09f4-42d6-ea3d6fecea6b", "objective",
     "Current measurements: height 84.9 cm",
     "Measurements 5 days earlier: height 84.9 cm"),
    ("dev", "strep_throat_f484f5a6-d76e-805a-4987-b0d39424b2ed", "objective",
     "Current labs: blood glucose 80 mg/dL",
     "Labs 8 days earlier: blood glucose 80 mg/dL"),
    ("dev", "viral_pharyngitis_14d5a418-1a57-464d-6b94-fc1735ecaae6", "objective",
     "Labs (current visit): glucose 98.2 mg/dL",
     "Labs 1 day earlier: glucose 98.2 mg/dL"),
    ("dev", "viral_pharyngitis_51d69a38-57ae-47c4-5529-6191f80947f5", "objective",
     "Current measurements: height 132.4 cm",
     "Measurements 3 days earlier: height 132.4 cm"),
    ("dev", "viral_pharyngitis_9ceddcb0-07b0-800f-baf4-3c3f8c8d0d32", "objective",
     "Current anthropometrics: height 86.4 cm",
     "Anthropometrics 24 days earlier: height 86.4 cm"),
    ("dev", "viral_pharyngitis_cfc8f8a7-45cc-abf8-7b4f-48c55dcf0ab2", "objective",
     "Current anthropometrics: height 90.3 cm",
     "Anthropometrics 24 days earlier: height 90.3 cm"),
    ("dev", "viral_pharyngitis_9d0b10c9-9b55-0ac1-41f3-d859f5d15864", "objective",
     "Current measurements: height 172.4 cm",
     "Measurements 1 day earlier: height 172.4 cm"),
    ("dev", "viral_pharyngitis_9d0b10c9-9b55-0ac1-41f3-d859f5d15864", "objective",
     "Labs (current visit): calcium 9.3 mg/dL",
     "Labs 1 day earlier: calcium 9.3 mg/dL"),
    ("dev", "viral_pharyngitis_b06de613-7e83-c236-eecf-0e5eb436689a", "objective",
     "Lab values (current visit): glucose 72.3 mg/dL",
     "Lab values 1 day earlier: glucose 72.3 mg/dL"),
    ("dev", "bacterial_sinusitis_3dbee36a-f698-f348-1193-616f6b855160", "objective",
     "Vitals (current visit): Blood pressure 99/72 mmHg",
     "Vitals 1 day earlier: Blood pressure 99/72 mmHg"),
    ("dev", "viral_pharyngitis_21a880c2-d923-c6b9-4900-447beaede194", "objective",
     "Leukocytes 9.7 10³/µL (current visit)",
     "Leukocytes 9.7 10³/µL (1 day earlier)"),
    ("dev", "viral_pharyngitis_d5a42573-f20f-cf34-8897-35516adf366e", "objective",
     "BMI 28.2 kg/m² (current)",
     "BMI 28.2 kg/m² (14 days earlier)"),
    ("dev", "viral_pharyngitis_36d23752-9ad8-68fb-e503-b67d48064c9b", "objective",
     "PHQ-9 modified for teens score 2 (current)",
     "PHQ-9 modified for teens score 2 (14 days earlier)"),
    ("dev", "viral_pharyngitis_5ea7dc74-2259-6089-019f-936c181bc81a", "objective",
     "Morse Fall Scale score: 8 (current visit)",
     "Morse Fall Scale score: 8 (14 days earlier)"),
    ("dev", "viral_pharyngitis_803b4f9f-6606-5468-5f14-f72689ffa88a", "objective",
     "DAST-10 score: 0 (current visit). Pain severity (0–10): 0 (current visit)",
     "DAST-10 score: 0 (7 days earlier). Pain severity (0–10): 0 (7 days earlier)"),
    ("dev", "viral_pharyngitis_ec990c7d-d8e7-ec8a-69a2-6812ad429563", "subjective",
     "Pain severity reported as 0 at this visit.",
     "Pain severity reported as 0 (14 days earlier)."),
    ("dev", "cystitis_d2fe2a28-5cce-e949-7e08-7fd689f44e94", "subjective",
     "PHQ-2 score of 1 at current visit (previously 5 five months ago)",
     "PHQ-2 score of 1 (28 days earlier; previously 5 five months ago)"),
    # "current visit" vital signs that mix in older vitals: visit-day value keeps the header
    ("dev", "viral_pharyngitis_18db769d-0497-35af-02fa-27469736ba13", "objective",
     "Vitals (current visit): temperature 37.4°C, heart rate 83/min,",
     "Vitals (current visit): temperature 37.4°C. Vitals 1 day earlier: heart rate 83/min,"),
    ("dev", "viral_pharyngitis_21a880c2-d923-c6b9-4900-447beaede194", "objective",
     "Vital signs at this visit: body temperature 38.0°C, heart rate 66/min,",
     "Vital signs at this visit: body temperature 38.0°C. Vital signs 1 day earlier: heart rate 66/min,"),
    ("dev", "viral_pharyngitis_39e2e8d9-7107-edc1-1bd0-60a384c45cfb", "objective",
     "Vital signs at current urgent care visit: temperature 37.9°C, heart rate 97/min,",
     "Vital signs at current visit: temperature 37.9°C. Vital signs at urgent care visit 14 days earlier: heart rate 97/min,"),
    ("dev", "viral_pharyngitis_fac577f4-7739-a049-aa4e-d2e233fd8c92", "objective",
     "Vitals at current urgent care visit: temperature 37.3°C, heart rate 94/min,",
     "Vitals at current visit: temperature 37.3°C. Vitals at urgent care visit 15 days earlier: heart rate 94/min,"),
    ("dev", "viral_pharyngitis_4bce846a-f09b-9807-9f58-f095343b5ca8", "objective",
     "Vital signs at current visit (3 days prior to record date): temperature 37.3°C, heart rate 91/min,",
     "Vital signs at current visit: temperature 37.3°C. Vital signs 3 days earlier: heart rate 91/min,"),
    ("dev", "viral_pharyngitis_96ef3911-6d88-f5b8-0f90-009a988e1ebf", "objective",
     "Vital signs at current visit: body temperature 37.2°C, heart rate 72/min,",
     "Vital signs at current visit: body temperature 37.2°C. Vital signs 1 day earlier: heart rate 72/min,"),
    ("dev", "viral_pharyngitis_9d0b10c9-9b55-0ac1-41f3-d859f5d15864", "objective",
     "Vital signs at current visit: body temperature 37.8°C, blood pressure 109/78 mmHg,",
     "Vital signs at current visit: body temperature 37.8°C. Vital signs 1 day earlier: blood pressure 109/78 mmHg,"),
    ("dev", "viral_pharyngitis_a8584e96-f522-c166-e4eb-8f7efde95e79", "objective",
     "Vital signs at current visit: temperature 37.6°C, blood pressure 128/63 mmHg,",
     "Vital signs at current visit: temperature 37.6°C. Vital signs 1 day earlier: blood pressure 128/63 mmHg,"),
    ("dev", "viral_pharyngitis_b014f376-c7a0-5c0e-7b83-9fbb834826aa", "objective",
     "Vital signs at current visit: body temperature 37.6°C, blood pressure 122/95 mmHg,",
     "Vital signs at current visit: body temperature 37.6°C. Vital signs 1 day earlier: blood pressure 122/95 mmHg,"),
    ("dev", "viral_pharyngitis_b06de613-7e83-c236-eecf-0e5eb436689a", "objective",
     "Vital signs at current visit: temperature 37.5°C, heart rate 78/min,",
     "Vital signs at current visit: temperature 37.5°C. Vital signs 1 day earlier: heart rate 78/min,"),
    ("dev", "viral_pharyngitis_b89fd874-2edb-bc10-ce8c-fc27d6dea158", "objective",
     "Vital signs at this visit: body temperature 37.3°C, heart rate 81/min,",
     "Vital signs at this visit: body temperature 37.3°C. Vital signs 1 day earlier: heart rate 81/min,"),
    ("dev", "viral_pharyngitis_c0d58c55-c55d-174c-076c-7e3deb6d6214", "objective",
     "Vital signs at current visit: body temperature 37.9°C, heart rate 77/min,",
     "Vital signs at current visit: body temperature 37.9°C. Vital signs 1 day earlier: heart rate 77/min,"),
    ("dev", "viral_pharyngitis_e11bb615-89e9-2a1a-d67b-4c8b7a27aa27", "objective",
     "Vital signs at current visit: body temperature 37.4°C (per presenting complaint field); heart rate 86/min;",
     "Vital signs at current visit: body temperature 37.4°C. Vital signs 1 day earlier: heart rate 86/min;"),
    ("dev", "viral_pharyngitis_f50b0a20-6cd0-f47f-2cc3-73ce4ce75a00", "objective",
     "Vital signs at current visit: temperature 37.4°C, heart rate 61/min,",
     "Vital signs at current visit: temperature 37.4°C. Vital signs 1 day earlier: heart rate 61/min,"),
    # diagnosis-day temperature labelled as an earlier visit's vital sign
    ("dev", "viral_pharyngitis_9339a434-6636-9a3b-698f-e9780deb738a", "objective",
     "Vital signs (7 days prior urgent care visit): temperature 37.9°C, blood pressure",
     "Body temperature at this visit: 37.9°C. Vital signs (7 days prior urgent care visit): blood pressure"),
    ("dev", "viral_pharyngitis_fdcac022-0492-018f-fa56-e0ddec2ba204", "objective",
     "Vital signs (7 days prior urgent care visit): temperature 37.9°C, heart rate",
     "Body temperature at this visit: 37.9°C. Vital signs (7 days prior urgent care visit): heart rate"),
    # other values given the wrong date
    ("dev", "viral_pharyngitis_95280113-0540-6a60-ed20-5de605f9428c", "objective",
     "GAD-7 score 3, PHQ-2 score 1, AUDIT-C score 1, DAST-10 score 0, HARK score 0 (all 10 months prior)",
     "GAD-7 score 3, PHQ-2 score 1, AUDIT-C score 1 (all 10 months prior); DAST-10 score 0, HARK score 0 (both 1 year prior)"),
    ("dev", "hypertension_fb4c9c92-f70e-49ab-1339-d178d2788067", "objective",
     "height 159.9 cm; BMI 30.1 kg/m² (1 year prior)",
     "height 159.9 cm (1 year prior); BMI 30.1 kg/m² (1 year prior)"),
    ("dev", "viral_pharyngitis_d1180c81-3f47-aa86-9d76-5637da1a2593", "objective",
     "body weight 74.2 kg, height 162 cm, BMI 27.7 kg/m².",
     "body weight 74.2 kg. Height 162 cm, BMI 27.7 kg/m² (12 months earlier)."),
    ("dev", "viral_pharyngitis_b63294d9-2c9c-2937-0cd8-9375b2f6a5a3", "subjective",
     "tooth extraction and denture fitting (approximately 11 months earlier)",
     "tooth extraction (approximately 11 months earlier) and denture fitting (approximately 10 months earlier)"),
    # time span the source never gives (dates are 0 and 1 day earlier)
    ("dev", "viral_pharyngitis_076ad5bb-9279-f3a0-6e7b-f7a707c2f0fa", "subjective",
     "Recent medications within the past 1–2 days included ticagrelor, reteplase, nitroglycerin, heparin sodium, "
     "aspirin 325 mg, and morphine sulfate injection.",
     "Recent medications included ticagrelor (current day) and reteplase, nitroglycerin, heparin sodium, aspirin "
     "325 mg, and morphine sulfate injection (1 day earlier)."),
    ("dev", "viral_pharyngitis_076ad5bb-9279-f3a0-6e7b-f7a707c2f0fa", "objective",
     "Emergency and inpatient visits within the past 1–2 days.",
     "Emergency and inpatient visits 1 day earlier."),
    ("dev", "viral_pharyngitis_ac9b55d9-6e54-b839-51f8-737e3fca2c31", "objective",
     "Recent procedures include dental restoration, oral health education, supragingival and subgingival scaling, "
     "gingival examination, and dental X-rays (within the past 1–2 days).",
     "Recent procedures include dental restoration and oral health education (current visit), and supragingival and "
     "subgingival scaling, gingival examination, and dental X-rays (1 day earlier)."),
    ("dev", "viral_pharyngitis_b5c80e84-e3be-de07-11e0-bb10be4a9ba5", "objective",
     "Recent dental procedures performed within the past 1–2 days.",
     "Recent dental procedures performed 1 day earlier, with oral health education at the current visit."),
    # severity given a 0-10 scale the source doesn't use (cough severity is 78)
    ("dev", "copd_2bb7b7b9-cf12-f3b4-4dc0-962cd405acbd", "subjective",
     "shortness of breath (severity 9/10)",
     "shortness of breath (severity 9)"),
    # corpus, second pass
    # "current visit" vital signs that mix in older vitals
    ("corpus", "strep_throat_40675a01-3dbc-03a2-29a8-9653b58c545e", "objective",
     "Vital signs at this visit: body temperature 38.0 Cel, heart rate 90 /min, blood pressure 110/77 mm[Hg], respiratory rate 16 /min.",
     "Vital signs at this visit: body temperature 38.0 Cel. Vital signs 1 day earlier: heart rate 90 /min, blood pressure 110/77 mm[Hg], respiratory rate 16 /min."),
    ("corpus", "viral_pharyngitis_081530ca-f4dd-c439-c617-102e98306b8e", "objective",
     "Vital signs at this visit: body temperature 37.9°C, heart rate 76/min, respiratory rate 14/min, systolic blood pressure 98 mmHg, diastolic blood pressure 85 mmHg.",
     "Vital signs at this visit: body temperature 37.9°C. Vital signs 1 day earlier: heart rate 76/min, respiratory rate 14/min, systolic blood pressure 98 mmHg, diastolic blood pressure 85 mmHg."),
    ("corpus", "viral_pharyngitis_1808d3eb-adb5-8424-6f9f-871bf1b5b5d1", "objective",
     "Vital signs at this visit: body temperature 37.2°C, heart rate 78/min, respiratory rate 12/min, blood pressure 120/76 mmHg, body weight 85.5 kg, height 174.5 cm, BMI 28.1 kg/m2.",
     "Vital signs at this visit: body temperature 37.2°C. Vital signs 1 day earlier: heart rate 78/min, respiratory rate 12/min, blood pressure 120/76 mmHg, body weight 85.5 kg, height 174.5 cm, BMI 28.1 kg/m2."),
    ("corpus", "viral_pharyngitis_63985fdd-d734-1be7-f620-6f474e2f609b", "objective",
     "Vital signs at this visit: body temperature 37.3 °C, heart rate 92/min, respiratory rate 15/min, systolic blood pressure 110 mmHg, diastolic blood pressure 66 mmHg.",
     "Vital signs at this visit: body temperature 37.3 °C. Vital signs 1 day earlier: heart rate 92/min, respiratory rate 15/min, systolic blood pressure 110 mmHg, diastolic blood pressure 66 mmHg."),
    ("corpus", "viral_pharyngitis_73e0d879-791b-96c8-bfa8-94d22d7577dd", "objective",
     "Vital signs at this visit: body temperature 37.1 °C, heart rate 64/min, respiratory rate 12/min, blood pressure 127/82 mmHg, body weight 79.9 kg, height 164.8 cm, BMI 29.4 kg/m².",
     "Vital signs at this visit: body temperature 37.1 °C. Vital signs 1 day earlier: heart rate 64/min, respiratory rate 12/min, blood pressure 127/82 mmHg, body weight 79.9 kg, height 164.8 cm, BMI 29.4 kg/m²."),
    ("corpus", "viral_pharyngitis_881e0ab0-a2f1-551a-8aa0-2223f7a9b76a", "objective",
     "Vital signs at this visit: body temperature 37.9 °C, heart rate 72/min, respiratory rate 14/min, systolic blood pressure 146 mmHg, diastolic blood pressure 89 mmHg.",
     "Vital signs at this visit: body temperature 37.9 °C. Vital signs 1 day earlier: heart rate 72/min, respiratory rate 14/min, systolic blood pressure 146 mmHg, diastolic blood pressure 89 mmHg."),
    ("corpus", "viral_pharyngitis_a5066c40-d53b-cbbb-4df3-fca825267a20", "objective",
     "Vital signs at this visit: body temperature 37.9°C, heart rate 88/min, respiratory rate 16/min, systolic blood pressure 109 mmHg, diastolic blood pressure 76 mmHg.",
     "Vital signs at this visit: body temperature 37.9°C. Vital signs 1 day earlier: heart rate 88/min, respiratory rate 16/min, systolic blood pressure 109 mmHg, diastolic blood pressure 76 mmHg."),
    ("corpus", "viral_pharyngitis_d7e7f221-271c-b133-61e5-12e1e3ee546f", "objective",
     "Vital signs at this visit: body temperature 37.5°C, heart rate 84/min, respiratory rate 13/min, systolic blood pressure 121 mmHg, diastolic blood pressure 95 mmHg.",
     "Vital signs at this visit: body temperature 37.5°C. Vital signs 1 day earlier: heart rate 84/min, respiratory rate 13/min, systolic blood pressure 121 mmHg, diastolic blood pressure 95 mmHg."),
    # earlier values called "current" / "at this visit"
    ("corpus", "hypertension_66333eb6-f0bb-f425-47fb-6bbec5ade727", "objective",
     "body weight 71 kg, BMI 30.3 kg/m2.", "body weight 71 kg. BMI 30.3 kg/m2 (11 months earlier)."),
    ("corpus", "hypertension_260e3a24-e40a-1908-4ff0-80c4ddf975b3", "objective",
     "current BMI 30.3 kg/m2", "most recent BMI 30.3 kg/m2, 5 months earlier"),
    ("corpus", "strep_throat_38b41ee8-f276-527a-4e87-3d210ab332d9", "objective",
     "glucose 86 mg/dL, HbA1c 6.4%, sodium", "glucose 86 mg/dL, HbA1c 6.4% (1 day earlier), sodium"),
    ("corpus", "hypertension_3e5229a0-8d17-cd71-ab87-facf73203e1c", "subjective",
     "Patient presents with no reported pain at this visit (pain score 0/10).",
     "Patient reported pain score 0/10 (2 months earlier)."),
    ("corpus", "hypertension_99631d3d-dc18-7502-54ea-daf6c3535ac5", "subjective",
     "51-year-old female presenting with no reported pain at this visit (pain severity score 0).",
     "51-year-old female presenting with pain severity score 0 (6 months earlier)."),
    ("corpus", "hypertension_7d528f59-9bda-0be7-b207-4d169fc79096", "subjective",
     "Pain severity rated 2 out of 10 at current visit.", "Pain severity rated 2 out of 10 (10 months earlier)."),
    ("corpus", "hypertension_93a4c131-20c4-f0a2-1546-27ad595b075f", "subjective",
     "Pain severity rated 1 out of 10 at current visit.", "Pain severity rated 1 out of 10 (15 days earlier)."),
    ("corpus", "viral_pharyngitis_c8bcc4ba-4410-9fb4-cef5-b3fbbe8776ae", "subjective",
     "Pain severity reported as 4 out of 10 at this visit.", "Pain severity reported as 4 out of 10 (14 days earlier)."),
    ("corpus", "cystitis_18ceb7dc-a9d5-0f80-0ddd-f37227916de1", "subjective",
     "Reports a current pain severity score of 1 out of 10.",
     "Reported a pain severity score of 1 out of 10 (1 day earlier)."),
    # other values given the wrong date
    ("corpus", "strep_throat_40675a01-3dbc-03a2-29a8-9653b58c545e", "objective",
     "DAST-10 score 0.", "DAST-10 score 0 (0 days earlier)."),
    ("corpus", "viral_pharyngitis_4a7b3b1d-b135-605c-dcbb-a02c9b29c245", "objective",
     "cholesterol 197.4 mg/dL, LDL 143.2 mg/dL, HDL 30.4 mg/dL, triglycerides 119.2 mg/dL (27 days earlier)",
     "cholesterol 197.4 mg/dL, LDL 143.2 mg/dL, HDL 30.4 mg/dL and triglycerides 119.2 mg/dL (all 27 days earlier)"),
    # "most recent" on vitals older than the visit-line vitals (or than a later reading)
    ("corpus", "copd_d6ea13fe-9e08-2411-4948-bc71a8c67ed0", "objective",
     "Most recent recorded labs and vitals:", "Labs and vitals 7 months earlier:"),
    ("corpus", "copd_a28f8e33-1ab1-d203-37d5-cf8095ba6263", "objective",
     "Most recent labs (1 month earlier unless noted):", "Labs (1 month earlier unless noted):"),
    ("corpus", "copd_b2a8fdba-c2af-df32-33db-eaeacd7fe625", "objective",
     "Most recent recorded labs/vitals (7 months earlier):", "Labs/vitals 7 months earlier:"),
    ("corpus", "hypertension_3c820294-9a75-7cb4-6c79-44ad830fd091", "objective",
     "Most recent lab and measurement findings (11 months earlier unless noted):",
     "Lab and measurement findings (11 months earlier unless noted):"),
    ("corpus", "hypertension_5e50a88f-a951-90a7-0afd-87ea592091da", "objective",
     "Most recent labs (14 days earlier unless noted):", "Labs (14 days earlier unless noted):"),
    ("corpus", "hypertension_64a26838-ffa3-c049-2a7e-c198cf2b70e6", "objective",
     "Most recent labs (11 months earlier unless noted):", "Labs (11 months earlier unless noted):"),
    ("corpus", "hypertension_961065b6-405a-1a0f-3c50-a3575a1a3d68", "objective",
     "Most recent recorded labs (1 year earlier):", "Labs 1 year earlier:"),
    ("corpus", "hypertension_9bd1114b-aaec-2959-ca33-92d8c84519c4", "objective",
     "Most recent lab values (11 months earlier):", "Lab values 11 months earlier:"),
    ("corpus", "hypertension_aa561437-9a15-0252-ed90-a8ba88757cf2", "objective",
     "Most recent lab and measurement values (7 months earlier unless noted):",
     "Lab and measurement values (7 months earlier unless noted):"),
    ("corpus", "hypertension_c257f005-456b-f846-321a-360c6ca718aa", "objective",
     "Most recent lab values (11 months earlier unless noted):", "Lab values (11 months earlier unless noted):"),
    ("corpus", "hypertension_d750d9f4-ded7-4fce-6c8c-d01cc09dd2ca", "objective",
     "Most recent labs (10 months earlier unless noted):", "Labs (10 months earlier unless noted):"),
    ("corpus", "hypertension_de4cd231-b267-7034-d067-f48c3fb2163a", "objective",
     "Most recent recorded measurements (11 months earlier):", "Measurements 11 months earlier:"),
    ("corpus", "hypertension_f5e26420-6145-c6ba-df8c-a30799f4daf8", "objective",
     "Most recent recorded vitals (5 months earlier):", "Vital signs 5 months earlier:"),
    ("corpus", "strep_throat_4785caed-d142-daa3-0641-22e2eeea78b3", "objective",
     "Most recent vitals (3 months earlier):", "Vital signs 3 months earlier:"),
    ("corpus", "viral_pharyngitis_4066cee9-2c89-f6f7-bd8f-20073eb15e68", "objective",
     "Most recent vitals (1 year earlier):", "Vital signs 1 year earlier:"),
    ("corpus", "viral_pharyngitis_6bfa542f-a9ba-5468-61b6-b9037bdadb36", "objective",
     "Most recent vitals (1 month earlier):", "Vital signs 1 month earlier:"),
    ("corpus", "viral_pharyngitis_cc03f27d-0962-d726-f54b-3c300bab848f", "objective",
     "Most recent recorded vitals (1 year earlier):", "Vital signs 1 year earlier:"),
    ("corpus", "viral_pharyngitis_f78c9e85-022f-2892-9403-34570959f354", "objective",
     "Most recent recorded vitals (9 months earlier):", "Vital signs 9 months earlier:"),
    ("corpus", "cystitis_1a3e6a3a-a388-84fb-5b06-1156ddc67b03", "objective",
     "Most recent recorded vitals (11 months earlier):", "Vital signs 11 months earlier:"),
    ("corpus", "cystitis_2cace539-dbac-9d79-98c6-b4ebaec83b9a", "objective",
     "Most recent labs (1 year earlier):", "Labs 1 year earlier:"),
    ("corpus", "cystitis_47323032-c162-a3fe-b064-bfa69feb8ccd", "objective",
     "Most recent recorded measurements (10 months earlier):", "Measurements 10 months earlier:"),
    ("corpus", "cystitis_821f4bef-109e-3e1a-9c97-eb07f47bbfc6", "objective",
     "Most recent recorded measurements (14 days earlier):", "Measurements 14 days earlier:"),
    ("corpus", "cystitis_8319cadc-e1b2-8172-1f81-ba538cfab585", "objective",
     "Most recent recorded vitals (2 years earlier):", "Vital signs 2 years earlier:"),
    # invented time spans
    ("corpus", "cystitis_ffec9005-4901-20a9-c697-a780634c2ac5", "subjective",
     "atopic dermatitis since age 3 months", "atopic dermatitis since 16 years earlier"),
    ("corpus", "hypertension_9cd138f5-25c5-9a77-0bdb-b863b5122758", "objective",
     "(4 visits over the past 10 months)", "(4 visits, most recent 10 months earlier)"),
    ("corpus", "hypertension_cf5ae43a-631b-2411-736a-f5c0afa9df73", "subjective",
     "(10 ambulatory visits in the past 23 days)", "(10 ambulatory visits, most recent 23 days earlier)"),
    ("corpus", "hypertension_afdf3c92-6216-dfd0-01fb-e7876eeffe09", "subjective",
     "multiple inpatient visits over the past month", "multiple inpatient visits (most recent 1 month earlier)"),
    ("corpus", "cystitis_cbe682d6-c9e9-e777-3c45-5924b918f06d", "objective",
     "Eight wisdom tooth extractions performed over the past several months, most recently 4 months prior.",
     "Eight wisdom tooth extractions performed, most recently 4 months prior."),
    # dev_nota, third pass
    # severity given a 0-10 scale the source doesn't use (source severities are 1)
    ("dev_nota", "nota_covid19_1a13196c-1e55-c3f9-a189-53dba18c743b", "subjective",
     "fever (severity 1/10) and cough (severity 1/10)", "fever (severity 1) and cough (severity 1)"),
    ("dev_nota", "nota_covid19_773ba2b7-4036-0a00-1799-efd7751961ba", "subjective",
     "fever (severity 1/10) and cough (severity 1/10)", "fever (severity 1) and cough (severity 1)"),
    ("dev_nota", "nota_covid19_bf3e958c-a269-e3b8-17f2-33b6da7852ad", "subjective",
     "a sore throat (severity 1/10)", "a sore throat (severity 1)"),
    # scores and vitals from an earlier visit labelled "at most recent visit" (the most recent visit is later)
    ("dev_nota", "nota_acute_bronchitis_0a8352d6-ad2a-1b5e-8765-ef809e58ebda", "subjective",
     "Pain severity rated 4/10 at most recent visit.", "Pain severity rated 4/10 (7 days earlier)."),
    ("dev_nota", "nota_acute_bronchitis_0a8352d6-ad2a-1b5e-8765-ef809e58ebda", "subjective",
     "PHQ-2 score 1 at most recent visit.", "PHQ-2 score 1 (7 days earlier)."),
    ("dev_nota", "nota_acute_bronchitis_0a8352d6-ad2a-1b5e-8765-ef809e58ebda", "subjective",
     "HARK score 2 at most recent visit.", "HARK score 2 (7 days earlier)."),
    ("dev_nota", "nota_acute_bronchitis_0e17c08f-1d04-04b5-0ffe-b3f78ebd2105", "subjective",
     "pain severity score of 3/10 at most recent visit", "pain severity score of 3/10 (1 month earlier)"),
    ("dev_nota", "nota_acute_bronchitis_154197d8-b190-f562-ded5-cbaa6c3a2366", "subjective",
     "pain severity of 3/10 at most recent visit", "pain severity of 3/10 (7 days earlier)"),
    ("dev_nota", "nota_acute_bronchitis_1e1036f2-b6cf-e7fa-65de-a15385cc4ae8", "subjective",
     "pain severity 4/10 at most recent visit", "pain severity 4/10 (10 months earlier)"),
    ("dev_nota", "nota_acute_bronchitis_23450009-ded9-12ad-5b78-8c29559ef512", "objective",
     "Pain score 0/10 at last visit.", "Pain score 0/10 (4 months earlier)."),
    ("dev_nota", "nota_covid19_02613e7a-48c9-3a26-d741-bace2666a286", "subjective",
     "pain severity of 1/10 at most recent visit", "pain severity of 1/10 (1 month earlier)"),
    ("dev_nota", "nota_covid19_02613e7a-48c9-3a26-d741-bace2666a286", "subjective",
     "PHQ-2 total score of 1 at most recent visit", "PHQ-2 total score of 1 (1 month earlier)"),
    ("dev_nota", "nota_covid19_02613e7a-48c9-3a26-d741-bace2666a286", "subjective",
     "DAST-10 total score of 2 at most recent visit", "DAST-10 total score of 2 (1 month earlier)"),
    ("dev_nota", "nota_covid19_12f27aff-5a59-9df8-08e4-48e0811e05a0", "subjective",
     "1/10 at most recent visit", "1/10 (14 days earlier)"),
    ("dev_nota", "nota_covid19_2ec420b6-a551-f9ce-952c-c35c8e527672", "subjective",
     "Reports no current pain (pain score 0/10 at most recent visit).",
     "Reported no pain (pain score 0/10, 14 days earlier)."),
    ("dev_nota", "nota_covid19_2ec420b6-a551-f9ce-952c-c35c8e527672", "subjective",
     "Medication review due at most recent visit.", "Medication review due (last noted 14 days earlier)."),
    ("dev_nota", "nota_covid19_7b8e13b0-52e6-9dd1-2088-c33fcc440d8d", "subjective",
     "4 out of 10 at the most recent visit", "4 out of 10 (1 month earlier)"),
    ("dev_nota", "nota_covid19_7b8e13b0-52e6-9dd1-2088-c33fcc440d8d", "subjective",
     "PHQ-2 total score of 1 reported at the most recent visit", "PHQ-2 total score of 1 reported 1 month earlier"),
    ("dev_nota", "nota_covid19_7b8e13b0-52e6-9dd1-2088-c33fcc440d8d", "subjective",
     "GAD-7 total score of 0 reported at the most recent visit", "GAD-7 total score of 0 reported 1 month earlier"),
    ("dev_nota", "nota_covid19_821ed16c-fb74-7131-c87f-e3cd2b08f332", "subjective",
     "pain severity of 3/10 at most recent visit", "pain severity of 3/10 (14 days earlier)"),
    ("dev_nota", "nota_covid19_821ed16c-fb74-7131-c87f-e3cd2b08f332", "subjective",
     "GAD-7 total score of 4 at most recent visit", "GAD-7 total score of 4 (14 days earlier)"),
    ("dev_nota", "nota_viral_sinusitis_0e042350-6de1-008c-fd31-c1d91959d45a", "subjective",
     "4/10 at most recent visit", "4/10 (14 days earlier)"),
    ("dev_nota", "nota_viral_sinusitis_10a04c53-e16c-346b-59f4-ab7828b19d7f", "subjective",
     "pain severity score of 4/10 at most recent visit", "pain severity score of 4/10 (14 days earlier)"),
    ("dev_nota", "nota_viral_sinusitis_10a04c53-e16c-346b-59f4-ab7828b19d7f", "subjective",
     "PHQ-2 score of 1 at most recent visit", "PHQ-2 score of 1 (14 days earlier)"),
    ("dev_nota", "nota_viral_sinusitis_126c711c-0bca-5068-30de-a22bf845b347", "subjective",
     "4 out of 10 at most recent visit", "4 out of 10 (7 days earlier)"),
    ("dev_nota", "nota_viral_sinusitis_17a7db29-c77f-78fd-d50c-27192fc28f79", "subjective",
     "2 out of 10 at most recent visit", "2 out of 10 (15 days earlier)"),
    ("dev_nota", "nota_viral_sinusitis_17a7db29-c77f-78fd-d50c-27192fc28f79", "subjective",
     "GAD-7 total score of 3 at most recent visit", "GAD-7 total score of 3 (15 days earlier)"),
    ("dev_nota", "nota_covid19_17a7db29-c77f-78fd-d50c-27192fc28f79", "subjective",
     "0 out of 10 at most recent visit (previously", "0 out of 10 (21 days earlier; previously"),
    ("dev_nota", "nota_covid19_17a7db29-c77f-78fd-d50c-27192fc28f79", "objective",
     "Vitals at most recent visit (21 days prior to last record):", "Vital signs 21 days earlier:"),
    ("dev_nota", "nota_covid19_17a7db29-c77f-78fd-d50c-27192fc28f79", "objective",
     "Labs at most recent visit:", "Labs (21 days earlier):"),
    ("dev_nota", "nota_viral_sinusitis_193e3159-a4a6-d920-7e03-96e580e1f65f", "subjective",
     "PHQ-9 total score of 3 at most recent visit (14 days prior)", "PHQ-9 total score of 3 (14 days prior)"),
    ("dev_nota", "nota_viral_sinusitis_193e3159-a4a6-d920-7e03-96e580e1f65f", "objective",
     "Vitals at most recent visit (14 days prior):", "Vital signs 14 days prior:"),
    ("dev_nota", "nota_viral_sinusitis_033b9dbc-c5d2-bcdf-ee99-a8bbab533172", "objective",
     "Vitals at most recent visit (14 days prior):", "Vital signs 14 days prior:"),
    ("dev_nota", "nota_acute_bronchitis_1ca84c4d-ccb7-100f-3136-5279d5411b1c", "objective",
     "Vitals at most recent visit (7 days prior):", "Vital signs 7 days prior:"),
    ("dev_nota", "nota_covid19_a0a5578c-719a-22dd-9436-e200dcfc25ed", "objective",
     "Vitals from most recent recorded visit (7 months prior):", "Vital signs 7 months prior:"),
    ("dev_nota", "nota_acute_bronchitis_25887d94-b1b1-edfc-eae9-ad00f45fb3cf", "objective",
     "Depression screening performed at most recent visit.", "Depression screening performed 7 days earlier."),
    # invented time spans and ranges (each item has its own date)
    ("dev_nota", "nota_acute_bronchitis_0a805673-54dc-634d-de6e-8d04c9038856", "objective",
     "including scaling, fluoride treatment, composite filling, and oral health education within the past 2 days",
     "including scaling (1 day earlier) and fluoride treatment, composite filling, and oral health education "
     "(current visit)"),
    ("dev_nota", "nota_viral_sinusitis_17ed8063-3c38-7f12-36df-408911470d13", "objective",
     "(all within last 15 days)", "(all 15 days earlier)"),
    ("dev_nota", "nota_viral_sinusitis_17ed8063-3c38-7f12-36df-408911470d13", "objective",
     "performed within last 1–2 days.", "performed 1 day earlier."),
    ("dev_nota", "nota_viral_sinusitis_0cad42a4-9147-a480-e867-73ea77f16be3", "subjective",
     "primary dental caries and gingivitis (noted within the past 8 days)",
     "primary dental caries (1 day earlier) and gingivitis (8 days earlier)"),
    ("dev_nota", "nota_viral_sinusitis_00ea1ff2-a235-3c2c-eeb0-0dce58a16722", "objective",
     "Pain severity 0–10: 3–4 (most recent).", "Pain severity 0–10: 4 (most recent)."),
    # visit counts given the date of the latest visit only
    ("dev_nota", "nota_viral_sinusitis_1b658199-982f-7307-938e-63a24747dde8", "objective",
     "two urgent care visits 14 days prior", "two urgent care visits (most recent 14 days prior)"),
    ("dev_nota", "nota_acute_bronchitis_07012a07-8108-d977-aabb-c538b2c0d588", "subjective",
     "two concussion-related encounters approximately 2 months ago",
     "two concussion-related encounters (approximately 2 months ago and at the current visit)"),
    # past problem given a resolution date the source never states (the date is when it was recorded)
    ("dev_nota", "nota_viral_sinusitis_0ecc1ec2-f0ac-40ad-cb89-0b622e2265be", "subjective",
     "gingivitis resolved approximately 14 days prior", "gingivitis approximately 14 days prior"),
    ("dev_nota", "nota_viral_sinusitis_1b2b5935-fff4-2ed8-4d1d-527fa689824c", "subjective",
     "gingivitis (resolved ~1 year ago)", "gingivitis (~1 year ago)"),
    ("dev_nota", "nota_acute_bronchitis_1f1031c5-2ace-e7ee-ba97-3c8fafb2e247", "subjective",
     "gingivitis (resolved 3 years ago)", "gingivitis (3 years ago)"),
    # corpus, third pass
    # leaked diagnosis: chief complaint / past diagnosis taken from the DIAGNOSIS line
    ("corpus", "strep_throat_20d0cf74-53a2-a394-3ef2-0405c984f530", "subjective",
     '40-year-old male presenting with a chief complaint consistent with sore throat.', '40-year-old male presenting.'),
    ("corpus", "cystitis_52b11457-711a-17a6-9efe-f7da95392904", "subjective",
     'Patient presents with a chief complaint consistent with acute infective cystitis.', 'Patient presents.'),
    ("corpus", "cystitis_fde9faf8-ee38-00f8-3f09-a411e7c1e694", "subjective",
     'Patient presents with a chief complaint consistent with acute infective cystitis.', 'Patient presents.'),
    ("corpus", "heart_failure_e386e770-c961-9d01-e528-fe01211e9fdf", "objective",
     ' Past diagnosis of Chronic congestive heart failure (disorder).', ''),
    # earlier pain scores called "at this visit"
    ("corpus", "hypertension_2d5a0544-4472-03e5-e343-82fadfd7328b", "subjective",
     'Patient presents with no reported pain at this visit (pain severity 0/10).', 'Patient reported pain severity 0/10 (1 year earlier).'),
    ("corpus", "bacterial_sinusitis_c395b3f0-d2ed-cd99-9608-3cb3df06c732", "objective",
     'Pain severity 0/10 on verbal numeric rating at this visit.', 'Pain severity 0/10 on verbal numeric rating (15 days earlier).'),
    # other values given the wrong date
    ("corpus", "hypertension_3ffde601-ea59-4401-ef47-c2800f845c3e", "objective",
     '96 mm[Hg] and 98 mm[Hg] (1 year earlier)', '96 mm[Hg] (11 months earlier) and 98 mm[Hg] (1 year earlier)'),
    # "currently pregnant" contradicted by a later termination or miscarriage
    ("corpus", "copd_fbf24933-262d-346a-414f-5bc5fe0e1b29", "subjective",
     'Patient is currently pregnant (noted 2 months earlier).', 'Pregnancy status: pregnant (2 months earlier).'),
    ("corpus", "hypertension_5f9f8e27-08ad-6eea-8869-5a54ed0730e3", "subjective",
     'who is currently pregnant (noted 2 months earlier)', 'who was recorded as pregnant 2 months earlier'),
    ("corpus", "strep_throat_9b9126c7-bfd1-53ef-760c-5c4241726e9c", "subjective",
     'Patient is currently pregnant (normal pregnancy noted 1 month earlier)', 'Patient was recorded as pregnant 3 months earlier (normal pregnancy noted 1 month earlier)'),
    ("corpus", "viral_pharyngitis_14c81a7b-35dc-5520-d214-b122e7fe30bb", "subjective",
     'Patient is currently pregnant (noted 2 months earlier);', 'Pregnancy status: pregnant (2 months earlier);'),
    ("corpus", "viral_pharyngitis_574f4a5c-e4ab-afb1-b6d7-abdd650de5aa", "subjective",
     'Patient is currently pregnant.', 'Pregnancy status: pregnant (2 months earlier).'),
    ("corpus", "viral_pharyngitis_6bfa542f-a9ba-5468-61b6-b9037bdadb36", "subjective",
     'Patient is currently pregnant (noted 2 months earlier).', 'Pregnancy status: pregnant (2 months earlier).'),
    ("corpus", "viral_pharyngitis_c2267a1d-e39f-136c-926d-aba919f3545d", "subjective",
     'Patient is currently pregnant (noted 2 months earlier).', 'Pregnancy status: pregnant (2 months earlier).'),
    ("corpus", "viral_pharyngitis_e4d65768-124e-2c5b-2a8c-8034a5d80527", "subjective",
     'Patient is currently pregnant and had', 'Patient was recorded as pregnant 3 months earlier and had'),
    ("corpus", "cystitis_7a29c34c-6407-563a-d428-2f1e83f16b4e", "subjective",
     'Patient is currently pregnant (as of 2 months earlier)', 'Patient was recorded as pregnant 2 months earlier'),
    # invented gestational age / counts / normal status
    ("corpus", "hypertension_1238d87f-8779-2af0-7733-ebc80f378026", "subjective",
     'Currently 8 months pregnant (normal pregnancy', 'Pregnancy status: pregnant 8 months earlier (normal pregnancy'),
    ("corpus", "cystitis_ea6b0c13-660f-fb81-22ca-171f036ba14b", "subjective",
     'Patient is currently pregnant (approximately 1 month gestation)', 'Pregnancy status: pregnant (1 month earlier)'),
    ("corpus", "viral_pharyngitis_3a333641-b2aa-1876-6b4d-717482736fd7", "subjective",
     'two prior miscarriages (complete', 'a prior miscarriage (complete'),
    ("corpus", "viral_pharyngitis_39e82d22-19bc-a57f-c37c-4967949082ae", "objective",
     'CBC from 2 years earlier within reported ranges.', 'CBC recorded 2 years earlier.'),
    ("corpus", "cystitis_be569c63-9bdc-e42a-00dd-63acd1ddd174", "objective",
     'CBC from 4 years earlier within documented ranges.', 'CBC recorded 4 years earlier.'),
    # text hygiene: repeated sentence, typos
    ("corpus", "viral_pharyngitis_09593cad-cf44-7a7d-feea-762268198d75", "objective",
     'GAD-7: 2. Morse Fall Scale: 94.', 'GAD-7: 2.'),
    ("corpus", "strep_throat_4436ca7c-44b0-c71e-714f-e70fec0d69da", "subjective",
     'swollen topsils', 'swollen tonsils'),
    ("corpus", "strep_throat_64164861-69ca-741a-c110-e08fe4e4014d", "subjective",
     'swollen torsils', 'swollen tonsils'),
    ("corpus", "strep_throat_8bc51b71-8144-2d29-bdfe-5a13fea25538", "subjective",
     'swollen torsils', 'swollen tonsils'),
    # dev, third pass
    # "most recent" on vitals older than the visit-line vitals
    ("dev", "hypertension_37c807a9-0484-9e5b-823a-076ae668555d", "objective", "Most recent recorded vitals (8 months earlier):", "Vital signs 8 months earlier:"),
    ("dev", "hypertension_3dbed350-e3a7-540b-c5b1-be5f2dab1ca2", "objective", "Most recent recorded vitals (1 year prior):", "Vital signs 1 year prior:"),
    ("dev", "hypertension_8b185f4e-f6aa-94a9-58ed-c9f96edf777e", "objective", "Most recent recorded vitals (1 year earlier):", "Vital signs 1 year earlier:"),
    ("dev", "hypertension_ac1600ec-3db6-ea14-7d07-dd75e3766ab9", "objective", "Most recent recorded vitals (one year prior):", "Vital signs one year prior:"),
    ("dev", "hypertension_ba205da5-6092-5c6c-d60f-709a315fdf7d", "objective", "Most recent recorded vitals (9 months earlier):", "Vital signs 9 months earlier:"),
    ("dev", "hypertension_c570a217-c63d-ea90-14ab-233f562e53cb", "objective", "Most recent labs (9 months prior unless noted):", "Labs (9 months prior unless noted):"),
    ("dev", "hypertension_d68a1c13-88f4-f94e-b1d5-c930db741575", "objective", "Most recent recorded measurements (9 months earlier):", "Measurements 9 months earlier:"),
    ("dev", "strep_throat_f0936094-3ecf-4a68-5c04-be6dfc532592", "objective", "Most recent vitals (1 month earlier):", "Vital signs 1 month earlier:"),
    ("dev", "viral_pharyngitis_23450009-ded9-12ad-5b78-8c29559ef512", "objective", "Most recent vitals (2 months prior):", "Vital signs 2 months prior:"),
    ("dev", "viral_pharyngitis_3844d2cf-4f3a-32ad-0f22-53d30e9c8add", "objective", "Most recent labs (1 year prior):", "Labs (1 year prior):"),
    ("dev", "cystitis_3ffb58e4-6e22-4bce-3d50-0605d52ff22e", "objective", "Most recent labs (1 day prior):", "Vital signs 1 day prior:"),
    ("dev", "cystitis_99848115-4645-ce10-8c30-ff454c0e6681", "objective", "Most recent measurements (6 months prior):", "Measurements 6 months prior:"),
    ("dev", "cystitis_fad4919d-30a4-ce22-7304-c3dc246ddfba", "objective", "Most recent recorded measurements (11 months prior):", "Measurements 11 months prior:"),
    # earlier values called "current" / "at this visit"
    ("dev", "hypertension_7437b253-c708-ca63-6e56-f3d6b668956a", "objective", "current BMI 28.2.", "BMI 28.2 (11 months earlier)."),
    ("dev", "hypertension_888d4524-6d44-bc45-7891-56536b282f1c", "subjective", "with a current severity of 8 out of 10 (up from", "with a severity of 8 out of 10 (1 month earlier; up from"),
    ("dev", "hypertension_908e9ad6-250e-31b5-df54-b5750c8186ed", "objective", "current BMI 28.3)", "BMI 28.3 11 months earlier)"),
    ("dev", "hypertension_af1cc7a8-b30d-9ddd-0130-2cf9e1ea550a", "objective", "current BMI 30.1 kg/m2)", "BMI 30.1 kg/m2 10 months earlier)"),
    ("dev", "hypertension_f45378ab-0f52-702a-9fb0-b57d0d84f0ff", "subjective", "a current pain severity score of 3 out of 10.", "a pain severity score of 3 out of 10 (1 year earlier)."),
    ("dev", "strep_throat_7eddc470-7318-d2ce-5f08-21e483ef6c0d", "objective", "Current labs (14 days earlier):", "Labs 14 days earlier:"),
    ("dev", "strep_throat_8ede6b48-d752-1f9f-ca54-8a9bd0652f75", "subjective", "Current pain severity reported as 3 out of 10.", "Pain severity reported as 3 out of 10 (5 days earlier)."),
    ("dev", "strep_throat_cb9f07f0-5436-45c9-38dc-63facb306e91", "subjective", "1 out of 10 at current visit.", "1 out of 10 (6 days earlier)."),
    ("dev", "strep_throat_dd67eccd-c361-8976-d6bc-013f7dd25223", "subjective", "pain severity of 1/10 at current visit.", "pain severity of 1/10 (14 days earlier)."),
    ("dev", "viral_pharyngitis_074b6442-58a4-9558-8740-d532bdf3e6f3", "subjective", "pain severity 1/10 at current visit.", "pain severity 1/10 (5 months earlier)."),
    ("dev", "viral_pharyngitis_14d39fe9-4aae-030f-76f2-d9021415e280", "objective", "current BMI 29.4 kg/m².", "BMI 29.4 kg/m² (14 days earlier)."),
    ("dev", "viral_pharyngitis_23ef44b2-0199-32ec-14c1-0d25216fc051", "subjective", "4 out of 10 at this visit.", "4 out of 10 (7 days earlier)."),
    ("dev", "viral_pharyngitis_6d4ed3c3-1e37-7d63-74e2-8bd8ef09dcf6", "objective", "score 23 (current visit)", "score 23 (7 days earlier)"),
    ("dev", "viral_pharyngitis_818a7e37-e12b-437e-601b-f68fe29c3960", "subjective", "Current pain severity reported as 3/10.", "Pain severity reported as 3/10 (3 months earlier)."),
    ("dev", "viral_pharyngitis_839ea010-a89f-8e46-b2f4-d7bcbe652c04", "subjective", "4 out of 10 at current visit.", "4 out of 10 (7 days earlier)."),
    ("dev", "viral_pharyngitis_8672d635-f67a-1e69-6e51-b1c562b9192b", "subjective", "Current pain severity rated 3/10.", "Pain severity rated 3/10 (14 days earlier)."),
    ("dev", "viral_pharyngitis_870ac4f5-9e34-74e9-de6d-090d7758d742", "objective", "Current BMI 29.9 kg/m² (height 173.1 cm, weight 89.5 kg).", "BMI 29.9 kg/m² (height 173.1 cm, weight 89.5 kg; 7 days earlier)."),
    ("dev", "viral_pharyngitis_9263eeea-78af-23a4-fea6-bbe3a158a986", "subjective", "Current pain severity rated 2/10.", "Pain severity rated 2/10 (14 days earlier)."),
    ("dev", "viral_pharyngitis_979bce63-063a-e6ba-9703-b42da7b0228f", "subjective", "1/10 at current visit.", "1/10 (1 day earlier)."),
    ("dev", "viral_pharyngitis_b6919b22-f0a9-bcd7-d2cc-a1d51b444790", "subjective", "a current pain severity of 4/10.", "a pain severity of 4/10 (14 days earlier)."),
    ("dev", "viral_pharyngitis_b955c3c8-0af4-7148-114f-0b9b0986457d", "subjective", "pain severity 3/10 at current visit.", "pain severity 3/10 (2 months earlier)."),
    ("dev", "viral_pharyngitis_c1818685-1952-385e-0115-40d50a309d1c", "subjective", "Pain severity rated 2/10 at current visit.", "Pain severity rated 2/10 (21 days earlier)."),
    ("dev", "viral_pharyngitis_d8ae71c6-c464-88a6-441b-b8d81d7f1c6c", "objective", "current BMI 27.7 kg/m²)", "BMI 27.7 kg/m² 3 months earlier)"),
    ("dev", "viral_pharyngitis_df0aee1b-82f2-78d6-d096-ad48e9d7ad9f", "objective", "Current weight 82.3 kg, height 176.8 cm, BMI 26.3 kg/m².", "Weight 82.3 kg, height 176.8 cm, BMI 26.3 kg/m² (2 months earlier)."),
    ("dev", "viral_pharyngitis_e3f8003d-6d0c-e936-685e-c09e8ca34827", "objective", "current BMI 28.2)", "BMI 28.2 7 months earlier)"),
    ("dev", "viral_pharyngitis_fb6300fb-4251-947e-0683-48bafdde92fd", "subjective", "Current pain severity rated 3/10.", "Pain severity rated 3/10 (14 days earlier)."),
    ("dev", "bacterial_sinusitis_5dab19a8-ba62-c4f8-19d3-34fb57276dc8", "subjective", "Current pain severity rated 3/10.", "Pain severity rated 3/10 (14 days earlier)."),
    ("dev", "cystitis_587a3ab9-dc43-5bd5-a3d3-ca48710c97eb", "objective", "current BMI 27.2 kg/m2.", "BMI 27.2 kg/m2 (5 months earlier)."),
    ("dev", "cystitis_d0a1c412-faf8-73c7-c5f8-73b95d64189d", "subjective", "a current pain severity score of 2 out of 10.", "a pain severity score of 2 out of 10 (10 months earlier)."),
    ("dev", "cystitis_d88c5636-cd71-ce55-f532-157b79d307e5", "objective", "Current weight 60.5 kg, height 159.2 cm.", "Weight 60.5 kg, height 159.2 cm (2 months earlier)."),
    ("dev", "heart_failure_019069c1-c2b9-165a-2ea3-bdcd2a3b5d8a", "subjective", "Current pain severity reported as 2/10.", "Pain severity reported as 2/10 (8 days earlier)."),
    ("dev", "heart_failure_585bae63-de40-caa6-9654-57aa050bc3a4", "subjective", "Current pain severity rated 1/10.", "Pain severity rated 1/10 (1 day earlier)."),
    ("dev", "heart_failure_9128304a-447d-bd98-110c-7f6d53d7f004", "objective", "current BMI 31.8 kg/m²)", "BMI 31.8 kg/m² 1 month earlier)"),
    ("dev", "heart_failure_e3cf3337-ab64-48e7-fd6e-4b821323750c", "objective", "Current BMI 27.7 kg/m2.", "BMI 27.7 kg/m2 (4 months earlier)."),
    # diagnosis-day temperature labelled as an earlier visit's vital sign
    ("dev", "viral_pharyngitis_5172dbfd-8480-a4b7-cf52-194935f71e9c", "objective", "Vital signs at urgent care visit: body temperature 37.1°C, heart rate", "Body temperature at this visit: 37.1°C. Vital signs at urgent care visit: heart rate"),
    # items given another item's date
    ("dev", "viral_pharyngitis_9d2fad10-fdaa-97c7-8f08-df6f206293e8", "objective", "tooth extraction, fitting of denture, and dental restorations (4–5 months prior)", "tooth extraction and fitting of denture (4–5 months prior) and dental restorations (9 months prior)"),
    # interpretation the source never states
    ("dev", "heart_failure_08f75e69-a9dc-12e2-47c5-d521cec2e83d", "objective", "Fecal immunochemical test positive (hemoglobin", "Fecal immunochemical test (hemoglobin"),
    # corpus_nota, second pass
    # "most recent" on vitals older than the visit-line vitals
    ("corpus_nota", "nota_covid19_31ded636-849d-0c93-cc26-d115433b602f", "objective",
     "Most recent measurements (26 days earlier):", "Measurements 26 days earlier:"),
    ("corpus_nota", "nota_covid19_7a28a34b-d24e-5d61-feef-e84504d7b7a2", "objective",
     "Most recent measurements (26 days earlier):", "Measurements 26 days earlier:"),
    ("corpus_nota", "nota_covid19_326bc803-f772-44f7-cfc3-ed110deba70d", "objective",
     "Most recent labs (14 days earlier):", "Measurements 14 days earlier:"),
    ("corpus_nota", "nota_covid19_e5951370-a7cd-6fb6-1eb6-701c3aa90938", "objective",
     "Most recent recorded labs and measurements (11 months earlier):", "Labs and measurements 11 months earlier:"),
    # earlier values called "current"
    ("corpus_nota", "nota_covid19_261aec35-26ef-712b-d143-9972e23a54cb", "subjective",
     "Current pain severity rated 4/10.", "Pain severity rated 4/10 (7 days earlier)."),
    ("corpus_nota", "nota_covid19_6a875332-617b-7820-b7c0-b9111ddddf5e", "subjective",
     "No current alcohol use (AUDIT-C 0).", "AUDIT-C score 0 (10 months earlier)."),
    ("corpus_nota", "nota_covid19_c7562ac2-ec72-6c10-2cb2-578a3836ae18", "subjective",
     "Patient reports a current pain severity score of 1 out of 10 and a PHQ-2 score of 1.",
     "Patient reported a pain severity score of 1 out of 10 and a PHQ-2 score of 1 (both 11 months earlier)."),
    ("corpus_nota", "nota_viral_sinusitis_ffcca9cb-bec1-72b0-76fd-2f0f51a9e3e7", "objective",
     "current BMI now 27.6", "most recent BMI 27.6, 23 days earlier"),
    # influenza vaccine dated earlier but called "at this visit"
    ("corpus_nota", "nota_viral_sinusitis_1afc86de-be87-26f6-ea2e-3c190e4c94e8", "objective",
     "Influenza vaccine administered at current visit.", "Influenza vaccine administered 9 days earlier."),
    ("corpus_nota", "nota_covid19_258b9b7d-8c4a-ce20-cfe7-fba0db140547", "objective",
     "Influenza vaccine administered at this visit.", "Influenza vaccine administered 14 days earlier."),
    ("corpus_nota", "nota_covid19_3108a353-100b-c0f9-362e-55e869b9c3d9", "objective",
     "Influenza vaccine administered at this visit.", "Influenza vaccine administered 1 day earlier."),
    ("corpus_nota", "nota_acute_bronchitis_6089f0a7-55d0-38fe-f6e5-82f4b7a27d88", "objective",
     "Influenza vaccine administered at this visit.", "Influenza vaccine administered 7 days earlier."),
    ("corpus_nota", "nota_covid19_912cc29d-29f1-b1c5-162b-ec3ba8d69a79", "objective",
     "Influenza vaccine administered at this visit.", "Influenza vaccine administered 1 day earlier."),
    # earlier visit type given as the current visit
    ("corpus_nota", "nota_covid19_31ded636-849d-0c93-cc26-d115433b602f", "subjective",
     "presenting for a wellness visit", "with a wellness visit 26 days earlier"),
    ("corpus_nota", "nota_covid19_7a28a34b-d24e-5d61-feef-e84504d7b7a2", "subjective",
     "Parent/guardian presents with a 4-year-old male for a wellness visit.",
     "4-year-old male; most recent wellness visit 26 days earlier."),
    ("corpus_nota", "nota_covid19_261aec35-26ef-712b-d143-9972e23a54cb", "subjective",
     "Patient presents to urgent care.", "Urgent care visit 7 days earlier."),
    ("corpus_nota", "nota_covid19_7f1e65df-c67b-26fa-bbdf-cbd235e0b471", "subjective",
     "48-year-old female presenting for wellness visit.", "48-year-old female; wellness visit 1 day earlier."),
    ("corpus_nota", "nota_covid19_e9854fbc-bb13-6452-0a94-87cd73f1ef65", "subjective",
     "6-year-old male presenting for a well child visit.", "6-year-old male; well child visit 14 days earlier."),
    # scores and vitals from an earlier visit labelled "at most recent / last visit" (a later visit exists)
    ("corpus_nota", "nota_covid19_31ded636-849d-0c93-cc26-d115433b602f", "subjective",
     "1 out of 10 at most recent visit", "1 out of 10 (26 days earlier)"),
    ("corpus_nota", "nota_covid19_7a28a34b-d24e-5d61-feef-e84504d7b7a2", "subjective",
     "3 out of 10 at most recent visit", "3 out of 10 (26 days earlier)"),
    ("corpus_nota", "nota_covid19_0a9c3691-1b21-8abb-4417-78239526ed72", "subjective",
     "Pain severity reported as 1/10 at most recent visit.", "Pain severity reported as 1/10 (7 days earlier)."),
    ("corpus_nota", "nota_covid19_28bfe362-17b0-d7d1-6819-222590e3fe11", "subjective",
     "Pain severity rated 3/10 at last recorded visit.", "Pain severity rated 3/10 (1 year earlier)."),
    ("corpus_nota", "nota_covid19_8fab8021-615c-ef5a-6c62-a20ca4a14e19", "subjective",
     "Pain severity rated 2 out of 10 at most recent visit.", "Pain severity rated 2 out of 10 (14 days earlier)."),
    ("corpus_nota", "nota_covid19_a99c8a6e-cb8e-4f37-f59b-ec5f9427870e", "subjective",
     "3 out of 10 at the most recent visit.", "3 out of 10 (3 months earlier)."),
    ("corpus_nota", "nota_covid19_dfb661b2-e7d2-3c1d-fa56-4442b92cd8e5", "subjective",
     "1 out of 10 at last recorded visit.", "1 out of 10 (1 year earlier)."),
    ("corpus_nota", "nota_covid19_dfb661b2-e7d2-3c1d-fa56-4442b92cd8e5", "subjective",
     "PHQ-2 total score of 1 at last recorded visit.", "PHQ-2 total score of 1 (1 year earlier)."),
    ("corpus_nota", "nota_viral_sinusitis_617a48bf-14f3-0b21-d7ed-85b3bdfeb36c", "objective",
     "Vitals at most recent visit (14 days prior to last record):", "Vital signs 14 days earlier:"),
    ("corpus_nota", "nota_covid19_1d486582-f6ba-14c8-9e62-8313e61a13c5", "objective",
     "Vitals at last visit (14 days earlier):", "Vital signs 14 days earlier:"),
    ("corpus_nota", "nota_acute_bronchitis_b8199cc9-6712-b775-231c-fa5b431cc2b3", "objective",
     "Vitals at last visit (7 days earlier):", "Vital signs 7 days earlier:"),
    ("corpus_nota", "nota_covid19_ce3a1395-8978-9378-1121-95e2e27b2106", "subjective",
     "PHQ-2 total score of 2 at last visit (14 days earlier)", "PHQ-2 total score of 2 (14 days earlier)"),
    # invented or misattributed dates, spans and status
    ("corpus_nota", "nota_acute_bronchitis_1238d87f-8779-2af0-7733-ebc80f378026", "subjective",
     "(approximately 7 months gestation)", "(pregnancy status recorded 7 months earlier)"),
    ("corpus_nota", "nota_covid19_b31b08ef-1800-cc6e-ae5b-31025d0d57e1", "subjective",
     "Currently pregnant (normal pregnancy last noted 2 months earlier).",
     "Normal pregnancy last noted 2 months earlier; induced termination of pregnancy 1 month earlier."),
    ("corpus_nota", "nota_covid19_ef33138a-8105-c72c-69c8-4ac9a80fcebe", "objective",
     "surgical manipulation of joint of knee and physical therapy (2 months earlier), with discharge from "
     "skilled nursing facility approximately 1 month earlier.",
     "surgical manipulation of joint of knee (2 months earlier) and physical therapy (4 times, last 1 month "
     "earlier), with a skilled nursing facility admission 2 months earlier."),
    ("corpus_nota", "nota_acute_bronchitis_237c7291-144c-267e-43a0-63880baf1adf", "objective",
     "echocardiography, plain X-ray of chest, electrocardiographic procedure, cardiovascular stress testing, "
     "and medication reconciliation (all within 20 days)",
     "echocardiography and medication reconciliation (0 days earlier), and plain X-ray of chest, "
     "electrocardiographic procedure and cardiovascular stress testing (20 days earlier)"),
    ("corpus_nota", "nota_covid19_e5951370-a7cd-6fb6-1eb6-701c3aa90938", "subjective",
     "Past problem of gingivitis resolved.", "Past problem of gingivitis (1 year earlier)."),
    # severity qualifier the source doesn't give (source severities are 1)
    ("corpus_nota", "nota_covid19_1d486582-f6ba-14c8-9e62-8313e61a13c5", "subjective",
     "each of mild severity (severity: 1).", "each with severity 1."),
    ("corpus_nota", "nota_covid19_d06cb73d-3bb4-bad0-07f5-45da98d4dc7e", "subjective",
     "each of mild severity.", "each with severity 1."),
    ("corpus_nota", "nota_covid19_f333c9a1-eec2-82e4-ea25-579329026a6b", "subjective",
     "each of mild severity.", "each with severity 1."),
    # active problems labelled "past diagnoses"
    ("corpus_nota", "nota_acute_bronchitis_071cd233-7be4-988d-6092-43e49d78c899", "objective",
     "Past diagnoses include chronic sinusitis", "Diagnoses include chronic sinusitis"),
    ("corpus_nota", "nota_acute_bronchitis_290c5f5b-b2e8-0881-fe09-f10026ce1c94", "objective",
     "Past diagnoses include chronic intractable", "Diagnoses include chronic intractable"),
    ("corpus_nota", "nota_viral_sinusitis_31db990e-0e68-7de8-0c35-2bda1a914eb8", "objective",
     "Past diagnoses include anemia", "Diagnoses and findings include anemia"),
    ("corpus_nota", "nota_viral_sinusitis_3ddbc7db-1517-9d38-94e6-28ed32fa7bd2", "objective",
     "Past diagnoses include atopic", "Diagnoses and findings include atopic"),
    ("corpus_nota", "nota_covid19_69ef873f-b272-d589-2c12-097b70241833", "objective",
     "Past diagnoses include prediabetes", "Diagnoses and findings include prediabetes"),
    ("corpus_nota", "nota_acute_bronchitis_95595649-b966-3025-632a-53870eaec950", "objective",
     "Past diagnoses include pulmonary", "Diagnoses and findings include pulmonary"),
    ("corpus_nota", "nota_covid19_99c55c46-435b-9991-cb91-40fbab98f11f", "objective",
     "Past diagnoses include chronic sinusitis", "Diagnoses and findings include chronic sinusitis"),
    ("corpus_nota", "nota_covid19_a6a4a4e9-9bb4-1990-e788-ba53b5047741", "objective",
     "Past diagnoses include chronic sinusitis", "Diagnoses and findings include chronic sinusitis"),
    ("corpus_nota", "nota_covid19_c7b0877a-2b7e-adfb-d89f-f2cb29c5c11d", "objective",
     "Past diagnosis of gingivitis", "Diagnosis of gingivitis"),
    # visit counts given the date of the latest visit only
    ("corpus_nota", "nota_viral_sinusitis_8dba88df-8112-df34-ef46-9badb3fe7b9c", "objective",
     "two emergency visits today for impacted molars", "two emergency visits for impacted molars (most recent today)"),
    ("corpus_nota", "nota_viral_sinusitis_8dba88df-8112-df34-ef46-9badb3fe7b9c", "objective",
     "three emergency visits two months earlier for chronic intractable migraine without aura",
     "three emergency visits for chronic intractable migraine without aura (most recent two months earlier)"),
    ("corpus_nota", "nota_acute_bronchitis_b5f4bd07-52e1-89cf-4037-e42e802b1e98", "objective",
     "wellness visits 6 months earlier", "wellness visits (2, most recent 6 months earlier)"),
    ("corpus_nota", "nota_acute_bronchitis_e3f5a45a-39f2-962f-67ef-1a504862805b", "objective",
     "wellness visits 4 months prior", "wellness visits (3, most recent 4 months prior)"),
    ("corpus_nota", "nota_viral_sinusitis_68ca7b0a-203d-1305-02d0-cf36c0e65637", "objective",
     "outpatient visits (7 days earlier)", "outpatient visits (2, most recent 7 days earlier)"),
    # invented time spans (each item has its own date)
    ("corpus_nota", "nota_acute_bronchitis_4c860d43-f760-8b6f-2ee3-2451cec7dec5", "subjective",
     "sinusitis within the past year.", "sinusitis (both 1 year earlier)."),
    ("corpus_nota", "nota_viral_sinusitis_cdf8454a-84a2-88c1-576f-fa35b88d6271", "subjective",
     "with recent readings of 7.8 and 8.8 over the past several days",
     "with earlier readings of 7.8 (3 days earlier) and 8.8 (6 days earlier)"),
    # dev_nota, fourth pass
    # "currently pregnant" for a pregnancy status recorded earlier
    ("dev_nota", "nota_viral_sinusitis_08e8fbae-93f8-ac20-865c-c07e47d6dffb", "subjective",
     "Patient is currently pregnant.", "Pregnancy status: pregnant (28 days earlier)."),
    ("dev_nota", "nota_viral_sinusitis_14d39fe9-4aae-030f-76f2-d9021415e280", "subjective",
     "Patient is currently pregnant (noted 6 months earlier).", "Pregnancy status: pregnant (6 months earlier)."),
    # invented pregnancy (the normal pregnancy noted 7 months earlier is the one terminated 6 months earlier)
    ("dev_nota", "nota_covid19_90232fe7-ce44-f994-de2f-cf76e51eca83", "subjective",
     "terminated approximately 6 months prior, and a prior normal pregnancy noted 7 months earlier",
     "terminated approximately 6 months prior (normal pregnancy noted 7 months earlier)"),
    # active problems listed as past diagnoses
    ("dev_nota", "nota_acute_bronchitis_2613c8b8-578a-5f9c-9e7f-c4159fbbb258", "objective",
     "loss of teeth (two years ago)", "loss of teeth (active since two years ago)"),
    ("dev_nota", "nota_covid19_43d3d559-6a8f-0f09-67d7-bc824bf82c83", "objective",
     "Past diagnoses include gingivitis (current), fracture of bone",
     "Active problem: gingivitis (since current visit). Past diagnoses include fracture of bone"),
    # interpretation the source never states (all source severities are 1)
    ("dev_nota", "nota_covid19_43d3d559-6a8f-0f09-67d7-bc824bf82c83", "subjective",
     "each of mild severity", "each severity 1"),
    ("dev_nota", "nota_covid19_9411af09-3a4f-b424-210e-d448d341f7f8", "subjective",
     "each of mild severity", "each severity 1"),
    ("dev_nota", "nota_covid19_b89fd874-2edb-bc10-ce8c-fc27d6dea158", "subjective",
     "each of mild severity", "each severity 1"),
    # time spans the source never gives (only the count and the latest date)
    ("dev", 'hypertension_1581eb11-ce2f-f0e0-2a6f-7980fbcd0864', 'objective',
     'three wellness/general examination visits within the past 3 years',
     'three wellness/general examination visits (most recent 3 years earlier)'),
    ("dev", 'hypertension_1d6d66d0-525b-3caf-2271-ca9f464f2541', 'objective',
     'documented within the past year.',
     'documented, most recent 1 year earlier.'),
    ("dev", 'hypertension_c026a95d-e0e7-3ddb-5d6a-7fa0c4acec7b', 'objective',
     'Three wellness visits documented within the past year.',
     'Three wellness visits documented, most recent at the current visit.'),
    ("dev", 'hypertension_e2f0523c-a58d-f429-b965-1d2d44b9e52e', 'objective',
     'Two wellness visits documented within the past 2 years.',
     'Two wellness visits documented, most recent 2 years earlier.'),
    ("dev", 'hypertension_54f6760c-10ec-3949-66b1-85b223015d3a', 'objective',
     'Two wellness visits recorded on the date of most recent record.',
     'Two wellness visits recorded, most recent on the date of most recent record.'),
    ("dev", 'strep_throat_4ade3c36-a064-b010-0f7c-fff29b63cb98', 'subjective',
     'for chronic pain over the past year',
     'for chronic pain (2 visits, most recent 1 year earlier)'),
    ("dev", 'strep_throat_d6b59f7f-4870-47cc-7b8b-2eae62603468', 'objective',
     'two EEGs performed within the past 6 months',
     'two EEGs performed (most recent 6 months earlier)'),
    ("dev", 'viral_pharyngitis_0b9e220a-75d1-275a-3076-b46b293bc9b9', 'objective',
     '(6 encounters within the past 3 days)',
     '(6 encounters, most recent 3 days earlier)'),
    ("dev", 'viral_pharyngitis_25887d94-b1b1-edfc-eae9-ad00f45fb3cf', 'objective',
     '21 times over the past year.',
     '21 times, most recently 1 year earlier.'),
    ("dev", 'viral_pharyngitis_44df5ef5-aa7b-c751-c362-35f40c750af0', 'objective',
     '5 urgent care visits in the past 7 days',
     '5 urgent care visits (most recent 7 days earlier)'),
    ("dev", 'viral_pharyngitis_4ed8a560-cac5-6c48-bf88-152ce864762c', 'objective',
     '(4 visits over the past 7 months)',
     '(4 visits, most recent 7 months earlier)'),
    ("dev", 'viral_pharyngitis_92c93438-0932-a442-f62c-23b5d6c7337b', 'objective',
     'wellness visits within the past year',
     'wellness visits (3 visits, most recent 1 year earlier)'),
    ("dev", 'cystitis_0e146c85-d39a-17f3-f82b-47e0456a2c87', 'objective',
     '10 times over the past 8 months',
     '10 times, most recently 8 months earlier'),
    ("dev", 'cystitis_eef1259a-b7b3-cfd5-eb4a-300ed001673f', 'objective',
     '(general examination) within the past year',
     '(general examination), most recent 1 year earlier'),
    ("dev", 'bacterial_sinusitis_862c9ebf-be68-90a4-633a-d444bf5997b8', 'objective',
     'two wellness visits within the past 7 days',
     'two wellness visits (most recent 7 days earlier)'),
    ("dev", 'bacterial_sinusitis_8026e154-2427-d4e1-935b-ac225a175279', 'objective',
     '(from 8–9 months prior)',
     '(from 9 months prior)'),
    # interpretations the source never states
    ("dev", 'viral_pharyngitis_02613e7a-48c9-3a26-d741-bace2666a286', 'objective',
     'notable for elevated ferritin (up to 557.9 ug/L), elevated LDH (up to 328.5 U/L), elevated CRP (up to 10.1 mg/L), elevated creatinine (up to 2.9 mg/dL), low GFR (as low as 6.8 mL/min), elevated bilirubin (up to 9.6 mg/dL), low leukocytes (3.2–3.7 × 10^3/uL), low platelets (107.7–139.1 × 10^3/uL), elevated INR (up to 1.9),',
     'include ferritin up to 557.9 ug/L, LDH up to 328.5 U/L, CRP up to 10.1 mg/L, creatinine up to 2.9 mg/dL, GFR as low as 6.8 mL/min, bilirubin up to 9.6 mg/dL, leukocytes 3.2–3.7 × 10^3/uL, platelets 107.7–139.1 × 10^3/uL, INR up to 1.9,'),
    ("dev", 'viral_pharyngitis_34349a57-b36a-1082-901d-7a86a3c275a2', 'objective',
     'leukocytes elevated (44.1',
     'leukocytes (44.1'),
    ("dev", 'heart_failure_d6c3e062-e9ea-b09b-7634-3c1f15970e91', 'objective',
     'Prior creatinine elevated at 1.6',
     'Prior creatinine 1.6'),
    ("dev", 'copd_9fbadbe5-9a1e-14a4-f0ca-40dbf07f359e', 'objective',
     'Prior BP readings elevated:',
     'Prior BP readings:'),
    ("dev", 'bacterial_sinusitis_8026e154-2427-d4e1-935b-ac225a175279', 'objective',
     'improving to 103/97',
     'then 103/97'),
    ("dev", 'viral_pharyngitis_a3a1bb3d-4642-e086-3679-e5fc487b1439', 'objective',
     ' vaccines consistent with age-appropriate schedule.',
     ' vaccines.'),
    ("dev", 'viral_pharyngitis_ed8423cc-71f3-a529-a135-9d3a53b10c65', 'subjective',
     'and reports no current pain beyond a self-reported pain score',
     'and self-reported a pain score'),
    # reaction attached to the wrong allergen (house dust mite has no recorded reaction)
    ("dev", 'hypertension_385528c7-6885-8a11-7927-8cfb04a4f755', 'subjective',
     'house dust mite (reaction: rhinoconjunctivitis), animal dander',
     'house dust mite, animal dander'),
    # corpus, third pass (cont.)
    # past problems listed as active / current
    ("corpus", "hypertension_02cfc815-0078-d472-44f4-1a3182284bc1", "objective",
     'Active diagnoses include disease caused by SARS-CoV-2.', 'Past problems include disease caused by SARS-CoV-2 (0 days earlier).'),
    ("corpus", "strep_throat_959b741c-746b-f85b-d124-25fd7359621b", "objective",
     'Active diagnosis of viral sinusitis (0 days earlier).', 'Past diagnosis of viral sinusitis (0 days earlier).'),
    ("corpus", "viral_pharyngitis_2d6f0589-6607-0ee7-362d-1a51778faa2a", "subjective",
     ', small cell carcinoma of lung (TNM stage 1), and recurrent gingivitis.', ', and small cell carcinoma of lung (TNM stage 1). Past problems include gingivitis (5 times, most recent 14 days earlier).'),
    ("corpus", "cystitis_14c81a7b-35dc-5520-d214-b122e7fe30bb", "subjective",
     ', which is also listed as a current problem.', ' (recorded as a past problem 0 days earlier).'),
    # findings (OTHER FINDINGS) listed as active problems/diagnoses
    ("corpus", "copd_c257f005-456b-f846-321a-360c6ca718aa", "objective",
     'Active diagnoses include hyperlipidemia (since 8 years earlier) and prediabetes.', 'Active diagnoses include hyperlipidemia (since 8 years earlier). Other findings: prediabetes.'),
    ("corpus", "hypertension_225f4941-2a61-2e9a-455f-c3fcfa338a56", "objective",
     'Active diagnoses include hyperlipidemia (disorder) and prediabetes (finding).', 'Active diagnoses include hyperlipidemia (disorder). Other findings: prediabetes (finding).'),
    ("corpus", "hypertension_2c25083c-ebef-ffec-a9aa-506b4cb3006b", "objective",
     'Active diagnoses on record: chronic sinusitis, anemia, prediabetes, obesity (BMI 30+).', 'Active diagnoses on record: chronic sinusitis and anemia. Other findings: prediabetes and obesity (BMI 30+).'),
    ("corpus", "hypertension_326bc803-f772-44f7-cfc3-ed110deba70d", "objective",
     'Active diagnoses include dependent drug abuse and prediabetes.', 'Active diagnoses include dependent drug abuse. Other findings: prediabetes.'),
    ("corpus", "hypertension_3bf33842-d65b-38c3-fbe4-bf4922746011", "subjective",
     'Active problems include anemia (ongoing for 7 years) and chronic pain, chronic neck pain, and chronic low back pain (all noted 3 years earlier).', 'Active problems include anemia (ongoing for 7 years). Other findings: chronic pain, chronic neck pain, and chronic low back pain (all noted 3 years earlier).'),
    ("corpus", "hypertension_8d01f7e0-df18-2186-f427-4bc018b6ef94", "subjective",
     'Active problems also include prediabetes and obesity.', 'Other findings: prediabetes and obesity.'),
    ("corpus", "hypertension_924b0d39-ec2c-84c3-2509-2a34964536f3", "subjective",
     'Active problems include anemia (onset 7 days prior) and prediabetes.', 'Active problems include anemia (onset 7 days prior). Other findings: prediabetes.'),
    ("corpus", "hypertension_924b0d39-ec2c-84c3-2509-2a34964536f3", "objective",
     'Active problems: Anemia (disorder), Prediabetes (finding).', 'Active problems: Anemia (disorder). Other findings: Prediabetes (finding).'),
    ("corpus", "hypertension_a8449755-5cc6-5c4a-346c-d62445d06bd8", "objective",
     'Active diagnoses on record: asthma, loss of teeth, anemia, prediabetes.', 'Active diagnoses on record: asthma, loss of teeth, and anemia. Other findings: prediabetes.'),
    ("corpus", "hypertension_f3906712-1aab-696a-cc07-375c4e0b1152", "objective",
     'Active diagnoses on record: chronic sinusitis (disorder), anemia (disorder), prediabetes (finding), body mass index 30+ - obesity (finding).', 'Active diagnoses on record: chronic sinusitis (disorder) and anemia (disorder). Other findings: prediabetes (finding) and body mass index 30+ - obesity (finding).'),
    ("corpus", "strep_throat_20d0cf74-53a2-a394-3ef2-0405c984f530", "objective",
     'Active diagnoses include anemia, prediabetes, obesity, chronic pain, chronic neck pain, and chronic low back pain.', 'Active diagnoses include anemia. Other findings: prediabetes, obesity, chronic pain, chronic neck pain, and chronic low back pain.'),
    ("corpus", "strep_throat_d380f61f-acca-5fb4-3aa7-d2834efaf9c4", "objective",
     'Active diagnoses include anemia (disorder) and prediabetes (finding).', 'Active diagnoses include anemia (disorder). Other findings: prediabetes (finding).'),
    ("corpus", "strep_throat_d60dafd2-3ce9-d8b2-3f74-1897929098c1", "objective",
     'Relevant active diagnoses include essential hypertension, ischemic heart disease, mitral valve regurgitation, anemia, and history of renal transplant.', 'Relevant active diagnoses include essential hypertension, ischemic heart disease, mitral valve regurgitation, and anemia. Other findings: history of renal transplant.'),
    ("corpus", "strep_throat_d7477532-5d92-d05c-2b6d-05fb8d98d3ca", "objective",
     'Active problems include anemia (disorder) and prediabetes (finding).', 'Active problems include anemia (disorder). Other findings: prediabetes (finding).'),
    ("corpus", "viral_pharyngitis_260e3a24-e40a-1908-4ff0-80c4ddf975b3", "objective",
     'Active diagnoses include anemia, recurrent urinary tract infection, gingivitis, and prediabetes.', 'Active diagnoses include anemia, recurrent urinary tract infection, and gingivitis. Other findings: prediabetes.'),
    ("corpus", "viral_pharyngitis_44f69c13-1103-01ab-0a12-6d2cc7f6466f", "objective",
     'Active diagnoses: chronic sinusitis, essential hypertension, normal pregnancy.', 'Active diagnoses: chronic sinusitis and essential hypertension. Other findings: normal pregnancy.'),
    ("corpus", "viral_pharyngitis_4b0661e4-a9ce-5c38-7a37-9faf42eae985", "objective",
     'Active diagnoses include essential hypertension, metabolic syndrome X, anemia, and prediabetes.', 'Active diagnoses include essential hypertension, metabolic syndrome X, and anemia. Other findings: prediabetes.'),
    ("corpus", "viral_pharyngitis_4b4e4da4-2846-b8b0-9e99-dab9e6eb45ea", "objective",
     'Active diagnoses: seizure disorder, essential hypertension, prediabetes (finding).', 'Active diagnoses: seizure disorder and essential hypertension. Other findings: prediabetes (finding).'),
    ("corpus", "viral_pharyngitis_7bc3372b-ed7b-bfa6-b2d2-7c8dde066f37", "objective",
     'Active problem: obesity (BMI 30+).', 'Other findings: obesity (BMI 30+).'),
    ("corpus", "viral_pharyngitis_93029974-c130-59a6-6eb3-ec7c124c5b60", "objective",
     'Active diagnoses include essential hypertension (disorder) and prediabetes (finding).', 'Active diagnoses include essential hypertension (disorder). Other findings: prediabetes (finding).'),
    ("corpus", "viral_pharyngitis_a844859f-3407-a3a9-56d1-8de5c87fcc7a", "objective",
     'Active problems include essential hypertension (13 years), ischemic heart disease (13 years), chronic kidney disease stage 4 (3 years), anemia (3 years), neoplasm of prostate (6 years), and awaiting kidney transplantation.', 'Active problems include essential hypertension (13 years), ischemic heart disease (13 years), chronic kidney disease stage 4 (3 years), anemia (3 years), and neoplasm of prostate (6 years). Other findings: awaiting kidney transplantation.'),
    ("corpus", "viral_pharyngitis_d92ece43-d4a7-d2d4-da65-fea91e4c9d96", "objective",
     'Active diagnoses include anemia (disorder) and prediabetes (finding).', 'Active diagnoses include anemia (disorder). Other findings: prediabetes (finding).'),
    ("corpus", "viral_pharyngitis_ee508733-c2c3-527e-af30-68b6dd793d7b", "objective",
     'Active diagnoses: anemia (disorder), hyperlipidemia (disorder), prediabetes (finding).', 'Active diagnoses: anemia (disorder) and hyperlipidemia (disorder). Other findings: prediabetes (finding).'),
    ("corpus", "viral_pharyngitis_fed74a07-da20-9d51-484f-0b475f58ae47", "objective",
     'Active diagnoses on record include prediabetes.', 'Other findings: prediabetes.'),
    ("corpus", "bacterial_sinusitis_63c651f3-9208-cbb1-1ce4-124ccd6fcef2", "objective",
     'Active diagnoses include chronic sinusitis, anemia, normal pregnancy, and prediabetes.', 'Active diagnoses include chronic sinusitis and anemia. Other findings: normal pregnancy and prediabetes.'),
    ("corpus", "cystitis_09386e6a-b488-bbe8-e82b-c331b30a002c", "objective",
     'Active problems include anemia (disorder) and prediabetes (finding, noted 1 year earlier).', 'Active problems include anemia (disorder). Other findings: prediabetes (finding, noted 1 year earlier).'),
    ("corpus", "cystitis_275d5c69-f012-f2bd-5c1c-58e8469506ed", "objective",
     'Active diagnoses: chronic intractable migraine without aura (since 20 years earlier), anemia (since 9 years earlier), prediabetes (finding, 9 years earlier).', 'Active diagnoses: chronic intractable migraine without aura (since 20 years earlier) and anemia (since 9 years earlier). Other findings: prediabetes (finding, 9 years earlier).'),
    ("corpus", "cystitis_62dbf64c-f8e6-de62-2d58-f179965baaab", "objective",
     'Active diagnoses include obstructive sleep apnea (on CPAP), ischemic heart disease, aortic valve stenosis, neoplasm of prostate, anemia, essential hypertension, chronic kidney disease stage 2, and history of renal transplant.', 'Active diagnoses include obstructive sleep apnea (on CPAP), ischemic heart disease, aortic valve stenosis, neoplasm of prostate, anemia, essential hypertension, and chronic kidney disease stage 2. Other findings: history of renal transplant.'),
    ("corpus", "cystitis_86b7049a-4990-96f7-fdfd-050a6622f5e1", "objective",
     'Active diagnoses include normal pregnancy.', 'Other findings: normal pregnancy.'),
    ("corpus", "cystitis_a8449755-5cc6-5c4a-346c-d62445d06bd8", "objective",
     'Active diagnoses include chronic kidney disease stage 1, ischemic heart disease, essential hypertension, anemia, hyperlipidemia, and history of renal transplant.', 'Active diagnoses include chronic kidney disease stage 1, ischemic heart disease, essential hypertension, anemia, and hyperlipidemia. Other findings: history of renal transplant.'),
    ("corpus", "cystitis_b7bfa96f-5e77-00c9-8522-9ef7972f26ff", "objective",
     'Active diagnoses: Anemia (disorder), Normal pregnancy (finding).', 'Active diagnoses: Anemia (disorder). Other findings: Normal pregnancy (finding).'),
    ("corpus", "cystitis_e0276e7c-f3eb-7e02-c947-2124ac3cf479", "objective",
     'Active diagnoses: anemia, essential hypertension, metabolic syndrome X, prediabetes (finding).', 'Active diagnoses: anemia, essential hypertension, and metabolic syndrome X. Other findings: prediabetes (finding).'),
    ("corpus", "viral_pharyngitis_5502095a-22b4-66da-ace2-766a7b3eef70", "objective",
     ', dependent drug abuse, and history of renal transplant.', ', and dependent drug abuse. Other findings: history of renal transplant.'),
    ("corpus", "viral_pharyngitis_f5e5a013-dd03-d53d-b405-f68da0855ddc", "objective",
     ', microalbuminuria and proteinuria due to type 2 diabetes mellitus, and nonproliferative diabetic retinopathy.', ', and microalbuminuria and proteinuria due to type 2 diabetes mellitus. Other findings: nonproliferative diabetic retinopathy.'),
    # "currently pregnant" for a pregnancy status recorded earlier
    ("corpus", "strep_throat_9a61d1d1-839e-f9cf-0d04-cabe0afc81de", "subjective",
     'Patient is currently pregnant', 'Pregnancy status: pregnant (2 months earlier)'),
    ("corpus", "strep_throat_b0ef6c6f-7a77-be94-4537-054ec1253261", "subjective",
     'Currently pregnant', 'Pregnancy status: pregnant (7 months earlier)'),
    ("corpus", "strep_throat_eff508c7-d69c-c122-f5bd-aa5e35142892", "subjective",
     'Currently pregnant', 'Pregnancy status: pregnant (5 months earlier)'),
    ("corpus", "viral_pharyngitis_0cc0978e-9eab-7fd2-e51b-27ea0b606e4a", "subjective",
     'Patient is currently pregnant', 'Pregnancy status: pregnant (7 months earlier)'),
    ("corpus", "viral_pharyngitis_3a333641-b2aa-1876-6b4d-717482736fd7", "subjective",
     'Patient is currently pregnant (noted 1 month earlier)', 'Pregnancy status: pregnant (1 month earlier)'),
    ("corpus", "viral_pharyngitis_4464601a-9058-4935-45ac-e5cdda42042f", "subjective",
     'Patient is currently pregnant', 'Pregnancy status: pregnant (7 months earlier)'),
    ("corpus", "viral_pharyngitis_44f69c13-1103-01ab-0a12-6d2cc7f6466f", "subjective",
     'Patient is currently pregnant', 'Pregnancy status: pregnant (1 month earlier)'),
    ("corpus", "viral_pharyngitis_63c651f3-9208-cbb1-1ce4-124ccd6fcef2", "subjective",
     'Patient is currently pregnant (noted 5 months earlier)', 'Pregnancy status: pregnant (5 months earlier)'),
    ("corpus", "viral_pharyngitis_9020a37c-2756-ec7b-c150-7d53d85e8380", "subjective",
     'Patient is currently pregnant', 'Pregnancy status: pregnant (2 months earlier)'),
    ("corpus", "viral_pharyngitis_b85bfae6-8ed9-7e14-902b-70a28b4d427f", "subjective",
     'Patient is currently pregnant (noted 3 months earlier)', 'Pregnancy status: pregnant (3 months earlier)'),
    ("corpus", "viral_pharyngitis_c0f0da32-292e-ffb0-d80b-fa3d129dab65", "subjective",
     'Currently pregnant (noted 4 months earlier)', 'Pregnancy status: pregnant (4 months earlier)'),
    ("corpus", "viral_pharyngitis_d290cbad-46c9-88a4-e8d4-0c87679b4708", "subjective",
     'Currently pregnant (7 months earlier)', 'Pregnancy status: pregnant (7 months earlier)'),
    ("corpus", "viral_pharyngitis_dc3b1550-3931-8632-be63-c743dd0aa657", "subjective",
     'Patient is currently pregnant (4 months)', 'Pregnancy status: pregnant (4 months earlier)'),
    ("corpus", "viral_pharyngitis_eb176939-f743-63b6-5aae-002953535125", "subjective",
     'Patient is currently pregnant', 'Pregnancy status: pregnant (5 months earlier)'),
    ("corpus", "viral_pharyngitis_fc2bbdfd-a525-f924-9755-98645a4574c5", "subjective",
     'Patient is currently pregnant (noted 8 months earlier)', 'Pregnancy status: pregnant (8 months earlier)'),
    ("corpus", "bacterial_sinusitis_0eb6f197-22bf-8e07-cbaa-9194eaa3a1df", "subjective",
     'Currently pregnant', 'Pregnancy status: pregnant (3 months earlier)'),
    ("corpus", "bacterial_sinusitis_3aff3eac-93f6-8813-3ec5-bcbdaa988148", "subjective",
     'Currently pregnant', 'Pregnancy status: pregnant (7 months earlier)'),
    ("corpus", "bacterial_sinusitis_63c651f3-9208-cbb1-1ce4-124ccd6fcef2", "subjective",
     'Patient is currently pregnant (noted 1 month earlier)', 'Pregnancy status: pregnant (1 month earlier)'),
    ("corpus", "bacterial_sinusitis_7b7a7485-071c-4dc0-8cfa-95faa72c3248", "subjective",
     'Currently pregnant (noted 8 months earlier)', 'Pregnancy status: pregnant (8 months earlier)'),
    ("corpus", "bacterial_sinusitis_807f4792-6d56-8fde-61e2-ed3ec6daad9a", "subjective",
     'Patient is currently pregnant', 'Pregnancy status: pregnant (2 months earlier)'),
    ("corpus", "bacterial_sinusitis_9bfa4f90-68b4-8069-7df6-6e137cb67491", "subjective",
     'Patient is currently pregnant', 'Pregnancy status: pregnant (1 month earlier)'),
    ("corpus", "bacterial_sinusitis_b7c52633-4342-d1e7-3a76-1fef71bb9b8b", "subjective",
     'Patient is currently pregnant', 'Pregnancy status: pregnant (3 months earlier)'),
    ("corpus", "cystitis_1c80d0d5-4ebd-5efc-e654-340c5599a734", "subjective",
     'Currently pregnant (5 months)', 'Pregnancy status: pregnant (5 months earlier)'),
    ("corpus", "cystitis_2062088f-97b3-5cd4-3f83-201daa6aad63", "subjective",
     'Currently pregnant', 'Pregnancy status: pregnant (9 months earlier)'),
    ("corpus", "cystitis_6bfa542f-a9ba-5468-61b6-b9037bdadb36", "subjective",
     'Patient is currently pregnant (4 months earlier noted)', 'Pregnancy status: pregnant (4 months earlier)'),
    ("corpus", "cystitis_76091734-7f86-e7b5-607f-65d14fad37ec", "subjective",
     'Patient is currently pregnant (noted 4 months earlier)', 'Pregnancy status: pregnant (4 months earlier)'),
    ("corpus", "cystitis_86b7049a-4990-96f7-fdfd-050a6622f5e1", "subjective",
     'Patient is currently pregnant (noted 8 months earlier)', 'Pregnancy status: pregnant (8 months earlier)'),
    ("corpus", "cystitis_9c933b61-71cb-99a7-b2a3-69148f154c76", "subjective",
     'Currently pregnant', 'Pregnancy status: pregnant (2 days earlier)'),
    ("corpus", "cystitis_b7bfa96f-5e77-00c9-8522-9ef7972f26ff", "subjective",
     'Patient is currently pregnant', 'Pregnancy status: pregnant (3 months earlier)'),
    ("corpus", "cystitis_b85bfae6-8ed9-7e14-902b-70a28b4d427f", "subjective",
     'Patient is currently pregnant (noted 4 months earlier)', 'Pregnancy status: pregnant (4 months earlier)'),
    ("corpus", "cystitis_c9a95290-f737-e341-ccfe-4c4b593942a2", "subjective",
     'Patient is currently pregnant (noted 15 days earlier)', 'Pregnancy status: pregnant (15 days earlier)'),
    ("corpus", "cystitis_d2c12c75-d07f-76aa-2104-2df282516d4b", "subjective",
     'Currently pregnant (noted 1 month earlier)', 'Pregnancy status: pregnant (1 month earlier)'),
    ("corpus", "cystitis_e15de763-4d76-ae66-2de3-9fcc7a954f5b", "subjective",
     'Currently pregnant (noted 4 months earlier)', 'Pregnancy status: pregnant (4 months earlier)'),
    ("corpus", "cystitis_e7bbfc7e-39db-9f7c-8f43-9d37fa5e1196", "subjective",
     'Patient is currently pregnant (noted 7 months earlier)', 'Pregnancy status: pregnant (7 months earlier)'),
    ("corpus", "cystitis_fc2bbdfd-a525-f924-9755-98645a4574c5", "subjective",
     'Patient is currently pregnant (noted 2 months earlier)', 'Pregnancy status: pregnant (2 months earlier)'),
    ("corpus", "cystitis_ea6b0c13-660f-fb81-22ca-171f036ba14b", "objective",
     'Currently pregnant with ultrasound scan', 'Pregnancy status: pregnant (1 month earlier). Ultrasound scan'),
    ("corpus", "cystitis_fc2bbdfd-a525-f924-9755-98645a4574c5", "objective",
     'currently pregnant with normal pregnancy', 'pregnancy status: pregnant (2 months earlier), with normal pregnancy'),
    # past problem given a resolution the source never states
    ("corpus", "hypertension_9bd1114b-aaec-2959-ca33-92d8c84519c4", "subjective",
     'gingivitis (disorder) resolved 1 year earlier', 'gingivitis (disorder) 1 year earlier'),
    ("corpus", "viral_pharyngitis_29d178cf-aa59-c1b3-8764-6b7d94045983", "objective",
     'Past diagnosis of gingivitis (resolved).', 'Past diagnosis of gingivitis.'),
    ("corpus", "viral_pharyngitis_c806c90b-5ca9-d47c-8fa9-addce7e15d87", "objective",
     'Past problem of gingivitis (resolved).', 'Past problem of gingivitis.'),
    ("corpus", "viral_pharyngitis_d1f3c85a-5d0b-93ea-60fa-afba13df1635", "subjective",
     'gingivitis (resolved 3 years earlier)', 'gingivitis (3 years earlier)'),
    # interpretations the source never states
    ("corpus", "hypertension_260e3a24-e40a-1908-4ff0-80c4ddf975b3", "subjective",
     'a chief complaint reflected by elevated blood pressure at this visit', 'blood pressure recorded at this visit'),
    ("corpus", "hypertension_662df830-75f8-77f2-2bc8-6eaf2254a1eb", "subjective",
     'a chief complaint reflected by elevated blood pressure at this visit', 'blood pressure recorded at this visit'),
    ("corpus", "hypertension_91d1a7b0-d6c1-5703-01da-6c68aaf21adb", "subjective",
     'a chief complaint reflected by elevated blood pressure at this visit', 'blood pressure recorded at this visit'),
    ("corpus", "hypertension_a7f42570-a263-4aca-a5bd-d936a7776a32", "subjective",
     'a chief complaint reflected by elevated blood pressure at this visit', 'blood pressure recorded at this visit'),
    ("corpus", "hypertension_c257f005-456b-f846-321a-360c6ca718aa", "subjective",
     'a chief complaint reflected by elevated blood pressure at this visit', 'blood pressure recorded at this visit'),
    ("corpus", "hypertension_e74a7540-ddb8-a775-8d1a-4b8385051f26", "subjective",
     'a chief complaint reflected by elevated blood pressure at this visit', 'blood pressure recorded at this visit'),
    ("corpus", "hypertension_64a26838-ffa3-c049-2a7e-c198cf2b70e6", "subjective",
     'presents with an elevated blood pressure noted at this visit', 'presents with blood pressure recorded at this visit'),
    ("corpus", "hypertension_7d208a3f-378d-7e13-5709-dadac9326d07", "subjective",
     'Patient reports presenting with elevated blood pressure.', 'Blood pressure recorded at this visit.'),
    ("corpus", "hypertension_99e1d312-3eed-a97b-2f55-aa5c43609734", "subjective",
     'presenting with elevated blood pressure at this visit', 'presenting with blood pressure recorded at this visit'),
    ("corpus", "strep_throat_5aa85d6f-7a0c-8e53-6282-c2a417dbcb19", "subjective",
     'elevated body temperature of 38.8°C', 'body temperature of 38.8°C'),
    ("corpus", "viral_pharyngitis_74e56077-3646-6a5e-bf57-87aa45e67799", "objective",
     'BMI percentile has been elevated over prior visits (96.6%–98%).', 'BMI percentile over prior visits: 96.6%–98%.'),
    ("corpus", "viral_pharyngitis_c60d2162-924b-0772-2cd8-a815dbb9a3b6", "objective",
     'notable for elevated systolic blood pressure', 'include systolic blood pressure'),
    ("corpus", "viral_pharyngitis_c60d2162-924b-0772-2cd8-a815dbb9a3b6", "objective",
     ', consistent with prior anemia diagnosis', ''),
    ("corpus", "bacterial_sinusitis_7f58ad3b-7f02-d661-b018-eed0db4b3223", "objective",
     'include elevated ferritin (476.9 ug/L), elevated LDH (345.3 U/L), elevated bilirubin (9.9 mg/dL), elevated creatinine (2.9 mg/dL), low GFR (5.4 mL/min), low oxygen saturation (84.9%),', 'include ferritin 476.9 ug/L, LDH 345.3 U/L, bilirubin 9.9 mg/dL, creatinine 2.9 mg/dL, GFR 5.4 mL/min, oxygen saturation 84.9%,'),
    ("corpus", "cystitis_5f07b566-38b5-8d51-2e1e-70b2dfabdd41", "subjective",
     'a notably elevated GAD-7 score of 14', 'a GAD-7 score of 14'),
    ("corpus", "cystitis_ee0ef782-cc69-c829-6646-59d2f96749d7", "subjective",
     ', indicating significant depressive symptoms', ''),
    ("corpus", "bacterial_sinusitis_04adcffd-72a5-f918-7760-86a5f6906708", "objective",
     'others low or undetectable.', 'others 0–0.2 kU/L.'),
    ("corpus", "cystitis_6828f563-bf90-9ec7-0346-58c8abd4e9f0", "objective",
     'with low-level sensitization to egg white, honey bee, latex, and shrimp (0.3 kU/L each)', 'with egg white, honey bee, latex, and shrimp at 0.3 kU/L each'),
    ("corpus", "cystitis_2b0d5c2f-37af-afd0-d40d-9e2979925505", "objective",
     ' with low-level results across multiple allergens.', '.'),
    ("corpus", "cystitis_c01b1a86-9013-7209-646d-6072f96fa73f", "objective",
     ' with low-level results across multiple allergens.', '.'),
    # leaked diagnosis / treatment
    ("corpus", "cystitis_6d3a306a-6575-9f79-7523-a4ec3ade0103", "subjective",
     '25-year-old male presenting with a chief complaint consistent with urinary symptoms.', '25-year-old male presenting.'),
    ("corpus", "cystitis_1d5b8492-5baa-cb75-75d6-2d5ea5f02e34", "subjective",
     '37-year-old male presenting with no treatment recorded.', '37-year-old male presenting.'),
    # wrong indication (hydrocortisone is for contact dermatitis)
    ("corpus", "cystitis_ee0ef782-cc69-c829-6646-59d2f96749d7", "objective",
     'hydrocortisone cream, chlorpheniramine, and epinephrine auto-injector for allergic disposition;', 'hydrocortisone cream for contact dermatitis; chlorpheniramine and epinephrine auto-injector;'),
    # items given another item's date
    ("corpus", "bacterial_sinusitis_7f58ad3b-7f02-d661-b018-eed0db4b3223", "objective",
     'acute viral pharyngitis, streptococcal sore throat, acute deep venous thrombosis', 'acute viral pharyngitis (most recent 1 year earlier), streptococcal sore throat (3 years earlier), acute deep venous thrombosis'),
    ("corpus", "bacterial_sinusitis_7f58ad3b-7f02-d661-b018-eed0db4b3223", "objective",
     '(all 4 years earlier or more)', '(all 4 years earlier)'),
    # invented time span (counts with only the latest date)
    ("corpus", "cystitis_cbe682d6-c9e9-e777-3c45-5924b918f06d", "objective",
     'medication courses over the past year.', 'medication courses (most recent 4 months earlier).'),
    # earlier pain scores framed as the presentation
    ("corpus", "hypertension_23797d60-3fbe-3feb-9107-bc01a9bfe0cb", "subjective",
     'Patient presents with no reported pain (pain severity score 0).', 'Patient reported no pain (pain severity score 0, 14 days earlier).'),
    ("corpus", "hypertension_5d4d1e64-9589-4fa2-50a0-14d11f37ff98", "subjective",
     'Patient presents with no reported pain (pain severity score 0).', 'Patient reported no pain (pain severity score 0, 2 years earlier).'),
    # scores from an earlier visit labelled "at most recent / last visit" (a later visit exists)
    ("corpus", "cystitis_52b11457-711a-17a6-9efe-f7da95392904", "subjective",
     '1 out of 10 at the most recent visit.', '1 out of 10 (14 days earlier).'),
    ("corpus", "hypertension_3c820294-9a75-7cb4-6c79-44ad830fd091", "subjective",
     'Pain severity rated 3/10 at most recent visit.', 'Pain severity rated 3/10 (11 months earlier).'),
    ("corpus", "hypertension_3c820294-9a75-7cb4-6c79-44ad830fd091", "subjective",
     'GAD-7 total score of 6 at most recent visit.', 'GAD-7 total score of 6 (11 months earlier).'),
    ("corpus", "hypertension_47e6bf50-88b3-7fa1-3222-d2443d3103c4", "subjective",
     '1 out of 10 at last visit.', '1 out of 10 (1 year earlier).'),
    ("corpus", "hypertension_c257f005-456b-f846-321a-360c6ca718aa", "subjective",
     '4 out of 10 at most recent lab visit.', '4 out of 10 (11 months earlier).'),
    ("corpus", "hypertension_de4cd231-b267-7034-d067-f48c3fb2163a", "subjective",
     '3 out of 10 at most recent visit.', '3 out of 10 (11 months earlier).'),
    ("corpus", "hypertension_de4cd231-b267-7034-d067-f48c3fb2163a", "subjective",
     'PHQ-9 score of 4 noted at most recent visit.', 'PHQ-9 score of 4 (both 11 months earlier).'),
    ("corpus", "hypertension_de4cd231-b267-7034-d067-f48c3fb2163a", "subjective",
     'GAD-7 score of 7 noted at most recent visit.', 'GAD-7 score of 7 (11 months earlier).'),
    ("corpus", "cystitis_37f03ddc-18b0-a9fb-26bd-64f657abbb07", "subjective",
     '2/10 at most recent visit.', '2/10 (3 months earlier).'),
    ("corpus", "cystitis_37f03ddc-18b0-a9fb-26bd-64f657abbb07", "subjective",
     'PHQ-2 score of 0 at most recent visit.', 'PHQ-2 score of 0 (both 3 months earlier).'),
    ("corpus", "cystitis_5f07b566-38b5-8d51-2e1e-70b2dfabdd41", "subjective",
     '0-10 scale at most recent visit.', '0-10 scale (14 days earlier).'),
    ("corpus", "cystitis_5f07b566-38b5-8d51-2e1e-70b2dfabdd41", "objective",
     '0 at most recent visit.', '0 (14 days earlier).'),
    ("corpus", "cystitis_7e69274c-6c5e-cacd-2334-2b9e1d58453f", "subjective",
     'PHQ-9 total score of 12 at most recent visit.', 'PHQ-9 total score of 12 (both 14 days earlier).'),
    ("corpus", "cystitis_cc6ff126-14fe-dd29-3d62-bbb6033bde67", "subjective",
     '2/10 at most recent visit.', '2/10 (14 days earlier).'),
    ("corpus", "cystitis_db55948d-fa7b-fb6e-c831-2ab6e8d45309", "subjective",
     '4/10 at most recent visit.', '4/10 (28 days earlier).'),
    ("corpus", "cystitis_db55948d-fa7b-fb6e-c831-2ab6e8d45309", "subjective",
     'PHQ-2 score of 1 at most recent visit.', 'PHQ-2 score of 1 (28 days earlier).'),
    # immunizations/procedures from an earlier visit called "at this visit"
    ("corpus", "strep_throat_85491dcd-ccbb-778b-c7ce-ce2ca03c6d80", "objective",
     'Immunizations administered at this visit:', 'Immunizations administered 7 days earlier:'),
    ("corpus", "strep_throat_897b3cdd-d541-8e51-6636-a343e0f297e4", "objective",
     'Influenza immunization administered at this visit.', 'Influenza immunization administered 14 days earlier.'),
    ("corpus", "viral_pharyngitis_17dd3774-9f6f-cd26-cb63-93e331958793", "objective",
     'Immunizations administered at this visit:', 'Immunizations administered 14 days earlier:'),
    ("corpus", "viral_pharyngitis_425ed0e1-ab2e-7b0c-90c0-d9012553d939", "objective",
     'Immunizations administered at this visit:', 'Immunizations administered 7 days earlier:'),
    ("corpus", "viral_pharyngitis_dcee36f7-ff9a-7274-aa31-68d406ea2761", "objective",
     'Influenza vaccine and IPV administered at this visit.', 'Influenza vaccine and IPV administered 7 days earlier.'),
    ("corpus", "viral_pharyngitis_7123e66a-a067-a9b2-988d-fa4221ce343a", "objective",
     'Medication reconciliation performed at this visit.', 'Medication reconciliation performed 3 days earlier.'),
    ("corpus", "viral_pharyngitis_7123e66a-a067-a9b2-988d-fa4221ce343a", "objective",
     'Influenza vaccine administered at this visit.', 'Influenza vaccine administered 3 days earlier.'),
    ("corpus", "bacterial_sinusitis_ecd176f1-a1ba-1956-6c86-d5a46ee355d1", "objective",
     'Medication reconciliation performed at current visit.', 'Medication reconciliation performed 9 days earlier.'),
    ("corpus", "strep_throat_cfba649e-1c2c-7431-2a14-66358b106bc9", "objective",
     'Procedures performed at this visit include', 'Procedures performed 1 month earlier include'),
    ("corpus", "viral_pharyngitis_0d90e0ce-dba9-de12-cf2d-12e7efd7edd1", "objective",
     'Depression screening and PHQ-9 performed at this visit.', 'Depression screening and PHQ-9 performed 14 days earlier.'),
    # corpus_nota, final pass
    # findings (prediabetes, obesity, history of MI) listed as active diagnoses
    ("corpus_nota", "nota_covid19_1b718ebf-39f4-3515-e346-c9fe932a73b4", "objective",
     'Active diagnoses include anemia and prediabetes.', 'Active diagnoses include anemia. Other findings: prediabetes.'),
    ("corpus_nota", "nota_covid19_261aec35-26ef-712b-d143-9972e23a54cb", "objective",
     'Active diagnoses include essential hypertension and prediabetes.', 'Active diagnoses include essential hypertension. Other findings: prediabetes.'),
    ("corpus_nota", "nota_acute_bronchitis_44006fc9-48bf-b05e-ac05-ecf114460907", "objective",
     'Active diagnoses: chronic sinusitis, anemia, prediabetes.', 'Active diagnoses: chronic sinusitis and anemia. Other findings: prediabetes.'),
    ("corpus_nota", "nota_covid19_51efae03-d695-5f54-1cbc-1a95f5bb0c25", "objective",
     'Active diagnoses: chronic sinusitis, anemia, prediabetes.', 'Active diagnoses: chronic sinusitis and anemia. Other findings: prediabetes.'),
    ("corpus_nota", "nota_acute_bronchitis_546eb117-8683-0453-fc48-09a45eec6caf", "objective",
     'Active diagnoses include anemia and prediabetes.', 'Active diagnoses include anemia. Other findings: prediabetes.'),
    ("corpus_nota", "nota_covid19_58118665-36fc-f625-6477-ba9e84dc51f0", "objective",
     'Active diagnoses include prediabetes (finding) and prior finding of obesity.', 'Other findings: prediabetes (finding) and prior finding of obesity.'),
    ("corpus_nota", "nota_covid19_6e482b47-34f4-80d0-5a02-915a16cdfd60", "objective",
     'Active diagnoses include ischemic heart disease, history of MI, anemia, osteoporosis, chronic sinusitis, and prediabetes.',
     'Active diagnoses include ischemic heart disease, anemia, osteoporosis, and chronic sinusitis. Other findings: history of MI and prediabetes.'),
    ("corpus_nota", "nota_acute_bronchitis_a27b5dda-ee8e-7c7b-1049-82ac56866ec4", "objective",
     'Active diagnoses: sleep disorder, obstructive sleep apnea syndrome, prediabetes (historical finding).',
     'Active diagnoses: sleep disorder and obstructive sleep apnea syndrome. Other findings: prediabetes (historical finding).'),
    ("corpus_nota", "nota_covid19_d380f61f-acca-5fb4-3aa7-d2834efaf9c4", "objective",
     'Active diagnoses include anemia (disorder) and prediabetes (finding).', 'Active diagnoses include anemia (disorder). Other findings: prediabetes (finding).'),
    ("corpus_nota", "nota_acute_bronchitis_ec156cea-eb47-8c7a-0873-fc722bec52ec", "objective",
     'Active diagnoses include anemia (disorder) and prediabetes (finding).', 'Active diagnoses include anemia (disorder). Other findings: prediabetes (finding).'),
    ("corpus_nota", "nota_acute_bronchitis_fdfc3f4d-d43e-fae4-b55b-d03e93148941", "objective",
     'Active diagnoses include essential hypertension, chronic kidney disease stage 3, and prediabetes.',
     'Active diagnoses include essential hypertension and chronic kidney disease stage 3. Other findings: prediabetes.'),
    # past problem called active
    ("corpus_nota", "nota_acute_bronchitis_afa35e45-c10e-6198-ba11-2b684695084d", "objective",
     'childhood asthma (active, with asthma self-management care plan)', 'childhood asthma (14 years earlier; asthma self-management care plan)'),
    # past medication called current
    ("corpus_nota", "nota_covid19_4947f1c7-9869-5aa4-efd3-7ed54e54f6d7", "subjective",
     'Currently on Acetaminophen', 'Past medication (0 days earlier): Acetaminophen'),
    # indication the source doesn't give
    ("corpus_nota", "nota_covid19_0a37d7ac-e80e-1202-7388-30230a8d3235", "subjective",
     'Ibuprofen 400 MG for pain management.', 'Ibuprofen 400 MG.'),
    ("corpus_nota", "nota_covid19_ce3a1395-8978-9378-1121-95e2e27b2106", "subjective",
     'opioid and analgesic agents for pain management.', 'opioid and analgesic agents.'),
    ("corpus_nota", "nota_acute_bronchitis_e3f5a45a-39f2-962f-67ef-1a504862805b", "subjective",
     'Percocet and tramadol for pain management.', 'Percocet and tramadol.'),
    # earlier values called "current"
    ("corpus_nota", "nota_acute_bronchitis_6089f0a7-55d0-38fe-f6e5-82f4b7a27d88", "subjective",
     'Current stress level reported as quite a bit.', 'Stress level reported as quite a bit (7 days earlier).'),
    ("corpus_nota", "nota_acute_bronchitis_b48d0f1b-735b-27c3-4cdc-99d1dd6c857c", "subjective",
     'Current stress level reported as quite a bit.', 'Stress level reported as quite a bit (1 month earlier).'),
    ("corpus_nota", "nota_acute_bronchitis_0cc0978e-9eab-7fd2-e51b-27ea0b606e4a", "subjective",
     'current quite a bit stress level', 'quite a bit stress level (2 months earlier)'),
    ("corpus_nota", "nota_covid19_d380f61f-acca-5fb4-3aa7-d2834efaf9c4", "subjective",
     'Patient presents with a reported pain severity of 3/10.', 'Patient reported a pain severity of 3/10 (5 months earlier).'),
    # scores labelled "at most recent visit" when a later visit exists
    ("corpus_nota", "nota_viral_sinusitis_68ca7b0a-203d-1305-02d0-cf36c0e65637", "subjective",
     'a pain severity score of 2/10 at the most recent recorded visit', 'a pain severity score of 2/10 (14 days earlier)'),
    # earlier visit type given as the current visit
    ("corpus_nota", "nota_acute_bronchitis_eedb4fdd-fede-a2eb-2bb9-50f198fad87e", "objective",
     'Encounter type: urgent care visit.', 'Urgent care visit 7 days earlier.'),
    # invented or misattributed dates
    ("corpus_nota", "nota_covid19_35d54707-2468-14c5-cf55-97840077573d", "objective",
     'Vitals (6 months prior to last visit):', 'Vitals (6 months earlier):'),
    ("corpus_nota", "nota_viral_sinusitis_8ff2c76c-03a1-4a5c-d032-1d2221666460", "objective",
     'Intrauterine System inserted 4 days earlier', 'Intrauterine System started 4 days earlier'),
    # "currently pregnant" for a pregnancy status recorded earlier
    ("corpus_nota", "nota_acute_bronchitis_09b13a6c-6724-adbe-c2d3-13c02df994fc", "subjective",
     'Patient is currently pregnant (noted 3 months earlier).', 'Pregnancy status: pregnant (3 months earlier).'),
    # corpus, final pass
    # earlier measurements labelled as current
    ("corpus", "strep_throat_ae575a01-5a3f-89bd-6c8a-45217d0eb6b4", "objective",
     'Current labs (8 days earlier):', 'Labs 8 days earlier:'),
    ("corpus", "viral_pharyngitis_006068d8-da63-6ba2-0fc2-d611826259dd", "objective",
     'Current measurements (9 days earlier):', 'Measurements 9 days earlier:'),
    ("corpus", "viral_pharyngitis_35d54707-2468-14c5-cf55-97840077573d", "objective",
     'Current visit (14 days earlier):', 'Vital signs 14 days earlier:'),
    # earlier vitals mixed into the current-visit sentence
    ("corpus", "strep_throat_d75ca155-5f46-3a42-5eb7-73c09fc1a0c5", "objective",
     'Vital signs at visit: temperature 37.8°C, blood pressure', 'Vital signs at visit: temperature 37.8°C. Vital signs 14 days earlier: blood pressure'),
    ("corpus", "viral_pharyngitis_64f17b66-d395-4424-1979-a550814f69fe", "objective",
     'Vital signs at visit: body temperature 37.4°C, heart rate', 'Vital signs at visit: body temperature 37.4°C. Vital signs 7 days earlier: heart rate'),
    # "most recent/last visit" values when a later visit exists
    ("corpus", "bacterial_sinusitis_a9e37cb5-c5a4-0aea-b940-dd2105a5488d", "objective",
     'Vitals at most recent visit (7 days earlier):', 'Vitals 7 days earlier:'),
    ("corpus", "bacterial_sinusitis_aca5045d-819c-cd1e-0822-067296639ad0", "objective",
     'Vitals at most recent visit (14 days earlier):', 'Vitals 14 days earlier:'),
    ("corpus", "bacterial_sinusitis_b76245b9-b32c-ed61-4cf4-52e132c99f3e", "objective",
     'Vitals at most recent visit (8 days earlier):', 'Vitals 8 days earlier:'),
    ("corpus", "heart_failure_843b00e6-3be7-aff8-f271-b2b220e43d6c", "objective",
     'Vitals at last visit (14 days earlier):', 'Vitals 14 days earlier:'),
    ("corpus", "cystitis_1b2fd46b-46fb-cf85-83e2-75cbe129631e", "subjective",
     '4 out of 10 reported at last visit.', '4 out of 10 reported 1 year earlier.'),
    ("corpus", "hypertension_546eb117-8683-0453-fc48-09a45eec6caf", "subjective",
     '1/10 at last visit (2 years earlier).', '1/10 (2 years earlier).'),
    ("corpus", "hypertension_723cb245-5a9f-be60-7975-c9af8030b659", "subjective",
     '3/10 at last recorded visit.', '3/10 (1 year earlier).'),
    ("corpus", "hypertension_81b74378-a378-c67f-021c-22dde23fe13b", "subjective",
     '2/10 at last recorded visit.', '2/10 (11 months earlier).'),
    # influenza vaccine given an earlier date called "at this visit"
    ("corpus", "strep_throat_006068d8-da63-6ba2-0fc2-d611826259dd", "objective",
     'Influenza vaccine administered at this visit', 'Influenza vaccine administered 14 days earlier'),
    ("corpus", "strep_throat_13180cf2-4f28-4143-ec9d-bf9abc40da6c", "objective",
     'Influenza vaccine administered at this visit', 'Influenza vaccine administered 15 days earlier'),
    ("corpus", "strep_throat_6b6533e2-14b0-085d-8724-2885864761b3", "objective",
     'Influenza vaccine administered at this visit', 'Influenza vaccine administered 1 day earlier'),
    ("corpus", "strep_throat_944d7287-b952-3bc3-1e3a-fdd63a570798", "objective",
     'Influenza vaccine administered at this visit', 'Influenza vaccine administered 7 days earlier'),
    ("corpus", "strep_throat_9a6b855c-d1dd-a985-b4ef-8adf2cd7bdff", "objective",
     'Influenza vaccine administered at this visit', 'Influenza vaccine administered 3 days earlier'),
    ("corpus", "strep_throat_ab5d5396-82e2-17d7-6e47-488e6e4dddd6", "objective",
     'Influenza vaccine administered at this visit', 'Influenza vaccine administered 2 days earlier'),
    ("corpus", "strep_throat_fc6769c1-7d84-90d9-aec0-5fd05ab908b3", "objective",
     'Influenza vaccine administered at this visit', 'Influenza vaccine administered 7 days earlier'),
    ("corpus", "viral_pharyngitis_006068d8-da63-6ba2-0fc2-d611826259dd", "objective",
     'Influenza vaccine administered at this visit', 'Influenza vaccine administered 9 days earlier'),
    ("corpus", "viral_pharyngitis_01091d90-7dea-6a1f-f99f-96aee288b930", "objective",
     'Influenza vaccine administered at this visit', 'Influenza vaccine administered 3 days earlier'),
    ("corpus", "viral_pharyngitis_03de2751-dace-33a3-7b08-73ef8d2c7755", "objective",
     'Influenza vaccine administered at this visit', 'Influenza vaccine administered 3 days earlier'),
    ("corpus", "viral_pharyngitis_0ed98963-6964-faee-7182-7e58ad6e26c2", "objective",
     'Influenza vaccine administered at this visit', 'Influenza vaccine administered 3 days earlier'),
    ("corpus", "viral_pharyngitis_122aad1d-d970-e249-40de-3cd3519089fc", "objective",
     'Influenza vaccine administered at this visit', 'Influenza vaccine administered 14 days earlier'),
    ("corpus", "viral_pharyngitis_13180cf2-4f28-4143-ec9d-bf9abc40da6c", "objective",
     'Influenza vaccine administered at this visit', 'Influenza vaccine administered 9 days earlier'),
    ("corpus", "viral_pharyngitis_1808d3eb-adb5-8424-6f9f-871bf1b5b5d1", "objective",
     'Influenza vaccine administered at this visit', 'Influenza vaccine administered 1 day earlier'),
    ("corpus", "viral_pharyngitis_18b7c7a7-be9e-ca86-a160-3f805e41ffa4", "objective",
     'Influenza vaccine administered at this visit', 'Influenza vaccine administered 9 days earlier'),
    ("corpus", "viral_pharyngitis_2a02d346-3f83-4d88-2922-fd91468c1e04", "objective",
     'Influenza vaccine administered at this visit', 'Influenza vaccine administered 28 days earlier'),
    ("corpus", "viral_pharyngitis_2d6f0589-6607-0ee7-362d-1a51778faa2a", "objective",
     'Influenza vaccine administered at this visit', 'Influenza vaccine administered 14 days earlier'),
    ("corpus", "viral_pharyngitis_30e883eb-7502-5590-cfb1-f34137f31c28", "objective",
     'Influenza vaccine administered at this visit', 'Influenza vaccine administered 3 days earlier'),
    ("corpus", "viral_pharyngitis_48ff19ee-135e-f45e-a4f1-b4cd6715bae9", "objective",
     'Influenza vaccine administered at this visit (9 days earlier)', 'Influenza vaccine administered 9 days earlier'),
    ("corpus", "viral_pharyngitis_55d6aaf9-56be-0862-d369-a0623c445e76", "objective",
     'Influenza vaccine administered at this visit', 'Influenza vaccine administered 25 days earlier'),
    ("corpus", "viral_pharyngitis_55e84570-094a-6b9a-10d0-eb941c963a24", "objective",
     'Influenza vaccine administered at this visit', 'Influenza vaccine administered 14 days earlier'),
    ("corpus", "viral_pharyngitis_61340079-1246-d4e7-74b4-05ba995e3db6", "objective",
     'Influenza vaccine administered at this visit', 'Influenza vaccine administered 7 days earlier'),
    ("corpus", "viral_pharyngitis_63985fdd-d734-1be7-f620-6f474e2f609b", "objective",
     'Influenza vaccine administered at this visit', 'Influenza vaccine administered 1 day earlier'),
    ("corpus", "viral_pharyngitis_64f17b66-d395-4424-1979-a550814f69fe", "objective",
     'Influenza vaccine administered at this visit', 'Influenza vaccine administered 7 days earlier'),
    ("corpus", "viral_pharyngitis_69653d27-2641-bf0d-7f62-16a11a8688ae", "objective",
     'Influenza vaccine administered at this visit', 'Influenza vaccine administered 1 day earlier'),
    ("corpus", "viral_pharyngitis_7e62cfe0-a515-6123-ac7f-5f81da2daba5", "objective",
     'Influenza vaccine administered at this visit', 'Influenza vaccine administered 14 days earlier'),
    ("corpus", "viral_pharyngitis_8319e29f-230d-7f5a-be4f-1de1c222fa95", "objective",
     'Influenza vaccine administered at this visit', 'Influenza vaccine administered 3 days earlier'),
    ("corpus", "viral_pharyngitis_8af5c769-e74f-be31-b0e7-94a51a7abe11", "objective",
     'Influenza vaccine administered at this visit', 'Influenza vaccine administered 3 months earlier'),
    ("corpus", "viral_pharyngitis_9081e8ad-1ee1-a605-0347-bbe8c983e780", "objective",
     'Influenza vaccine administered at this visit', 'Influenza vaccine administered 14 days earlier'),
    ("corpus", "viral_pharyngitis_90842be7-7934-5aeb-3611-5c087be050db", "objective",
     'Influenza vaccine administered at this visit', 'Influenza vaccine administered 15 days earlier'),
    ("corpus", "viral_pharyngitis_912cc29d-29f1-b1c5-162b-ec3ba8d69a79", "objective",
     'Influenza vaccine administered at this visit', 'Influenza vaccine administered 7 days earlier'),
    ("corpus", "viral_pharyngitis_96216a02-8de6-638b-2fa5-124f996e9601", "objective",
     'Influenza vaccine administered at this visit', 'Influenza vaccine administered 1 day earlier'),
    ("corpus", "viral_pharyngitis_9be5621e-e78a-e06c-e99b-c42f5be5f382", "objective",
     'Influenza vaccine administered at this visit', 'Influenza vaccine administered 8 days earlier'),
    ("corpus", "viral_pharyngitis_af5c9b37-3c78-2155-aea8-9aacdbe59e7a", "objective",
     'Influenza vaccine administered at this visit', 'Influenza vaccine administered 7 days earlier'),
    ("corpus", "viral_pharyngitis_b4e4bfe4-2802-6b00-a6d4-14e12e3c684c", "objective",
     'Influenza vaccine administered at this visit', 'Influenza vaccine administered 1 day earlier'),
    ("corpus", "viral_pharyngitis_bb2a593a-8c68-0dc5-9f18-c2187ffaa18c", "objective",
     'Influenza vaccine administered at this visit', 'Influenza vaccine administered 14 days earlier'),
    ("corpus", "viral_pharyngitis_bc200bee-307d-cf34-588d-1816a2d6a8bc", "objective",
     'Influenza vaccine administered at this visit', 'Influenza vaccine administered 12 days earlier'),
    ("corpus", "viral_pharyngitis_c7ecf2ad-6279-cc41-b027-94b740ffae53", "objective",
     'Recent influenza vaccination administered at this visit', 'Recent influenza vaccination administered 15 days earlier'),
    ("corpus", "viral_pharyngitis_d87953e2-5807-a0f8-9422-87be8c738edc", "objective",
     'Influenza vaccine administered at this visit', 'Influenza vaccine administered 9 days earlier'),
    ("corpus", "viral_pharyngitis_d87a68b4-f815-32f9-8dbb-16aff9b9dccc", "objective",
     'Influenza vaccine administered at this visit', 'Influenza vaccine administered 14 days earlier'),
    # misattributed dates for findings/screenings
    ("corpus", "hypertension_cb0dcba9-91e9-720f-b3e5-3700506517f3", "objective",
     'PHQ-2 score 2, GAD-7', 'PHQ-2 score 2 (0 days earlier), GAD-7'),
    ("corpus", "viral_pharyngitis_07cc3c8d-5fc7-21c0-fb9c-f442bb915b58", "objective",
     'DAST-10 score 0, HARK', 'DAST-10 score 0 (1 month earlier), HARK'),
    ("corpus", "cystitis_e0276e7c-f3eb-7e02-c947-2124ac3cf479", "objective",
     'PHQ-2 score 0, GAD-7', 'PHQ-2 score 0 (14 days earlier), GAD-7'),
    ("corpus", "cystitis_36798473-94eb-0c1a-d2e0-2f72b82375ef", "subjective",
     'impacted molars and a fracture/fracture subluxation of the wrist (both noted at current visit)', 'impacted molars (since 9 years earlier) and a fracture/fracture subluxation of the wrist (noted at current visit)'),
    ("corpus", "bacterial_sinusitis_9bfa4f90-68b4-8069-7df6-6e137cb67491", "objective",
     '(normal pregnancy finding, 1 month earlier)', '(normal pregnancy finding, last noted 0 days earlier)'),
    # "currently pregnant" for a pregnancy status recorded earlier
    ("corpus", "viral_pharyngitis_84adc262-74de-6904-459b-051f345a890d", "subjective",
     'Current pregnancy status was noted 2 months earlier.', 'Pregnancy status: pregnant (2 months earlier).'),
    ("corpus", "cystitis_27cbef3a-7677-e392-0e55-df55b299dc36", "subjective",
     'Current pregnancy status noted (1 year earlier).', 'Pregnancy status: pregnant (1 year earlier).'),
    ("corpus", "cystitis_c9a95290-f737-e341-ccfe-4c4b593942a2", "objective",
     'Current pregnancy status confirmed 15 days earlier.', 'Pregnancy status: pregnant (15 days earlier).'),
    # active problems called past/prior diagnoses
    ("corpus", "copd_cf844ffc-964c-a2f9-c165-7d92dec808a8", "objective",
     'Past diagnosis includes chronic sinusitis (disorder).', 'Diagnosis includes chronic sinusitis (disorder).'),
    ("corpus", "copd_dbf0b7ab-8c48-df36-421c-603d773f3dd3", "objective",
     'Past diagnoses include loss of teeth (disorder) and ', 'Diagnoses and findings include loss of teeth (disorder) and '),
    ("corpus", "hypertension_3c18e172-f4f2-868b-ad03-3f4a2cd6cabe", "objective",
     'Past diagnoses include osteoarthritis of knee, prima', 'Diagnoses include osteoarthritis of knee, prima'),
    ("corpus", "hypertension_66333eb6-f0bb-f425-47fb-6bbec5ade727", "objective",
     'Prior diagnoses include ischemic heart disease, malig', 'Diagnoses and findings include ischemic heart disease, malig'),
    ("corpus", "hypertension_6d2044cd-7725-6121-2762-a620212a0003", "objective",
     'Past diagnoses and findings include adolescent idiopathic scolios', 'Diagnoses and findings include adolescent idiopathic scolios'),
    ("corpus", "hypertension_752a4744-f97b-63b5-ca41-0705de7fc1e2", "objective",
     'Past diagnoses include sleep disorder, sleep apnea, ', 'Diagnoses include sleep disorder, sleep apnea, '),
    ("corpus", "hypertension_7d528f59-9bda-0be7-b207-4d169fc79096", "objective",
     'Past diagnosis of chronic sinusitis.', 'Diagnosis of chronic sinusitis.'),
    ("corpus", "hypertension_881e0ab0-a2f1-551a-8aa0-2223f7a9b76a", "objective",
     'Past diagnoses include anemia, chronic sinusitis, pr', 'Diagnoses and findings include anemia, chronic sinusitis, pr'),
    ("corpus", "hypertension_a68dbb13-c784-52c3-49c1-df8427250fca", "objective",
     'Past diagnoses include epilepsy, seizure disorder, a', 'Diagnoses and findings include epilepsy, seizure disorder, a'),
    ("corpus", "strep_throat_0ad26dc5-0704-9767-73ac-fbbce7710568", "objective",
     'Past diagnoses include chronic sinusitis (disorder),', 'Diagnoses include chronic sinusitis (disorder),'),
    ("corpus", "strep_throat_34103e3c-77be-bb59-d149-6ac08583a1c0", "objective",
     'Past diagnoses include anemia (disorder), prediabete', 'Diagnoses and findings include anemia (disorder), prediabete'),
    ("corpus", "strep_throat_4e4878af-f053-0489-3772-5403ede9e6d6", "objective",
     'Past diagnoses include chronic sinusitis, acute bact', 'Diagnoses include chronic sinusitis, acute bact'),
    ("corpus", "strep_throat_760bd4c7-30d1-171a-ae7a-86cd5d9451f6", "objective",
     'Past diagnoses include acute bacterial sinusitis (ac', 'Diagnoses include acute bacterial sinusitis (ac'),
    ("corpus", "strep_throat_9e632398-8d56-1d7f-dbda-b32772bf589b", "objective",
     'Past diagnoses include chronic sinusitis, gingival d', 'Diagnoses include chronic sinusitis, gingival d'),
    ("corpus", "strep_throat_c4544788-2d19-5872-8b76-45ddde6f452d", "objective",
     'Past diagnoses include adolescent idiopathic scolios', 'Diagnoses include adolescent idiopathic scolios'),
    ("corpus", "strep_throat_c5c12f2e-300b-269d-35fa-639d857b4581", "objective",
     'Past diagnoses include anemia (active), prediabetes,', 'Diagnoses and findings include anemia (active), prediabetes,'),
    ("corpus", "viral_pharyngitis_268efd1b-4950-47be-75c3-9064cf3166a1", "objective",
     'Past diagnoses include chronic sinusitis, prediabete', 'Diagnoses and findings include chronic sinusitis, prediabete'),
    ("corpus", "viral_pharyngitis_56b5a8ee-5802-74ef-7b66-82e934fab106", "objective",
     'Past diagnoses include ischemic heart disease, colon', 'Diagnoses and findings include ischemic heart disease, colon'),
    ("corpus", "viral_pharyngitis_574f4a5c-e4ab-afb1-b6d7-abdd650de5aa", "objective",
     'Past diagnoses include anemia (disorder), prediabete', 'Diagnoses and findings include anemia (disorder), prediabete'),
    ("corpus", "viral_pharyngitis_5c932547-44d3-6faa-043d-88a21c305b85", "objective",
     'Past diagnosis of gingivitis.', 'Diagnosis of gingivitis.'),
    ("corpus", "viral_pharyngitis_65566b00-9bb1-cf33-9e69-386e89e31ee4", "objective",
     'Past diagnoses include transformed migraine, sprain ', 'Diagnoses and findings include transformed migraine, sprain '),
    ("corpus", "viral_pharyngitis_8b475336-5986-71eb-f248-ced7fbd185b6", "objective",
     'Past diagnoses include sepsis (disorder) active sinc', 'Diagnoses and findings include sepsis (disorder) active sinc'),
    ("corpus", "viral_pharyngitis_8b51159d-2c79-b015-3796-0fa860e478f9", "objective",
     'Past diagnosis of chronic sinusitis (disorder).', 'Diagnosis of chronic sinusitis (disorder).'),
    ("corpus", "viral_pharyngitis_a048f312-f74f-499d-823f-06debe9e7daa", "objective",
     'Past diagnoses include chronic sinusitis (active), g', 'Diagnoses include chronic sinusitis (active), g'),
    ("corpus", "viral_pharyngitis_a1deabfa-a924-65f4-3463-154584eaa3a8", "objective",
     'Past diagnoses include chronic sinusitis, streptococ', 'Diagnoses and findings include chronic sinusitis, streptococ'),
    ("corpus", "viral_pharyngitis_b60437f4-776a-b78c-d8b5-a29739354057", "objective",
     'Past diagnoses include anemia (ongoing), prediabetes', 'Diagnoses and findings include anemia (ongoing), prediabetes'),
    ("corpus", "viral_pharyngitis_d8a7867d-73f9-65bf-ed42-edfb060cec0b", "objective",
     "Past diagnoses include Alzheimer's disease, essentia", "Diagnoses and findings include Alzheimer's disease, essentia"),
    ("corpus", "viral_pharyngitis_e63d5d65-0703-1cc9-917f-af5b50aef057", "objective",
     'Past diagnoses include anemia (disorder), prediabete', 'Diagnoses and findings include anemia (disorder), prediabete'),
    ("corpus", "viral_pharyngitis_e7c80b62-33c5-ad9f-5b8f-38554783096e", "objective",
     'Past diagnoses include recurrent urinary tract infec', 'Diagnoses include recurrent urinary tract infec'),
    ("corpus", "bacterial_sinusitis_3733e76a-7799-bc26-c1bc-2fa6f0fe7b1e", "objective",
     'Past diagnoses include essential hypertension, gingi', 'Diagnoses and findings include essential hypertension, gingi'),
    ("corpus", "cystitis_5f07b566-38b5-8d51-2e1e-70b2dfabdd41", "objective",
     'Past diagnoses include transformed migraine, gingiva', 'Diagnoses include transformed migraine, gingiva'),
    ("corpus", "cystitis_62719a80-56da-c740-8445-1fec268fb10b", "objective",
     'Past diagnoses include ischemic heart disease, essen', 'Diagnoses and findings include ischemic heart disease, essen'),
    ("corpus", "cystitis_9fd38068-53b2-b3aa-d9c0-b3a3b04e8a2a", "objective",
     'Past diagnoses include chronic sinusitis, primary de', 'Diagnoses include chronic sinusitis, primary de'),
    ("corpus", "heart_failure_00cc5f58-1c10-d04c-97fd-13454c4dd75b", "objective",
     'Past diagnoses include anemia, pulmonary emphysema, ', 'Diagnoses and findings include anemia, pulmonary emphysema, '),
    ("corpus", "heart_failure_0bdded59-aa19-243a-4dbe-ae62a6d325db", "objective",
     'Past diagnoses include sepsis (1 year earlier), prim', 'Diagnoses and findings include sepsis (1 year earlier), prim'),
    ("corpus", "heart_failure_36866e3e-62e8-71d0-06f7-2aab8560cb65", "objective",
     'Past diagnosis of anemia (disorder), prediabete', 'Diagnosis of anemia (disorder), prediabete'),
    ("corpus", "heart_failure_578527da-203c-a4d3-6797-ae46fba8af89", "objective",
     'Past history of ischemic heart disease (active, 3 years earlier).', 'Active problem: ischemic heart disease (since 3 years earlier).'),
    ("corpus", "viral_pharyngitis_71913559-84ec-98b6-ad9e-a3a090ecd2f5", "objective",
     'Past diagnoses include normal pregnancy (last noted 5 months earlier)', 'Other findings include normal pregnancy (last noted 5 months earlier)'),
    # findings listed as active diagnoses
    ("corpus", "viral_pharyngitis_14d786f1-72a6-4366-3dd8-12491fc3acad", "objective",
     'Active conditions include prediabetes;', 'Other findings include prediabetes;'),
    ("corpus", "cystitis_505ed6e8-3f8a-e1cc-7476-18713e9377d5", "objective",
     'Active diagnoses of anemia and prediabetes;', 'Active diagnosis of anemia; other findings: prediabetes;'),
    ("corpus", "viral_pharyngitis_1974d1a3-8dea-8607-158f-06d1c22ee3a1", "objective",
     'Active problems include ischemic heart disease and history of coronary artery bypass grafting.', 'Active problems include ischemic heart disease. Other findings: history of coronary artery bypass grafting.'),
    # earlier visit type given as the current visit
    ("corpus", "hypertension_fbf24933-262d-346a-414f-5bc5fe0e1b29", "subjective",
     'presenting at a wellness visit.', 'with a wellness visit 14 days earlier.'),
    ("corpus", "cystitis_04adcffd-72a5-f918-7760-86a5f6906708", "subjective",
     '17-year-old male presenting at a wellness visit.', '17-year-old male; well child visit 14 days earlier.'),
    ("corpus", "cystitis_7e69274c-6c5e-cacd-2334-2b9e1d58453f", "subjective",
     'Patient presents at a wellness visit.', 'Wellness visit 14 days earlier.'),
    ("corpus", "cystitis_e386e770-c961-9d01-e528-fe01211e9fdf", "subjective",
     '18-year-old female presenting at a wellness visit.', '18-year-old female; wellness visit 14 days earlier.'),
    ("corpus", "cystitis_137291cf-f1ac-c019-b65d-325c24087dc8", "subjective",
     '14-year-old female presenting at a wellness visit.', '14-year-old female; well child visit 15 days earlier.'),
    ("corpus", "hypertension_575987c0-3cbe-b6d0-e141-f790c569be95", "subjective",
     '53-year-old male presenting for wellness visit.', '53-year-old male; wellness visit 7 months earlier.'),
    ("corpus", "hypertension_80f535e7-b256-3165-802a-f70a9646f21c", "subjective",
     '74-year-old male presenting for check-up.', '74-year-old male; check-up 7 months earlier.'),
    ("corpus", "hypertension_9bd1114b-aaec-2959-ca33-92d8c84519c4", "subjective",
     'The patient presents for a check-up.', 'Check-up 11 months earlier.'),
    ("corpus", "hypertension_9cd138f5-25c5-9a77-0bdb-b863b5122758", "subjective",
     '54-year-old male presenting for wellness visit.', '54-year-old male; wellness visit 10 months earlier.'),
    ("corpus", "hypertension_ec60593d-0a71-23c0-40d3-6390759d828a", "subjective",
     '44-year-old female presenting for check-up.', '44-year-old female; check-up 2 months earlier.'),
    ("corpus", "cystitis_a2aaf1b0-41a2-24fd-1a22-2c145d8d158b", "subjective",
     '14-year-old female presenting for a check-up.', '14-year-old female; check-up 1 day earlier.'),
    ("corpus", "cystitis_b94b61aa-7ae9-dad1-af80-62c7a1b0995c", "subjective",
     '14-year-old female presenting for a well child visit.', '14-year-old female; well child visit 1 day earlier.'),
    ("corpus", "cystitis_c6d95351-b948-c1f3-1ea5-00a68307fb0e", "subjective",
     '44-year-old male presenting to urgent care.', '44-year-old male; urgent care visit 1 month earlier.'),
    ("corpus", "cystitis_d188135d-1236-838b-0c72-4b95d378cccb", "subjective",
     'presenting at a wellness visit.', 'with a wellness visit 5 days earlier.'),
    ("corpus", "cystitis_18ceb7dc-a9d5-0f80-0ddd-f37227916de1", "subjective",
     'Patient presents for a check-up.', 'Check-up 1 day earlier.'),
    ("corpus", "viral_pharyngitis_762a4f62-c8a3-dccf-6fff-97098c8b7220", "subjective",
     '49-year-old female presenting to urgent care.', '49-year-old female; urgent care visit 1 day earlier.'),
    ("corpus", "viral_pharyngitis_06fc213d-dfff-d854-e3f4-fa6d9ef57598", "subjective",
     'Visit type was urgent care.', 'Urgent care visit 7 days earlier.'),
    ("corpus", "viral_pharyngitis_8f9caee5-6abe-8764-3626-08d15a102e76", "subjective",
     'Visit type was urgent care.', 'Urgent care visit 14 days earlier.'),
    ("corpus", "viral_pharyngitis_909d8664-f283-3dcd-0b67-14f32786f8bd", "objective",
     'Visit type: urgent care.', 'Urgent care visit 7 days earlier.'),
    ("corpus", "viral_pharyngitis_d92be68e-9455-4b0a-814d-93bdd2802ad8", "objective",
     'Visit type: urgent care.', 'Urgent care visit 14 days earlier.'),
    ("corpus", "bacterial_sinusitis_a084ac44-e80e-5ff6-7510-6e2ea189c3fe", "objective",
     'Visit type: urgent care.', 'Urgent care visit 15 days earlier.'),
    ("corpus", "viral_pharyngitis_69c0b218-374b-5347-7032-fbee864a575e", "subjective",
     'Patient presents to urgent care with', 'Patient presents with'),
    # wrong medication indication
    ("corpus", "heart_failure_95a8683e-2c76-e035-c04a-3a3db61ac1c5", "subjective",
     'hydrocortisone topical cream, loratadine, and epinephrine auto-injector for allergic disposition', 'hydrocortisone topical cream for contact dermatitis, loratadine, and epinephrine auto-injector'),
    ("corpus", "viral_pharyngitis_82898c5f-1dc8-6f1f-28c3-4a520243947d", "subjective",
     'hydrocortisone 10 MG/ML topical cream and fexofenadine hydrochloride 60 MG oral tablet for contact dermatitis', 'hydrocortisone 10 MG/ML topical cream for contact dermatitis, fexofenadine hydrochloride 60 MG oral tablet'),
    # medication indication not stated in source
    ("corpus", "hypertension_4b6293d6-92bd-d01e-25d5-a3adad0432d3", "subjective",
     'ferrous sulfate 325 MG Oral Tablet for anemia.', 'ferrous sulfate 325 MG Oral Tablet.'),
    ("corpus", "hypertension_5d8fa469-1af0-87dc-a505-942499683969", "subjective",
     'Chlorpheniramine Maleate 4 MG Oral Tablet for allergic disposition.', 'Chlorpheniramine Maleate 4 MG Oral Tablet.'),
    ("corpus", "hypertension_b10aa2b1-740e-13a7-9b1b-cb984cbe9b37", "subjective",
     'ferrous sulfate 325 MG oral tablet for anemia.', 'ferrous sulfate 325 MG oral tablet.'),
    ("corpus", "hypertension_dd32116f-170a-39a5-3d0d-7f6f7fa078a6", "subjective",
     'Hydrocodone Bitartrate 5 MG Oral Tablet for chronic pain.', 'Hydrocodone Bitartrate 5 MG Oral Tablet.'),
    ("corpus", "hypertension_eed8fcc9-b5b1-ec8c-5688-09f7c00348c0", "objective",
     'On ferrous sulfate 325 MG oral tablet for anemia.', 'On ferrous sulfate 325 MG oral tablet.'),
    ("corpus", "strep_throat_5fd9787a-ac00-d766-1ffd-983536ad9359", "subjective",
     'ferrous sulfate 325 MG Oral Tablet (ongoing for anemia).', 'ferrous sulfate 325 MG Oral Tablet.'),
    ("corpus", "viral_pharyngitis_6828f563-bf90-9ec7-0346-58c8abd4e9f0", "subjective",
     ' (both for shellfish allergy with reaction of allergic angioedema)', ''),
    ("corpus", "viral_pharyngitis_89c45abd-9afa-dbfa-10fc-e97ef2c12537", "objective",
     'ferrous sulfate and Vitamin B12 for anemia,', 'ferrous sulfate, Vitamin B12,'),
    ("corpus", "viral_pharyngitis_e25a53d0-a3eb-9254-9c5e-d446b50c6b36", "subjective",
     'ferrous sulfate 325 MG Oral Tablet for anemia.', 'ferrous sulfate 325 MG Oral Tablet.'),
    ("corpus", "cystitis_9c933b61-71cb-99a7-b2a3-69148f154c76", "subjective",
     'acetaminophen 325 MG oral tablet for chronic low back pain.', 'acetaminophen 325 MG oral tablet.'),
    ("corpus", "cystitis_bccbd7ef-4b83-3a9c-18e9-fc3116ca1487", "subjective",
     'epinephrine auto-injector (for allergies to grass pollen and fish)', 'epinephrine auto-injector'),
    ("corpus", "heart_failure_db66c0fe-f672-07b9-b429-90bccb3b2e28", "subjective",
     'ferrous sulfate 325 MG Oral Tablet (for anemia).', 'ferrous sulfate 325 MG Oral Tablet.'),
    # invented interpretations/trends
    ("corpus", "heart_failure_5c2c05ec-241a-dd5f-6999-046bc9c8860c", "objective",
     'Notable weight increase from', 'Weight increased from'),
    ("corpus", "hypertension_7d208a3f-378d-7e13-5709-dadac9326d07", "objective",
     '(7 months earlier), down from 0.8 mg/dL', '(7 months earlier); previously 0.8 mg/dL'),
    ("corpus", "viral_pharyngitis_43f276a3-25f1-7132-eefb-5e2123478a51", "subjective",
     'and recurrent sinusitis.', 'and sinusitis (8 years earlier).'),
    ("corpus", "viral_pharyngitis_57daea5c-629f-c6e0-66e8-d2b09c541d36", "subjective",
     'recurrent acute bronchitis, and active tobacco ex-smoker status.', 'acute bronchitis (1 year earlier), and ex-smoker status.'),
    # invented "fasting" qualifier
    ("corpus", "hypertension_5645eafd-3f39-3ec3-9100-b5993c25f9ee", "objective",
     'fasting glucose', 'glucose'),
    ("corpus", "hypertension_8e7e80c1-3715-fbaf-ff69-631d39d3ad2f", "objective",
     'fasting glucose', 'glucose'),
    ("corpus", "hypertension_f598a1ca-1df0-551e-da97-4fec26b79d56", "objective",
     'fasting glucose', 'glucose'),
    # invented spans/counts
    ("corpus", "hypertension_bd743394-52bf-0588-17a9-7ea83e6f8133", "objective",
     'wellness examinations within the past year.', 'wellness examinations (2 visits, most recent 1 year earlier).'),
    ("corpus", "bacterial_sinusitis_88d88570-5917-1cde-b3ac-5529fcd54bf7", "objective",
     'Recent wellness visits 5 months ago.', 'Recent wellness visits (2 visits, most recent 5 months ago).'),
    # "in the record"-style hedges and "others as noted"
    ("corpus", "copd_0c18daf7-7436-17fa-5baa-8c0723d6031d", "objective",
     'No vital signs or lab values documented in record.', 'No vital signs or lab values documented.'),
    ("corpus", "copd_1fef7202-2f9c-8fbc-add5-56556898f418", "objective",
     'No vital signs or lab values documented in the record.', 'No vital signs or lab values documented.'),
    ("corpus", "copd_24121936-5643-5786-16ec-c8cc3a49c77a", "objective",
     'No vital signs or lab values documented in this record.', 'No vital signs or lab values documented.'),
    ("corpus", "copd_87a93397-7f44-27b4-f2fb-eae06fc77828", "objective",
     'No vital signs or lab values documented in the record.', 'No vital signs or lab values documented.'),
    ("corpus", "hypertension_056e1a61-d1b4-e433-0b35-448f697b1037", "subjective",
     'Patient presents with no explicitly stated chief complaint in the record.', 'Patient presents with no explicitly stated chief complaint.'),
    ("corpus", "hypertension_088119fa-5f4a-842a-ff37-053afc37921d", "subjective",
     'The patient presents with no explicitly stated verbal complaints in the record.', 'The patient presents with no explicitly stated verbal complaints.'),
    ("corpus", "hypertension_1125fa9c-b297-bc94-ec26-143bebefe5e1", "subjective",
     'Patient presents with no explicitly documented chief complaint in the record.', 'Patient presents with no explicitly documented chief complaint.'),
    ("corpus", "hypertension_240fbf17-4658-15d0-5ec4-f8637c8473e7", "subjective",
     'Patient presents with no explicitly stated chief complaint in the record.', 'Patient presents with no explicitly stated chief complaint.'),
    ("corpus", "hypertension_32498796-fa93-4648-0d2f-e6c770205886", "subjective",
     'Patient presents with no explicitly stated chief complaint in the record.', 'Patient presents with no explicitly stated chief complaint.'),
    ("corpus", "hypertension_3e9864ef-b4a9-22a8-c218-04e44360b435", "subjective",
     'Patient presents with no explicitly stated complaints in the record.', 'Patient presents with no explicitly stated complaints.'),
    ("corpus", "hypertension_511f36d7-e2d5-2d0a-3ecc-6a7bf6e30cbe", "subjective",
     'Patient presents with no explicitly stated chief complaint in the record.', 'Patient presents with no explicitly stated chief complaint.'),
    ("corpus", "hypertension_51e5383b-6775-e2c3-bf0d-ed6f7a247a0a", "subjective",
     'History of tubal ligation, past pregnancy history of miscarriage, and prediabetes noted in record.', 'History of tubal ligation, past pregnancy history of miscarriage, and prediabetes noted.'),
    ("corpus", "hypertension_55c03c08-9b59-a60a-c80b-24f84f7dc46f", "objective",
     'No vital signs or lab values documented in the record.', 'No vital signs or lab values documented.'),
    ("corpus", "hypertension_6b6533e2-14b0-085d-8724-2885864761b3", "subjective",
     'Patient reports no specific symptoms documented in the record.', 'Patient reports no specific symptoms documented.'),
    ("corpus", "hypertension_7e76ee1b-6605-353e-0e9d-9caed9e34c7f", "objective",
     'No vital signs or lab values documented in the record.', 'No vital signs or lab values documented.'),
    ("corpus", "hypertension_7f1e65df-c67b-26fa-bbdf-cbd235e0b471", "subjective",
     'Patient presents with no explicitly stated verbal complaints in the record.', 'Patient presents with no explicitly stated verbal complaints.'),
    ("corpus", "hypertension_81b74378-a378-c67f-021c-22dde23fe13b", "subjective",
     'Patient presents with no explicitly stated chief complaint in the record.', 'Patient presents with no explicitly stated chief complaint.'),
    ("corpus", "hypertension_81ce372b-bbd6-0da6-5796-6ac8a21cc626", "subjective",
     'Patient presents with no explicitly stated chief complaint documented in the record.', 'Patient presents with no explicitly stated chief complaint documented.'),
    ("corpus", "hypertension_89b7d5c4-f418-f78b-6e89-0523d9254b75", "subjective",
     'History of prediabetes and obesity noted in record.', 'History of prediabetes and obesity noted.'),
    ("corpus", "hypertension_a53354dd-0f30-d574-43bb-db8be4674404", "subjective",
     'The patient presents with no explicitly stated complaints in the record.', 'The patient presents with no explicitly stated complaints.'),
    ("corpus", "hypertension_acb54479-1160-20f2-1fd5-9b971b77dffe", "subjective",
     'Patient presents with no explicitly stated chief complaint in the record.', 'Patient presents with no explicitly stated chief complaint.'),
    ("corpus", "hypertension_ae575a01-5a3f-89bd-6c8a-45217d0eb6b4", "subjective",
     'History of prediabetes, obesity (BMI 30+), and transport/access to transportation problems noted in record.', 'History of prediabetes, obesity (BMI 30+), and transport/access to transportation problems noted.'),
    ("corpus", "hypertension_b2b0862e-9829-501f-7840-ffa3b73992e7", "subjective",
     'Patient presents with no explicitly stated chief complaint in the record.', 'Patient presents with no explicitly stated chief complaint.'),
    ("corpus", "hypertension_b305d7d7-d5e4-eda1-5735-065d15a164a2", "subjective",
     'Patient presents with no explicitly stated chief complaint in the record.', 'Patient presents with no explicitly stated chief complaint.'),
    ("corpus", "hypertension_bd743394-52bf-0588-17a9-7ea83e6f8133", "subjective",
     'No specific symptoms reported in the record.', 'No specific symptoms reported.'),
    ("corpus", "hypertension_cfbd6a59-3238-04f8-719c-2cf7b0f3ff53", "objective",
     'No vital signs or lab values documented in the record.', 'No vital signs or lab values documented.'),
    ("corpus", "hypertension_d7b612be-94da-e3f0-6b77-b51164ac3521", "subjective",
     'History of prediabetes and obesity noted in record.', 'History of prediabetes and obesity noted.'),
    ("corpus", "hypertension_de6c0a99-282c-1da1-b462-d464f2f34608", "objective",
     'No vital signs or lab values documented in the record.', 'No vital signs or lab values documented.'),
    ("corpus", "hypertension_e4f86c0b-5d26-c495-8b68-985ae1a8e63d", "subjective",
     'Patient presents with no explicitly stated chief complaint in the record.', 'Patient presents with no explicitly stated chief complaint.'),
    ("corpus", "hypertension_ea6b6328-0c54-91a2-9c99-1ea3b77e3dd1", "subjective",
     'History of transport problem and lack of access to transportation noted in record.', 'History of transport problem and lack of access to transportation noted.'),
    ("corpus", "hypertension_ec8dde48-666d-f497-3ab0-e62a1a5390b2", "objective",
     'Received higher education noted in record.', 'Received higher education noted.'),
    ("corpus", "hypertension_ef263224-c39b-4dd4-be4e-4876aa9171c8", "subjective",
     'Patient presents with no documented verbal complaints in the record.', 'Patient presents with no documented verbal complaints.'),
    ("corpus", "hypertension_fd7977b3-c7f4-41e3-31c2-8bc88da216ce", "objective",
     'No vital signs or lab values documented in the record.', 'No vital signs or lab values documented.'),
    ("corpus", "strep_throat_23797d60-3fbe-3feb-9107-bc01a9bfe0cb", "subjective",
     'History of chronic pain, chronic neck pain, chronic low back pain, prediabetes, and obesity noted in record.', 'History of chronic pain, chronic neck pain, chronic low back pain, prediabetes, and obesity noted.'),
    ("corpus", "viral_pharyngitis_05d186fd-98e8-0511-1221-01a2819dccc7", "objective",
     'No vital signs or lab values documented in this record.', 'No vital signs or lab values documented.'),
    ("corpus", "viral_pharyngitis_1a48d638-bcfb-e266-a635-c89ebde14dfb", "subjective",
     'History of stress, prediabetes, and obesity noted in record.', 'History of stress, prediabetes, and obesity noted.'),
    ("corpus", "viral_pharyngitis_85e7781f-43e9-ca14-1cae-de06d713fe7a", "objective",
     'No vital signs or lab values documented in the record.', 'No vital signs or lab values documented.'),
    ("corpus", "viral_pharyngitis_a6dcb864-a3d9-bd53-1a1d-f2cb0c9d60cb", "objective",
     'History of tubal ligation, stress, social isolation, prediabetes, and chronic pain noted in record.', 'History of tubal ligation, stress, social isolation, prediabetes, and chronic pain noted.'),
    ("corpus", "cystitis_2cace539-dbac-9d79-98c6-b4ebaec83b9a", "subjective",
     'Patient presents with no explicitly stated chief complaint in the record.', 'Patient presents with no explicitly stated chief complaint.'),
    ("corpus", "cystitis_9f41ad76-5e44-5ac4-3b2d-e75b669e8070", "subjective",
     'Patient presents with no explicitly stated complaints in the record.', 'Patient presents with no explicitly stated complaints.'),
    ("corpus", "cystitis_dee22e64-f7b3-f498-a950-8d44fa437214", "objective",
     'No vital signs or lab values provided in record.', 'No vital signs or lab values provided.'),
    ("corpus", "cystitis_e99eb13a-540f-045d-5b5a-d60d236641c8", "subjective",
     'Patient presents with no explicitly stated chief complaint in the record.', 'Patient presents with no explicitly stated chief complaint.'),
    ("corpus", "heart_failure_4251f02c-e647-2dc8-a79e-87167a6b9c52", "subjective",
     'History of prediabetes, obesity, and stress noted in record.', 'History of prediabetes, obesity, and stress noted.'),
    ("corpus", "strep_throat_84737eaa-1a82-e759-6e51-abc5bc11274b", "objective",
     'acute bronchitis, and others as noted in record.', 'and acute bronchitis.'),
    ("corpus", "viral_pharyngitis_d8a7867d-73f9-65bf-ed42-edfb060cec0b", "objective",
     'osteoporosis, and others as listed in active problems.', 'and osteoporosis.'),
    ("corpus", "cystitis_e94b975e-249f-22a7-1d10-fec64a5fc61f", "objective",
     'hyperlipidemia, and others as listed in active problems.', 'and hyperlipidemia.'),
]

SEVERITY_RE = re.compile(r"(severity:? \d+)/100\b")
IMMUN_RE = re.compile(r"(?:Current |Recent )?[Ii]mmuniz\w*(?: history)?(?: are| is)? (?:up[ -]to[ -]date|current)"
                      r"(?: per record)?(?:,? including |(?P<end>\.))")
LEFTOVER_RE = re.compile(r"up[ -]to[ -]date|immuniz\w*(?: history)?(?: are| is)? current\b(?! visit)", re.I)


def fix_immunizations(s):
    if not IMMUN_RE.search(s):
        return s

    def sub(m):
        return "" if m.group("end") else "Immunizations: "
    out = IMMUN_RE.sub(sub, s)
    out = re.sub(r"(Immunizations: [^.]*?) per record\.", r"\1.", out)
    return re.sub(r"  +", " ", out).replace(" .", ".").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--split", nargs="+", choices=SPLITS, default=["dev", "corpus", "dev_nota", "corpus_nota"])
    ap.add_argument("--dir", default="soap")
    ap.add_argument("--data", default=ROOT)
    a = ap.parse_args()
    out_dir = os.path.join(os.path.abspath(a.data), a.dir)

    log, problems, pending = [], [], []
    for split in a.split:
        path = os.path.join(out_dir, f"{split}-soap.jsonl")
        if not os.path.exists(path):
            print(f"{split}: no {os.path.basename(path)}, skipped")
            continue
        with open(path) as f:
            lines = f.readlines()
        rows = [json.loads(l) for l in lines]
        by_id = {r["id"]: r for r in rows}
        changed = set()

        for sp, rid, field, old, new in PATCHES:
            if sp != split:
                continue
            r = by_id.get(rid)
            if r is None:
                problems.append(f"{split} {rid}: id not found")
                continue
            n = r[field].count(old)
            if n == 0 and new in r[field]:
                continue  # already applied
            if n != 1:
                problems.append(f"{split} {rid}: expected 1 match of {old!r}, found {n}")
                continue
            r[field] = r[field].replace(old, new)
            changed.add(rid)
            log.append(f"{split} {rid} [{field}] patch: {old!r} -> {new!r}")

        n_sev = n_imm = 0
        for r in rows:
            for field in TEXT_FIELDS:
                before = r[field]
                after = SEVERITY_RE.sub(r"\1", before)
                if after != before:
                    n_sev += 1
                    log.append(f"{split} {r['id']} [{field}] severity: dropped /100 x{len(SEVERITY_RE.findall(before))}")
                imm = fix_immunizations(after)
                if imm != after:
                    n_imm += 1
                    for m in IMMUN_RE.finditer(after):
                        s = after.rfind(". ", 0, m.start()) + 2 if after.rfind(". ", 0, m.start()) >= 0 else 0
                        e = after.find(".", m.end() - 1 if m.group("end") else m.end())
                        log.append(f"{split} {r['id']} [{field}] immunization: {after[s:e + 1]!r}")
                if imm != before:
                    r[field] = imm
                    changed.add(r["id"])
                for m in LEFTOVER_RE.finditer(r[field]):
                    problems.append(f"{split} {r['id']} [{field}] unhandled: ...{r[field][max(0, m.start() - 80):m.end() + 40]}...")

        print(f"{split}: {len(changed)} rows changed ({n_sev} severity fields, {n_imm} immunization fields)")
        if changed:
            pending.append((path, lines, rows, changed))

    if a.apply and not problems:
        for path, lines, rows, changed in pending:
            backup = path + ".prepatch"
            if not os.path.exists(backup):  # keep the very first original
                with open(backup, "w") as f:
                    f.writelines(lines)
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                for r, l in zip(rows, lines):
                    f.write(json.dumps(r, ensure_ascii=False) + "\n" if r["id"] in changed else l)
            os.replace(tmp, path)

    for line in log:
        print("  " + line)
    if problems:
        print("\nPROBLEMS (fix PATCHES/rules; --apply writes nothing until this list is empty):")
        for p in problems:
            print("  " + p)
    if a.apply and log and not problems:
        with open(os.path.join(out_dir, "patches.log"), "a") as f:
            f.write(f"# {datetime.now():%Y-%m-%d %H:%M:%S}\n" + "\n".join(log) + "\n")
    if not a.apply:
        print("\nDry run. Rerun with --apply to write.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
