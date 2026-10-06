# Synthea Module Scan

Static analysis of 242 Synthea module JSON files. For each unique ConditionOnset SNOMED code, traces all paths from `Initial` to the onset state and classifies what gates whether onset can occur.

## Method

For each `ConditionOnset` state, a backward BFS identifies all ancestor states. That set is then intersected with a forward BFS from `Initial` that stops before entering any other `ConditionOnset` in the same module. This *pre-first-onset filter* removes states that are only reachable after a different condition has already been diagnosed — eliminating false ancestors from post-onset care-management cycles (e.g. `wellness_encounters.json`) and unrelated injury branches in multi-condition modules (e.g. `injuries.json`). Condition checks on transitions within the filtered ancestor set are collected and classified by type. `CallSubmodule` ancestors are expanded inline: all leaf conditions in the submodule are collected as modifiers.

### Gate classification

| Class | Condition types |
|-------|-----------------|
| **demographic** | Age, Gender, Race, Socioeconomic Status, Date |
| **visible history** | Active Condition, Active Medication, Observation, Vital Sign, MultiObservation, Active CarePlan; Attribute where the attribute is set by a ConditionOnset (`assign_to_attribute`) or by a recorded observation (`smoker`, tobacco-related) |
| **hidden history** | Attribute where the attribute is an internal flag set by demographic logic only (e.g. `veteran`, `atopic`, `diabetes` flag in metabolic_syndrome_disease.json) |

The overall class for each code is the highest-precedence class found across all gates: visible > hidden > demographic. Conditions flagged as `PriorState`, `Symptom`, `True`, or `False` are structural and are excluded from classification.

### Cross-module attributes

Attributes set by a `ConditionOnset` in one module (via `assign_to_attribute`) and read as an `Attribute` condition in another module are classified as **visible**: the originating condition appears in `conditions.csv`. For example, `obesity` is set by a BMI ConditionOnset in `wellness_encounters.json` and is therefore visible.

Attributes set by demographic logic only (`veteran`, `atopic`, `diabetes` flag from `metabolic_syndrome_disease.json`) are classified as **hidden**: they are internal state, not observable records.

### Known approximations

The pre-first-onset filter eliminates the main source of over-inclusion — post-onset care-management cycles and unrelated disease branches in multi-condition modules — but does not resolve three remaining cases:

- **Observation checks that follow selection, not predict it**: In `metabolic_syndrome_care.json`, HbA1c and glucose are recorded as part of the diagnosis encounter after the patient is already selected by the hidden `diabetes`/`prediabetes` attribute. These checks are technically on the pre-first-onset path, so type 2 diabetes and prediabetes remain classified as visible. The `hidden_gates` field shows the true structural gate (`diabetes`, `prediabetes` — set by demographic logic in `metabolic_syndrome_disease.json`).
- **Multi-injury module (injuries.json)**: The broken-jaw submodule (`injuries/broken_jaw.json`) is callable on the first pass through `injuries.json` and contains `dental_referral` attribute checks. These propagate as soft modifiers to all other injury types, including gunshot wounds and fractures, even though prior dental care has no causal role. The `hidden_gates` field lists `osteoporosis` and prescription attributes that do modulate injury incidence rates in the model.
- **Sequential disease progression (e.g. colorectal cancer)**: The filter stops before `Detect_Adenoma` (an in-module ConditionOnset in `colorectal_cancer.json`), which is also the causal prerequisite for cancer onset. The prior-polyp gate is therefore excluded from the visible-gates list; only the `smoker` incidence-rate modifier is reported. The `colorectal_cancer_stage` attribute (representing polyp-to-cancer progression) appears in `hidden_gates`.

### Exclusion flags

Conditions are flagged as *excluded* when they are known to cause label-leakage issues per `SCRUBBING.md` §Known limitations, or by heuristic detection of the same patterns:

- **symptom_type** — the SNOMED code is a finding or the description matches symptom keywords; the symptom may be recorded before the formal diagnosis.
- **precursor_names_target** — a precursor or variant condition whose description names the target survives before the scrub cutoff.
- **screening_work_up** — pre-diagnosis assessment or referral records name the target condition.
- **situation_code** — SNOMED `(situation)` tag typically marks referrals, history-of records, or administrative findings.

### Sanity check

CHF (88805009) classifies as **demographic-only** with no exclusion flags. ✓  Gates found: Age, Gender.

## History-dependent (visible)  (127 conditions)

| Code | Description | Module | Gates / notes | Cases (10k) |
|------|-------------|--------|---------------|------------|
| 75498004 | Acute bacterial sinusitis | sinusitis | vis: Active Condition, Active Medication | 656 |
| 10509002 | Acute bronchitis | bronchitis | vis: Attribute:smoker · hid: OTC_Bronchitis_Med · dem: Age | 4582 |
| 401314000 | Acute non-ST segment elevation myocardial infarction | heart/nsteacs_pathway | vis: Observation | 264 |
| 195662009 | Acute viral pharyngitis | sore_throat | vis: Active Condition, Active Medication, Observation · dem: Age | 5108 |
| 61804006 | Alveolitis of jaw | dental_and_oral_examination | vis: Active Condition, Attribute:dental_referral · hid: orthodontic_appliance | 369 |
| 271737000 | Anemia | hiv/hiv_baseline | vis: Active Condition | 4154 |
| 225444004 | At increased risk for suicide | veteran_ptsd | vis: Attribute:mTBI · hid: veteran · dem: Gender | 1 |
| 287185009 | Attempted suicide by cutting or stabbing | veteran_self_harm | vis: Attribute:mdd, Attribute:ptsd · hid: opioid_addiction, veteran · dem: Gender | — |
| 287182007 | Attempted suicide by suffocation | veteran_self_harm | vis: Attribute:mdd, Attribute:ptsd · hid: opioid_addiction, veteran · dem: Gender | — |
| 87628006 | Bacterial infectious disease | covid19/diagnose_bacterial_infection | vis: Attribute:covid19_bacterial_infection | 7 |
| 6072007 | Bleeding from anus | colorectal_cancer | vis: Attribute:smoker | 26 |
| 60951000119105 | Blindness due to type 2 diabetes mellitus | metabolic_syndrome/diabetic_retinopathy_diagnoses | vis: Active Condition · hid: blindness | 5 |
| 162864005 | Body mass index 30+ - obesity | wellness_encounters | vis: Active CarePlan, Active Condition, Attribute:criminal_record +6 · hid: birth_country, education, first_language +15 · dem: Age, Gender, Race | 5531 |
| 408512008 | Body mass index 40+ - severely obese | wellness_encounters | vis: Active CarePlan, Active Condition, Attribute:criminal_record +6 · hid: birth_country, education, first_language +15 · dem: Age, Gender, Race | 78 |
| 262574004 | Bullet wound | injuries | vis: Active Condition, Attribute:dental_referral · hid: antibiotic_prescription, osteoporosis, otc_pain_reliever · dem: Age | 76 |
| 48333001 | Burn injury | injuries | vis: Active Condition, Attribute:dental_referral · hid: antibiotic_prescription, osteoporosis, otc_pain_reliever · dem: Age | 444 |
| 92691004 | Carcinoma in situ of prostate | veteran_prostate_cancer | vis: Observation · hid: veteran · dem: Gender | 153 |
| 128302006 | Chronic hepatitis C | hiv/hiv_baseline | vis: Active Condition · hid: ckd · dem: Gender | 5 |
| 124171000119105 | Chronic intractable migraine without aura | opioid_addiction | vis: Attribute:ptsd · hid: opioid_prescription · dem: Age, Date, Socioeconomic Status | 528 |
| 185086009 | Chronic obstructive bronchitis | copd | vis: Attribute:smoker · dem: Age, Socioeconomic Status | 158 |
| 698754002 | Chronic paralysis due to lesion of spinal cord | injuries | vis: Active Condition, Attribute:dental_referral · hid: antibiotic_prescription, osteoporosis, otc_pain_reliever · dem: Age | 12 |
| 61977001 | Chronic type B viral hepatitis | hiv/hiv_baseline | vis: Active Condition · hid: ckd · dem: Gender | 3 |
| 359817006 | Closed fracture of hip | injuries | vis: Active Condition, Attribute:dental_referral · hid: antibiotic_prescription, osteoporosis, otc_pain_reliever · dem: Age | 105 |
| 110030002 | Concussion injury of brain | injuries | vis: Active Condition, Attribute:dental_referral · hid: antibiotic_prescription, osteoporosis, otc_pain_reliever · dem: Age | 827 |
| 62564004 | Concussion with loss of consciousness | injuries | vis: Active Condition, Attribute:dental_referral · hid: antibiotic_prescription, osteoporosis, otc_pain_reliever · dem: Age | 162 |
| 62106007 | Concussion with no loss of consciousness | injuries | vis: Active Condition, Attribute:dental_referral · hid: antibiotic_prescription, osteoporosis, otc_pain_reliever · dem: Age | 617 |
| 278558000 | Dental filling lost | dental_and_oral_examination | vis: Active Condition, Attribute:dental_referral · hid: orthodontic_appliance | 1233 |
| 6525002 | Dependent drug abuse | opioid_addiction | vis: Attribute:ptsd · hid: opioid_prescription · dem: Age, Date, Socioeconomic Status | 569 |
| 840539006 | Disease caused by severe acute respiratory syndrome coronavirus 2 | covid19/infection | vis: Attribute:Cystic_Fibrosis, Attribute:asthma_condition, Attribute:breast_cancer_condition +4 · hid: colorectal_cancer_stage, coronary_heart_disease, diabetes +5 | 948 |
| 127013003 | Disorder of kidney due to diabetes mellitus | metabolic_syndrome/kidney_conditions | vis: Active Condition · hid: nephropathy | 1136 |
| 46177005 | End-stage renal disease | metabolic_syndrome/kidney_conditions | vis: Active Condition · hid: ckd, microalbuminuria, nephropathy +1 | 208 |
| 403190006 | Epidermal burn of skin | injuries | vis: Active Condition, Attribute:dental_referral · hid: antibiotic_prescription, osteoporosis, otc_pain_reliever · dem: Age | 253 |
| 59621000 | Essential hypertension | hypertension | vis: Active Condition, Attribute:chf, Attribute:smoker · hid: ckd, coronary_heart_disease, hypertension +2 · dem: Age, Gender, Race | 2650 |
| 370247008 | Facial laceration | injuries | vis: Active Condition, Attribute:dental_referral · hid: antibiotic_prescription, osteoporosis, otc_pain_reliever · dem: Age | 320 |
| 16114001 | Fracture of ankle | injuries | vis: Active Condition, Attribute:dental_referral · hid: antibiotic_prescription, osteoporosis, otc_pain_reliever · dem: Age | 300 |
| 125605004 | Fracture of bone | injuries | vis: Active Condition, Attribute:dental_referral · hid: antibiotic_prescription, osteoporosis, otc_pain_reliever · dem: Age | 1504 |
| 58150001 | Fracture of clavicle | injuries | vis: Active Condition, Attribute:dental_referral · hid: antibiotic_prescription, osteoporosis, otc_pain_reliever · dem: Age | 332 |
| 65966004 | Fracture of forearm | injuries | vis: Active Condition, Attribute:dental_referral · hid: antibiotic_prescription, osteoporosis, otc_pain_reliever · dem: Age | 397 |
| 33737001 | Fracture of rib | injuries | vis: Active Condition, Attribute:dental_referral · hid: antibiotic_prescription, osteoporosis, otc_pain_reliever · dem: Age | 175 |
| 1734006 | Fracture of vertebral column with spinal cord injury | injuries | vis: Active Condition, Attribute:dental_referral · hid: antibiotic_prescription, osteoporosis, otc_pain_reliever · dem: Age | 12 |
| 15724005 | Fracture of vertebral column without spinal cord injury | injuries | vis: Active Condition, Attribute:dental_referral · hid: antibiotic_prescription, osteoporosis, otc_pain_reliever · dem: Age | 11 |
| 263102004 | Fracture subluxation of wrist | injuries | vis: Active Condition, Attribute:dental_referral · hid: antibiotic_prescription, osteoporosis, otc_pain_reliever · dem: Age | 315 |
| 278588009 | Fractured dental filling | dental_and_oral_examination | vis: Active Condition, Attribute:dental_referral · hid: orthodontic_appliance | 1219 |
| 403192003 | Full thickness burn | injuries | vis: Active Condition, Attribute:dental_referral · hid: antibiotic_prescription, osteoporosis, otc_pain_reliever · dem: Age | 18 |
| 160903007 | Full-time employment | encounter/sdoh_hrsn | vis: Attribute:employment_condition · hid: education, first_language, household_size +3 · dem: Age, Race | 9076 |
| 18718003 | Gingival disease | dental_and_oral_examination | vis: Active Condition, Attribute:dental_referral · hid: orthodontic_appliance | 5299 |
| 66383009 | Gingivitis | wellness_encounters | vis: Active CarePlan, Active Condition, Attribute:criminal_record +5 · hid: birth_country, education, first_language +15 · dem: Age, Gender, Race | 9322 |
| 283545005 | Gunshot wound | injuries | vis: Active Condition, Attribute:dental_referral · hid: antibiotic_prescription, osteoporosis, otc_pain_reliever · dem: Age | 76 |
| 266948004 | Has a criminal record | encounter/sdoh_hrsn | vis: Active Condition, Attribute:employment_condition · hid: education, first_language, food_insecurity +7 · dem: Age, Race | 2272 |
| 32911000 | Homeless | homelessness | vis: Active Condition · hid: homelessness_category, instances_of_homelessness, sexual_orientation · dem: Socioeconomic Status | 200 |
| 80394007 | Hyperglycemia | metabolic_syndrome_care | vis: Active CarePlan, Active Condition, Active Medication +3 · hid: ckd, diabetes, diabetes_severity +6 · dem: Gender | 391 |
| 302870006 | Hypertriglyceridemia | metabolic_syndrome_care | vis: Active CarePlan, Active Condition, Active Medication +3 · hid: ckd, diabetes, diabetes_severity +6 · dem: Gender | 825 |
| 196416002 | Impacted molars | opioid_addiction | vis: Attribute:ptsd · hid: opioid_prescription · dem: Age, Date, Socioeconomic Status | 569 |
| 427898007 | Infection of tooth | dental_and_oral_examination | vis: Active Condition, Attribute:dental_referral · hid: missing_teeth, orthodontic_appliance | 1721 |
| 444470001 | Injury of anterior cruciate ligament | injuries | vis: Active Condition, Attribute:dental_referral · hid: antibiotic_prescription, osteoporosis, otc_pain_reliever · dem: Age | 52 |
| 125601008 | Injury of knee | injuries | vis: Active Condition, Attribute:dental_referral · hid: antibiotic_prescription, osteoporosis, otc_pain_reliever · dem: Age | 245 |
| 444448004 | Injury of medial collateral ligament of knee | injuries | vis: Active Condition, Attribute:dental_referral · hid: antibiotic_prescription, osteoporosis, otc_pain_reliever · dem: Age | 65 |
| 90460009 | Injury of neck | injuries | vis: Active Condition, Attribute:dental_referral · hid: antibiotic_prescription, osteoporosis, otc_pain_reliever · dem: Age | 541 |
| 307731004 | Injury of tendon of the rotator cuff of shoulder | injuries | vis: Active Condition, Attribute:dental_referral · hid: antibiotic_prescription, opioid_prescription, osteoporosis +1 · dem: Age, Date | 158 |
| 312608009 | Laceration - injury | injuries | vis: Active Condition, Attribute:dental_referral · hid: antibiotic_prescription, osteoporosis, otc_pain_reliever · dem: Age | 1507 |
| 284551006 | Laceration of foot | injuries | vis: Active Condition, Attribute:dental_referral · hid: antibiotic_prescription, osteoporosis, otc_pain_reliever · dem: Age | 305 |
| 283371005 | Laceration of forearm | injuries | vis: Active Condition, Attribute:dental_referral · hid: antibiotic_prescription, osteoporosis, otc_pain_reliever · dem: Age | 321 |
| 284549007 | Laceration of hand | injuries | vis: Active Condition, Attribute:dental_referral · hid: antibiotic_prescription, osteoporosis, otc_pain_reliever · dem: Age | 317 |
| 283385000 | Laceration of thigh | injuries | vis: Active Condition, Attribute:dental_referral · hid: antibiotic_prescription, osteoporosis, otc_pain_reliever · dem: Age | 328 |
| 713458007 | Lack of access to transportation | encounter/sdoh_hrsn | vis: Active Condition, Attribute:employment_condition · hid: education, first_language, food_insecurity +7 · dem: Age, Race | 1080 |
| 278598003 | Leaking dental filling | dental_and_oral_examination | vis: Active Condition, Attribute:dental_referral · hid: orthodontic_appliance | 1184 |
| 423315002 | Limited social contact | encounter/sdoh_hrsn | vis: Active Condition, Attribute:employment_condition · hid: education, first_language, food_insecurity +7 · dem: Age, Race | 6256 |
| 278602001 | Loose dental filling | dental_and_oral_examination | vis: Active Condition, Attribute:dental_referral · hid: orthodontic_appliance | 1231 |
| 97331000119101 | Macular edema and retinopathy due to type 2 diabetes mellitus | metabolic_syndrome/diabetic_retinopathy_diagnoses | vis: Active Condition · hid: macular_edema | 17 |
| 237602007 | Metabolic syndrome X | metabolic_syndrome_care | vis: Active CarePlan, Active Condition, Active Medication +6 · hid: ckd, diabetes, diabetes_severity +7 · dem: Gender | 1694 |
| 94260004 | Metastatic malignant neoplasm to colon | colorectal_cancer | vis: Attribute:smoker · dem: Age | 12 |
| 94503003 | Metastatic malignant neoplasm to prostate | veteran_prostate_cancer | vis: Observation · hid: veteran · dem: Gender | 25 |
| 90781000119102 | Microalbuminuria due to type 2 diabetes mellitus | metabolic_syndrome/kidney_conditions | vis: Active Condition · hid: microalbuminuria, nephropathy | 948 |
| 22298006 | Myocardial infarction | myocardial_infarction | vis: Active Condition, Active Medication, Observation · hid: ACS_CABG_referral, ace_arb, acs_antiplatelet +4 | 269 |
| 126906006 | Neoplasm of prostate | veteran_prostate_cancer | vis: Observation · hid: veteran · dem: Gender | 168 |
| 368581000119106 | Neuropathy due to type 2 diabetes mellitus | metabolic_syndrome_care | vis: Active CarePlan, Active Condition, Active Medication +2 · hid: ckd, diabetes, diabetes_severity +6 · dem: Gender | 256 |
| 424132000 | Non-small cell carcinoma of lung, TNM stage 1 | lung_cancer | vis: Attribute:quit smoking age, Attribute:smoker · dem: Gender | 71 |
| 425048006 | Non-small cell carcinoma of lung, TNM stage 2 | lung_cancer | vis: Attribute:quit smoking age, Attribute:smoker · dem: Gender | — |
| 422968005 | Non-small cell carcinoma of lung, TNM stage 3 | lung_cancer | vis: Attribute:quit smoking age, Attribute:smoker · dem: Gender | — |
| 423121009 | Non-small cell carcinoma of lung, TNM stage 4 | lung_cancer | vis: Attribute:quit smoking age, Attribute:smoker · dem: Gender | — |
| 254637007 | Non-small cell lung cancer | lung_cancer | vis: Attribute:quit smoking age, Attribute:smoker · dem: Gender | 71 |
| 1551000119108 | Nonproliferative diabetic retinopathy due to type II diabetes mellitus | metabolic_syndrome/diabetic_retinopathy_diagnoses | vis: Active Condition · hid: diabetic_retinopathy_stage | 210 |
| 72892002 | Normal pregnancy | pregnancy | vis: Attribute:anemia, Attribute:pregnancy_complication, Observation · hid: RH_NEG, anemia_pregnancy, birth_type +1 · dem: Age, Gender | 2090 |
| 741062008 | Not in labor force | encounter/sdoh_hrsn | vis: Attribute:employment_condition · hid: education, first_language, household_size +3 · dem: Age, Race | 5455 |
| 64859006 | Osteoporosis | injuries | vis: Active Condition, Attribute:dental_referral · hid: antibiotic_prescription, osteoporosis, otc_pain_reliever · dem: Age | 489 |
| 443165006 | Osteoporotic fracture of bone | injuries | vis: Active Condition, Attribute:dental_referral · hid: antibiotic_prescription, osteoporosis, otc_pain_reliever · dem: Age | 138 |
| 1149222004 | Overdose | opioid_addiction | vis: Attribute:ptsd · hid: opioid_addiction_careplan, opioid_prescription · dem: Age, Date, Socioeconomic Status | 530 |
| 109838007 | Overlapping malignant neoplasm of colon | colorectal_cancer | vis: Attribute:smoker · dem: Age | 82 |
| 160904001 | Part-time employment | encounter/sdoh_hrsn | vis: Attribute:employment_condition · hid: education, first_language, household_size +3 · dem: Age, Race | 7388 |
| 403191005 | Partial thickness burn | injuries | vis: Active Condition, Attribute:dental_referral · hid: antibiotic_prescription, osteoporosis, otc_pain_reliever · dem: Age | 170 |
| 68496003 | Polyp of colon | colorectal_cancer | vis: Active Condition, Attribute:smoker · dem: Age | 789 |
| 47505003 | Posttraumatic stress disorder | veteran_ptsd | vis: Attribute:mTBI · hid: veteran · dem: Gender | 15 |
| 714628002 | Prediabetes | metabolic_syndrome_care | vis: Active CarePlan, Active Condition, Active Medication +2 · hid: ckd, diabetes, diabetes_severity +6 · dem: Gender | 4400 |
| 109570002 | Primary dental caries | dental_and_oral_examination | vis: Active Condition, Attribute:dental_referral · hid: orthodontic_appliance | 4775 |
| 93761005 | Primary malignant neoplasm of colon | colorectal_cancer | vis: Attribute:smoker · dem: Age | 21 |
| 67811000119102 | Primary small cell malignant neoplasm of lung, TNM stage 1 | lung_cancer | vis: Attribute:quit smoking age, Attribute:smoker · dem: Gender | 14 |
| 67821000119109 | Primary small cell malignant neoplasm of lung, TNM stage 2 | lung_cancer | vis: Attribute:quit smoking age, Attribute:smoker · dem: Gender | — |
| 67831000119107 | Primary small cell malignant neoplasm of lung, TNM stage 3 | lung_cancer | vis: Attribute:quit smoking age, Attribute:smoker · dem: Gender | — |
| 67841000119103 | Primary small cell malignant neoplasm of lung, TNM stage 4 | lung_cancer | vis: Attribute:quit smoking age, Attribute:smoker · dem: Gender | — |
| 1501000119109 | Proliferative diabetic retinopathy due to type II diabetes mellitus | metabolic_syndrome/diabetic_retinopathy_diagnoses | vis: Active Condition · hid: diabetic_retinopathy_stage | 16 |
| 157141000119108 | Proteinuria due to type 2 diabetes mellitus | metabolic_syndrome/kidney_conditions | vis: Active Condition · hid: microalbuminuria, nephropathy, proteinuria | 670 |
| 87433001 | Pulmonary emphysema | copd | vis: Attribute:smoker · dem: Age, Socioeconomic Status | 226 |
| 713197008 | Recurrent rectal polyp | colorectal_cancer | vis: Attribute:smoker · dem: Age | 249 |
| 446654005 | Refugee | encounter/sdoh_hrsn | vis: Active Condition, Attribute:employment_condition · hid: education, first_language, food_insecurity +7 · dem: Age, Race | 500 |
| 424393004 | Reports of violence in the environment | encounter/sdoh_hrsn | vis: Active Condition, Attribute:employment_condition · hid: education, first_language, food_insecurity +7 · dem: Age, Race | 3803 |
| 30832001 | Rupture of patellar tendon | injuries | vis: Active Condition, Attribute:dental_referral · hid: antibiotic_prescription, osteoporosis, otc_pain_reliever · dem: Age | 74 |
| 254632001 | Small cell carcinoma of lung | lung_cancer | vis: Attribute:quit smoking age, Attribute:smoker · dem: Gender | 14 |
| 422650009 | Social isolation | encounter/sdoh_hrsn | vis: Active Condition, Attribute:employment_condition · hid: education, first_language, food_insecurity +7 · dem: Age, Race | 6189 |
| 384709000 | Sprain | injuries | vis: Active Condition, Attribute:dental_referral · hid: antibiotic_prescription, osteoporosis, otc_pain_reliever · dem: Age | 1555 |
| 44465007 | Sprain of ankle | injuries | vis: Active Condition, Attribute:dental_referral · hid: antibiotic_prescription, osteoporosis, otc_pain_reliever · dem: Age | 1103 |
| 70704007 | Sprain of wrist | injuries | vis: Active Condition, Attribute:dental_referral · hid: antibiotic_prescription, osteoporosis, otc_pain_reliever · dem: Age | 504 |
| 43878008 | Streptococcal sore throat | sore_throat | vis: Active Condition, Active Medication, Observation · dem: Age | 1499 |
| 73595000 | Stress | encounter/sdoh_hrsn | vis: Active Condition, Attribute:employment_condition · hid: education, first_language, food_insecurity +7 · dem: Age, Race | 9076 |
| 86849004 | Suicidal poisoning | veteran_self_harm | vis: Attribute:mdd, Attribute:ptsd · hid: opioid_addiction, veteran · dem: Gender | 1 |
| 6471006 | Suicidal thoughts | veteran_self_harm | vis: Attribute:mdd, Attribute:ptsd · hid: opioid_addiction, veteran · dem: Gender | 1 |
| 44301001 | Suicide | veteran_self_harm | vis: Attribute:mdd, Attribute:ptsd · hid: opioid_addiction, veteran · dem: Gender | — |
| 287191006 | Suicide by suffocation | veteran_self_harm | vis: Attribute:mdd, Attribute:ptsd · hid: opioid_addiction, veteran · dem: Gender | — |
| 287193009 | Suicide using firearm | veteran_self_harm | vis: Attribute:mdd, Attribute:ptsd · hid: opioid_addiction, veteran · dem: Gender | — |
| 239720000 | Tear of meniscus of knee | injuries | vis: Active Condition, Attribute:dental_referral · hid: antibiotic_prescription, osteoporosis, otc_pain_reliever · dem: Age | 57 |
| 428915008 | Toxoplasma gondii antibody detected | hiv/hiv_baseline | vis: Active Condition · hid: ckd · dem: Gender | 2 |
| 266934004 | Transport problem | encounter/sdoh_hrsn | vis: Active Condition, Attribute:employment_condition · hid: education, first_language, food_insecurity +7 · dem: Age, Race | 1096 |
| 262521009 | Traumatic injury of spinal cord and/or vertebral column | injuries | vis: Active Condition, Attribute:dental_referral · hid: antibiotic_prescription, osteoporosis, otc_pain_reliever · dem: Age | 23 |
| 127294003 | Traumatic or nontraumatic brain injury | injuries | vis: Active Condition, Attribute:dental_referral · hid: antibiotic_prescription, osteoporosis, otc_pain_reliever · dem: Age | 67 |
| 73438004 | Unemployed | encounter/sdoh_hrsn | vis: Attribute:employment_condition · hid: education, first_language, household_size +4 · dem: Age, Race | 3496 |
| 706893006 | Victim of intimate partner abuse | encounter/sdoh_hrsn | vis: Active Condition, Attribute:employment_condition · hid: education, first_language, food_insecurity +7 · dem: Age, Race | 5047 |
| 444814009 | Viral sinusitis | sinusitis | vis: Active Condition, Active Medication | 7367 |
| 39848009 | Whiplash injury to neck | injuries | vis: Active Condition, Attribute:dental_referral · hid: antibiotic_prescription, osteoporosis, otc_pain_reliever · dem: Age | 541 |

## History-dependent (hidden only)  (61 conditions)

| Code | Description | Module | Gates / notes | Cases (10k) |
|------|-------------|--------|---------------|------------|
| 132281000119108 | Acute deep venous thrombosis | covid19/diagnose_blood_clot | hid: covid19_ddimer, covid19_diagnosed_clot | 80 |
| 706870000 | Acute pulmonary embolism | covid19/diagnose_blood_clot | hid: covid19_ddimer, covid19_diagnosed_clot | 59 |
| 129721000119106 | Acute renal failure on dialysis | heart/avrr/outcomes | hid: operative_status | 1 |
| 65710008 | Acute respiratory failure | heart/avrr/outcomes | hid: operative_status | 94 |
| 7200002 | Alcoholism | veteran_substance_abuse_treatment | hid: alcoholism, opioid_addiction · dem: Age | 65 |
| 278365007 | Anticoagulant-induced bleeding | heart/avrr/outcomes | hid: operative_status | — |
| 24079001 | Atopic dermatitis | dermatitis | hid: atopic | 222 |
| 35999006 | Blighted ovum | pregnancy | hid: pregnant · dem: Gender | 183 |
| 230690007 | Cerebrovascular accident | heart/avrr/outcomes | hid: operative_status | 62 |
| 233678006 | Childhood asthma | asthma | hid: atopic | 239 |
| 43724002 | Chill | covid19/symptoms | hid: covid19_severity | 113 |
| 431855005 | Chronic kidney disease stage 1 | metabolic_syndrome/kidney_conditions | hid: ckd | 1010 |
| 431856006 | Chronic kidney disease stage 2 | metabolic_syndrome/kidney_conditions | hid: ckd | 868 |
| 433144002 | Chronic kidney disease stage 3 | metabolic_syndrome/kidney_conditions | hid: ckd | 601 |
| 431857002 | Chronic kidney disease stage 4 | metabolic_syndrome/kidney_conditions | hid: ckd | 357 |
| 156073000 | Complete miscarriage | pregnancy | hid: pregnant · dem: Gender | 375 |
| 609496007 | Complication occurring during pregnancy | pregnancy | hid: pregnant · dem: Gender | 16 |
| 37849005 | Congenital uterine anomaly | pregnancy | hid: pregnant · dem: Gender | 32 |
| 876882001 | Died in hospice | hospice_treatment | hid: days_until_death, hospice, hospice_days | 79 |
| 198992004 | Eclampsia in pregnancy | pregnancy | hid: pregnant · dem: Gender | 240 |
| 473461003 | Educated to high school level | encounter/sdoh_hrsn | hid: education, first_language, household_size +3 · dem: Age, Race | 2830 |
| 267253006 | Fetus with chromosomal abnormality | pregnancy | hid: pregnant · dem: Gender | 64 |
| 66857006 | Hemoptysis | covid19/symptoms | hid: covid19_severity | 5 |
| 47200007 | High risk pregnancy | pregnancy | hid: pregnant · dem: Gender | 2 |
| 55822004 | Hyperlipidemia | veteran_hyperlipidemia | hid: veteran | 1136 |
| 312157006 | Infectious mediastinitis | heart/avrr/outcomes | hid: operative_status | 21 |
| 57676002 | Joint pain | covid19/symptoms | hid: covid19_severity | 131 |
| 213150003 | Kidney transplant failure and rejection | kidney_transplant | hid: ckd | 21 |
| 201834006 | Localized, primary osteoarthritis of the hand | osteoarthritis | hid: veteran · dem: Gender | 226 |
| 36955009 | Loss of taste | covid19/symptoms | hid: covid19_severity | 526 |
| 37320007 | Loss of teeth | dentures | hid: missing_teeth | 1536 |
| 36923009 | Major depression, single episode | veteran_mdd | hid: veteran · dem: Age, Gender | 3 |
| 370143000 | Major depressive disorder | veteran_mdd | hid: veteran · dem: Age, Gender | 58 |
| 19169002 | Miscarriage in first trimester | pregnancy | hid: pregnant · dem: Gender | 576 |
| 85116003 | Miscarriage in second trimester | pregnancy | hid: pregnant · dem: Gender | 110 |
| 68235000 | Nasal congestion | covid19/symptoms | hid: covid19_severity | 53 |
| 224295006 | Only received primary school education | encounter/sdoh_hrsn | hid: education, first_language, household_size +3 · dem: Age, Race | 1098 |
| 5602001 | Opioid abuse | veteran_substance_abuse_treatment | hid: opioid_addiction · dem: Age | 126 |
| 239872002 | Osteoarthritis of hip | osteoarthritis | hid: veteran · dem: Gender | 215 |
| 239873007 | Osteoarthritis of knee | osteoarthritis | hid: veteran · dem: Gender | 551 |
| 65363002 | Otitis media | ear_infections | hid: antibiotic_prescription, otc_pain_reliever · dem: Age, Date | 1289 |
| 246677007 | Passive conjunctival congestion | covid19/symptoms | hid: covid19_severity | 9 |
| 446096008 | Perennial allergic rhinitis | allergic_rhinitis | hid: atopic | 185 |
| 232353008 | Perennial allergic rhinitis with seasonal variation | allergic_rhinitis | hid: atopic | 190 |
| 129574000 | Postoperative myocardial infarction | heart/avrr/outcomes | hid: operative_status | — |
| 275408006 | Postoperative renal failure | heart/avrr/outcomes | hid: operative_status | 1 |
| 698819004 | Postoperative sepsis | heart/cabg/outcomes | hid: operative_status | — |
| 398254007 | Pre-eclampsia | pregnancy | hid: pregnant · dem: Gender | 214 |
| 224299000 | Received higher education | encounter/sdoh_hrsn | hid: first_language, household_size, income +2 · dem: Age, Race | 5318 |
| 1255252008 | Resorption of alveolar process due to dental trauma | dentures | hid: missing_teeth | 158 |
| 367498001 | Seasonal allergic rhinitis | allergic_rhinitis | hid: atopic | 100 |
| 1187604002 | Serving in military service | encounter/sdoh_hrsn | hid: income, state, veteran · dem: Age, Race | 523 |
| 449868002 | Smokes tobacco daily | veteran_substance_abuse_treatment | hid: opioid_addiction, smoking, smoking_medication · dem: Age | 88 |
| 160701002 | Social migrant | encounter/sdoh_hrsn | hid: income, state · dem: Age, Race | 8 |
| 248595008 | Sputum finding | covid19/symptoms | hid: covid19_severity | 347 |
| 67787004 | Tongue tie | dentures | hid: missing_teeth | 28 |
| 11625007 | Torus mandibularis | dentures | hid: missing_teeth | 9 |
| 46752004 | Torus palatinus | dentures | hid: missing_teeth | 172 |
| 127295002 | Traumatic brain injury | mTBI | hid: veteran | 5 |
| 79586000 | Tubal pregnancy | pregnancy | hid: pregnant · dem: Gender | 75 |
| 56018004 | Wheezing | covid19/symptoms | hid: covid19_severity | 207 |

## Demographic-only  (100 conditions)

| Code | Description | Module | Gates / notes | Cases (10k) |
|------|-------------|--------|---------------|------------|
| 274531002 | Abnormal findings diagnostic imaging heart+coronary circulat | heart/cabg/cabg_referral | — | 1857 |
| 234466008 | Acquired coagulation disorder | covid19/outcomes | — | 14 |
| 62479008 | Acquired immune deficiency syndrome | hiv_diagnosis | — | 5 |
| 241929008 | Acute allergic reaction | allergies/severe_allergic_reaction | — | 124 |
| 65275009 | Acute cholecystitis | gallstones | dem: Gender, Race | 29 |
| 307426000 | Acute infective cystitis | urinary_tract_infections | — | 1160 |
| 91861009 | Acute myeloid leukemia | acute_myeloid_leukemia | dem: Gender | 4 |
| 67782005 | Acute respiratory distress syndrome | covid19/outcomes | — | 67 |
| 401303003 | Acute ST segment elevation myocardial infarction | heart/stemi_pathway | — | 201 |
| 203646004 | Adolescent idiopathic scoliosis | AIS_From_School_Screening_to_SOSORT_Recommendations | dem: Age | 300 |
| 1003755004 | Allergy to Hevea brasiliensis latex protein | spina_bifida | — | 3 |
| 26929004 | Alzheimer's disease | dementia | dem: Socioeconomic Status | 270 |
| 60234000 | Aortic valve regurgitation | vhd_aortic | — | 50 |
| 60573004 | Aortic valve stenosis | vhd_aortic | — | 78 |
| 74400008 | Appendicitis | appendicitis | dem: Gender | 91 |
| 49436004 | Atrial fibrillation | atrial_fibrillation | — | 98 |
| 5758002 | Bacteremia | acute_myeloid_leukemia | — | — |
| 128188000 | Cerebral palsy | cerebral_palsy | dem: Age, Gender | 38 |
| 373587001 | Chiari malformation type II | spina_bifida | — | 4 |
| 192127007 | Child attention deficit disorder | attention_deficit_disorder | dem: Gender | 204 |
| 88805009 | Chronic congestive heart failure | congestive_heart_failure | dem: Age, Gender | 313 |
| 40055000 | Chronic sinusitis | sinusitis | — | 2384 |
| 302297009 | Congenital deformity of foot | spina_bifida | — | 4 |
| 14760008 | Constipation | cerebral_palsy | dem: Age, Gender | 6 |
| 40275004 | Contact dermatitis | dermatitis | — | 60 |
| 190905008 | Cystic fibrosis | cystic_fibrosis | dem: Race | 5 |
| 427089005 | Diabetes mellitus due to cystic fibrosis | cystic_fibrosis | — | 3 |
| 157265008 | Dislocation of hip joint | cerebral_palsy | dem: Age, Gender | 6 |
| 62718007 | Dribbling from mouth | cerebral_palsy | dem: Age, Gender | 6 |
| 47318007 | Drug-induced neutropenia | acute_myeloid_leukemia | — | 3 |
| 15802004 | Dystonia | cerebral_palsy | dem: Age, Gender | 3 |
| 84757009 | Epilepsy | cerebral_palsy | dem: Age, Gender | 224 |
| 53827007 | Excessive salivation | cerebral_palsy | dem: Age, Gender | 14 |
| 230265002 | Familial Alzheimer's disease of early onset | dementia | — | 77 |
| 409089005 | Febrile neutropenia | acute_myeloid_leukemia | — | 1 |
| 707577004 | Female infertility due to cystic fibrosis | cystic_fibrosis | — | 3 |
| 203082005 | Fibromyalgia | prescribing_opioids_for_chronic_pain_and_treatment_of_oud | dem: Age, Date | 297 |
| 263172003 | Fracture of mandible | injuries/broken_jaw | — | 44 |
| 235919008 | Gallbladder calculus | gallstones | dem: Gender, Race | 29 |
| 235595009 | Gastroesophageal reflux disease | cerebral_palsy | dem: Age, Gender | 26 |
| 90560007 | Gout | gout | dem: Gender | 75 |
| 84114007 | Heart failure | covid19/outcomes | — | 13 |
| 105531004 | Housing unsatisfactory | encounter/sdoh_hrsn | — | 1448 |
| 86406008 | Human immunodeficiency virus infection | hiv_diagnosis | — | 23 |
| 230745008 | Hydrocephalus | spina_bifida | — | 3 |
| 389087006 | Hypoxemia | covid19/outcomes | — | 177 |
| 83664006 | Idiopathic atrophic hypothyroidism | hypothyroidism | dem: Gender | 96 |
| 11218009 | Infection caused by Pseudomonas aeruginosa | cystic_fibrosis | — | 2 |
| 406602003 | Infection caused by Staphylococcus aureus | cystic_fibrosis | — | 1 |
| 86175003 | Injury of heart | covid19/outcomes | — | 5 |
| 40095003 | Injury of kidney | covid19/outcomes | — | 12 |
| 110359009 | Intellectual disability | cerebral_palsy | dem: Age, Gender | 17 |
| 414545008 | Ischemic heart disease | stable_ischemic_heart_disease | — | 1853 |
| 414564002 | Kyphosis deformity of spine | spina_bifida | — | 2 |
| 93143009 | Leukemia, disease | trigger_bone_marrow_transplant | — | 24 |
| 200936003 | Lupus erythematosus | lupus | dem: Gender | 1 |
| 707418001 | Male infertility due to cystic fibrosis | cystic_fibrosis | — | 2 |
| 254837009 | Malignant neoplasm of breast | breast_cancer | dem: Gender, Race | 182 |
| 206523001 | Meconium ileus | cystic_fibrosis | — | — |
| 171131006 | Meningocele | spina_bifida | — | — |
| 414667000 | Meningomyelocele | spina_bifida | — | 4 |
| 361055000 | Misuses drugs | encounter/substance_use_screening | dem: Age | 956 |
| 48724000 | Mitral valve regurgitation | vhd_mitral | — | 52 |
| 79619009 | Mitral valve stenosis | vhd_mitral | — | 7 |
| 109989006 | Multiple myeloma | trigger_bone_marrow_transplant | — | 20 |
| 78275009 | Obstructive sleep apnea syndrome | sleep_apnea | — | 341 |
| 233604007 | Pneumonia | cerebral_palsy | dem: Age, Gender | 272 |
| 398152000 | Poor muscle tone | cerebral_palsy | dem: Age, Gender | 21 |
| 4557003 | Preinfarction syndrome | heart/nsteacs_pathway | — | 30 |
| 1163220007 | Pressure injury stage II | spina_bifida | — | 4 |
| 95417003 | Primary fibromyalgia syndrome | fibromyalgia | dem: Date, Gender | 36 |
| 91434003 | Pulmonic valve regurgitation | vhd_pulmonic | — | 4 |
| 56786000 | Pulmonic valve stenosis | vhd_pulmonic | — | 27 |
| 45816000 | Pyelonephritis | urinary_tract_infections | — | 25 |
| 197927001 | Recurrent urinary tract infection | urinary_tract_infections | — | 479 |
| 204949001 | Renal dysplasia | chronic_kidney_disease | — | 19 |
| 271825005 | Respiratory distress | covid19/outcomes | — | 177 |
| 267064002 | Retention of urine | cerebral_palsy | dem: Age, Gender | 1 |
| 69896004 | Rheumatoid arthritis | rheumatoid_arthritis | — | 27 |
| 160968000 | Risk activity involvement | encounter/substance_use_screening | — | 2811 |
| 47693006 | Rupture of appendix | appendicitis | dem: Gender | 27 |
| 298382003 | Scoliosis deformity of spine | spina_bifida | — | 2 |
| 128613002 | Seizure disorder | epilepsy | dem: Gender | 403 |
| 91302008 | Sepsis | sepsis | dem: Age | 405 |
| 448813005 | Sepsis caused by Pseudomonas | cystic_fibrosis | — | 2 |
| 448417001 | Sepsis caused by Staphylococcus aureus | cystic_fibrosis | — | 1 |
| 770349000 | Sepsis caused by virus | covid19/outcomes | — | 73 |
| 76571007 | Septic shock | covid19/outcomes | — | 91 |
| 80583007 | Severe anxiety (panic) | encounter/anxiety_screening | dem: Gender | 1319 |
| 27942005 | Shock | congestive_heart_failure | dem: Age, Gender | 21 |
| 39898005 | Sleep disorder | sleep_apnea | dem: Gender | 503 |
| 221360009 | Spasticity | cerebral_palsy | dem: Age, Gender | 31 |
| 76916001 | Spina bifida occulta | spina_bifida | — | 5 |
| 427419006 | Transformed migraine | prescribing_opioids_for_chronic_pain_and_treatment_of_oud | dem: Age, Date | 93 |
| 81629009 | Traumatic dislocation of temporomandibular joint | injuries/broken_jaw | — | 49 |
| 111287006 | Tricuspid valve regurgitation | vhd_tricuspid | — | 6 |
| 49915006 | Tricuspid valve stenosis | vhd_tricuspid | — | 1 |
| 288959006 | Unable to swallow saliva | cerebral_palsy | dem: Age, Gender | 10 |
| 10939881000119105 | Unhealthy alcohol drinking behavior | encounter/substance_use_screening | dem: Age, Gender | 1742 |
| 68566005 | Urinary tract infectious disease | spina_bifida | — | 4 |

## Excluded  (45 conditions)

| Code | Description | Module | Gates / notes | Cases (10k) |
|------|-------------|--------|---------------|------------|
| 195967001 | Asthma | asthma | hid: atopic · excl: precursor_names_target:known | 387 |
| 698303004 | Awaiting transplantation of bone marrow | bone_marrow_transplant | vis: Attribute:dental_referral · hid: bone_marrow · excl: situation_code:heuristic | 27 |
| 698306007 | Awaiting transplantation of kidney | kidney_transplant | vis: Attribute:dental_referral · hid: ckd · excl: situation_code:heuristic | 391 |
| 278860009 | Chronic low back pain | prescribing_opioids_for_chronic_pain_and_treatment_of_oud | dem: Age, Date · excl: symptom_type:heuristic | 2116 |
| 1121000119107 | Chronic neck pain | prescribing_opioids_for_chronic_pain_and_treatment_of_oud | dem: Age, Date · excl: symptom_type:heuristic | 1348 |
| 82423001 | Chronic pain | opioid_addiction | vis: Attribute:ptsd · hid: opioid_prescription · dem: Age, Date, Socioeconomic Status · excl: symptom_type:heuristic | 2892 |
| 49727002 | Cough | covid19/symptoms | hid: covid19_severity · excl: symptom_type:known | 644 |
| 44054006 | Diabetes mellitus type 2 | metabolic_syndrome_care | vis: Active CarePlan, Active Condition, Active Medication +2 · hid: ckd, diabetes, diabetes_severity +6 · dem: Gender · excl: precursor_names_target:known | 848 |
| 267060006 | Diarrhea symptom | covid19/symptoms | hid: covid19_severity · excl: symptom_type:heuristic | 31 |
| 267036007 | Dyspnea | covid19/symptoms | hid: covid19_severity · excl: symptom_type:known | 207 |
| 84229001 | Fatigue | covid19/symptoms | hid: covid19_severity · excl: symptom_type:known | 416 |
| 386661006 | Fever | covid19/symptoms | hid: covid19_severity · excl: symptom_type:known | 871 |
| 25064002 | Headache | covid19/symptoms | hid: covid19_severity · excl: symptom_type:known | 150 |
| 152621000119105 | History of allotransplantation of bone marrow | bone_marrow_transplant | hid: bone_marrow · excl: situation_code:heuristic | 3 |
| 429280009 | History of amputation of foot | metabolic_syndrome/amputations | excl: situation_code:heuristic | 3 |
| 119481000119105 | History of aortic valve repair | heart/avrr/savrr_operation | hid: cardiac_surgery · excl: situation_code:heuristic | 1 |
| 1231000119100 | History of aortic valve replacement | heart/avrr/savrr_operation | hid: cardiac_surgery · excl: situation_code:heuristic | 80 |
| 428251008 | History of appendectomy | appendicitis | dem: Gender · excl: situation_code:heuristic | 471 |
| 161679004 | History of artificial joint | total_joint_replacement | hid: assessment_done, joint_replacement · dem: Age · excl: situation_code:heuristic | 9 |
| 108631000119101 | History of autologous bone marrow transplant | bone_marrow_transplant | hid: bone_marrow · excl: situation_code:heuristic | 24 |
| 399261000 | History of coronary artery bypass grafting | heart/cabg/operation | vis: Attribute:hyperglycemia · hid: diabetes, operative_status · excl: situation_code:heuristic | 770 |
| 698423002 | History of disarticulation at wrist | metabolic_syndrome/amputations | excl: situation_code:heuristic | — |
| 161622006 | History of lower limb amputation | metabolic_syndrome/amputations | excl: situation_code:heuristic | 7 |
| 399211009 | History of myocardial infarction | heart/cabg_sequence | vis: Attribute:cardiac_surgery_reason · excl: situation_code:heuristic | 494 |
| 153351000119102 | History of peripheral stem cell transplant | bone_marrow_transplant | hid: bone_marrow · excl: situation_code:heuristic | 15 |
| 161665007 | History of renal transplant | kidney_transplant | hid: ckd · excl: situation_code:heuristic | 355 |
| 1290882004 | History of seizure | epilepsy | dem: Gender · excl: situation_code:heuristic | 403 |
| 267020005 | History of tubal ligation | contraceptives/female_sterilization | excl: situation_code:heuristic | 1260 |
| 120991000119102 | History of undergoing in utero procedure while a fetus | spina_bifida | excl: situation_code:heuristic | — |
| 161621004 | History of upper limb amputation | metabolic_syndrome/amputations | excl: situation_code:heuristic | — |
| 363406005 | Malignant neoplasm of colon | colorectal_cancer | vis: Attribute:smoker · dem: Age · excl: screening_work_up:known | 77 |
| 314529007 | Medication review due | med_rec | excl: situation_code:heuristic | 11504 |
| 68962001 | Muscle pain | covid19/symptoms | hid: covid19_severity · excl: symptom_type:heuristic | 131 |
| 422587007 | Nausea | covid19/symptoms | hid: covid19_severity · excl: symptom_type:known | 44 |
| 22253000 | Pain | cerebral_palsy | dem: Age, Gender · excl: symptom_type:heuristic | 10 |
| 161744009 | Past pregnancy history of miscarriage | pregnancy | hid: pregnant · dem: Gender · excl: situation_code:heuristic | 2220 |
| 236077008 | Protracted diarrhea | colorectal_cancer | vis: Attribute:smoker · excl: symptom_type:heuristic | 26 |
| 36971009 | Sinusitis | sinusitis | vis: Active Condition, Active Medication · excl: precursor_names_target:known | 671 |
| 73430006 | Sleep apnea | sleep_apnea | excl: screening_work_up:known | 161 |
| 267102003 | Sore throat | covid19/symptoms | hid: covid19_severity · excl: symptom_type:known; precursor_names_target:known | 153 |
| 183996000 | Sterilization requested | contraceptives/female_sterilization | excl: situation_code:heuristic | 314 |
| 840544004 | Suspected disease caused by Severe acute respiratory coronavirus 2 | covid19/infection | vis: Attribute:Cystic_Fibrosis, Attribute:asthma_condition, Attribute:breast_cancer_condition +4 · hid: colorectal_cancer_stage, coronary_heart_disease, diabetes +5 · excl: situation_code:heuristic | 977 |
| 162573006 | Suspected lung cancer | lung_cancer | vis: Attribute:quit smoking age, Attribute:smoker · dem: Gender · excl: situation_code:heuristic | 85 |
| 315268008 | Suspected prostate cancer | veteran_prostate_cancer | vis: Attribute:prostate_cancer, Observation · hid: veteran · dem: Gender · excl: situation_code:heuristic | 140 |
| 249497008 | Vomiting symptom | covid19/symptoms | hid: covid19_severity · excl: symptom_type:heuristic | 44 |

## Summary

| Group | Count |
|-------|-------|
| History-dependent (visible) | 127 |
| History-dependent (hidden only) | 61 |
| Demographic-only | 100 |
| Excluded | 45 |
| **Total** | **333** |

_Generated by `scripts/scan_modules.py`. Source: `.synthea/src/main/resources/modules/`._
