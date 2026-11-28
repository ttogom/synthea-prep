"""Descriptive diagnostics behind the agentic-versus-fixed retrieval talking points.

The poster matrices and significance tests come from ``poster_statistics.py`` and
``compare_agentic_variants.py``. This script explains *why* the agentic cells look the
way they do. It reads the fixed and agentic run JSONLs named in ``sources.yaml`` plus
the combined ``experiment_results.csv`` and saves one long-format CSV of descriptive
numbers: search behaviour, evidence choice, cost, guideline text size, failures,
question focus, and per-condition plan results. No statistical tests are run here.

Keyword rules (``TREATMENT_QUESTION``, ``CAUSE_QUESTION``, ``PLAN_TEST``) are simple,
documented regular expressions. Percentages that depend on them are approximate
classifications, not clinician labels.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import statistics
import sys

import pandas as pd
import yaml

# Direct execution must resolve repository modules before importing them.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS_DIR = Path('experiments/results_all_variants')
OUTPUT_NAME = 'agentic_diagnostics.csv'
MODELS = ('gpt_4o_mini', 'ollama_gemma3_4b', 'ollama_llama3_2_3b', 'ollama_ministral_3_3b')
AGENT_COMPONENTS = {'agent_patients': 'agentic_patient_retrieval',
                    'agent_guidelines': 'agentic_guidelines', 'agent_both': 'agentic_both'}
PATIENT_TOOL, GUIDELINE_TOOL = 'search_similar_patients', 'search_guidelines'

_UUID_SUFFIX = re.compile(r'_([0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12})$')
# A guideline question is "treatment" if it asks about treating or managing, and
# "cause" if it asks what could explain the findings. A question can be both.
TREATMENT_QUESTION = re.compile(
    r'treat|manag|therap|medication|first-line|drug|prescri|antibiotic', re.I)
CAUSE_QUESTION = re.compile(r'caus|differential|distinguish|etiolog|diagnos', re.I)
# A generated rank-1 plan "includes a test" if any item names a diagnostic procedure.
PLAN_TEST = re.compile(
    r'culture|swab|\btest|ecg|ekg|electrocardiogra|echocardiogra|nyha|spirometr|'
    r'urinalysis|x-ray|radiograph|\bcbc\b|blood count|\bbnp\b|monospot|imaging|ultrasound',
    re.I)
# Conditions discussed in the talking points, matched against condition_label.
FOCUS_CONDITIONS = {'viral pharyngitis': 'viral pharyngitis', 'heart failure': 'heart failure',
                    'hypertension': 'hypertension', 'sinusitis': 'sinusitis'}


def condition_prefix(case_id):
    """Return the condition part of '<condition>_<patient uuid>' identifiers."""
    return _UUID_SUFFIX.sub('', case_id)


def patient_uuid(case_id):
    match = _UUID_SUFFIX.search(case_id)
    return match.group(1).lower() if match else case_id


def total_tokens(responses):
    return sum((item.get('token_usage') or {}).get('total_tokens') or 0 for item in responses or [])


def share(flags):
    flags = list(flags)
    return (sum(flags) / len(flags) if flags else None), len(flags)


class Rows:
    """Collect long-format rows: one number per row, with its denominator and note."""

    def __init__(self):
        self.rows = []

    def add(self, section, metric, value, n=None, variant='', group='', model='', note=''):
        self.rows.append({'section': section, 'metric': metric, 'variant': variant,
                          'group': group, 'model': model,
                          'value': None if value is None else float(value), 'n': n, 'note': note})

    def frame(self):
        return pd.DataFrame(self.rows)


def run_paths(results_dir):
    """Locate the fixed and agentic run JSONLs from the committed sources.yaml."""
    sources = yaml.safe_load((results_dir / 'sources.yaml').read_text(encoding='utf-8'))
    fixed = agentic = None
    for run in sources['runs']:
        if 'guidelines' in run['variants']:
            fixed = ROOT / run['path'] / 'results.jsonl'
        if 'agentic_guidelines' in run['variants']:
            agentic = ROOT / run['path'] / 'results.jsonl'
    if fixed is None or agentic is None:
        raise ValueError('sources.yaml must name one fixed run and one agentic run')
    return fixed, agentic


def load_fixed(path):
    """Per-case evidence and token use from the fixed (non-agentic) run."""
    fixed = {}
    with path.open(encoding='utf-8') as source:
        for line in source:
            record = json.loads(line)
            components = record['components']
            patients = components['patient_retrieval']
            guidelines = components['guidelines']
            fixed[record['id']] = {
                'patient': (patients['capture'].get('selected_reference_patient') or {}).get('id'),
                'guideline_bytes': len((guidelines['outputs'].get('Key_info') or '').encode('utf-8')),
                'questions': guidelines['capture'].get('guideline_questions') or [],
                'keyword_tokens': total_tokens(patients['capture']['model_responses'].get('keyword_extraction')),
                'question_tokens': total_tokens(guidelines['capture']['model_responses'].get('guideline_questions')),
                'graphrag_tokens': total_tokens(guidelines['capture']['model_responses'].get('guideline_evidence')),
            }
    return fixed


def failure_reason(error):
    message = str((error or {}).get('message', '')) if isinstance(error, dict) else str(error or '')
    if 'Selected guidelines use' in message:
        return 'guideline_byte_budget'
    if 'Unrecorded diagnosis terms' in message:
        return 'diagnosis_term_rule'
    return 'other_error'


def agentic_diagnostics(agentic_path, fixed, rows):
    """Stream the agentic run and record agent behaviour, cost, and failures."""
    searches = defaultdict(list)
    calls = defaultdict(Counter)
    control_tokens = defaultdict(list)
    graphrag_tokens = defaultdict(list)
    keyword_tokens = defaultdict(list)
    failures = Counter()
    failed_ids = defaultdict(set)
    guideline_bytes = {}
    screens, both_mix, uses_all = [], Counter(), []
    same_patient, condition_match_agentic, condition_match_fixed = [], [], []
    not_first_top, from_later_search = [], []
    agentic_questions = []

    with agentic_path.open(encoding='utf-8') as source:
        for line in source:
            record = json.loads(line)
            case_id = record['id']
            for name, component in record['components'].items():
                if name not in AGENT_COMPONENTS:
                    continue
                if component['status'] != 'completed':
                    failures[(name, failure_reason(component.get('error')))] += 1
                    failed_ids[name].add(case_id)
                    continue
                capture = component['capture']
                agent, responses = capture['agent'], capture['model_responses']
                trace = agent.get('trace') or []
                searches[name].append(agent['searches'])
                calls[name][len(responses.get('agent_control') or [])] += 1
                control_tokens[name].append(total_tokens(responses.get('agent_control')))
                keyword_tokens[name].append(total_tokens(responses.get('keyword_extraction')))
                graphrag_tokens[name].append(total_tokens(responses.get('guideline_evidence')))

                if name != 'agent_patients':
                    screens.append(len(agent.get('screened_questions') or []))
                if name == 'agent_guidelines':
                    uses_all.append(agent['searches'] == component['settings']['max_searches'])
                    guideline_bytes[case_id] = len(component['outputs']['Key_info'].encode('utf-8'))
                    agentic_questions += [step['input'] for step in trace if step['tool'] == GUIDELINE_TOOL]
                if name == 'agent_both':
                    tools = Counter(step['tool'] for step in trace)
                    both_mix[(tools[PATIENT_TOOL], tools[GUIDELINE_TOOL])] += 1
                if name == 'agent_patients':
                    chosen = capture['selected_reference_patient']['id']
                    fixed_choice = fixed[case_id]['patient']
                    target = condition_prefix(case_id)
                    same_patient.append(chosen == fixed_choice)
                    condition_match_agentic.append(condition_prefix(chosen) == target)
                    condition_match_fixed.append(
                        fixed_choice is not None and condition_prefix(fixed_choice) == target)
                    # Reference ids are numbered in search order (P1, P2, ...), so the
                    # first search returned P1..Pk with k = results_per_search.
                    first_search_size = component['settings']['results_per_search']
                    number = int(agent['patient_ref'][1:])
                    not_first_top.append(number != 1)
                    from_later_search.append(number > first_search_size)

    section = 'searches'
    for name, variant in AGENT_COMPONENTS.items():
        rows.add(section, 'mean_searches_per_case', statistics.mean(searches[name]),
                 len(searches[name]), variant, note='maximum is max_searches (4)')
    value, n = share(uses_all)
    rows.add(section, 'share_using_all_searches', value, n, 'agentic_guidelines')
    total_both = sum(both_mix.values())
    for (patient_searches, guideline_searches), count in sorted(both_mix.items()):
        rows.add(section, f'share_{patient_searches}_patient_{guideline_searches}_guideline_searches',
                 count / total_both, total_both, 'agentic_both')

    section = 'patient_choice'
    for metric, flags, note in (
            ('same_patient_as_fixed', same_patient, 'agent pick equals the fixed pipeline pick'),
            ('agent_pick_same_condition', condition_match_agentic, 'reference condition equals target condition'),
            ('fixed_pick_same_condition', condition_match_fixed, 'reference condition equals target condition'),
            ('pick_not_top_of_first_search', not_first_top, 'agent did not keep P1'),
            ('pick_from_later_search', from_later_search, 'pick came after the first search results')):
        value, n = share(flags)
        rows.add(section, metric, value, n, 'agentic_patient_retrieval', note=note)

    section = 'cost'
    for name, variant in AGENT_COMPONENTS.items():
        counter = calls[name]
        n = sum(counter.values())
        rows.add(section, 'min_agent_calls', min(counter), n, variant)
        rows.add(section, 'median_agent_calls', statistics.median(counter.elements()), n, variant)
        rows.add(section, 'max_agent_calls', max(counter), n, variant)
        for count, cases in sorted(counter.items()):
            rows.add(section, f'share_with_{count}_agent_calls', cases / n, n, variant)
        rows.add(section, 'mean_agent_control_tokens', statistics.mean(control_tokens[name]), n, variant,
                 note='GPT-4o-mini agent calls only, including retries')
        rows.add(section, 'median_agent_control_tokens', statistics.median(control_tokens[name]), n, variant)
        rows.add(section, 'mean_keyword_extraction_tokens', statistics.mean(keyword_tokens[name]), n, variant)
        rows.add(section, 'mean_graphrag_answer_tokens', statistics.mean(graphrag_tokens[name]), n, variant)
    fixed_values = list(fixed.values())
    rows.add(section, 'mean_keyword_extraction_tokens', statistics.mean(v['keyword_tokens'] for v in fixed_values),
             len(fixed_values), 'patient_retrieval', note='fixed controller tokens')
    rows.add(section, 'mean_guideline_question_tokens', statistics.mean(v['question_tokens'] for v in fixed_values),
             len(fixed_values), 'guidelines', note='fixed controller tokens')
    rows.add(section, 'mean_graphrag_answer_tokens', statistics.mean(v['graphrag_tokens'] for v in fixed_values),
             len(fixed_values), 'guidelines', note='one GraphRAG search per case')

    section = 'guideline_text'
    fixed_bytes = [fixed[case_id]['guideline_bytes'] for case_id in guideline_bytes]
    rows.add(section, 'median_guideline_bytes', statistics.median(guideline_bytes.values()),
             len(guideline_bytes), 'agentic_guidelines')
    rows.add(section, 'median_guideline_bytes', statistics.median(fixed_bytes), len(fixed_bytes), 'guidelines',
             note='same cases as the agentic row')

    section = 'diagnosis_rule'
    rows.add(section, 'mean_rejected_questions_per_case', statistics.mean(screens), len(screens),
             'agentic_guidelines+agentic_both', note='question named a diagnosis and had to be rewritten')

    section = 'failures'
    for (name, reason), count in sorted(failures.items()):
        rows.add(section, 'failed_cases', count, None, AGENT_COMPONENTS[name], group=reason)
    rows.add(section, 'failed_cases_total', sum(failures.values()), None, 'all_agentic')

    section = 'question_focus'
    fixed_questions = [question for value in fixed_values for question in value['questions']]
    for variant, questions in (('agentic_guidelines', agentic_questions), ('guidelines', fixed_questions)):
        value, n = share(bool(TREATMENT_QUESTION.search(q)) for q in questions)
        rows.add(section, 'share_treatment_questions', value, n, variant, note='keyword rule TREATMENT_QUESTION')
        value, n = share(bool(CAUSE_QUESTION.search(q)) for q in questions)
        rows.add(section, 'share_cause_questions', value, n, variant, note='keyword rule CAUSE_QUESTION')

    return guideline_bytes, failed_ids


def score_diagnostics(results, guideline_bytes, failed_ids, rows):
    """Per-condition plan results, failure groups, and text-length quarters from the scored CSV."""
    results = results.copy()
    results['plan'] = pd.to_numeric(results['plan1_score'], errors='coerce').where(
        results['plan1_status'].eq('scored'))
    results['diagnosis'] = pd.to_numeric(results['biolord_score_diagnosis1'], errors='coerce').where(
        results['diagnosis1_status'].eq('scored'))
    cases = results.drop_duplicates('id').set_index('id')

    section = 'conditions'
    for condition, count in cases['condition_label'].value_counts().items():
        rows.add(section, 'cases', count, len(cases), group=condition)
    for label, needle in FOCUS_CONDITIONS.items():
        subset = cases[cases['condition_label'].str.contains(needle, case=False)]
        plans = subset['plan_label'].value_counts(normalize=True)
        for plan, value in plans.head(3).items():
            rows.add(section, 'reference_plan_share', value, len(subset), group=label, note=plan)

    # Fixed vs agentic guideline plans, paired on case and model.
    wide = results[results['variant'].isin(['guidelines', 'agentic_guidelines'])].pivot_table(
        index=['id', 'model'], columns='variant', values='plan', aggfunc='first')
    plans = results[results['variant'].isin(['guidelines', 'agentic_guidelines'])].pivot_table(
        index=['id', 'model'], columns='variant', values='plan1', aggfunc='first')
    paired = wide.dropna().join(plans, rsuffix='_text')
    paired = paired.join(cases['condition_label'], on='id')
    section = 'plan_by_condition'
    for label, needle in FOCUS_CONDITIONS.items():
        subset = paired[paired['condition_label'].str.contains(needle, case=False)]
        for model in ('all_models',) + MODELS:
            part = subset if model == 'all_models' else subset[subset.index.get_level_values('model') == model]
            if part.empty:
                continue
            n = len(part)
            rows.add(section, 'mean_plan_score', part['guidelines'].mean(), n, 'guidelines', label, model)
            rows.add(section, 'mean_plan_score', part['agentic_guidelines'].mean(), n, 'agentic_guidelines', label, model)
            rows.add(section, 'plan_change_points', 100 * (part['agentic_guidelines'] - part['guidelines']).mean(),
                     n, 'agentic_guidelines-guidelines', label, model)
            for variant in ('guidelines', 'agentic_guidelines'):
                text = part[f'{variant}_text'].fillna('').astype(str)
                rows.add(section, 'share_plans_with_test', text.str.contains(PLAN_TEST).mean(), n,
                         variant, label, model, note='keyword rule PLAN_TEST')

    # Text-length quarters of agentic guideline evidence, pooled over models.
    section = 'guideline_text'
    paired = paired.reset_index()
    paired = paired[paired['id'].isin(guideline_bytes)]
    paired['bytes'] = paired['id'].map(guideline_bytes)
    paired['quarter'] = pd.qcut(paired['bytes'], 4, labels=['Q1_shortest', 'Q2', 'Q3', 'Q4_longest'])
    for quarter, part in paired.groupby('quarter', observed=True):
        rows.add(section, 'plan_change_points_by_text_quarter',
                 100 * (part['agentic_guidelines'] - part['guidelines']).mean(), len(part),
                 'agentic_guidelines-guidelines', str(quarter),
                 note=f'{int(part["bytes"].min())}-{int(part["bytes"].max())} bytes')

    # Failed agent_both cases: which groups, and were they harder without retrieval?
    section = 'failures'
    failed = failed_ids['agent_both']
    for group, part in cases.groupby('bin'):
        rows.add(section, 'share_failed_by_group', part.index.isin(failed).mean(), len(part),
                 'agentic_both', group)
    baseline = results[results['variant'].eq('llm_only')]
    for model in MODELS:
        part = baseline[baseline['model'].eq(model)]
        is_failed = part['id'].isin(failed)
        rows.add(section, 'llm_only_diagnosis_failed_cases', part.loc[is_failed, 'diagnosis'].mean(),
                 int(is_failed.sum()), 'llm_only', model=model, note='cases whose agentic_both run failed')
        rows.add(section, 'llm_only_diagnosis_other_cases', part.loc[~is_failed, 'diagnosis'].mean(),
                 int((~is_failed).sum()), 'llm_only', model=model)

    section = 'cohort'
    rows.add(section, 'cases', len(cases), None)
    rows.add(section, 'patients', cases.index.map(patient_uuid).nunique(), None)


def generate_agentic_diagnostics(results_dir=ROOT / RESULTS_DIR):
    results_dir = Path(results_dir).expanduser().resolve()
    fixed_path, agentic_path = run_paths(results_dir)
    rows = Rows()
    fixed = load_fixed(fixed_path)
    guideline_bytes, failed_ids = agentic_diagnostics(agentic_path, fixed, rows)
    results = pd.read_csv(results_dir / 'experiment_results.csv', low_memory=False)
    score_diagnostics(results, guideline_bytes, failed_ids, rows)
    destination = results_dir / 'analysis' / OUTPUT_NAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = rows.frame()
    frame.to_csv(destination, index=False, na_rep='')
    return frame, destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--results-dir', type=Path, default=ROOT / RESULTS_DIR)
    args = parser.parse_args(argv)
    frame, destination = generate_agentic_diagnostics(args.results_dir)
    print(f'Saved {len(frame)} descriptive diagnostics to {destination}')


if __name__ == '__main__':
    main()
