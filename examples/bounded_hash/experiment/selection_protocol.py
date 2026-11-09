"""Predeclared finite version-selection panel. Change only before a new study."""
from __future__ import annotations
import hashlib
import json

VERSION = 'infinilm-version-selection-20261103-single-pass'
PROFILES = ['reuse8', 'reuse16', 'longdecode', 'cold']
LEVELS = [16, 32, 64]
SEEDS = [33962]
ARMS = ['nh_mean_bounded', 'nh_cache_bounded_v2', 'vllm_least_active']


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def make_spec():
    from cache_workload import PROFILES as PARAMETERS
    return dict(
        version=VERSION,
        execution_protocol='infinilm-cache-paired-confirm-v2-fresh-http',
        profiles=list(PROFILES), profile_parameters={p: dict(PARAMETERS[p]) for p in PROFILES},
        concurrency=list(LEVELS), seeds=list(SEEDS), arms=list(ARMS),
        rounds=1, learning_requests=64, measured_requests=128,
        primary_phase='measured', secondary_phase='learn_plus_measured',
        scenarios=[dict(profile=p, concurrency=c) for p in PROFILES for c in LEVELS],
        actual_cells=len(PROFILES)*len(LEVELS)*len(ARMS),
        learning_plus_measured_requests=len(PROFILES)*len(LEVELS)*len(ARMS)*(64+128),
        expected_paired_trials_per_scenario=1,
        evidence_level='SINGLE_SEED_SINGLE_ROUND_SCREEN',
        rules=dict(min_strict_pair_wins=1, min_pooled_throughput_ratio=1.02,
                   max_weighted_mean_ttft_ratio=0.98,
                   majority_scenarios=7, scenario_count=12,
                   joint_win_requires=['vllm_least_active', 'other_candidate']),
        interpretation=dict(
            scenario_weight='One equal vote for each declared profile/concurrency; no token-weighted cross-scenario score.',
            margin='2% is a predeclared practical difference, not statistical significance.',
            seed_scope='One fixed seed (33962), one measured observation per arm per scenario.',
            order_scope='Each arm occupies each position once per workload across the three concurrency levels, four times per position over the panel.',
            tail='Report p95 and slow requests separately; p95 is not a hidden veto or a winning claim.',
            cold_start='Learning-inclusive sum excludes model startup, probes and calibration; it is not deployment cold-start time.',
            no_winner='Archive both versions and all evidence; do not tune, drop cases or automatically extend the panel.',
            selection_scope='Primary measured winner only. Same combined winner supports both phases; opposite combined winner prevents unique selection.'
        ))


def validate_spec(spec):
    expected = make_spec()
    for key, value in expected.items():
        if spec.get(key) != value:
            raise ValueError('Frozen selection protocol differs: '+key)
    if 'spec_sha256' in spec:
        payload = {k: v for k, v in spec.items() if k != 'spec_sha256'}
        if spec['spec_sha256'] != digest(payload):
            raise ValueError('Selection plan hash mismatch')
    return True
