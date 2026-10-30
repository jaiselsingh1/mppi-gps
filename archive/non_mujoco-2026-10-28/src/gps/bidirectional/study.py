"""Finite neural Ending development: learning, trust, acquisition, next actors.

This is one open-development pilot, not the three-seed physical comparison.
All gates and budgets are fixed before launch; a failed gate retains its data.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch

from .ending import EndingExecutor,EndingSnapshot,ScalarArchive,Task,identity
from .evidence import verify_experiment,file_sha256
from .learning import (LearningConfig,NeuralPolicy,OutcomeLearner,collect_owned,
    context_from_episode,load_pt,make_actor,save_pt)
from .loop import IndependentMemorySearch,controller_episode
from .policy import FamilyGaussianActor,policy_hash
from .search import SearchConfig,TrajectoryMemory,state_json
from .trust import CandidateTrustManager,Outcome,TrustManager
from .coupling import select_teaching_pairs
from .meter import optimization_totals


DEVELOP=dict(schema='outcome-coupled-neural-ending-v0',mode='develop',seed=0,
    task=asdict(Task()),learning=asdict(LearningConfig()),
    teacher_roots=[1000,1001,1002,1003],teacher_streams=[0,1],
    long_search=asdict(SearchConfig(K=32,H=25,temperature=.2,sigma=.8,
        smooth_candidates=8,memory_candidates=8,knot_spacing=4)),
    short_search=asdict(SearchConfig(K=32,H=4,temperature=.2,sigma=.8,
        smooth_candidates=8,memory_candidates=16,knot_spacing=4)),
    full_memory_candidates=16,planner_seed=910000,agreement=1.,
    initial_updates=96,child_updates=24,rounds=2,qualification_every=8,
    ordinary_replicates=8,acquired_replicates=4,acquired_times=[4,8],
    ordinary_qualification_roots=16,standalone_roots=64,
    handoff_times=[0,4,8],handoff_perturbations=[-.01,0.,.01],support_radius=.15,
    control_trial_roots=8,control_confirmation_roots=8,control_evaluation_roots=16,
    acquisition_roots=4,agreement_ladder=[.1,1.,10.],capture_min=.9,
    minimum_control_gain=.001,minimum_child_gain=.001,
    max_controls=10000000,max_neural_rows=5000000,max_optimizer_calls=800,max_backward_calls=810,
    max_verification_controls=10000000,max_verification_neural_rows=5000000,
    max_verification_optimizer_calls=800,max_verification_backward_calls=810,
    transition_atol=0.,neural_atol=1e-6,optimizer_atol=0.,
    scope='One-seed finite open development; no physical-task or all-cost efficiency claim')

INTERFACE=deepcopy(DEVELOP)
INTERFACE.update(schema='outcome-coupled-neural-interface-v0',mode='interface',initial_updates=2,
    child_updates=0,rounds=0,qualification_every=1,ordinary_qualification_roots=4,
    standalone_roots=16,max_controls=400000,max_neural_rows=100000,max_optimizer_calls=16,max_backward_calls=20,
    max_verification_controls=400000,max_verification_neural_rows=100000,
    max_verification_optimizer_calls=16,max_verification_backward_calls=20,
    short_search=asdict(SearchConfig(K=16,H=4,temperature=.2,sigma=.8,
        smooth_candidates=4,memory_candidates=4,knot_spacing=4)),
    scope='Neural interface, ownership, independent fallback and optimizer replay; no performance gate')

# One declared revision: execute the prefix/actor program that was scored.
# Learning, independent proposals, memory and outcome gates stay unchanged.
CONTINUATION=deepcopy(DEVELOP)
CONTINUATION.update(schema='outcome-coupled-neural-commitment-v1',mode='continuation',
    initial_updates=0,root_offset=2000000,seed_offset=2000000,
    execution_mode='commit_prefix_then_actor',commitment_state_atol=1e-5,
    max_controls=8000000,max_neural_rows=4000000,max_optimizer_calls=400,max_backward_calls=400,
    max_verification_controls=8000000,max_verification_neural_rows=4000000,
    max_verification_optimizer_calls=400,max_verification_backward_calls=400,
    scope='One-seed two-round continuation from a closed neural parent; unchanged learning; no physical or efficiency claim')

ACQUISITION=deepcopy(CONTINUATION)
ACQUISITION.update(schema='outcome-coupled-qualified-acquisition-v2',rounds=1,
    root_offset=4000000,seed_offset=4000000,acquisition_strategy='qualified_trial_pairs',
    teaching_minimum_gain=1e-6,teaching_state_atol=1e-8,
    max_controls=4000000,max_neural_rows=2000000,max_optimizer_calls=200,max_backward_calls=200,
    max_verification_controls=4000000,max_verification_neural_rows=2000000,
    max_verification_optimizer_calls=200,max_verification_backward_calls=200,
    scope='One-round conditional teaching intervention; unchanged learner; no unconditional acquisition, efficiency or completed-loop claim')

ARCHIVE_TRANSFER=deepcopy(ACQUISITION)
ARCHIVE_TRANSFER.update(schema='outcome-coupled-archived-transfer-v3',mode='archive_transfer',
    acquisition_strategy='archived_qualified_pairs',root_offset=6000000,seed_offset=6000000,
    teacher_round=0,max_controls=200000,max_neural_rows=1000000,
    max_verification_controls=200000,max_verification_neural_rows=1000000,
    scope='One-round archived teacher-to-learner transfer; no current search influence, fresh acquisition or full-loop claim')

ARCHIVE_INTERFACE=deepcopy(ARCHIVE_TRANSFER)
ARCHIVE_INTERFACE.update(schema='outcome-coupled-archive-interface-v3',interface_only=True,
    child_updates=1,qualification_every=1,ordinary_qualification_roots=4,standalone_roots=4,
    max_controls=20000,max_neural_rows=100000,max_optimizer_calls=16,max_backward_calls=16,
    max_verification_controls=20000,max_verification_neural_rows=100000,
    max_verification_optimizer_calls=16,max_verification_backward_calls=16,
    scope='Archive/learner/import-replay software interface only; no learning-benefit pass criterion')

DETERMINISTIC_CREDIT=deepcopy(ARCHIVE_TRANSFER)
DETERMINISTIC_CREDIT.update(schema='outcome-coupled-deterministic-credit-v4',credit_diagnostic=True,
    root_offset=8000000,seed_offset=8000000,rounds=0,child_updates=0,
    max_controls=20000,max_neural_rows=100000,max_optimizer_calls=0,max_backward_calls=32,
    max_verification_controls=20000,max_verification_neural_rows=100000,
    max_verification_optimizer_calls=0,max_verification_backward_calls=32,
    credit=dict(epsilons=[.001,.0005],line_steps=[.01,.003,.001,.0003],
        derivative_atol=.0001,derivative_rtol=.02,minimum_gradient_contrast=1e-8,minimum_training_gain=1e-8),
    scope='Deterministic-feedback gradient and one full-network line-step only; no current search or full-loop claim')

ENDING_PROJECTION=deepcopy(DETERMINISTIC_CREDIT)
ENDING_PROJECTION.update(schema='outcome-coupled-ending-projection-v5',root_offset=10000000,seed_offset=10000000,
    scope='One ending-constrained matched actor step; no current search, new acquisition or full-loop claim')
ENDING_PROJECTION['credit'].update(ending_projection=True,ending_margin=.0002,
    ending_derivative_atol=.00001,ending_derivative_rtol=.02)
PRESERVED_ENDING=deepcopy(ENDING_PROJECTION)
PRESERVED_ENDING.update(schema='outcome-coupled-preserved-ending-v6',root_offset=12000000,seed_offset=12000000,
    scope='One ending-and-ordinary constrained matched actor step; no current search or full-loop claim')
PRESERVED_ENDING['credit'].update(ordinary_preservation=True,projection_feasibility_atol=1e-10,
    projection_condition_limit=1e10,ordinary_training_atol=0.)
FRESH_SEARCH=deepcopy(ACQUISITION)
FRESH_SEARCH.update(schema='outcome-coupled-fresh-search-v7',fresh_search_only=True,
    child_updates=0,root_offset=14000000,seed_offset=14000000,
    max_optimizer_calls=0,max_backward_calls=0,
    max_verification_optimizer_calls=0,max_verification_backward_calls=0,
    scope='Fresh search qualification and teaching acquisition by an improved actor; no next-policy or completed-loop claim')
POINTWISE_TRANSFER=deepcopy(PRESERVED_ENDING)
POINTWISE_TRANSFER.update(schema='outcome-coupled-pointwise-transfer-v8',teacher_evidence='pointwise_trial',
    root_offset=16000000,seed_offset=16000000,
    scope='Conditional learning from measured successful trial experience; no qualified search or full-loop claim')
NONLINEAR_RESTORATION=deepcopy(POINTWISE_TRANSFER)
NONLINEAR_RESTORATION.update(schema='outcome-coupled-nonlinear-restoration-v9',
    root_offset=18000000,seed_offset=18000000,max_backward_calls=128,max_verification_backward_calls=128,
    scope='Conditional matched actor steps with bounded measured nonlinear restoration; no current search or full-loop claim')
NONLINEAR_RESTORATION['credit']['restoration_max_steps']=3
RESTORED_POLICY_SEARCH=deepcopy(FRESH_SEARCH)
RESTORED_POLICY_SEARCH.update(schema='outcome-coupled-restored-policy-search-v10',compare_previous_actor=True,
    root_offset=20000000,seed_offset=20000000,max_controls=6000000,max_verification_controls=6000000,
    scope='Fresh restored-policy search and previous-actor counterfactual with all-raw independent memory; no next learner or full-loop claim')
REPEATED_CREDIT=deepcopy(NONLINEAR_RESTORATION)
REPEATED_CREDIT.update(schema='outcome-coupled-learning-depth-v11',repeated_credit=True,
    one_step_reference_run='runs/2026-10-27_coupled_nonlinear_restoration_001',
    root_offset=22000000,seed_offset=22000000,qualification_every=4,
    max_controls=500000,max_verification_controls=500000,
    max_neural_rows=1000000,max_verification_neural_rows=1000000,
    max_backward_calls=4096,max_verification_backward_calls=4096,
    scope='Bounded repeated outcome-constrained learning on unchanged teachers; no current search or full-loop claim')
REPEATED_CREDIT['credit']['max_iterations']=32
REPEATED_CREDIT_INTERFACE=deepcopy(REPEATED_CREDIT)
REPEATED_CREDIT_INTERFACE.update(schema='outcome-coupled-learning-depth-interface-v11',interface_only=True,
    root_offset=24000000,seed_offset=24000000,
    max_controls=50000,max_verification_controls=50000,
    max_neural_rows=200000,max_verification_neural_rows=200000,
    max_backward_calls=256,max_verification_backward_calls=256,
    scope='Two-sweep repeated-learning software interface, current-owner checks and v9 hash reproduction; no depth-benefit claim')
REPEATED_CREDIT_INTERFACE['credit']['max_iterations']=2
DEEP_POLICY_SEARCH=deepcopy(RESTORED_POLICY_SEARCH)
DEEP_POLICY_SEARCH.update(schema='outcome-coupled-depth-policy-search-v12',
    root_offset=26000000,seed_offset=26000000,
    scope='Fresh deep-policy search versus independent memory and the one-step policy; no next learner or full-loop claim')
FEEDBACK_LEARNING=deepcopy(REPEATED_CREDIT)
FEEDBACK_LEARNING.pop('one_step_reference_run')
FEEDBACK_LEARNING.update(schema='outcome-coupled-fresh-feedback-learning-v13',feedback_learning=True,
    teacher_evidence='promoted',root_offset=28000000,seed_offset=28000000,
    scope='Matched next actors from fresh qualified independent/guided teachers; no repeated-loop or efficiency claim')
FEEDBACK_LEARNING_INTERFACE=deepcopy(FEEDBACK_LEARNING)
FEEDBACK_LEARNING_INTERFACE.update(schema='outcome-coupled-fresh-feedback-interface-v13',interface_only=True,
    root_offset=30000000,seed_offset=30000000,
    max_controls=50000,max_verification_controls=50000,
    max_neural_rows=200000,max_verification_neural_rows=200000,
    max_backward_calls=256,max_verification_backward_calls=256,
    scope='Two-sweep promoted-teacher/current-parent feedback interface only; no next-policy benefit claim')
FEEDBACK_LEARNING_INTERFACE['credit']['max_iterations']=2
HYBRID_FEEDBACK=deepcopy(FEEDBACK_LEARNING)
HYBRID_FEEDBACK.update(schema='outcome-coupled-hybrid-feedback-learning-v14',hybrid_credit=True,
    root_offset=32000000,seed_offset=32000000,
    scope='Matched fresh-teacher learning with explicit smooth-credit/discrete-jump qualification; no repeated-loop or efficiency claim')
HYBRID_FEEDBACK['credit']['hybrid_arithmetic_atol']=1e-9
HYBRID_FEEDBACK_INTERFACE=deepcopy(HYBRID_FEEDBACK)
HYBRID_FEEDBACK_INTERFACE.update(schema='outcome-coupled-hybrid-feedback-interface-v14',interface_only=True,
    root_offset=34000000,seed_offset=34000000,
    max_controls=50000,max_verification_controls=50000,
    max_neural_rows=200000,max_verification_neural_rows=200000,
    max_backward_calls=256,max_verification_backward_calls=256,
    scope='Two-sweep hybrid-credit interface; original task/capture gates remain; no next-policy benefit claim')
HYBRID_FEEDBACK_INTERFACE['credit']['max_iterations']=2
EVENT_REPLAY=deepcopy(HYBRID_FEEDBACK)
EVENT_REPLAY.update(schema='outcome-coupled-event-replay-learning-v17',event_aligned_replay=True,
    root_offset=38000000,seed_offset=38000000,
    scope='Matched complete-outcome learning at recorded preparation-event states; no action labels or inherited search authority')
EVENT_REPLAY_INTERFACE=deepcopy(EVENT_REPLAY)
EVENT_REPLAY_INTERFACE.update(schema='outcome-coupled-event-replay-interface-v17',interface_only=True,
    root_offset=40000000,seed_offset=40000000,
    max_controls=50000,max_verification_controls=50000,
    max_neural_rows=200000,max_verification_neural_rows=200000,
    max_backward_calls=256,max_verification_backward_calls=256,
    scope='Two-sweep event-state replay interface; no next-policy benefit claim')
EVENT_REPLAY_INTERFACE['credit']['max_iterations']=2
FEASIBILITY_CURRICULUM=deepcopy(EVENT_REPLAY)
FEASIBILITY_CURRICULUM.update(schema='outcome-coupled-feasibility-curriculum-v18',feasibility_curriculum=True,
    root_offset=42000000,seed_offset=42000000,
    curriculum=dict(radius=.01,feasibility_alpha=1.,cost_alphas=[.01,.003,.001,.0003],
        fractions=[1.,.5,.25,.125],minimum_feasibility_gain=1e-6,minimum_cost_gain=1e-8,
        solver=dict(max_iterations=200,solver_ftol=1e-12,feasibility_atol=1e-8,
            certificate_atol=1e-8,certificate_rtol=1e-8),
        max_solver_calls=256,max_parameter_constructions=1032),
    scope='Bounded ending-feasibility curriculum and own-return cost learning on event states; no inherited authority or action labels')
FEASIBILITY_CURRICULUM_INTERFACE=deepcopy(FEASIBILITY_CURRICULUM)
FEASIBILITY_CURRICULUM_INTERFACE.update(schema='outcome-coupled-feasibility-curriculum-interface-v18',interface_only=True,
    root_offset=44000000,seed_offset=44000000,
    max_controls=50000,max_verification_controls=50000,
    max_neural_rows=200000,max_verification_neural_rows=200000,
    max_backward_calls=256,max_verification_backward_calls=256,
    scope='Two accepted bounded curriculum updates and paid replay; no deployment or next-policy benefit claim')
FEASIBILITY_CURRICULUM_INTERFACE['credit']['max_iterations']=2
FEASIBILITY_CURRICULUM_INTERFACE['curriculum'].update(max_solver_calls=16,max_parameter_constructions=72)
NORMALIZED_CURRICULUM=deepcopy(FEASIBILITY_CURRICULUM)
NORMALIZED_CURRICULUM.update(schema='outcome-coupled-normalized-feasibility-v19',dual_feasibility_normalization=True,
    root_offset=46000000,seed_offset=46000000,
    scope='Same bounded curriculum with positive Phase I dual objective conditioning; task and original-unit outcome gates unchanged')
NORMALIZED_CURRICULUM_INTERFACE=deepcopy(FEASIBILITY_CURRICULUM_INTERFACE)
NORMALIZED_CURRICULUM_INTERFACE.update(schema='outcome-coupled-normalized-feasibility-interface-v19',
    dual_feasibility_normalization=True,root_offset=48000000,seed_offset=48000000,
    scope='Two accepted conditioned-curriculum updates and paid replay; no standalone benefit claim')
FEEDBACK_POLICY_SEARCH=deepcopy(DEEP_POLICY_SEARCH)
FEEDBACK_POLICY_SEARCH.update(schema='outcome-coupled-feedback-policy-search-v20',
    root_offset=50000000,seed_offset=50000000,
    scope='Fresh search by a qualification-selected feedback child versus independent memory and its incoming policy; no new actor learning')
SECOND_FEEDBACK=deepcopy(NORMALIZED_CURRICULUM)
SECOND_FEEDBACK.update(schema='outcome-coupled-second-feedback-learning-v21',
    feedback_source_schema=FEEDBACK_POLICY_SEARCH['schema'],measured_descent_acceptance=True,
    root_offset=52000000,seed_offset=52000000,
    scope='Second matched own-outcome learner round from freshly requalified search; no automatic efficiency or generalization claim')
SECOND_FEEDBACK_INTERFACE=deepcopy(NORMALIZED_CURRICULUM_INTERFACE)
SECOND_FEEDBACK_INTERFACE.update(schema='outcome-coupled-second-feedback-interface-v21',
    feedback_source_schema=FEEDBACK_POLICY_SEARCH['schema'],measured_descent_acceptance=True,
    root_offset=54000000,seed_offset=54000000,
    scope='Second-round two-update ownership and paid-replay interface; no standalone benefit claim')
MEASURED_DESCENT=deepcopy(NORMALIZED_CURRICULUM)
MEASURED_DESCENT.update(schema='outcome-coupled-measured-descent-v22',measured_descent_acceptance=True,
    root_offset=56000000,seed_offset=56000000,
    scope='Same conditioned learner with feasible proposals and measured executed descent; optimizer optimality remains diagnostic')
MEASURED_DESCENT_INTERFACE=deepcopy(NORMALIZED_CURRICULUM_INTERFACE)
MEASURED_DESCENT_INTERFACE.update(schema='outcome-coupled-measured-descent-interface-v22',
    measured_descent_acceptance=True,root_offset=58000000,seed_offset=58000000,
    scope='Two actual measured-descent updates and paid replay; no deployment or next-policy advantage claim')
FEASIBILITY_RAY=deepcopy(MEASURED_DESCENT)
FEASIBILITY_RAY.update(schema='outcome-coupled-feasibility-ray-v23',phase_one_gradient_ray=True,
    root_offset=60000000,seed_offset=60000000,
    scope='Bounded measured ending-gradient restoration, then unchanged own-outcome cost learning; no subproblem-optimality or inherited authority claim')
FEASIBILITY_RAY['curriculum']['max_feasibility_ray_calls']=64
FEASIBILITY_RAY_INTERFACE=deepcopy(MEASURED_DESCENT_INTERFACE)
FEASIBILITY_RAY_INTERFACE.update(schema='outcome-coupled-feasibility-ray-interface-v23',phase_one_gradient_ray=True,
    root_offset=62000000,seed_offset=62000000,
    scope='Two bounded ending-gradient updates and paid replay; no standalone advantage claim')
FEASIBILITY_RAY_INTERFACE['curriculum']['max_feasibility_ray_calls']=4
for protocol in (SECOND_FEEDBACK,SECOND_FEEDBACK_INTERFACE):
    protocol['phase_one_gradient_ray']=True
    protocol['curriculum']['max_feasibility_ray_calls']=4 if protocol.get('interface_only') else 64
QUALIFIED_CHILD_SEARCH=deepcopy(FEEDBACK_POLICY_SEARCH)
QUALIFIED_CHILD_SEARCH.update(schema='outcome-coupled-qualified-child-search-v25',
    feedback_parent_schema=HYBRID_FEEDBACK['schema'],root_offset=66000000,seed_offset=66000000,
    scope='Fresh zero-authority search from the v14 qualification-selected child, its actual incoming policy and unchanged RAW memory; v14 guided-data negative result is retained')
SECOND_HYBRID_FEEDBACK=deepcopy(HYBRID_FEEDBACK)
SECOND_HYBRID_FEEDBACK.update(schema='outcome-coupled-second-hybrid-feedback-learning-v26',
    feedback_source_schema=QUALIFIED_CHILD_SEARCH['schema'],requires_learning_interface=True,
    root_offset=68000000,seed_offset=68000000,
    scope='Second matched learning round using the unchanged v14 complete-outcome learner and newly qualified v25 teachers; no automatic beneficial-loop claim')
SECOND_HYBRID_FEEDBACK_INTERFACE=deepcopy(HYBRID_FEEDBACK_INTERFACE)
SECOND_HYBRID_FEEDBACK_INTERFACE.update(schema='outcome-coupled-second-hybrid-feedback-interface-v26',
    feedback_source_schema=QUALIFIED_CHILD_SEARCH['schema'],requires_learning_interface=True,
    root_offset=70000000,seed_offset=70000000,
    scope='Two actual accepted second-round hybrid updates and paid replay before the bounded full learner')
CONDITIONAL_FEEDBACK=deepcopy(SECOND_HYBRID_FEEDBACK)
CONDITIONAL_FEEDBACK.update(schema='outcome-coupled-conditional-feedback-learning-v27',
    feedback_source_gate='qualified_teacher_packet',root_offset=72000000,seed_offset=72000000,
    scope='Conditional next-policy test from promoted complete teachers after a retained negative old-policy search comparison; unchanged matched learner, no repaired search or full-loop claim')
CONDITIONAL_FEEDBACK_INTERFACE=deepcopy(SECOND_HYBRID_FEEDBACK_INTERFACE)
CONDITIONAL_FEEDBACK_INTERFACE.update(schema='outcome-coupled-conditional-feedback-interface-v27',
    feedback_source_gate='qualified_teacher_packet',root_offset=74000000,seed_offset=74000000,
    scope='Two actual accepted updates and paid replay for the conditional teacher-packet follow-up; old-policy search comparison remains negative')
JOINT_FEEDBACK=deepcopy(FEASIBILITY_RAY)
JOINT_FEEDBACK.update(schema='outcome-coupled-joint-feedback-learning-v28',
    joint_feedback_constraints=True,fixed_feedback_contexts=True,event_aligned_replay=False,
    feedback_source_schema=QUALIFIED_CHILD_SEARCH['schema'],feedback_source_gate='qualified_teacher_packet',
    requires_learning_interface=True,root_offset=76000000,seed_offset=76000000,
    scope='Joint all-context ending and fixed ordinary-cost constraints on the same v27 teacher data; actual actor outcomes, no sequential repair or action labels')
JOINT_FEEDBACK['curriculum']['ordinary_anchor_atol']=1e-8
JOINT_FEEDBACK_INTERFACE=deepcopy(JOINT_FEEDBACK)
JOINT_FEEDBACK_INTERFACE.update(schema='outcome-coupled-joint-feedback-interface-v28',interface_only=True,
    root_offset=78000000,seed_offset=78000000,max_controls=50000,max_verification_controls=50000,
    max_neural_rows=200000,max_verification_neural_rows=200000,
    max_backward_calls=256,max_verification_backward_calls=256,
    scope='Two accepted anchored updates including joint cost improvement, with paid replay, before full learning')
JOINT_FEEDBACK_INTERFACE['credit']['max_iterations']=2
JOINT_FEEDBACK_INTERFACE['curriculum'].update(max_solver_calls=16,max_parameter_constructions=72,max_feasibility_ray_calls=4)
JOINT_CHILD_SEARCH=deepcopy(FEEDBACK_POLICY_SEARCH)
JOINT_CHILD_SEARCH.update(schema='outcome-coupled-joint-child-search-v29',
    feedback_parent_schema=JOINT_FEEDBACK['schema'],root_offset=80000000,seed_offset=80000000,
    scope='Fresh zero-trust search by a changed qualification-selected joint-feedback child versus independent RAW memory and its exact incoming policy')
CONTINUATION_PROTOCOLS=(CONTINUATION,ACQUISITION,ARCHIVE_TRANSFER,ARCHIVE_INTERFACE,
    DETERMINISTIC_CREDIT,ENDING_PROJECTION,PRESERVED_ENDING,FRESH_SEARCH,POINTWISE_TRANSFER,
    NONLINEAR_RESTORATION,RESTORED_POLICY_SEARCH,REPEATED_CREDIT,REPEATED_CREDIT_INTERFACE,DEEP_POLICY_SEARCH,
    FEEDBACK_LEARNING,FEEDBACK_LEARNING_INTERFACE,HYBRID_FEEDBACK,HYBRID_FEEDBACK_INTERFACE,
    EVENT_REPLAY,EVENT_REPLAY_INTERFACE,FEASIBILITY_CURRICULUM,FEASIBILITY_CURRICULUM_INTERFACE,
    NORMALIZED_CURRICULUM,NORMALIZED_CURRICULUM_INTERFACE,FEEDBACK_POLICY_SEARCH,
    SECOND_FEEDBACK,SECOND_FEEDBACK_INTERFACE,MEASURED_DESCENT,MEASURED_DESCENT_INTERFACE,
    FEASIBILITY_RAY,FEASIBILITY_RAY_INTERFACE,QUALIFIED_CHILD_SEARCH,
    SECOND_HYBRID_FEEDBACK,SECOND_HYBRID_FEEDBACK_INTERFACE,
    CONDITIONAL_FEEDBACK,CONDITIONAL_FEEDBACK_INTERFACE,JOINT_FEEDBACK,JOINT_FEEDBACK_INTERFACE,JOINT_CHILD_SEARCH)


def validate_continuation(config):
    value=deepcopy(config);ancestor=value.pop('ancestor',None);origin=value.pop('teacher_origin',None)
    reference=value.pop('one_step_reference',None)
    interface=value.pop('interface_admission',None)
    if value not in tuple(state_json(p) for p in CONTINUATION_PROTOCOLS) or not isinstance(ancestor,dict):
        raise ValueError('Only the declared finite round-boundary continuation is accepted')
    if value['mode']=='archive_transfer':
        if not isinstance(origin,dict) or set(origin)!={'manifest_sha256','ledger_sha256','config_sha256'}:
            raise ValueError('Pinned archived teacher origin required')
    elif origin is not None:raise ValueError('Unexpected archived teacher binding')
    if value.get('repeated_credit') and not value.get('feedback_learning'):
        if not isinstance(reference,dict) or reference.get('schema')!='qualification-selected-one-step-reference-v1':
            raise ValueError('Repeated learning requires the pinned one-step comparator')
    elif reference is not None:raise ValueError('Unexpected one-step comparator')
    required=(value.get('event_aligned_replay') or value.get('requires_learning_interface')) and not value.get('interface_only')
    if required and (not isinstance(interface,dict) or interface.get('schema')!='accepted-learning-interface-v1'
            or interface.get('accepted_updates',0)<2):
        raise ValueError('Full event replay requires a bound interface with two actual accepted updates')
    if not required and interface is not None:raise ValueError('Unexpected interface admission')
    return ancestor


def _check_learning_interface(config, docs):
    old=docs['config.json'];study=docs['study.json'];outcome=docs['result.json']
    curriculum=bool(config.get('feasibility_curriculum'))
    interface_pass=outcome.get('curriculum_interface_pass' if curriculum else 'feedback_interface_pass')
    credit_checks=study.get('initial_credit_checks',{})
    credit_pass=bool(credit_checks) and all(credit_checks.values()) if curriculum else study.get('credit_diagnostic_pass')
    counts={arm:sum(row['arm']==arm and row['accepted'] is True and row['selected_line'] is not None
        for row in study['iterations']) for arm in ('raw','guided')}
    if (not (old.get('event_aligned_replay') or old.get('requires_learning_interface')) or not old.get('interface_only')
            or old['credit']['max_iterations']!=2 or docs['audit.json']['passed'] is not True
            or outcome['producer_failed'] or interface_pass is not True
            or credit_pass is not True or max(counts.values())<2
            or counts!=study.get('accepted_updates',study.get('accepted_cost_steps'))
            or old['ancestor']!=config['ancestor'] or old['teacher_origin']!=config['teacher_origin']):
        raise ValueError('Interface admission requires paid replay and two actual accepted updates in an arm')
    if config.get('joint_feedback_constraints') and not any(row['accepted'] is True
            and row['phase']=='improvement' for row in study['iterations']):
        raise ValueError('Joint interface admission requires an actual joint cost improvement')
    for key in ('task','learning','short_search','long_search','acquired_times','event_aligned_replay',
                'feasibility_curriculum','phase_one_gradient_ray','dual_feasibility_normalization',
                'joint_feedback_constraints','fixed_feedback_contexts',
                'measured_descent_acceptance','feedback_source_schema','feedback_source_gate','requires_learning_interface',
                'hybrid_credit','teacher_evidence','qualification_every',
                'capture_min','minimum_child_gain'):
        if old.get(key)!=config.get(key):raise ValueError('Interface algorithm component changed: '+key)
    previous_credit=deepcopy(old['credit']);current_credit=deepcopy(config['credit'])
    previous_credit.pop('max_iterations');current_credit.pop('max_iterations')
    if previous_credit!=current_credit:raise ValueError('Interface credit method changed')
    for key in ('curriculum',):
        before=deepcopy(old.get(key,{}));after=deepcopy(config.get(key,{}))
        for cap in ('max_solver_calls','max_parameter_constructions','max_feasibility_ray_calls'):
            before.pop(cap,None);after.pop(cap,None)
        if before!=after:raise ValueError('Interface update method changed: '+key)
    if config['credit']['max_iterations']!=32:
        raise ValueError('Interface admits only the declared 32-attempt full protocol')
    return max(counts.values())


def admit_learning_interface(source,config):
    """Fail before a full-run directory or model query if its interface stalled."""
    source=Path(source).resolve();names=('config.json','study.json','audit.json','result.json','manifest.json','launch.json')
    docs={name:json.loads((source/name).read_text()) for name in names}
    if docs['manifest.json']['config_sha256']!=file_sha256(source/'config.json'):
        raise ValueError('Interface configuration binding changed')
    accepted=_check_learning_interface(config,docs)
    value=deepcopy(config);value['interface_admission']=dict(schema='accepted-learning-interface-v1',run=str(source),
        accepted_updates=accepted,files={name:file_sha256(source/name) for name in (*names,'producer/ledger.jsonl')},
        original_interface_costs_remain_charged=True)
    return value


def copy_learning_interface(out,config):
    import shutil
    admission=config.get('interface_admission')
    if admission is None:return
    source=Path(admission['run']);target=out/'inputs/learning_interface';target.mkdir()
    for name,digest in admission['files'].items():
        if file_sha256(source/name)!=digest:raise ValueError('Learning interface changed during freezing')
        dest=target/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,dest)
    verify_learning_interface(out,config)


def verify_learning_interface(out,config):
    admission=config.get('interface_admission')
    if admission is None:
        if (config.get('event_aligned_replay') or config.get('requires_learning_interface')) and not config.get('interface_only'):
            raise ValueError('Full event replay lacks its required accepted-update interface')
        return
    target=out/'inputs/learning_interface'
    for name,digest in admission['files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or file_sha256(target/name)!=digest:
            raise ValueError('Frozen learning-interface binding changed')
    docs={name:json.loads((target/name).read_text()) for name in admission['files'] if name.endswith('.json')}
    if (docs['manifest.json']['config_sha256']!=file_sha256(target/'config.json')
            or _check_learning_interface(config,docs)!=admission['accepted_updates']):
        raise ValueError('Frozen learning-interface admission changed')


def qualified_feedback_acquisition(study,outcome,config):
    """Admit recorded teachers without relabeling a failed search comparison.

    The origin's paid audit, commit and full teacher-packet reconstruction are
    mandatory separately. Held outcomes can admit the experiment; they never
    supply its training contexts or actor-return targets.
    """
    if (config.get('schema')!='outcome-coupled-qualified-child-search-v25'
            or config.get('capture_min')!=.9 or config.get('minimum_control_gain')!=.001
            or study.get('search_feedback_pass') is not False or outcome.get('search_feedback_pass') is not False
            or len(study.get('rounds',[]))!=1):return False
    row=study['rounds'][0];control=row.get('control_evaluation',{});prior=row.get('previous_actor_counterfactual',{})
    if (row.get('round')!=0 or row.get('trust',{}).get('promoted') is not True
            or row.get('teaching_contexts')!=4 or row.get('data_identical') is not False
            or type(row.get('changed_controls')) is not int or row['changed_controls']<=0
            or row.get('promoted_then_revoked_exact') is not True
            or control.get('improves') is not True or prior.get('current_improves_previous') is not False):return False
    try:
        a,b=control['raw'],control['guided']
        for panel in (a,b):
            costs=np.asarray(panel['costs'],dtype=np.float64);flags=panel['capture_flags']
            if (panel['episodes']!=16 or costs.shape!=(16,) or len(flags)!=16
                    or not np.isfinite(costs).all() or any(type(flag) is not bool for flag in flags)
                    or panel['captures']!=sum(flags) or abs(panel['mean_cost']-float(costs.mean()))>1e-12):return False
        return bool(b['captures']>=a['captures'] and b['captures']>=np.ceil(config['capture_min']*16)
            and a['mean_cost']-b['mean_cost']>config['minimum_control_gain'])
    except (KeyError,TypeError,ValueError):return False


def continuation_config(source,protocol=None):
    from .execution import diagnostic_config
    source=Path(source).resolve();protocol=CONTINUATION if protocol is None else protocol
    if state_json(protocol) not in tuple(state_json(p) for p in CONTINUATION_PROTOCOLS):
        raise ValueError('Unknown continuation protocol')
    previous=json.loads((source/'config.json').read_text())
    for key in ('task','learning','short_search','long_search','full_memory_candidates','planner_seed',
                'agreement','acquired_times','ordinary_replicates','acquired_replicates'):
        if previous[key]!=state_json(protocol[key]):raise ValueError('Continuation component changed: '+key)
    if not json.loads((source/'audit.json').read_text())['passed']:raise ValueError('Parent raw replay must pass')
    if json.loads((source/'result.json').read_text())['producer_failed']:raise ValueError('Parent producer was incomplete')
    config=deepcopy(protocol);config['ancestor']=diagnostic_config(source)['ancestor']
    hybrid_child=config['ancestor'].get('parent_training_schema')==HYBRID_FEEDBACK['schema']
    joint_child=config['ancestor'].get('parent_training_schema')==JOINT_FEEDBACK['schema']
    if hybrid_child:
        if (protocol['schema']!=QUALIFIED_CHILD_SEARCH['schema']
                or protocol.get('feedback_parent_schema')!=HYBRID_FEEDBACK['schema']):
            raise ValueError('The qualified v14 child requires its explicit v25 source profile')
    elif joint_child:
        if (protocol['schema']!=JOINT_CHILD_SEARCH['schema']
                or protocol.get('feedback_parent_schema')!=JOINT_FEEDBACK['schema']):
            raise ValueError('The qualified joint child requires its explicit v29 source profile')
    elif protocol.get('feedback_parent_schema') is not None:
        raise ValueError('Declared feedback-parent profile does not match the selected source')
    if protocol['mode']=='archive_transfer':
        config['teacher_origin']=dict(manifest_sha256=file_sha256(source/'manifest.json'),
            ledger_sha256=file_sha256(source/'producer/ledger.jsonl'),config_sha256=file_sha256(source/'config.json'))
    if protocol.get('feedback_learning'):
        result=json.loads((source/'result.json').read_text())
        admitted=(qualified_feedback_acquisition(json.loads((source/'study.json').read_text()),result,previous)
            if protocol.get('feedback_source_gate')=='qualified_teacher_packet' else result.get('search_feedback_pass') is True)
        if (previous.get('schema')!=protocol.get('feedback_source_schema','outcome-coupled-depth-policy-search-v12')
                or previous.get('interface_only') or not admitted
                or not (source/'round_0_commit.json').is_file()):
            raise ValueError('Fresh feedback learning requires its declared closed search admission and committed RAW memory')
    if protocol.get('repeated_credit') and not protocol.get('feedback_learning'):
        from .execution import bind_one_step_reference
        reference=Path(__file__).resolve().parents[3]/protocol['one_step_reference_run']
        config['one_step_reference']=bind_one_step_reference(reference,source,config)
    return state_json(config)


def root_panel(config,base,count):
    start=base+config.get('root_offset',0)
    return list(range(start,start+count))


def write(path,value):
    with Path(path).open('x') as stream:json.dump(state_json(value),stream,indent=2,sort_keys=True);stream.write('\n')


def load_config(path):
    config=json.loads(Path(path).read_text())
    if config not in tuple(state_json(p) for p in (DEVELOP,INTERFACE,*CONTINUATION_PROTOCOLS)):
        raise ValueError('Only declared finite neural protocols are accepted')
    return config


def configure_archive(archive,config,verification=False):
    prefix='max_verification_' if verification else 'max_'
    archive.max_neural_rows=config[prefix+'neural_rows']
    archive.max_optimizer_calls=config[prefix+'optimizer_calls']
    archive.max_backward_calls=config[prefix+'backward_calls']
    archive.compact_transitions=True
    archive.append('transition_evidence_layout',layout='complete_query_npz_and_durable_step_digests',
        failed_queries='Input/checkpoint/RNG retained; known completed slots journaled; unresolved remainder never called complete')


def add_memory(memory,episode,task,label):
    observations=task.observation(EndingSnapshot(episode['states'],task.task_hash))
    memory.add(observations,episode['actions'],task.config.T,label,family=0,stride=5)


def discover(executor,config):
    memory=TrajectoryMemory();episodes=[]
    for root in config['teacher_roots']:
        for stream in config['teacher_streams']:
            temporary=dict(config,short_search=config['long_search'],planner_seed=config['planner_seed']+stream*10000)
            tag=f'discovery_{root}_{stream}'
            episode=controller_episode(executor,root,temporary,memory,tag)
            episodes.append(episode);add_memory(memory,episode,executor.task,tag)
    executor.archive.append('discovery_complete',episodes=len(episodes),captures=sum(e['capture'] for e in episodes),
        memory_fragments=len(memory.fragments),learned_influence=0.)
    return episodes,memory


def successful_sources(episodes,roots):
    selected=[]
    for root in roots:
        success=[e for e in episodes if e['root']==root and e['capture'] and e['complete']]
        if not success:return []
        selected.append(min(success,key=lambda e:float(e['components'].sum())))
    return selected


def training_contexts(episodes,task,config):
    contexts=[]
    ordinary=config['ordinary_replicates'];acquired=config['acquired_replicates']
    if config.get('acquisition_strategy') in ('qualified_trial_pairs','archived_qualified_pairs'):
        if len(episodes) not in (1,2,4):raise ValueError('Declared 1/2/4 teaching contexts required')
        ordinary=32//len(episodes);acquired=16//len(episodes)
    for episode in episodes:
        contexts.append(context_from_episode(episode,task,replicates=ordinary))
        for t in config['acquired_times']:
            contexts.append(context_from_episode(episode,task,t,acquired))
    if sum(c.replicates for c in contexts)!=64:raise ValueError('Protocol requires 32 ordinary + 32 acquired trajectories')
    return contexts


def actor_panel(executor,actor,roots,tag,initial_states=None,*,inference_contract='native'):
    initial=np.stack([executor.task.reset(r).state for r in roots]) if initial_states is None else np.asarray(initial_states)
    policy=NeuralPolicy(actor,executor.archive,executor.task,inference_contract=inference_contract)
    result=executor.feedback(initial,policy,kind=tag,mode='executed',deterministic=True)
    outcomes=[]
    for i,state in enumerate(initial):
        snapshot=EndingSnapshot(state,executor.task.task_hash)
        cost=float(result.components[i].sum());capture=bool(result.states[i,-1,1]<=executor.task.dynamics.ending_y)
        reference=executor.archive.append('complete_actor_outcome',query_id=result.query_id,row=i,
            actor_hash=policy.digest,context_key=snapshot.context_id,remaining=executor.task.config.T-snapshot.elapsed,
            cost=cost,capture=capture,complete=True,deterministic=True)
        outcomes.append(Outcome(snapshot.context_id,cost,capture,True,policy.digest,policy.digest,0,
            executor.task.config.T-snapshot.elapsed,executor.task.task_hash,(reference,),'actor_suffix'))
    return outcomes,result


def summary(outcomes):
    return dict(episodes=len(outcomes),captures=sum(o.capture for o in outcomes),
        mean_cost=float(np.mean([o.cost for o in outcomes])),costs=[o.cost for o in outcomes],
        capture_flags=[o.capture for o in outcomes])


def improves(base,candidate,minimum=1e-6):
    return sum(o.capture for o in candidate)>=sum(o.capture for o in base) and (
        np.mean([o.cost for o in base])-np.mean([o.cost for o in candidate])>minimum)


def restore_learner(executor,checkpoint,config):
    state=load_pt(executor.archive.path,checkpoint)
    started=time.perf_counter()
    actor=FamilyGaussianActor.from_checkpoint(state['actor'])
    executor.archive.append('policy_restore',policy_hash=policy_hash(actor),checkpoint=checkpoint,
        restore_wall_seconds=time.perf_counter()-started,neural_forward_rows=0)
    learner=OutcomeLearner(actor,executor.archive,executor.task,LearningConfig(**config['learning']),state['optimizer'])
    learner.update_index=state['update_index']
    torch.set_rng_state(state['torch_rng'])
    return learner


def train(executor,learner,sources,config,*,updates,seed_base,qual_base,tag):
    contexts=training_contexts(sources,executor.task,config)
    best=learner.checkpoint(f'checkpoints/{tag}_initial.pt');rows=[];probe=None
    for update in range(updates):
        groups=collect_owned(executor,learner.actor,contexts,seed=seed_base+update*100,tag=f'{tag}_{update:03d}')
        if update==0 and tag=='initial':probe=learner.gradient_probe(groups,f'{tag}_probe')
        row=learner.update(groups,f'{tag}_{update:03d}')
        if (update+1)%config['qualification_every']==0 or update+1==updates:
            roots=list(range(qual_base+update*100,qual_base+update*100+config['ordinary_qualification_roots']))
            retained=restore_learner(executor,best,config)
            baseline,_=actor_panel(executor,retained.actor,roots,f'{tag}_retained_{update:03d}')
            changed,_=actor_panel(executor,learner.actor,roots,f'{tag}_candidate_{update:03d}')
            accepted=(sum(o.capture for o in changed)>=np.ceil(config['capture_min']*len(changed))
                and improves(baseline,changed))
            if accepted:best=row['after']
            executor.archive.append('actor_promotion',tag=tag,update=update+1,accepted=bool(accepted),
                baseline=summary(baseline),candidate=summary(changed),retained_checkpoint=best,
                partial_acquired_success_cannot_promote=True)
            row['qualification']=summary(changed);row['promoted']=bool(accepted)
        rows.append(row)
    return best,rows,probe,groups


def control_panel(executor,config,memory,roots,tag,policy=None,trust=None,trial_weight=None):
    episodes=[controller_episode(executor,root,config,memory,f'{tag}_{i:03d}',
        policy=policy,trust=trust,trial_weight=trial_weight) for i,root in enumerate(roots)]
    outcomes=[]
    for episode in episodes:
        outcomes.append(Outcome(episode['context_key'],float(episode['components'].sum()),episode['capture'],
            True,episode['controller_hash'],None if policy is None else policy.digest,0,executor.task.config.T,
            executor.task.task_hash,tuple(episode['query_ids']),episode['semantics']))
    return outcomes,episodes


def previous_actor_search(executor,out,config,memory,sources,round_index,roots,current_actor,current,weight,
                          *,previous_actor=None,previous_checkpoint=None):
    """One paid old-policy counterfactual; its authority must qualify separately."""
    if previous_actor is None:
        path=out/'inputs/previous_parent.pt'
        state=torch.load(path,map_location='cpu',weights_only=True)
        actor=FamilyGaussianActor.from_checkpoint(state['actor'])
        previous_reference=dict(file='inputs/previous_parent.pt',sha256=file_sha256(path))
    else:
        if previous_checkpoint is None:raise ValueError('An actual incoming-policy checkpoint must bind the counterfactual')
        actor=previous_actor;previous_reference=previous_checkpoint
        if load_pt(executor.archive.path,previous_checkpoint)['actor']['policy_hash']!=policy_hash(actor):
            raise ValueError('The old-policy counterfactual differs from its current-round incoming checkpoint')
    eligibility,record=handoff_eligibility(executor,actor,sources,config,f'previous_actor_{round_index}')
    eligible=record.state=='eligible'
    manager=CandidateTrustManager.from_eligible(eligibility,weight) if eligible else eligibility
    manager.inference_contract=config.get('policy_inference_contract','native')
    previous,_=control_panel(executor,config,memory,roots,f'round{round_index}_previous_eval',
        NeuralPolicy(actor,executor.archive,executor.task,
            inference_contract=config.get('policy_inference_contract','native')),manager,weight if eligible else None)
    result=dict(policy_hash=policy_hash(actor),eligibility_rate=record.handoff_success_rate,
        eligible=eligible,used_trial_weight=float(weight) if eligible else 0.,
        comparison=summary(previous),current_improves_previous=improves(previous,current,config['minimum_control_gain']))
    executor.archive.append('previous_actor_search_comparison',round=round_index,
        previous_parent=previous_reference,
        current_actor_hash=policy_hash(current_actor),roots=roots,result=result,
        semantics='Fresh old-actor eligibility; experimental trial only when eligible, otherwise executed zero-influence fallback',
        old_deployment_promotion=False)
    return result


def handoff_eligibility(executor,actor,sources,config,tag):
    contract=config.get('policy_inference_contract','native')
    policy=NeuralPolicy(actor,executor.archive,executor.task,inference_contract=contract);task=executor.task
    states=[]
    for t in config['handoff_times']:
        for episode in sources:
            for delta in config['handoff_perturbations']:
                state=episode['states'][t].copy();state[1]=np.clip(state[1]+delta,-1,1);states.append(state)
    unique={EndingSnapshot(s,task.task_hash).context_id:s for s in states}
    states=list(unique.values());anchors=task.observation(EndingSnapshot(np.stack(states),task.task_hash))
    manager=TrustManager();manager.set_version(policy.digest,task.task_hash)
    manager.inference_contract=contract
    times=config['handoff_times'];lo,hi=min(times),max(times)
    record=manager.register(actor_hash=policy.digest,actor_controller_hash=policy.digest,task_hash=task.task_hash,
        family_id=0,total_steps=task.config.T,time_interval=(lo,hi),
        remaining_interval=(task.config.T-hi,task.config.T-lo),support_anchors=anchors,
        observation_scale=np.ones(8),support_radius=config['support_radius'],anchor_context_ids=tuple(unique))
    outcomes=[]
    for t in times:
        batch=np.stack([s for s in states if int(s[0])==t])
        panel,_=actor_panel(executor,actor,[],f'{tag}_handoff_t{t}',batch,
            inference_contract=contract);outcomes.extend(panel)
    qualified=manager.qualify(record.record_id,outcomes)
    executor.archive.append('handoff_qualification',tag=tag,record=state_json(qualified),
        empirical_only=True,controller_influence=0.,
        **({} if contract=='native' else dict(inference_contract=contract)))
    return manager,qualified


def qualify_guidance(executor,actor,sources,memory,config,round_index,acquisition_pairs=None):
    contract=config.get('policy_inference_contract','native')
    tag=f'round{round_index}';policy=NeuralPolicy(actor,executor.archive,executor.task,inference_contract=contract)
    eligibility,record=handoff_eligibility(executor,actor,sources,config,tag)
    if record.state!='eligible':return eligibility,dict(promoted=False,reason='actor_suffix_failure',handoff_rate=record.handoff_success_rate)
    roots=root_panel(config,100000+round_index*10000,config['control_trial_roots'])
    raw,raw_episodes=control_panel(executor,config,memory,roots,f'{tag}_trial_raw')
    for weight in config['agreement_ladder']:
        manager=CandidateTrustManager.from_eligible(eligibility,weight)
        manager.inference_contract=contract
        guided,episodes=control_panel(executor,config,memory,roots,f'{tag}_trial_w{weight}',policy,manager,weight)
        if acquisition_pairs is not None:
            acquisition_pairs.extend(dict(raw=a,guided=b,phase='trial',weight=weight,
                raw_tag=f'{tag}_trial_raw_{i:03d}',guided_tag=f'{tag}_trial_w{weight}_{i:03d}')
                for i,(a,b) in enumerate(zip(raw_episodes,episodes)))
        changed=sum(e['changed_controls'] for e in episodes)
        passes=improves(raw,guided,config['minimum_control_gain']) and changed>0
        executor.archive.append('guidance_trial',round=round_index,weight=weight,
            baseline=summary(raw),guided=summary(guided),changed_controls=changed,passed=bool(passes),
            **({} if contract=='native' else dict(inference_contract=contract)))
        if not passes:
            manager.revoke(record.record_id,'failed declared executed-control trial');continue
        # Fresh confirmation uses a fresh eligible packet, not the same trials
        # as both selection and proof of deployment performance.
        confirmed=CandidateTrustManager.from_eligible(eligibility,weight)
        confirmed.inference_contract=contract
        fresh=root_panel(config,120000+round_index*10000,config['control_confirmation_roots'])
        base,base_episodes=control_panel(executor,config,memory,fresh,f'{tag}_confirm_raw')
        guide,guide_episodes=control_panel(executor,config,memory,fresh,f'{tag}_confirm_guide',policy,confirmed,weight)
        if acquisition_pairs is not None:
            acquisition_pairs.extend(dict(raw=a,guided=b,phase='confirmation',weight=weight,
                raw_tag=f'{tag}_confirm_raw_{i:03d}',guided_tag=f'{tag}_confirm_guide_{i:03d}')
                for i,(a,b) in enumerate(zip(base_episodes,guide_episodes)))
        if (improves(base,guide,config['minimum_control_gain']) and sum(e['changed_controls'] for e in guide_episodes)>0):
            promoted=confirmed.promote(record.record_id,weight,base,guide,
                baseline_controller_hash=base[0].controller_hash,guided_controller_hash=guide[0].controller_hash,
                guided_semantics=guide[0].semantics)
            executor.archive.append('guidance_promotion',round=round_index,record=state_json(promoted),
                baseline=summary(base),guided=summary(guide),
                **({} if contract=='native' else dict(inference_contract=contract)))
            return confirmed,dict(promoted=promoted.state=='promoted',weight=weight,
                baseline=summary(base),guided=summary(guide))
        confirmed.revoke(record.record_id,'fresh executed confirmation failed')
        return confirmed,dict(promoted=False,reason='fresh_confirmation_failure',weight=weight,
            baseline=summary(base),guided=summary(guide))
    eligibility.revoke(record.record_id,'all finite settings failed executed-control trial')
    return eligibility,dict(promoted=False,reason='no_control_improvement')


def acquire_qualified_teachers(executor,pairs,trust_result,config,round_index):
    """Reuse complete executions, preserving their original query identities."""
    candidates=[p for p in pairs if trust_result['promoted'] and p['weight']==trust_result['weight']]
    selected,decisions=select_teaching_pairs(candidates,minimum_gain=config['teaching_minimum_gain'],
        state_atol=config['teaching_state_atol'])
    executor.archive.append('teaching_selection',round=round_index,
        controller_promoted=trust_result['promoted'],selected_indices=selected,decisions=decisions,
        candidates=[{k:p[k] for k in ('phase','weight','raw_tag','guided_tag')} for p in candidates],
        training_objective='unchanged actor-owned outcome envelope; each arm keeps own acquired states',
        conditional_on_measured_teacher_improvement=True,new_executed_controls=0)
    raw=[];guided=[]
    for i,index in enumerate(selected):
        pair=candidates[index]
        for arm,target in (('raw',raw),('guided',guided)):
            source=pair[arm+'_tag'];alias=f'round{round_index}_{arm}_data_{i:03d}'
            executor.archive.append('teaching_source_selected',round=round_index,arm=arm,alias_tag=alias,
                source_tag=source,context=pair[arm]['context_key'],query_ids=pair[arm]['query_ids'],
                new_executed_controls=0,source_controller_hash=pair[arm]['controller_hash'])
            target.append(pair[arm])
    return raw,guided


def matched_children(executor,parent,parent_checkpoint,raw_data,guided_data,config,round_index,curves):
    """Shared unchanged learner and held-out comparison for both data arms."""
    children={};child_qual={};child_eval={}
    for label,data in [('raw',raw_data),('guided',guided_data)]:
        child=restore_learner(executor,parent_checkpoint,config)
        checkpoint,rows,_,_=train(executor,child,data,config,updates=config['child_updates'],
            seed_base=500000+round_index*100000+config.get('seed_offset',0),
            qual_base=600000+round_index*100000+config.get('root_offset',0),
            tag=f'round{round_index}_{label}')
        curves.extend(rows);children[label]=checkpoint
        candidate=restore_learner(executor,checkpoint,config)
        fresh=root_panel(config,710000+round_index*10000,config['ordinary_qualification_roots'])
        child_qual[label],_=actor_panel(executor,candidate.actor,fresh,f'round{round_index}_{label}_child_qual')
        evaluation=root_panel(config,740000+round_index*10000,config['standalone_roots'])
        child_eval[label],_=actor_panel(executor,candidate.actor,evaluation,f'round{round_index}_{label}_child_eval')
    fresh=root_panel(config,710000+round_index*10000,config['ordinary_qualification_roots'])
    parent_qual,_=actor_panel(executor,parent.actor,fresh,f'round{round_index}_parent_qual')
    chosen='parent';chosen_outcome=parent_qual
    for label in ('raw','guided'):
        if improves(chosen_outcome,child_qual[label],config['minimum_child_gain']):
            chosen=label;chosen_outcome=child_qual[label]
    parent_eval,_=actor_panel(executor,parent.actor,evaluation,f'round{round_index}_parent_eval')
    row=dict(child_qualification={k:summary(v) for k,v in child_qual.items()},
        child_evaluation={k:summary(v) for k,v in child_eval.items()},parent_evaluation=summary(parent_eval),
        guided_child_improves_raw=improves(child_eval['raw'],child_eval['guided'],config['minimum_child_gain']),
        guided_child_improves_parent=improves(parent_eval,child_eval['guided'],config['minimum_child_gain']),
        selected_next_parent=chosen,
        child_hashes={k:load_pt(executor.archive.path,v)['actor']['policy_hash'] for k,v in children.items()})
    return row,children,chosen


def produce_temporal_cycle(executor,incoming,teachers,windows,memory,config):
    """Matched temporal teaching, then two actual search/teacher/actor rounds.

    The temporal profile uses four ordinary teacher starts and prefix costs;
    the complete-outcome profile acquires eight stratified ordinary starts
    before learning. Neither uses action/state regression. Every
    changed actor owns its complete returns. The caller owns the sole archive,
    finite executor budgets, input bindings and final work summary. This
    function returns a JSON-safe result and the actual retained full state.
    """
    from .teaching import temporal_step,panel,outcome_step,outcome_scales,outcome_coupling_checks
    archive=executor.archive;task=executor.task;arms=('raw','guided')
    complete=bool(config.get('complete_outcome_teaching'))
    global_learning=bool(config.get('global_outcome_teaching'))
    unit_step=config['schema'] in ('complete-outcome-unit-step-cycle-v35','global-outcome-reference-cycle-v38','global-outcome-active-endpoints-cycle-v40','global-outcome-group-endpoints-cycle-v41','global-outcome-realized-admission-cycle-v42','global-outcome-interior-ending-cycle-v43')
    if complete and (config['schema'] not in ('complete-outcome-coupled-cycle-v34',
            'complete-outcome-unit-step-cycle-v35','global-outcome-reference-cycle-v38','global-outcome-active-endpoints-cycle-v40','global-outcome-group-endpoints-cycle-v41','global-outcome-realized-admission-cycle-v42','global-outcome-interior-ending-cycle-v43')
            or config.get('unit_proposal_gradient',False) is not unit_step
            or not config.get('bootstrap_search') or config.get('bootstrap_round_index')!=100
            or config.get('teaching_contexts')!=8 or config.get('acquisition_seed_pool')!=512
            or config['updates']!=16):
        raise ValueError('The declared search-first complete-outcome cycle is required')
    count=config['teaching_contexts'] if complete else 4
    stepper=outcome_step if complete else temporal_step
    if global_learning:
        from .global_learning import global_step,coupling_checks as global_coupling_checks
        if (config['schema'] not in ('global-outcome-reference-cycle-v38','global-outcome-active-endpoints-cycle-v40','global-outcome-group-endpoints-cycle-v41','global-outcome-realized-admission-cycle-v42','global-outcome-interior-ending-cycle-v43') or config['ordinary_training_roots']!=32
                or config.get('unit_feasibility_metric') is not True or config.get('stage_reference_budget') is not True
                or config['acquired_training_clocks']!=[2,4,6,8]):
            raise ValueError('Declared global 32+32 actor rollout curriculum required')
        stepper=global_step
    if (not config.get('temporal_teaching') or config.get('policy_inference_contract')!='native_batch1_rows'
            or config['rounds']!=2 or not 1<=config['updates']<=32 or config['qualification_every']!=4):
        raise ValueError('The finite canonical temporal cycle contract is required')

    def better(base,candidate):
        return bool(candidate['captures']>=max(base['captures'],np.ceil(config['capture_min']*candidate['episodes']))
            and base['mean_cost']-candidate['mean_cost']>config['minimum_child_gain'])

    def stage(state,data,bounds,index):
        parent=FamilyGaussianActor.from_checkpoint(state['actor']);owner=policy_hash(parent)
        if (len(bounds)!=count or any(len(data[a])!=count for a in arms)
                or any(e['guiding_policy_hash']!=owner for e in data['guided'])
                or any(e['guiding_policy_hash'] is not None for e in data['raw'])
                or any(not all(e['complete'] and e['capture'] for e in (a,b))
                    or not np.array_equal(a['states'][0],b['states'][0]) for a,b in zip(data['raw'],data['guided']))):
            raise ValueError(('Eight' if complete else 'Four')+' successful paired teachers must bind this actual incoming actor')
        if complete:
            scales=outcome_scales(executor,parent,data,f'complete_stage{index}')
        else:
            scales=dict(complete=float(np.mean([e['components'].sum() for a in arms for e in data[a]])),
                window=float(np.mean([e['components'][lo:hi,:2].sum() for a in arms for e,(lo,hi) in zip(data[a],bounds)])))
            if not np.isfinite(list(scales.values())).all() or min(scales.values())<=0.:
                raise ValueError('Shared positive measured original-cost normalization is required')
        actors={a:FamilyGaussianActor.from_checkpoint(state['actor']) for a in arms}
        incumbent={a:deepcopy(state) for a in arms};stopped={a:False for a in arms}
        accepted={a:0 for a in arms};last={a:0 for a in arms};tested={a:set() for a in arms}
        records=[];history=[];first={};base=2000000+index*1000000
        archive.append('temporal_stage_requested',stage=index,incoming_actor=owner,windows=bounds,scales=scales,
            teacher_queries={a:[e['query_ids'] for e in data[a]] for a in arms},
            training='64 current-actor executions:32 fresh ordinary+32 actual teacher/clock slots; eight auxiliary exact-context complete-outcome bounds'
                if global_learning else 'Eight stratified ordinary complete executions per arm; complete-outcome witnesses'
                if complete else 'Four same-context ordinary complete executions per arm; prefix outcomes only',planning_influence=0.)

        def full(actor,arm,qualification_iteration=0):
            value=deepcopy(state);value['actor']=actor.policy_state()
            key='complete_outcome_teaching' if complete else 'temporal_teaching'
            value[key]=dict(stage=index,arm=arm,training_iteration=last[arm],
                qualification_iteration=qualification_iteration,accepted_updates=accepted[arm],windows=bounds,
                scales=scales,planning_influence=0.,inference_contract='native_batch1_rows',
                original_optimizer_history_untouched=True,optimizer_steps=0,
                previous=deepcopy(state.get(key)))
            return value

        for iteration in range(1,config['updates']+1):
            for arm in arms:
                if stopped[arm]:continue
                tag=f'temporal_stage{index}_{arm}_iter{iteration:03d}'
                before=save_pt(archive,f'checkpoints/{tag}_before.pt',full(actors[arm],arm))
                actors[arm],row=stepper(executor,actors[arm],data[arm],bounds,scales,config,tag,
                    check_credit=iteration==1,**(dict(stage=index,iteration=iteration,reference_actor=parent) if global_learning else {}))
                if row['accepted']:accepted[arm]+=1;last[arm]=iteration
                stopped[arm]=not row['accepted']
                after=save_pt(archive,f'checkpoints/{tag}_after.pt',full(actors[arm],arm))
                row.update(stage=index,arm=arm,iteration=iteration,before_checkpoint=before,after_checkpoint=after)
                records.append(row)
                if iteration==1:first[arm]=row
            if iteration%config['qualification_every']==0 or iteration==config['updates'] or all(stopped.values()):
                roots=root_panel(config,base+iteration*1000,config['ordinary_qualification_roots'])
                for arm in arms:
                    current=policy_hash(actors[arm]);attempt=current not in tested[arm];tested[arm].add(current)
                    old=FamilyGaussianActor.from_checkpoint(incumbent[arm]['actor'])
                    baseline=panel(executor,old,roots,f'temporal_stage{index}_{arm}_qual{iteration:03d}_incumbent')
                    candidate=panel(executor,actors[arm],roots,f'temporal_stage{index}_{arm}_qual{iteration:03d}_candidate')
                    promoted=bool(attempt and better(baseline,candidate))
                    if promoted:incumbent[arm]=full(actors[arm],arm,iteration)
                    row=dict(stage=index,arm=arm,iteration=iteration,roots=roots,promotion_attempt=attempt,
                        baseline=baseline,candidate=candidate,promoted=promoted,
                        retained_actor=incumbent[arm]['actor']['policy_hash'])
                    history.append(row);archive.append('temporal_stage_qualification',**row)
            if all(stopped.values()):break
        if global_learning:
            checks,contrast=global_coupling_checks(archive,first,scales,config)
        elif complete:
            checks,contrast=outcome_coupling_checks(archive,first,scales,config)
        else:
            owned={}
            for arm in arms:
                ref=first[arm]['proposal_inputs']
                if file_sha256(archive.path/ref['file'])!=ref['sha256']:raise ValueError('Owned gradient packet changed')
                with np.load(archive.path/ref['file'],allow_pickle=False) as packet:
                    owned[arm]={key:packet[key].copy() for key in packet.files}
            raw,guide=owned['raw'],owned['guided']
            exchanged=raw['original_cost_gradient']/scales['complete']+config['beta']*(
                2*np.maximum(raw['prefix_costs']-guide['prefix_budgets'],0.)[:,None]*raw['prefix_gradients']/scales['window']**2).mean(axis=0)
            checks=dict(same_owned_cost_gradient=np.array_equal(raw['original_cost_gradient'],guide['original_cost_gradient']),
                same_owned_prefix_costs=np.array_equal(raw['prefix_costs'],guide['prefix_costs']),
                same_owned_prefix_gradients=np.array_equal(raw['prefix_gradients'],guide['prefix_gradients']),
                budget_exchange_reproduces_guided_gradient=np.array_equal(exchanged,guide['gradient']),
                beta_zero_removes_teacher_dependence=np.array_equal(raw['original_cost_gradient']/scales['complete'],guide['original_cost_gradient']/scales['complete']),
                window_derivatives_pass=all(first[a]['credit_check']['passed'] for a in arms))
            contrast=float(np.linalg.norm(exchanged-raw['gradient']))
            archive.append('temporal_stage_coupling_controls',stage=index,checks=checks,new_queries=0,
                gradient_contrast=contrast,active_guided_prefixes=int(np.sum(guide['prefix_costs']>guide['prefix_budgets'])))
        candidates=dict(parent=parent,**{a:FamilyGaussianActor.from_checkpoint(incumbent[a]['actor']) for a in arms})
        qroots=root_panel(config,base+100000,config['ordinary_qualification_roots'])
        eroots=root_panel(config,base+200000,config['standalone_roots'])
        qualification={role:panel(executor,actor,qroots,f'temporal_stage{index}_{role}_final_qual') for role,actor in candidates.items()}
        evaluation={role:panel(executor,actor,eroots,f'temporal_stage{index}_{role}_final_eval') for role,actor in candidates.items()}
        chosen='parent'
        for arm in arms:
            if better(qualification[chosen],qualification[arm]):chosen=arm
        states=dict(parent=deepcopy(state),**incumbent)
        checkpoints={role:save_pt(archive,f'checkpoints/temporal_stage{index}_{role}_qualified.pt',value) for role,value in states.items()}
        frontier={a:save_pt(archive,f'checkpoints/temporal_stage{index}_{a}_frontier.pt',full(actors[a],a)) for a in arms}
        science=bool(all(checks.values()) and chosen=='guided'
            and all(better(evaluation[a],evaluation['guided']) for a in ('parent','raw')))
        result=dict(stage=index,incoming_actor=owner,windows=bounds,scales=scales,steps=records,
            qualification_history=history,qualification=qualification,evaluation=evaluation,accepted_updates=accepted,
            stopped=stopped,coupling_checks=checks,interface_pass=bool(all(checks.values())),
            gradient_contrast=contrast,
            selected_next_parent=chosen,selected_actor=states[chosen]['actor']['policy_hash'],
            qualified_checkpoints=checkpoints,frontier_checkpoints=frontier,
            guided_next_policy_advantage=science,planning_influence=0.,teacher_action_labels=False,teacher_state_targets=False)
        if config.get('ordinary_teacher_constraints'):
            active={arm:first[arm]['phase']=='improvement' for arm in arms}
            result.update(gradient_contrast_scope='mixed_F_diagnostic_only',actual_update_influence_claimed=False,
                teacher_constraints_active_by_arm=active,ordinary_objective_active_by_arm=active)
        archive.append('temporal_stage_complete',result=result)
        return result,states[chosen]

    def search_packet(actor,sources,index,old=None,old_checkpoint=None):
        """Acquire a fixed state-only roster; never rank roots by outcomes."""
        pairs=[];manager,trust=qualify_guidance(executor,actor,sources,memory,config,index,pairs)
        policy=NeuralPolicy(actor,archive,task,inference_contract=config['policy_inference_contract'])
        row=dict(round=index,bootstrap=old is None,incoming_actor=policy_hash(actor) if old is None else policy_hash(old),
            current_actor=policy_hash(actor),trust=trust,memory_before=identity(state_json(memory.fragments)),
            memory_fragments_before=len(memory.fragments),control_evaluation=None,
            previous_actor_counterfactual=None,promoted_then_revoked_exact=False,
            coverage_packet_passed=False,coverage_failure='guidance_not_promoted',acquisition=None)
        packet={a:[] for a in arms};bounds=[]
        if trust['promoted']:
            held=root_panel(config,800000+index*10000,config['control_evaluation_roots'])
            independent,_=control_panel(executor,config,memory,held,f'round{index}_raw_eval')
            controlled,controlled_episodes=control_panel(executor,config,memory,held,f'round{index}_guided_eval',policy,manager)
            row['control_evaluation']=dict(raw=summary(independent),guided=summary(controlled),
                improves=improves(independent,controlled,config['minimum_control_gain']),
                capture_floor=bool(sum(o.capture for o in controlled)>=np.ceil(config['capture_min']*len(controlled))),
                changed_controls=sum(e['changed_controls'] for e in controlled_episodes))
            if old is not None:
                row['previous_actor_counterfactual']=previous_actor_search(executor,None,config,memory,sources,index,
                    held,actor,controlled,trust['weight'],previous_actor=old,previous_checkpoint=old_checkpoint)
            invalid=deepcopy(manager);invalid.set_version('deliberately-invalid-policy-version',task.task_hash)
            revoke=root_panel(config,850000+index,1)
            _,a=control_panel(executor,config,memory,revoke,f'round{index}_revocation_raw')
            _,b=control_panel(executor,config,memory,revoke,f'round{index}_revoked',policy,invalid)
            row['promoted_then_revoked_exact']=all(np.array_equal(a[0][key],b[0][key]) for key in ('states','actions','components'))
            pool=root_panel(config,200000+index*10000,config['acquisition_seed_pool'])
            request=archive.append('complete_acquisition_resets_requested',round=index,roots=pool,
                bins=count,interval=[.2,.4],selection='First pool seed in each equal state bin; no outcome access')
            started=time.perf_counter();states=[]
            try:
                for root in pool:states.append(task.reset(root).state.copy())
            except BaseException as error:
                partial=archive.save_npz(f'complete_acquisition/round{index}_partial_reset_pool.npz',
                    roots=np.asarray(pool[:len(states)],dtype=np.int64),states=np.asarray(states).reshape(-1,2))
                archive.append('complete_acquisition_resets_error',request_id=request,error_type=type(error).__name__,
                    partial=partial,reset_work=dict(requested=len(pool),completed=len(states),
                        uniform_draws=len(states),scalar_controls=0,neural_rows=0,wall_seconds=time.perf_counter()-started))
                raise
            states=np.asarray(states);edges=np.linspace(.2,.4,count+1)
            bins=np.searchsorted(edges,states[:,1],side='right')-1
            selected=[int(np.flatnonzero(bins==i)[0]) for i in range(count) if np.any(bins==i)]
            selection=archive.save_npz(f'complete_acquisition/round{index}_reset_pool.npz',
                roots=np.asarray(pool,dtype=np.int64),states=states,bin_edges=edges,bin_indices=bins,
                selected_pool_indices=np.asarray(selected,dtype=np.int64))
            reset_work=dict(requested=len(pool),completed=len(states),uniform_draws=len(states),
                scalar_controls=0,neural_rows=0,wall_seconds=time.perf_counter()-started)
            archive.append('complete_acquisition_resets_complete',request_id=request,round=index,
                raw=selection,reset_work=reset_work,selection='First seed per state bin, before acquisition outcomes')
            chosen=[pool[i] for i in selected]
            row['acquisition']=dict(pool=selection,roots=chosen,reset_work=reset_work,weight=trust['weight'],
                no_favorable_pair_filter=True,selected_before_outcomes=True)
            row['coverage_failure']='state_bin_not_covered'
            if len(chosen)==count:
                _,packet['raw']=control_panel(executor,config,memory,chosen,f'round{index}_coverage_raw')
                _,packet['guided']=control_panel(executor,config,memory,chosen,f'round{index}_coverage_guided',policy,manager)
                decisions=[]
                for i,(a,b) in enumerate(zip(packet['raw'],packet['guided'])):
                    handovers=sorted(b.get('handover_windows',[]),key=lambda value:value['actual_handover'])
                    checks=dict(complete_pair=bool(a['complete'] and b['complete']),
                        successful_pair=bool(a['capture'] and b['capture']),
                        same_initial_state=np.array_equal(a['states'][0],b['states'][0]),
                        chosen_state=bool(np.array_equal(a['states'][0],states[selected[i]]) and a['root']==b['root']==chosen[i]),
                        own_controllers=bool(a['guiding_policy_hash'] is None and b['guiding_policy_hash']==policy.digest),
                        actual_handover=bool(handovers and handovers[0]['actual_handover']==handovers[0]['stop']
                            and 0<handovers[0]['stop']<task.config.T))
                    if checks['actual_handover']:bounds.append((0,int(handovers[0]['stop'])))
                    decisions.append(dict(index=i,root=chosen[i],checks=checks,passed=bool(all(checks.values())),
                        raw_query_ids=a['query_ids'],guided_query_ids=b['query_ids'],handovers=handovers,
                        original_cost_gain=float(a['components'].sum()-b['components'].sum())))
                row['coverage_packet_passed']=bool(len(decisions)==count and all(d['passed'] for d in decisions))
                row['coverage_failure']=None if row['coverage_packet_passed'] else 'incomplete_unsuccessful_or_no_actual_handover'
                row['acquisition'].update(decisions=decisions,windows=bounds,
                    teacher_queries={a:[e['query_ids'] for e in packet[a]] for a in arms})
                archive.append('complete_outcome_coverage_packet',round=index,result=row['acquisition'],
                    passed=row['coverage_packet_passed'],failure=row['coverage_failure'])
        observed=[]
        for episode in [p['raw'] for p in pairs]+packet['raw']:
            key=tuple(episode['query_ids'])
            if key in seen_raw:continue
            seen_raw.add(key);label=f'complete_round{index}_raw_{len(observed)}'
            observed.append(dict(label=label,query_ids=list(key)));add_memory(memory,episode,task,label)
        archive.append('independent_memory_ingestion',round=index,raw_query_groups=[r['query_ids'] for r in observed],
            labels=[r['label'] for r in observed],rule='All distinct RAW trial/confirmation/fixed-acquisition traces; no quality filter or guided append')
        row.update(teaching_contexts=len(packet['raw']),changed_controls=sum(e['changed_controls'] for e in packet['guided']),
            raw_memory_ingestion=observed,memory_after=identity(state_json(memory.fragments)),
            memory_fragments_after=len(memory.fragments),data_identical=all(all(np.array_equal(a[key],b[key])
                for key in ('states','actions','components')) for a,b in zip(packet['raw'],packet['guided'])))
        if not row['coverage_packet_passed']:
            manager.revoke_actor(policy.digest,'Fixed coverage packet failed; no root retry or favorable filtering')
        return manager,row,packet,bounds

    stages=[];rounds=[];seen_raw=set();current=deepcopy(incoming);data={a:list(teachers[a]) for a in arms}
    bootstrap=None;stop_reason=None
    if complete:
        actor=FamilyGaussianActor.from_checkpoint(current['actor'])
        qualification=panel(executor,actor,root_panel(config,1500000,config['ordinary_qualification_roots']),
            'complete_bootstrap_incoming_qual')
        bootstrap=dict(round=config['bootstrap_round_index'],incoming_actor=policy_hash(actor),
            current_actor=policy_hash(actor),trust=dict(promoted=False,reason='incoming_qualification_failed'),
            incoming_qualification=qualification,coverage_packet_passed=False,acquisition=None,
            stopped='incoming_qualification_failed',coverage_failure='incoming_qualification_failed',bootstrap=True)
        if qualification['captures']>=np.ceil(config['capture_min']*qualification['episodes']):
            manager,bootstrap,data,windows=search_packet(actor,data['guided'],config['bootstrap_round_index'])
            bootstrap['incoming_qualification']=qualification
            manager.revoke_actor(policy_hash(actor),'Bootstrap evidence imports no authority into learning')
            bootstrap['stopped']=None if bootstrap['coverage_packet_passed'] else bootstrap['coverage_failure']
        archive.append('complete_outcome_bootstrap',result=bootstrap,scientific_round=False)
        if not bootstrap['coverage_packet_passed']:stop_reason=bootstrap['stopped']
    if not complete or bootstrap['coverage_packet_passed']:
        first,current=stage(current,data,windows,0);stages.append(first)
    for round_index in range(config['rounds'] if stages else 0):
        previous=stages[-1]
        if not previous['interface_pass']:
            stop_reason='coupling_interface_failed';break
        if previous['selected_next_parent']=='parent' or previous['selected_actor']==previous['incoming_actor']:
            stop_reason='no_qualified_changed_child';break
        old=FamilyGaussianActor.from_checkpoint(incoming['actor']);actor=FamilyGaussianActor.from_checkpoint(current['actor'])
        source=data[previous['selected_next_parent']]
        if complete:
            manager,row,packet,fresh_windows=search_packet(actor,source,round_index,old,
                previous['qualified_checkpoints']['parent'])
            policy=NeuralPolicy(actor,archive,task,inference_contract=config['policy_inference_contract'])
            raw_data,guided_data=packet['raw'],packet['guided']
            if not row['coverage_packet_passed']:
                row['stopped']=row['coverage_failure'];rounds.append(row);stop_reason=row['stopped'];break
        else:
            pairs=[]
            manager,trust=qualify_guidance(executor,actor,source,memory,config,round_index,pairs)
            policy=NeuralPolicy(actor,archive,task,inference_contract=config['policy_inference_contract'])
            row=dict(round=round_index,incoming_actor=policy_hash(old),current_actor=policy_hash(actor),trust=trust,
                memory_before=identity(state_json(memory.fragments)),memory_fragments_before=len(memory.fragments),
                control_evaluation=None,previous_actor_counterfactual=None,promoted_then_revoked_exact=False)
            if trust['promoted']:
                roots=root_panel(config,800000+round_index*10000,config['control_evaluation_roots'])
                independent,_=control_panel(executor,config,memory,roots,f'round{round_index}_raw_eval')
                controlled,controlled_episodes=control_panel(executor,config,memory,roots,f'round{round_index}_guided_eval',policy,manager)
                row['control_evaluation']=dict(raw=summary(independent),guided=summary(controlled),
                    improves=improves(independent,controlled,config['minimum_control_gain']),
                    capture_floor=bool(sum(o.capture for o in controlled)>=np.ceil(config['capture_min']*len(controlled))),
                    changed_controls=sum(e['changed_controls'] for e in controlled_episodes))
                row['previous_actor_counterfactual']=previous_actor_search(executor,None,config,memory,source,round_index,
                    roots,actor,controlled,trust['weight'],previous_actor=old,
                    previous_checkpoint=previous['qualified_checkpoints']['parent'])
                invalid=deepcopy(manager);invalid.set_version('deliberately-invalid-policy-version',task.task_hash)
                revoke=root_panel(config,850000+round_index,1)
                _,raw_revoke=control_panel(executor,config,memory,revoke,f'round{round_index}_revocation_raw')
                _,guide_revoke=control_panel(executor,config,memory,revoke,f'round{round_index}_revoked',policy,invalid)
                row['promoted_then_revoked_exact']=all(np.array_equal(raw_revoke[0][key],guide_revoke[0][key])
                    for key in ('states','actions','components'))
            observed=[]
            for pair in pairs:
                key=tuple(pair['raw']['query_ids'])
                if key in seen_raw:continue
                seen_raw.add(key);label=f'temporal_round{round_index}_raw_{len(observed)}'
                observed.append(dict(label=label,query_ids=list(key)));add_memory(memory,pair['raw'],task,label)
            archive.append('independent_memory_ingestion',round=round_index,raw_query_groups=[r['query_ids'] for r in observed],
                labels=[r['label'] for r in observed],rule='All distinct observed RAW trial/confirmation traces before teacher filtering; no guided append')
            raw_data,guided_data=acquire_qualified_teachers(executor,pairs,trust,config,round_index)
            row.update(teaching_contexts=len(raw_data),changed_controls=sum(e['changed_controls'] for e in guided_data),
                raw_memory_ingestion=observed,memory_after=identity(state_json(memory.fragments)),
                memory_fragments_after=len(memory.fragments),data_identical=all(all(np.array_equal(a[key],b[key])
                    for key in ('states','actions','components')) for a,b in zip(raw_data,guided_data)))
            fresh_windows=[]
            for episode in guided_data:
                actual=sorted(episode.get('handover_windows',[]),key=lambda value:value['actual_handover'])
                if not actual or not 0<actual[0]['stop']<task.config.T:break
                fresh_windows.append((0,int(actual[0]['stop'])))
            if len(raw_data)!=4 or len(guided_data)!=4 or len(fresh_windows)!=4:
                row['stopped']='no_four_qualified_temporal_teachers'
                rounds.append(row);stop_reason=row['stopped']
                manager.revoke_actor(policy.digest,'Round ended without a qualified teacher packet');break
            archive.append('temporal_acquired_windows',round=round_index,windows=fresh_windows,
                handovers=[e['handover_windows'] for e in guided_data],raw_query_ids=[e['query_ids'] for e in raw_data],
                guided_query_ids=[e['query_ids'] for e in guided_data],selection='First actual handover, before student outcomes; common RAW/GUIDE deadline')
        incoming=deepcopy(current);data=dict(raw=raw_data,guided=guided_data)
        learned,current=stage(current,data,fresh_windows,round_index+1);stages.append(learned)
        row.update(learning_stage=learned['stage'],selected_next_parent=learned['selected_next_parent'],
            guided_next_policy_advantage=learned['guided_next_policy_advantage'],
            guided_control_improves_independent=bool(row['control_evaluation'] and row['control_evaluation']['improves']),
            guided_control_improves_previous=bool(row['previous_actor_counterfactual'] and row['previous_actor_counterfactual']['current_improves_previous']))
        row['scientific_pass']=bool(row['guided_next_policy_advantage'] and row['guided_control_improves_independent']
            and row['guided_control_improves_previous'] and row['changed_controls']>0
            and row['control_evaluation']['changed_controls']>0 and row['control_evaluation']['capture_floor']
            and row['promoted_then_revoked_exact'])
        manager.revoke_actor(policy.digest,'Learning stage completed; no authority is carried into another round')
        manager.set_version(current['actor']['policy_hash'],task.task_hash)
        archive.append('temporal_cycle_authority_reset',round=round_index,actor_hash=current['actor']['policy_hash'],
            records=state_json(manager.records),planning_influence=0.)
        row['next_parent_checkpoint']=learned['qualified_checkpoints'][learned['selected_next_parent']]
        committed_row=row
        if complete:
            committed_row=deepcopy(row)
            if committed_row.get('acquisition'):
                committed_row['acquisition']['reset_work'].pop('wall_seconds',None)
        commit=archive.save_npz(f'temporal_cycle/round{round_index}_commit.npz',
            metadata_json=np.array(json.dumps(state_json(dict(round=committed_row,memory=memory.fragments,
                next_parent=row['next_parent_checkpoint'],complete=True)),sort_keys=True)))
        row['commit']=commit;rounds.append(row);archive.append('temporal_cycle_round_complete',round=round_index,result=row)
    final=FamilyGaussianActor.from_checkpoint(current['actor'])
    final_panel=panel(executor,final,root_panel(config,950000,config['standalone_roots']),'temporal_cycle_final_standalone')
    checkpoint=save_pt(archive,'checkpoints/temporal_cycle_retained.pt',current)
    result=dict(schema=config['schema'],stages=stages,rounds=rounds,stop_reason=stop_reason,
        interface_pass=bool(stages and all(s['interface_pass'] for s in stages)),
        any_accepted_updates=any(sum(s['accepted_updates'].values())>0 for s in stages),
        guided_next_policy_advantage=bool(rounds and len(rounds)==config['rounds']
            and all(r.get('guided_next_policy_advantage',False) for r in rounds)),
        beneficial_bidirectional_loop=bool(len(rounds)==2 and all(s['interface_pass'] for s in stages)
            and all(r.get('scientific_pass',False) for r in rounds)
            and final_panel['captures']>=np.ceil(config['capture_min']*config['standalone_roots'])),
        final_checkpoint=checkpoint,final_actor_hash=policy_hash(final),final_standalone=final_panel,
        memory_fragments=len(memory.fragments),memory_signature=identity(state_json(memory.fragments)),
        planning_influence=0.,teacher_action_labels=False,teacher_state_targets=False,
        scope=config['scope'],ordinary_training_contexts=32 if global_learning else count,
        acquired_state_training=global_learning)
    if global_learning:result.update(global_outcome_teaching=True,acquired_source_slots=32,auxiliary_teacher_contexts=8,
        training_distribution='Mixed resets, distinct from ordinary deployment; duplicate physical contexts remain paid')
    if complete:result.update(bootstrap=bootstrap,complete_outcome_teaching=True,bootstrap_search=True)
    archive.append('temporal_cycle_complete',result=result)
    return result,current


def archive_transfer(executor,out,config,parent_checkpoint,curves):
    """A teacher-data learning intervention; no planner/trust authority used."""
    from .experience import import_qualified_teachers
    imported=import_qualified_teachers(executor.archive,out/'inputs/teachers',executor.task.task_hash)
    origin=imported['provenance']['packet']
    if (origin['origin_manifest_sha256']!=config['teacher_origin']['manifest_sha256']
            or origin['origin_ledger_sha256']!=config['teacher_origin']['ledger_sha256']
            or file_sha256(out/'inputs/teachers/origin/config.json')!=config['teacher_origin']['config_sha256']):
        raise ValueError('Pinned teaching origin differs')
    executor.archive.append('archived_teaching_loaded',provenance=imported['provenance'],
        parent_checkpoint=parent_checkpoint,trust_authority_imported=False,planning_influence=0.)
    raw_data,guided_data=imported['raw'],imported['guided']
    if not raw_data:raise ValueError('No archived teaching contrast')
    parent=restore_learner(executor,parent_checkpoint,config)
    initial,_=actor_panel(executor,parent.actor,root_panel(config,900000,config['standalone_roots']),
        'initial_standalone_qualification')
    row=dict(round=0,teaching_contexts=len(raw_data),
        trust=dict(promoted=False,reason='archive evidence imports no current authority'),
        historical_changed_controls=sum(e['changed_controls'] for e in guided_data),
        changed_controls=0,planning_influence=0.,
        data_identical=all(all(np.array_equal(a[k],b[k]) for k in ('states','actions','components'))
            for a,b in zip(raw_data,guided_data)))
    result=dict(initial_standalone_qualification=summary(initial),rounds=[])
    if sum(o.capture for o in initial)<np.ceil(config['capture_min']*len(initial)):
        row['stopped']='initial_standalone_qualification_failed'
    else:
        comparison,children,chosen=matched_children(executor,parent,parent_checkpoint,raw_data,guided_data,config,0,curves)
        row.update(comparison)
        if chosen!='parent':parent_checkpoint=children[chosen]
    row['next_parent_checkpoint']=parent_checkpoint;result['rounds'].append(row)
    final=restore_learner(executor,parent_checkpoint,config)
    panel,_=actor_panel(executor,final.actor,root_panel(config,950000,config['standalone_roots']),
        'final_standalone_evaluation')
    result.update(final_standalone=summary(panel),final_checkpoint=parent_checkpoint,
        archive_transfer_pass=bool(not row['data_identical'] and row.get('guided_child_improves_raw',False)
            and row.get('guided_child_improves_parent',False)
            and row.get('selected_next_parent')=='guided'
            and sum(o.capture for o in panel)>=np.ceil(config['capture_min']*len(panel))),
        beneficial_bidirectional_loop=False,current_planning_influence=0.,
        scope='Conditional archived teacher-to-learner transfer only; no new search-control or full-loop result')
    if config.get('interface_only'):
        from .audit import read_npz
        events=[json.loads(s) for s in executor.archive.ledger_path.read_text().splitlines()]
        updates=[e for e in events if e['kind']=='neural_actor_update']
        replies={e['query_id']:e for e in events if e['kind']=='query_complete'}
        batches={e['tag']:e for e in updates}
        ordinary=[read_npz(executor.archive.path,replies[batches[f'round0_{arm}_000']['query_ids'][0]]['result'])
            for arm in ('raw','guided')]
        checks=dict(changed_teacher_experience=not row['data_identical'],
            no_planning_authority=not any(e['kind'] in ('guidance_promotion','coupled_selection',
                'independent_memory_decision','executed_decision') for e in events)
                and all(not e['trust_authority_imported'] for e in events if e['kind']=='inherited_executed_source'),
            new_actor_owned_updates=len(updates)==2,
            same_ordinary_actor_data=all(np.array_equal(ordinary[0][k],ordinary[1][k])
                for k in ('pre_states','states','actions','components','latent','log_probs')),
            teaching_changes_update=batches['round0_raw_000']['changed_actor']!=batches['round0_guided_000']['changed_actor'])
        result['archive_interface_checks']=checks;result['archive_interface_pass']=all(checks.values())
    return result


def produce(out,config):
    started=time.perf_counter();curves=[];result=dict(schema=config['schema'],rounds=[])
    with ScalarArchive(out/'producer') as archive:
        configure_archive(archive,config)
        executor=EndingExecutor(archive,max_controls=config['max_controls'],task_config=config['task'])
        if config['mode'] in ('continuation','archive_transfer'):
            from .execution import load_inputs
            from .audit import compare_states
            validate_continuation(config)
            actor,memory,sources=load_inputs(out,config,validate_continuation)
            state=torch.load(out/'inputs/parent.pt',map_location='cpu',weights_only=True)
            if state_json(state['config'])!=config['learning']:raise ValueError('Inherited learning objective changed')
            learner=OutcomeLearner(actor,archive,executor.task,LearningConfig(**config['learning']),state['optimizer'])
            learner.update_index=state['update_index'];torch.set_rng_state(state['torch_rng'])
            # A deterministic step preserves the optimizer/RNG and records its
            # own metadata alongside the canonical learner fields. Keep that
            # full checkpoint; validate every canonical field without treating
            # the additional provenance as a learner-state mismatch.
            canonical=learner.state()
            if not compare_states(canonical,{k:state[k] for k in canonical}):
                raise ValueError('Inherited learner/optimizer/RNG changed')
            best=save_pt(archive,'checkpoints/ancestor.pt',state)
            archive.append('learner_continuation',checkpoint=best,ancestor=config['ancestor'],
                exact_actor_optimizer_rng=True,ancestor_support_only_not_actor_budget=True,
                memory_fragments=len(memory.fragments),fresh_initial_updates=0)
            parent_study=json.loads((out/'inputs/parent_study.json').read_text())
            result['inherited_discovery']=parent_study.get('discovery',parent_study.get('inherited_discovery'))
            result['ancestor_work']=dict(producer=parent_study['work'],
                verification=json.loads((out/'inputs/parent_audit.json').read_text())['work'],
                full_entry_wall_seconds=json.loads((out/'inputs/parent_launch.json').read_text())['full_entry_wall_seconds'],
                scope='Inherited search/learning/replay costs remain charged; earlier development costs are additional')
        else:
            episodes,memory=discover(executor,config)
            sources=successful_sources(episodes,config['teacher_roots'])
            result['discovery']=dict(episodes=len(episodes),captures=sum(e['capture'] for e in episodes))
            if len(sources)!=4:
                result.update(stopped='independent_discovery_failed',work=archive.totals(),wall_seconds=time.perf_counter()-started)
                write(out/'study.json',result);return result
            actor=make_actor(archive,LearningConfig(**config['learning']),config['seed'])
            learner=OutcomeLearner(actor,archive,executor.task,LearningConfig(**config['learning']))
            best,initial_rows,probe,last_groups=train(executor,learner,sources,config,updates=config['initial_updates'],
                seed_base=300000,qual_base=400000,tag='initial')
            curves.extend(initial_rows);result['neural_constraint_gradient_difference']=probe
        if config.get('credit_diagnostic'):
            from .sensitivity import produce_credit,produce_repeated_credit
            result.update((produce_repeated_credit if config.get('repeated_credit') else produce_credit)(executor,out,config,best))
        elif config['mode']=='archive_transfer':
            result.update(archive_transfer(executor,out,config,best,curves))
        elif config['mode']=='interface':
            try:learner.update(last_groups,'stale_owner_must_be_rejected')
            except ValueError:
                stale_rejected=True;archive.append('expected_ownership_rejection',before_optimizer=True)
            else:stale_rejected=False
            current=learner.checkpoint('checkpoints/interface_current.pt')
            restored=restore_learner(executor,current,config)
            root=executor.task.reset(16001)
            protected=IndependentMemorySearch(executor,config['short_search'],seed=55001,memory=memory,
                memory_candidates=config['full_memory_candidates'])
            saved=protected.snapshot();bank=protected.propose(root,'interface_before')
            policy=NeuralPolicy(restored.actor,archive,executor.task,
                inference_contract=config.get('policy_inference_contract','native'))
            trust=TrustManager();trust.set_version(policy.digest,executor.task.task_hash)
            trust.inference_contract=config.get('policy_inference_contract','native')
            from .loop import CoupledSelector
            action,changed,info=CoupledSelector(executor,policy,trust).select(bank)
            protected.restore(saved);repeated=protected.propose(root,'interface_after')
            raw,raw_eps=control_panel(executor,config,memory,[16002],'interface_raw')
            guided,guide_eps=control_panel(executor,config,memory,[16002],'interface_zero_guide',policy,trust)
            means,_=actor_panel(executor,restored.actor,list(range(17000,17016)),'interface_standalone',
                inference_contract=config.get('policy_inference_contract','native'))
            result['interface_checks']=dict(neural_constraint_changes_gradient=probe is not None and probe>1e-8,
                stale_behavior_owner_rejected=stale_rejected,
                checkpoint_hash_exact=policy_hash(restored.actor)==policy_hash(learner.actor),
                proposal_bank_exact=bank.digest==repeated.digest,
                zero_influence_fallback_exact=np.array_equal(action,bank.plan[0]) and not changed and info['influence']==0,
                executed_zero_fallback_exact=all(np.array_equal(raw_eps[0][k],guide_eps[0][k]) for k in ('states','actions','components')))
            result['interface_pass']=all(result['interface_checks'].values());result['fixture_neural_outcomes']=summary(means)
        else:
            parent=restore_learner(executor,best,config)
            roots=root_panel(config,900000,config['standalone_roots'])
            qualified,_=actor_panel(executor,parent.actor,roots,'initial_standalone_qualification',
                inference_contract=config.get('policy_inference_contract','native'))
            result['initial_standalone_qualification']=summary(qualified)
            if sum(o.capture for o in qualified)<np.ceil(config['capture_min']*len(qualified)):
                result['stopped']='standalone_learning_failed'
            else:
                parent_checkpoint=best
                for round_index in range(config['rounds']):
                    parent=restore_learner(executor,parent_checkpoint,config)
                    pairs=[]
                    manager,trust_result=qualify_guidance(executor,parent.actor,sources,memory,config,round_index,
                        pairs if config.get('acquisition_strategy')=='qualified_trial_pairs' else None)
                    policy=NeuralPolicy(parent.actor,archive,executor.task,
                        inference_contract=config.get('policy_inference_contract','native'))
                    if config.get('acquisition_strategy')=='qualified_trial_pairs':
                        raw_data,guided_data=acquire_qualified_teachers(executor,pairs,trust_result,config,round_index)
                    else:
                        acquisition=root_panel(config,200000+round_index*10000,config['acquisition_roots'])
                        _,raw_data=control_panel(executor,config,memory,acquisition,f'round{round_index}_raw_data')
                        _,guided_data=control_panel(executor,config,memory,acquisition,f'round{round_index}_guided_data',policy,manager)
                    row=dict(round=round_index,trust=trust_result,changed_controls=sum(e['changed_controls'] for e in guided_data),
                        teaching_contexts=len(raw_data),
                        data_identical=all(all(np.array_equal(a[k],b[k]) for k in ('states','actions','components'))
                            for a,b in zip(raw_data,guided_data)))
                    if not raw_data:
                        row['stopped']='no_qualified_teaching_contrast';result['rounds'].append(row);break
                    if not all(e['capture'] for e in raw_data+guided_data):
                        row['stopped']='teaching_acquisition_failure';result['rounds'].append(row);break
                    if config.get('fresh_search_only'):
                        comparison={};children={};chosen='parent'
                    else:
                        comparison,children,chosen=matched_children(executor,parent,parent_checkpoint,
                            raw_data,guided_data,config,round_index,curves)
                    row.update(comparison)
                    control_roots=root_panel(config,800000+round_index*10000,config['control_evaluation_roots'])
                    independent,_=control_panel(executor,config,memory,control_roots,f'round{round_index}_raw_eval')
                    controlled,_=control_panel(executor,config,memory,control_roots,f'round{round_index}_guided_eval',policy,manager)
                    row['control_evaluation']=dict(raw=summary(independent),guided=summary(controlled),
                        improves=improves(independent,controlled,config['minimum_control_gain']))
                    if config.get('compare_previous_actor') and trust_result['promoted']:
                        row['previous_actor_counterfactual']=previous_actor_search(executor,out,config,memory,sources,
                            round_index,control_roots,parent.actor,controlled,trust_result['weight'])
                    if trust_result['promoted']:
                        invalid=deepcopy(manager);invalid.set_version('deliberately-invalid-policy-version',executor.task.task_hash)
                        revoke_root=root_panel(config,850000+round_index,1)
                        _,a=control_panel(executor,config,memory,revoke_root,f'round{round_index}_revocation_raw')
                        _,b=control_panel(executor,config,memory,revoke_root,f'round{round_index}_revoked',policy,invalid)
                        row['promoted_then_revoked_exact']=all(np.array_equal(a[0][k],b[0][k]) for k in ('states','actions','components'))
                    else:row['promoted_then_revoked_exact']=None
                    # Teaching selection must not filter the independent
                    # memory according to learned/guided success. In v2 ingest
                    # every distinct observed raw trial/confirmation trace.
                    raw_memory=raw_data
                    if config.get('acquisition_strategy')=='qualified_trial_pairs':
                        raw_memory=[];seen_queries=set()
                        for pair in pairs:
                            key=tuple(pair['raw']['query_ids'])
                            if key not in seen_queries:
                                raw_memory.append(pair['raw']);seen_queries.add(key)
                        archive.append('independent_memory_ingestion',round=round_index,
                            raw_query_groups=[e['query_ids'] for e in raw_memory],
                            rule='all distinct observed raw trial/confirmation executions; no guided-quality filter')
                    for i,episode in enumerate(raw_memory):add_memory(memory,episode,executor.task,f'round{round_index}_raw_{i}')
                    sources=guided_data if chosen=='guided' else raw_data
                    if chosen!='parent':parent_checkpoint=children[chosen]
                    manager.set_version(load_pt(archive.path,parent_checkpoint)['actor']['policy_hash'],executor.task.task_hash)
                    row['next_parent_checkpoint']=parent_checkpoint
                    result['rounds'].append(row)
                    write(out/f'round_{round_index}_commit.json',dict(round=row,memory=state_json(memory.fragments),
                        next_parent=parent_checkpoint,source_config=config,complete=True))
                final=restore_learner(executor,parent_checkpoint,config)
                final_panel,_=actor_panel(executor,final.actor,root_panel(config,950000,config['standalone_roots']),
                    'final_standalone_evaluation',inference_contract=config.get('policy_inference_contract','native'))
                result['final_standalone']=summary(final_panel);result['final_checkpoint']=parent_checkpoint
        result['work']=dict(archive.totals(),**optimization_totals(archive));result['wall_seconds']=time.perf_counter()-started
        conjunction=(len(result['rounds'])==config['rounds'] and config['rounds']>0
            and all(r.get('guided_child_improves_raw',False) and r.get('control_evaluation',{}).get('improves',False)
                and r.get('guided_child_improves_parent',False) and r['changed_controls']>0
                and r.get('promoted_then_revoked_exact') for r in result['rounds'])
            and result.get('final_standalone',{}).get('captures',0)>=np.ceil(config['capture_min']*config['standalone_roots']))
        result['beneficial_bidirectional_loop']=bool(conjunction and config['rounds']>=2)
        if config.get('fresh_search_only'):
            result['search_feedback_pass']=bool(len(result['rounds'])==1 and all(
                r['trust']['promoted'] and r['teaching_contexts']==4 and not r['data_identical']
                and r['changed_controls']>0 and r.get('control_evaluation',{}).get('improves',False)
                and r.get('promoted_then_revoked_exact',False)
                and (not config.get('compare_previous_actor') or r.get('previous_actor_counterfactual',{}).get(
                    'current_improves_previous',False)) for r in result['rounds']))
        if config.get('acquisition_strategy')=='qualified_trial_pairs':
            result['acquisition_transfer_pass']=bool(conjunction)
            result['scope']=(config['scope'] if config.get('fresh_search_only') else
                'Conditional one-round transfer test; full two-round beneficial loop remains required')
        write(out/'learning_curve.json',curves);write(out/'study.json',result)
    return result


def preflight_feedback_source(out,config):
    """Check frozen acquisition evidence before any new actor execution."""
    if config.get('feedback_source_gate')!='qualified_teacher_packet':return
    from .audit import _audit_conditional_teacher_source
    admission_started=time.perf_counter();inputs=out/'inputs';origin_root=inputs/'teachers/origin'
    incoming=torch.load(inputs/'parent.pt',map_location='cpu',weights_only=True)
    admission=_audit_conditional_teacher_source(inputs,config,
        json.loads((origin_root/'study.json').read_text()),json.loads((origin_root/'config.json').read_text()),
        json.loads((inputs/'teachers/manifest.json').read_text()),incoming['actor']['policy_hash'])
    write(out/'source_admission_audit.json',dict(passed=True,checks=admission,
        wall_seconds=time.perf_counter()-admission_started,new_controls=0,new_neural_rows=0,
        new_backward_calls=0,new_optimizer_steps=0))


def run(root,out,config):
    verify_experiment(root,out)
    verify_learning_interface(out,config)
    if config.get('feasibility_curriculum'):
        from .curriculum import run as curriculum_run
        preflight_feedback_source(out,config)
        return curriculum_run(root,out,config)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    # Native engines are deliberately unavailable in this scalar development.
    import importlib.abc,sys
    class NoPhysics(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname.split('.')[0] in ('mujoco','mujoco_warp','warp'):
                raise RuntimeError('Scalar neural study forbids native physics')
    sys.meta_path.insert(0,NoPhysics())
    started=time.perf_counter();failure=None
    from types import SimpleNamespace
    from .meter import ledger_work_snapshot
    def durable_work(name):
        return ledger_work_snapshot(SimpleNamespace(ledger_path=out/name/'ledger.jsonl'))
    try:
        preflight_feedback_source(out,config)
        result=produce(out,config)
    except BaseException as error:
        import traceback
        write(out/'producer_failure.json',dict(type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),
            durable_accounting=durable_work('producer'),wall_seconds=time.perf_counter()-started))
        failure=error;result={}
    audit_failure=None;failure_work=None;audit_started=time.perf_counter()
    try:
        from .audit import audit_neural
        audit=audit_neural(out,config)
    except BaseException as error:
        import traceback
        audit_failure=error;failure_work=dict(producer=durable_work('producer'),verification=durable_work('verification'))
        record=dict(type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),
            passed=False,durable_accounting=failure_work,partial_work_must_not_be_treated_as_zero=True,
            wall_seconds=time.perf_counter()-audit_started)
        write(out/'audit_failure.json',record)
        audit=dict(passed=False,errors=['Verification raised '+type(error).__name__+': '+str(error)],checks={},
            work=failure_work['verification']['work'],work_status=failure_work['verification']['status'],
            scope='Incomplete verification; durable lower bounds and unknown partial work only',
            wall_seconds=time.perf_counter()-audit_started)
        # Preserve an already written auditor artifact, even if an exception
        # happened while closing it. The failure receipt overrides any pass.
        if not (out/'audit.json').exists():write(out/'audit.json',audit)
        if not (out/'study.json').exists():write(out/'study_incomplete.json',dict(passed=False,
            producer_failed=failure is not None,work=failure_work['producer']['work'],
            work_status=failure_work['producer']['status'],scope='Failed/incomplete producer; no scientific pass'))
    success=(not failure and audit['passed'] and (result.get('search_feedback_pass',False) if config.get('fresh_search_only')
        else result.get('feedback_interface_pass',False) if config.get('feedback_learning') and config.get('interface_only')
        else result.get('feedback_training_pass',False) if config.get('feedback_learning')
        else result.get('depth_interface_pass',False) if config.get('repeated_credit') and config.get('interface_only')
        else result.get('depth_training_pass',False) if config.get('repeated_credit')
        else result.get('pointwise_transfer_pass',False) if config.get('teacher_evidence')=='pointwise_trial'
        else result.get('ending_projection_pass',False) if config.get('credit',{}).get('ending_projection')
        else result.get('credit_diagnostic_pass',False) if config.get('credit_diagnostic')
        else result.get('interface_pass',False) if config['mode']=='interface'
        else result.get('archive_interface_pass',False) if config.get('interface_only')
        else result.get('archive_transfer_pass',False) if config['mode']=='archive_transfer'
        else result.get('acquisition_transfer_pass',False) if config.get('acquisition_strategy')=='qualified_trial_pairs'
        else result.get('beneficial_bidirectional_loop',False)))
    write(out/'result.json',dict(passed=bool(success),producer_failed=failure is not None,audit_pass=audit['passed'],
        mode=config['mode'],beneficial_bidirectional_loop=result.get('beneficial_bidirectional_loop',False),
        acquisition_transfer_pass=result.get('acquisition_transfer_pass'),
        archive_transfer_pass=result.get('archive_transfer_pass'),
        archive_interface_pass=result.get('archive_interface_pass'),
        search_feedback_pass=result.get('search_feedback_pass'),
        feedback_training_pass=result.get('feedback_training_pass'),
        feedback_interface_pass=result.get('feedback_interface_pass'),
        pointwise_transfer_pass=result.get('pointwise_transfer_pass'),
        depth_training_pass=result.get('depth_training_pass'),
        depth_interface_pass=result.get('depth_interface_pass'),
        credit_diagnostic_pass=result.get('credit_diagnostic_pass'),
        ending_projection_pass=result.get('ending_projection_pass'),
        ending_restoration_pass=result.get('ending_restoration_pass'),
        **({} if failure_work is None else dict(audit_failed=True,audit_failure_file='audit_failure.json',
            producer_flags_unverified=True,durable_failure_accounting=failure_work)),
        total_worker_wall_seconds=time.perf_counter()-started,
        verdict='Execution or verification failed; durable work and unknown partials preserved' if failure or audit_failure
            else 'Fresh search gate only; next learner and full feedback remain unestablished' if config.get('fresh_search_only')
            else 'Fresh-teacher learning interface only; no next-policy benefit result' if config.get('feedback_learning') and config.get('interface_only')
            else 'One fresh matched feedback-learning round; repeated-loop and efficiency claims remain unestablished' if config.get('feedback_learning')
            else 'Repeated-learning interface only; no depth-benefit result' if config.get('repeated_credit') and config.get('interface_only')
            else 'Bounded learning-depth test; fresh search and full feedback remain unestablished' if config.get('repeated_credit')
            else 'Conditional pointwise teaching transfer; no qualified search or full loop' if config.get('teacher_evidence')=='pointwise_trial'
            else 'One ending-constrained transfer step; full feedback remains unestablished' if config.get('credit',{}).get('ending_projection')
            else 'Deterministic credit mechanism only; full feedback remains unestablished' if config.get('credit_diagnostic')
            else 'Interface only' if config['mode']=='interface' or config.get('interface_only')
            else 'Conditional archived transfer; full feedback remains unestablished' if config['mode']=='archive_transfer'
            else 'One-seed development; physical/general efficiency conclusions unestablished'))
    return success
