"""Independent audit of the closed selective-criterion initial negative.

No producer, actor, simulator or optimizer import. The first version's75%gate
and its frozen auditor remain separate. This adapter certifies no unexecuted
confirmation, child update, memory revision or two-round success.
"""
from __future__ import annotations
import math
import numpy as np
from .conditional_audit import require,validate_conditional_initial

ROOTS=dict(competence=801000,initial_qualification=802000,initial_confirmation=803000,
    round0_acquisition=811000,round0_training=812000,round0_actor_qualification=813000,
    round0_controller_qualification=814000,round0_controller_confirmation=815000,
    round1_acquisition=821000,round1_training=822000,round1_actor_qualification=823000,
    round1_controller_qualification=824000,round1_controller_confirmation=825000,final=831000)


def criteria(panels,proposed,references,*,teacher=True):
    """Reconstruct the complete prospective mean/no-observed-loss criterion."""
    rows=panels[proposed];require(bool(rows),'Empty selective outcome panel');threshold=.01 if teacher else .001
    comparisons={}
    for name in references:
        base=panels[name]
        require(len(base)==len(rows) and [r['context_id'] for r in base]==[r['context_id'] for r in rows]
            and len({r['context_id'] for r in rows})==len(rows),'Selective reset panels are not paired')
        require(all(r['complete'] is True and r['deterministic'] is True and r['family_id']==0 and r['remaining_steps']==25
            and math.isfinite(r['cost']) for r in base+rows),'Selective returns are not complete actual mean control')
        require(len({r['task_hash'] for r in base+rows})==1,'Selective task differs')
        aq={q for r in base for q in r['query_ids']};bq={q for r in rows for q in r['query_ids']}
        require(not aq.intersection(bq),'Selective comparison reuses actual control evidence')
        gains=[a['cost']-b['cost'] for a,b in zip(base,rows)];require(all(math.isfinite(g) for g in gains),'Nonfinite selective gain')
        mean=math.fsum(gains)/len(gains);positive=sum(g>=threshold for g in gains)
        gained=sum(g>1e-9 for g in gains);tied=sum(abs(g)<=1e-9 for g in gains);lost=sum(g < -1e-9 for g in gains)
        nonreg=sum(r['capture'] for r in rows)>=sum(r['capture'] for r in base)
        comparisons[name]=dict(mean_cost_gain=mean,paired_cost_gains=gains,positive_pairs=positive,total_pairs=len(gains),
            gain_pairs=gained,tie_pairs=tied,loss_pairs=lost,paired_cost_tolerance=1e-9,
            improvement_prevalence=gained/len(gains),threshold_improvement_prevalence=positive/len(gains),zero_observed_paired_loss=lost==0,
            paired_capture_losses=sum(a['capture'] and not b['capture'] for a,b in zip(base,rows)),
            paired_capture_gains=sum(b['capture'] and not a['capture'] for a,b in zip(base,rows)),
            capture_nonregression=bool(nonreg),passed=bool(nonreg and mean>=threshold and (not teacher or lost==0)))
    capture=sum(r['capture'] for r in rows)/len(rows)
    return dict(passed=bool(capture>=.9 and all(r['passed'] for r in comparisons.values())),capture_rate=capture,comparisons=comparisons,
        minimum_mean_gain=threshold,minimum_pair_fraction=None,teacher_pair_rule='zero_observed_paired_loss' if teacher else None,
        minimum_paired_cost_gain=-1e-9 if teacher else None,population_noninterference_guarantee=False,
        capture_nonregression='aggregate matched count, not a per-root never-lose requirement',mode='deterministic_mean_only')


def validate_selective_initial(e,base,rows,work,result,config,checkpoints,episodes,qreq,qdone,task_hash):
    require(config['schema']=='bidirectional-ending-conditional-mean-feedback-v2-selective'
        and config['roots']==ROOTS and config['teacher_pair_rule']=='zero_observed_paired_loss',
        'Only the exact fresh selective v2 namespace/rule is supported')
    require(result['teacher_pair_rule']=='zero_observed_paired_loss' and result['selective_noninterference_variant'] is True
        and result['criterion_disclosure']==dict(outcome_informed=True,motivating_run='2026-10-27_bidirectional_ending_conditional_001',
            motivating_result='failed_predeclared_75_percent_improvement_prevalence',original_v1_negative_unchanged=True,
            original_v1_rows_are_not_fresh_confirmation=True,population_noninterference_guarantee=False,efficiency_claim=False),
        'Selective outcome-informed development revision is not disclosed')
    require(result['stage']=='failed_initial_complete_teacher_qualification' and result['rounds']==[] and result['passed'] is False
        and result['full_feedback_established'] is False,'This certificate is restricted to the closed initial selective negative')
    forbidden=('conditional_features_requested','conditional_svd_requested','conditional_candidate_evaluation_requested',
        'conditional_promotion_requested','conditional_controller_confirmation','conditional_final_assessment_requested',
        'conditional_memory_appended','conditional_actor_promoted','conditional_revocation_checks_requested')
    require(not any(r['kind'] in forbidden for r in rows),'Initial selective failure incurred later unreported work')
    report=validate_conditional_initial(e,base,rows,work,result,config,checkpoints,episodes,qreq,qdone,task_hash,allow_later=True)
    context=report.pop('_prefix_context');checked=criteria(context['initial_panels'],'GUIDE',('RAW','actor'),teacher=True)
    require(checked['passed'] is False and result['initial_controller_gate']['passed'] is False
        and result['initial_controller_gate']['phase']=='qualification','Selective actual paired-loss failure was not retained')
    failures=[r for r in rows if r['kind']=='conditional_gate_failed']
    require(len(failures)==1 and failures[0]['stage']==result['stage'] and failures[0]['no_retry'] is True,'Selective stop is not durable')
    require(len(episodes)==80 and sum(r['controls'] for r in episodes)==1808,'Selective actual execution roster differs')
    require(context['used_model_queries']=={q for q,r in qreq.items() if r['mode']=='model'},'Selective initial evidence omitted paid model queries')
    require(result['accounting']['executed_controls']==1808 and result['accounting']['model_controls']==sum(r['requested_controls'] for r in qreq.values() if r['mode']=='model'),
        'Selective separate actual/model accounting differs')
    report.update(scope='Only initial selective no-observed-paired-loss gate; no confirmation/child/memoryupdate/feedback executed or certified',
        teacher_qualification=checked,teacher_pair_rule='zero_observed_paired_loss',fresh_root_namespace=ROOTS,
        original75percent_negative_unchanged=True,population_noninterference_guarantee=False,
        scientific_success=False,full_feedback_established=False)
    return report
