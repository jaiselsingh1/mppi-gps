"""Prospective, bounded Ending rollout-improvement and student-feedback test.

The completion is an exact frozen actor's deterministic continuation. It is
never a measured return of the future replanning controller or a training label
for a different policy. Only real uninterrupted actor traces supply MC labels.
Classical one-step rollout is a component, not a novelty claim.
"""
from copy import deepcopy
from dataclasses import asdict
import numpy as np
from .policy import policy_hash,controller_hash
from .learning import AuxiliaryBatch
from .meter import attach_meter
from .search import IndependentSearch,SearchConfig
from .trust import CandidateTrustManager
from .feedback import _digest


def work_bound(config):
    """Conservative before-call caps; every acquired context counted as T25."""
    T=25;short=sum(min(4,r) for r in range(1,T+1));full=T*(T+1)//2
    width=48;aux=200;own=64*T;minibatches=4*int(np.ceil(own/256))
    initial_model=8*17*full
    initial_live=8*T+2*own+16*T+2*4*16*T+16*T
    initial_rows=2000*aux+2*own+2*(own+4*own+minibatches*aux)+16*T+2*4*16*T+16*T
    # Two 16-root qualification panels, 8 acquisition roots, 64 held roots,
    # and one revoked root. Both online controllers are executed independently.
    search_roots=16+16+8+64+1
    model_per_round=2*search_roots*(width+1)*short+search_roots*5*full
    handoff=8*(T+1)
    live_per_round=handoff+2*search_roots*T+(16+16+8+64)*T+own+2*3*64*T
    actor_per_round=handoff+search_roots*(5*(full-T)+T)+(16+16+8+64)*T+own+2*(own+4*own+minibatches*aux)+2*3*64*T
    rounds=config['feedback_rounds']
    return dict(model_controls=initial_model+rounds*model_per_round,
        executed_controls=initial_live+rounds*live_per_round+3*64*T,
        total_controls=initial_model+initial_live+rounds*(model_per_round+live_per_round)+3*64*T,
        actor_rows=initial_rows+rounds*actor_per_round+3*64*T,
        adam_steps=2000+2*minibatches+rounds*2*minibatches,
        diagnostic_backward_passes=2+2*rounds,
        scope='Fixed maximum two rounds; all acquired starts counted as ordinary full T25; unused reservations are not spent; no retry/retuning')


def _summary(rows):
    return dict(capture_rate=float(np.mean([r.capture for r in rows])),
        mean_cost=float(np.mean([r.cost for r in rows])),outcomes=[asdict(r) for r in rows])


def _comparison(base,proposed,minimum):
    if [r.context_id for r in base]!=[r.context_id for r in proposed]:
        raise ValueError('Complete outcome contexts are not paired')
    gains=[a.cost-b.cost for a,b in zip(base,proposed)]
    return dict(mean_cost_gain=float(np.mean(gains)),paired_cost_gains=gains,
        positive_pairs=sum(g>=minimum for g in gains),total_pairs=len(gains),
        capture_nonregression=sum(r.capture for r in proposed)>=sum(r.capture for r in base),
        improved=bool(sum(r.capture for r in proposed)>=sum(r.capture for r in base) and np.mean(gains)>=minimum))


def _auxiliary(traces,rows):
    successful=[t for t in traces if t['successful']]
    if not successful:raise ValueError('No successful actual teacher data')
    observations=np.concatenate([t['observations'][:-1] for t in successful])
    actions=np.concatenate([t['actions'] for t in successful])
    idx=np.linspace(0,len(actions)-1,rows,dtype=np.int64)
    return AuxiliaryBatch(observations[idx].copy(),actions[idx].copy(),np.zeros(rows,dtype=int)),idx


def teaching_auxiliary(shared,new,rows):
    """Identical shared half; arm-specific half with explicit shared fallback."""
    if rows%2:raise ValueError('Two fixed equal auxiliary strata require even rows')
    common,common_idx=_auxiliary(shared,rows//2)
    successful=[t for t in new if t['successful']]
    specific,specific_idx=_auxiliary(successful or shared,rows//2)
    batch=AuxiliaryBatch(np.concatenate((common.observations,specific.observations)),
        np.concatenate((common.actions,specific.actions)),np.zeros(rows,dtype=int))
    return batch,dict(common=common_idx,specific=specific_idx,
        specific_is_shared_fallback=not bool(successful))


def _panel(executor,actor,trust,memory,seeds,tag,config,*,guided,trial=False):
    from .ending_pilot import _episode
    from .first_action_completion import FirstActionCompletionSelector
    outcomes=[];traces=[];changed=0
    for i,seed in enumerate(seeds):
        initial=executor.task.reset(seed)
        planner=IndependentSearch(executor,SearchConfig(**config['search']),seed=9000+seed,memory=deepcopy(memory))
        selector=FirstActionCompletionSelector(executor,actor,trust) if guided else None
        row,trace,_,changes=_episode(executor,initial,f'{tag}_{i:03d}',planner=planner,
            selector=selector,trial_weight=1. if trial else None)
        outcomes.append(row);traces.append(trace);changed+=changes
    return outcomes,traces,changed


def run_feedback(config,executor,parent,teachers,memory,out,phase):
    from .ending_pilot import _episode,_actor_panel,EndingSnapshot
    from .workflow import checkpoint
    from .first_action_completion import FirstActionCompletionSelector
    archive=executor.archive;minimum=config['minimum_complete_gain'];rounds=[]
    teacher_minimum=config['minimum_teacher_gain']
    def teacher_pass(comparison):
        return comparison['improved'] and comparison['positive_pairs']/comparison['total_pairs']>=config['minimum_positive_fraction']
    shared=[t for t in teachers if t['successful']]
    result=dict(passed=False,stage='completion_setup',next_generation_learning_tested=False,
        agreement_used=False,completion_semantics='one_candidate_action_then_deterministic_frozen_actor',
        classical_component='exact one-step rollout/policy improvement',feedback_rounds=rounds)
    def finish(stage,**fields):
        result.update(stage=stage,**fields)
        if not result.get('passed'):result['negative_retained']=True
        return result
    for round_id in range(config['feedback_rounds']):
        prefix=f'completion_r{round_id}';offset=10000*round_id;actor=parent.actor;ah=policy_hash(actor)
        parent_state=_digest(parent.state_dict());entry=dict(round=round_id,parent_actor_hash=ah,parent_state_sha256=parent_state)
        rounds.append(entry)
        checkpoint(out,prefix+'_parent',dict(learner=parent.state_dict(),promotion_claim=False),archive)
        manager=CandidateTrustManager(1.);anchors=[]
        for i in range(8):
            trace=shared[i%len(shared)]
            for t in (0,24):anchors.append(EndingSnapshot(trace['states'][t].copy(),t,f'{prefix}_handoff:{i}:{t}'))
        record=manager.register(actor_hash=ah,actor_controller_hash=controller_hash(actor,deterministic=True),
            task_hash=executor.task.task_hash,family_id=0,total_steps=25,time_interval=(0,24),remaining_interval=(1,25),
            support_anchors=np.stack([executor.task.observation(a) for a in anchors]),observation_scale=np.ones(8),
            support_radius=config['support_radius'],anchor_context_ids=tuple(a.context_id for a in anchors))
        if record.weight!=0:raise RuntimeError('New actor version carried influence before qualification')
        phase(prefix+'_suffix_eligibility')
        handoffs=[_episode(executor,a,f'{prefix}_handoff_{i:03d}',actor=actor,semantics='actor_suffix')[0] for i,a in enumerate(anchors)]
        eligible=manager.qualify(record.record_id,handoffs)
        if eligible.state!='eligible':return finish('failed_completion_suffix_qualification')
        # The protocol identity is bound into every actual controller hash.
        entry['selector_protocol']=FirstActionCompletionSelector(executor,actor,manager).protocol_identity
        for panel_index,(label,start) in enumerate((('qualification',61000),('confirmation',62000))):
            phase(prefix+'_'+label);seeds=list(range(offset+start,offset+start+16))
            tested=CandidateTrustManager.from_eligible(manager,1.)
            rec=tested.records[0]
            raw,_,_=_panel(executor,actor,tested,memory,seeds,prefix+'_'+label+'_raw',config,guided=False)
            guided,_,changed=_panel(executor,actor,tested,memory,seeds,prefix+'_'+label+'_guided',config,guided=True,trial=True)
            actor_rows=_actor_panel(executor,actor,seeds,prefix+'_'+label+'_actor')
            vs_raw=_comparison(raw,guided,teacher_minimum);vs_actor=_comparison(actor_rows,guided,teacher_minimum)
            entry[label]=dict(raw=_summary(raw),guided=_summary(guided),actor=_summary(actor_rows),
                guided_vs_raw=vs_raw,guided_vs_actor=vs_actor,changed_controls=changed)
            ok=teacher_pass(vs_raw) and teacher_pass(vs_actor) and np.mean([r.capture for r in guided])>=.9 and changed>0
            if not ok:
                tested.revoke(rec.record_id,'complete executed outcome criterion failed')
                archive.append('completion_qualification',round=round_id,panel=label,passed=False,result=entry[label])
                return finish('failed_completion_'+label)
            promoted=tested.promote(rec.record_id,1.,raw,guided,
                baseline_controller_hash=raw[0].controller_hash,guided_controller_hash=guided[0].controller_hash)
            archive.append('trust_state',purpose='completion_'+label,selector_protocol=entry['selector_protocol'],record=asdict(promoted))
            if promoted.state!='promoted':return finish('failed_completion_promotion')
            # Confirmation reconstructs eligibility from the ORIGINAL suffix
            # evidence, so trial promotion is never its authorization source.
            if panel_index==1:manager=tested;record=promoted
        phase(prefix+'_acquisition');seeds=list(range(offset+63000,offset+63008))
        raw,raw_traces,_=_panel(executor,actor,manager,memory,seeds,prefix+'_acquisition_raw',config,guided=False)
        guided,guided_traces,changed=_panel(executor,actor,manager,memory,seeds,prefix+'_acquisition_guided',config,guided=True)
        actor_acquisition=_actor_panel(executor,actor,seeds,prefix+'_acquisition_actor')
        quality=dict(guided_vs_raw=_comparison(raw,guided,teacher_minimum),guided_vs_actor=_comparison(actor_acquisition,guided,teacher_minimum),
            raw_captures=sum(r.capture for r in raw),guided_captures=sum(r.capture for r in guided))
        entry['teaching_quality']=quality
        if not teacher_pass(quality['guided_vs_raw']) or not teacher_pass(quality['guided_vs_actor']):
            return finish('failed_complete_teacher_quality')
        phase(prefix+'_fresh_executed_evaluation');seeds=list(range(offset+65000,offset+65064))
        baseline,_,_=_panel(executor,actor,manager,memory,seeds,prefix+'_held_raw',config,guided=False)
        influenced,_,held_changes=_panel(executor,actor,manager,memory,seeds,prefix+'_held_guided',config,guided=True)
        parent_rows=_actor_panel(executor,actor,seeds,prefix+'_held_actor')
        entry['executed_control']=dict(guided_vs_raw=_comparison(baseline,influenced,teacher_minimum),
            guided_vs_actor=_comparison(parent_rows,influenced,teacher_minimum),raw=_summary(baseline),
            guided=_summary(influenced),actor=_summary(parent_rows),changed_controls=held_changes)
        ctrl=entry['executed_control']
        control_ok=teacher_pass(ctrl['guided_vs_raw']) and teacher_pass(ctrl['guided_vs_actor']) and ctrl['guided']['capture_rate']>=.9 and held_changes>0
        contrast=ctrl['raw']['capture_rate']<=config['baseline_capture_max']
        if not control_ok or not contrast:return finish('failed_completion_held_control')
        phase(prefix+'_owned_mc');episodes=[]
        contexts=[executor.task.reset(offset+64000+i) for i in range(8)]
        for i in range(8):
            t=(6,12,18,21)[i%4];trace=shared[i%len(shared)]
            contexts.append(EndingSnapshot(trace['states'][t].copy(),t,f'{prefix}_acquired:{i}'))
        # Clone sampling state too: collecting own behavior must not change the
        # immutable training parent used for the two matched proposals.
        sample_generator=deepcopy(parent.torch_generator)
        for i,initial in enumerate(contexts):
            for j in range(4):
                _,_,ep,_=_episode(executor,initial,f'{prefix}_own_c{i:02d}_r{j}',actor=actor,
                    deterministic=False,generator=sample_generator);episodes.append(ep)
        if _digest(parent.state_dict())!=parent_state:raise RuntimeError('Parent changed before matched children')
        behavior_hash=_digest(episodes);children={name:parent.clone_for_trial() for name in ('raw','guided')}
        if any(_digest(child.state_dict())!=parent_state for child in children.values()):raise RuntimeError('Unequal child initialization')
        aux={};indices={}
        for name,traces in (('raw',raw_traces),('guided',guided_traces)):
            aux[name],indices[name]=teaching_auxiliary(shared,traces,config['child_auxiliary_rows'])
        half=config['child_auxiliary_rows']//2
        if not np.array_equal(aux['raw'].observations[:half],aux['guided'].observations[:half]) or not np.array_equal(aux['raw'].actions[:half],aux['guided'].actions[:half]):
            raise RuntimeError('Matched children received different common teacher rows')
        aux_raw=archive.save_npz(f'feedback/{prefix}_matched_aux.npz',**{f'{name}_{field}':getattr(batch,field)
            for name,batch in aux.items() for field in ('observations','actions','families')},
            **{f'{name}_{stratum}':value for name,details in indices.items() for stratum,value in details.items()})
        phase(prefix+'_child_fit');learning={}
        for name,child in children.items():
            attach_meter(child.actor,archive,prefix+'_child_'+name)
            learning[name]=child.update(episodes,ah,aux=aux[name])
            archive.append('fit_work',purpose=prefix+'_matched_child_'+name,**learning[name])
            if _digest(episodes)!=behavior_hash:raise RuntimeError('Behavior data changed between child updates')
            checkpoint(out,prefix+'_child_'+name,dict(learner=child.state_dict(),parent_actor_hash=ah,
                parent_state_sha256=parent_state,behavior_sha256=behavior_hash,learning=learning[name],auxiliary_raw=aux_raw,
                shared_teacher_refs=[t['raw'] for t in shared],new_teacher_refs=[t['raw'] for t in (raw_traces if name=='raw' else guided_traces)],
                promotion_claim=False),archive)
        if learning['raw']['gradient_steps']!=learning['guided']['gradient_steps']:raise RuntimeError('Unequal child optimizer work')
        phase(prefix+'_fresh_child_evaluation');seeds=list(range(offset+66000,offset+66064))
        panels={'parent':_actor_panel(executor,actor,seeds,prefix+'_child_eval_parent')}
        for name,child in children.items():panels[name]=_actor_panel(executor,child.actor,seeds,prefix+'_child_eval_'+name)
        stochastic={'parent':_actor_panel(executor,actor,seeds,prefix+'_child_stochastic_parent',deterministic=False,stochastic_seed=offset+68000)}
        for name,child in children.items():stochastic[name]=_actor_panel(executor,child.actor,seeds,
            prefix+'_child_stochastic_'+name,deterministic=False,stochastic_seed=offset+68000)
        benefit=dict(guided_vs_parent=_comparison(panels['parent'],panels['guided'],minimum),
            guided_vs_raw=_comparison(panels['raw'],panels['guided'],minimum))
        stochastic_benefit=dict(guided_vs_parent=_comparison(stochastic['parent'],stochastic['guided'],minimum),
            guided_vs_raw=_comparison(stochastic['raw'],stochastic['guided'],minimum))
        child_ok=benefit['guided_vs_parent']['improved'] and benefit['guided_vs_raw']['improved'] and np.mean([r.capture for r in panels['guided']])>=.9 and all(c['improved'] for c in stochastic_benefit.values())
        entry['matched_child']=dict(parent_state_sha256=parent_state,behavior_sha256=behavior_hash,
            panel_role='Fresh promotion/validation; only the separate end-only panel is pipeline holdout',
            auxiliary_rows_per_child=config['child_auxiliary_rows'],auxiliary_raw=aux_raw,learning=learning,
            auxiliary_strata='Identical fixed100 common rows; fixed100 successful new arm rows or declared shared fallback; deterministic index resampling',
            panels={name:_summary(rows) for name,rows in panels.items()},benefit=benefit,
            stochastic_panels={name:_summary(rows) for name,rows in stochastic.items()},stochastic_benefit=stochastic_benefit,passed=bool(child_ok),
            scope='Same parent/optimizer/RNG/own MC/update work/common discovery; only newly executed auxiliary teachers differ')
        result['next_generation_learning_tested']=True
        if _digest(parent.state_dict())!=parent_state:raise RuntimeError('Matched trial changed parent')
        phase(prefix+'_revocation');manager.revoke(record.record_id,'predeclared exact fallback test')
        a,at,_=_panel(executor,actor,manager,memory,[offset+67000],prefix+'_revoked_raw',config,guided=False)
        b,bt,changes=_panel(executor,actor,manager,memory,[offset+67000],prefix+'_revoked_guided',config,guided=True)
        revoked=np.array_equal(at[0]['actions'],bt[0]['actions']) and a[0].cost==b[0].cost and changes==0
        entry['revocation_restored_baseline']=bool(revoked)
        archive.append('completion_feedback_round',**entry)
        if not revoked:return finish('failed_completion_revocation')
        if not child_ok:return finish('failed_matched_child_improvement')
        # Promotion of the student is an empirical complete deterministic
        # outcome result here; broader/stochastic deployment remains unclaimed.
        parent=children['guided']
        # Protected independent memory retains independent discovery only.
        # Teacher data is retained for future rounds; a family is fixed (0).
        shared=shared+guided_traces
        checkpoint(out,prefix+'_promoted_student',dict(learner=parent.state_dict(),result=entry,
            promotion_claim='finite deterministic Ending panel only',trust_influence=0.),archive)
    phase('completion_end_only_actor_holdout');seeds=list(range(90000,90064))
    final_panels={name:_actor_panel(executor,network,seeds,'completion_final_'+name) for name,network in
        (('parent',actor),('raw',children['raw'].actor),('guided',parent.actor))}
    final_benefit=dict(guided_vs_parent=_comparison(final_panels['parent'],final_panels['guided'],minimum),
        guided_vs_raw=_comparison(final_panels['raw'],final_panels['guided'],minimum))
    final_ok=all(c['improved'] for c in final_benefit.values()) and np.mean([r.capture for r in final_panels['guided']])>=.9
    result['end_only_holdout']=dict(panels={name:_summary(rows) for name,rows in final_panels.items()},
        benefit=final_benefit,used_for_training_or_parent_selection=False)
    if not final_ok:return finish('failed_end_only_actor_holdout')
    return finish('complete',passed=True,negative_retained=False,
        full_task_claim=False,efficiency_claim=False,ordinary_physical_task_tested=False,
        completed_feedback_rounds=len(rounds),final_actor_hash=policy_hash(parent.actor))
