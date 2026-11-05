"""Frozen arithmetic only: twenty predeclared old near-upright banks; no native imports."""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import math
import os
import resource
import signal
import sys
import time

import h5py
import numpy as np

OUT = Path(__file__).resolve().parents[1]
ROOT = Path('/Users/jaisel/Documents/workbench/mppi-full-trajectory-pi')
START = time.process_time()
WALL = time.monotonic()

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as s:
        for b in iter(lambda:s.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def cpu():
    child=resource.getrusage(resource.RUSAGE_CHILDREN)
    return time.process_time()-START+child.ru_utime+child.ru_stime

def guard():
    if cpu()>30:raise RuntimeError('Frozen30CPU budget exhausted')
    if sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())>8*1024**2:raise RuntimeError('Frozen8MiB output exhausted')

def weights(c,lam):
    x=np.exp(-(c-c.min())/lam)
    return x/x.sum()

def ess(w):return float(1/np.dot(w,w))

def independent_lambda(c):
    d=float(np.ptp(c));lo,hi=math.log(d*1e-12),math.log(d*1e12)
    for _ in range(120):
        mid=(lo+hi)/2
        if ess(weights(c,math.exp(mid)))<32:lo=mid
        else:hi=mid
    return math.exp((lo+hi)/2)

def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def main():
    cfg=json.loads((OUT/'config.json').read_text());manifest=json.loads((OUT/'manifest.json').read_text())
    assert sha(OUT/'config.json')==manifest['config_sha256']
    for n,b in manifest['files'].items():assert sha(OUT/'source'/n)==b['sha256']
    spec=importlib.util.spec_from_file_location('frozen_math',OUT/'source/src/utils/math.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    roster=cfg['input_roster'][0];raw=ROOT/roster['raw'];assert sha(raw)==roster['sha256']
    rows=[];all_c=[];all_w=[];all_a=[]
    with h5py.File(raw,'r') as f:
        for t in roster['control_indices']:
            guard();g=f[f"{roster['trial']}/planning/control_{t:04d}/pass_000"]
            c=g['candidate_costs'][:];old=g['weights'][:]
            assert c.shape==(256,) and np.isfinite(c).all() and np.ptp(c)>0
            assert g.attrs['lam']==3.0
            w3=weights(c,3.0);assert np.max(np.abs(old-w3))<=1e-10
            lam=independent_lambda(c);stock=module.temperature_for_neff_fraction(c,.125)
            adaptive=weights(c,lam);stockw=module.softmin_weights(c,stock)
            assert abs(ess(adaptive)-32)<=1e-10 and abs(ess(stockw)-32)<=1e-10
            assert np.max(np.abs(adaptive-stockw))<=1e-10
            std=float(np.std(c));gap=float(np.partition(c,1)[1]-c.min())
            rows.append({'control':t,'pass':0,'cost_min':float(c.min()),'cost_max':float(c.max()),'cost_std':std,'cost_range':float(np.ptp(c)),'best_to_second_gap':gap,'fixed_temperature':3.,'fixed_ess':ess(w3),'fixed_top_weight':float(w3.max()),'target_ess':32.,'target_temperature':lam,'stock_temperature':stock,'temperature_over_cost_std':lam/std,'achieved_ess':ess(adaptive),'adaptive_top_weight':float(adaptive.max()),'weight_L1_change':float(np.abs(adaptive-w3).sum()),'weighted_cost_fixed':float(np.dot(w3,c)),'weighted_cost_adaptive':float(np.dot(adaptive,c)),'same_top_candidate':bool(np.argmax(w3)==np.argmax(adaptive)),'maximum_stock_weight_error':float(np.max(np.abs(adaptive-stockw)))})
            all_c.append(c);all_w.append(w3);all_a.append(adaptive)
    with (OUT/'banks.csv').open('x',newline='') as s:
        w=csv.DictWriter(s,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    np.savez_compressed(OUT/'counterfactual_weights.npz',costs=np.asarray(all_c),fixed_weights=np.asarray(all_w),adaptive_weights=np.asarray(all_a),control_indices=np.array(roster['control_indices']))
    os.environ['MPLCONFIGDIR']=str(OUT/'mpl-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.6))
    for i,name in enumerate(('fixed_ess','achieved_ess')):axes[0].plot([r['control'] for r in rows],[r[name] for r in rows],'.-',label=('fixed λ3','ESS-target32')[i])
    axes[0].axhspan(10,50,color='gray',alpha=.1);axes[0].set(xlabel='Recorded control index',ylabel='Effective sample size');axes[0].legend()
    axes[1].scatter([r['cost_std'] for r in rows],[r['target_temperature'] for r in rows]);axes[1].set(xlabel='Candidate cost standard deviation',ylabel='Temperature for ESS32')
    order=np.argsort(all_c[0]);axes[2].plot(np.arange(256),all_w[0][order],label='fixed λ3');axes[2].plot(np.arange(256),all_a[0][order],label='ESS32');axes[2].set(xlabel='First bank, sorted candidate cost rank',ylabel='Weight');axes[2].legend()
    fig.suptitle('Saved-bank reweighting only: weight breadth is not controller competence',fontsize=11);fig.tight_layout();fig.savefig(OUT/'figure.png',dpi=130);plt.close(fig)
    guard()
    result={'status':'PASSED','banks':len(rows),'scope':'Saved arithmetic only, firstnearcase0 controls0,25,...475 pass0; no action or physics prediction','fixed_ess':{'minimum':min(r['fixed_ess'] for r in rows),'median':float(np.median([r['fixed_ess'] for r in rows])),'maximum':max(r['fixed_ess'] for r in rows)},'ESS32_temperatures':{'minimum':min(r['target_temperature'] for r in rows),'median':float(np.median([r['target_temperature'] for r in rows])),'maximum':max(r['target_temperature'] for r in rows)},'cost_std':{'minimum':min(r['cost_std'] for r in rows),'median':float(np.median([r['cost_std'] for r in rows])),'maximum':max(r['cost_std'] for r in rows)},'achieved_ESS_maximum_absolute_error':max(abs(r['achieved_ess']-32) for r in rows),'maximum_stock_weight_error':max(r['maximum_stock_weight_error'] for r in rows),'all_top_candidate_rankings_unchanged':all(r['same_top_candidate'] for r in rows),'recommendation':{'lam_target_neff_fraction':.125,'target_ess':32,'use_existing_core_implementation':True,'absolute_temperature_clip':'none; spread-scaled bracket1e-8 to1e8','warning':'HigherESS changes update averaging but does not establish successful balance. All candidates are physically valid yetnone havequalifiedhold in full priorprobe.'},'cpu_seconds_including_children':cpu(),'wall_seconds':time.monotonic()-WALL,'new_physics_controls':0,'network_forwards':0,'fitting_updates':0}
    write(OUT/'result.json',result)
    write(OUT/'output_manifest.json',{'files':{str(p.relative_to(OUT)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in OUT.rglob('*') if p.is_file() and p.name!='output_manifest.json'},'source_sha256':sha(Path(__file__))})
    print(json.dumps(result))

if __name__=='__main__':
    signal.alarm(120)
    resource.setrlimit(resource.RLIMIT_CPU,(30,resource.RLIM_INFINITY))
    try:main()
    except Exception as e:
        write(OUT/'failure.json',{'status':'FAILED','error':type(e).__name__+': '+str(e),'cpu_seconds':cpu(),'wall_seconds':time.monotonic()-WALL,'no_new_physics':True})
        raise
