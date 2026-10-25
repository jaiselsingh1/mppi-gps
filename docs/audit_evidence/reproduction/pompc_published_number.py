"""Re-run official PO-MPC result analysis, not a training reproduction."""
import hashlib,importlib.util,json,pickle,site,subprocess,sys
from pathlib import Path
site.addsitedir('/private/tmp/mppi-competitor-audit-tools')
repo=Path('/tmp/mppi-literature/code/pompc')
script=repo/'paper_results/data_load_plots.py'
spec=importlib.util.spec_from_file_location('official_plot_analysis',script)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
data_path=repo/'paper_results/pompc_humanoid_bench_plot_data.pkl'
with data_path.open('rb') as f: data=pickle.load(f)
row=data['h1hand-walk-v0']['bcnmpc2_rkl_9-0_final']['data']
values=row.iloc[-1].to_numpy(dtype=float)
point,low,high=module.bootstrap_final_iqm(values,n_boot=1000,ci=95,random_state=0)
result={'scope':'Official released run-data reanalysis; no fresh policy training or evaluation',
'git_head':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
'git_remote':subprocess.check_output(['git','-C',str(repo),'remote','get-url','origin'],text=True).strip(),
'script_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),
'data_sha256':hashlib.sha256(data_path.read_bytes()).hexdigest(),
'task':'H1hand-walk','method':'PO-MPC lambda=9','data_final_step':float(row.index[-1]),
'released_seed_values':values.tolist(),'official_iqm':float(point),'official_ci95':[float(low),float(high)],
'paper_table1':'934 [544, 936]','rounded_result':f'{point:.0f} [{low:.0f}, {high:.0f}]',
'paper_match':f'{point:.0f} [{low:.0f}, {high:.0f}]'=='934 [544, 936]'}
result['repository']=result['git_remote']
assert result['paper_match']
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
