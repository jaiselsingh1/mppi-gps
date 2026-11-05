import json,pathlib,xml.etree.ElementTree as ET,hashlib,time,resource,math
import numpy as np
ROOT=pathlib.Path.cwd(); OUT=ROOT/'runs/2026-11-05_acrobot_external_reference_001/model_review_001'
start=time.process_time(); resource.setrlimit(resource.RLIMIT_CPU,(15,15))
def canonical(node):return [node.tag,sorted(node.attrib.items()),[canonical(c) for c in node]]
ours=ET.parse(ROOT/'assets/acrobot.xml').getroot(); upstream=ET.parse(OUT/'upstream/dm_control/suite/acrobot.xml').getroot()
sensors=[canonical(s) for s in ours.findall('sensor')]
for s in ours.findall('sensor'):ours.remove(s)
energy=json.loads((ROOT/'runs/2026-11-05_acrobot_energy_discovery_001/preflight/energy_validation.json').read_text())
lqr=json.loads((ROOT/'runs/2026-11-05_acrobot_attractor_001/recovered-lqr/lqr.json').read_text())
p=energy['model_parameters']; low=p['I2']+.25; M=np.array([[p['I1']+.25+low+2,low+.5],[low+.5,low]])
G=p['gravity']*np.array([[2,.5],[.5,.5]]);D=.05*np.eye(2)
A=np.block([[np.zeros((2,2)),np.eye(2)],[np.linalg.solve(M,G),np.linalg.solve(M,-D)]]);B=np.r_[np.zeros((2,1)),np.linalg.solve(M,[[0],[2]])]
cfg=json.loads((ROOT/'runs/2026-11-05_acrobot_attractor_001/near/config.json').read_text());starts=[]
for s in cfg['lqr']['historical_starts']:
 x=np.array(s['initial_state'][1:]);raw=float((-np.array(lqr['K'])@x).item());q0,q1=x[:2]
 starts.append({'reset_seed':s['reset_seed'],'state':s['initial_state'],'initial_lqr_normalized_unclipped':raw,'initial_lqr_physical_torque_unclipped_nm':2*raw,'shoulder_gravity_nm':-9.81*(1.5*math.sin(q0)+.5*math.sin(q0+q1)),'elbow_gravity_nm':-4.905*math.sin(q0+q1)})
manifest=json.loads((OUT/'upstream_manifest.json').read_text());common=[]
for name in ['visual.xml','skybox.xml','materials.xml']:
 a=ROOT/'assets/common'/name;b=OUT/'upstream/dm_control/suite/common'/name
 common.append({'name':name,'bytes_equal':a.read_bytes()==b.read_bytes(),'ours_sha256':hashlib.sha256(a.read_bytes()).hexdigest(),'upstream_sha256':hashlib.sha256(b.read_bytes()).hexdigest()})
result={'upstream_commit':manifest['commit'],'physical_xml_structurally_equal_after_removing_local_sensor':canonical(ours)==canonical(upstream),'removed_local_sensors':sensors,'common_assets':common,'compiled_parameters_source':'runs/2026-11-05_acrobot_energy_discovery_001/preflight/energy_validation.json','parameters':p,'M_upright':M.tolist(),'gravity_stiffness_upright':G.tolist(),'continuous_A':A.tolist(),'continuous_B_normalized_input':B.tolist(),'upright_energy_j':58.86,'hanging_energy_j':19.62,'energy_raise_j':39.24,'maximum_elbow_torque_nm':2,'static_elbow_angle_necessary_bound_rad':math.asin(2/4.905),'historical_starts_lqr_demand':starts,'new_physics_controls':0,'new_native_forwards':0,'analysis_cpu_seconds':time.process_time()-start}
with (OUT/'model_evidence.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
