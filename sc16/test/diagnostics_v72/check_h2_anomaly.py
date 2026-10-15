import sys, pandas as pd, numpy as np
df = pd.read_csv(sys.argv[1] + "/v71_qssa_best_oracle_h128/trajectory_states.csv")
pd.set_option('display.width',250); pd.set_option('display.max_rows',100)
g = df[(df.trajectory_id==2)&(df.step==9)].set_index('species')
print(g[['true','oracle_raw','oracle_clamped','oracle','pure_ar_raw','pure_ar']].to_string(float_format=lambda x:f'{x:.3e}'))
print('sum true', g.true.sum(), 'sum raw', g.oracle_raw.sum(), 'sum proj', g.oracle.sum())
g4 = df[(df.trajectory_id==4)&(df.step==0)].set_index('species')
print('traj4 inert:', g4.loc[['AR','HE'],'true'].to_dict() if 'AR' in g4.index else 'no AR', 'sum', g4.true.sum())
# how often H2 projected << raw across all trajectories
d = df[df.species=='H2']
bad = d[(d.oracle < 1e-3*d.oracle_raw)]
print('H2 rows where projection kills >99.9%:', len(bad), 'of', len(d), ' traj ids', bad.trajectory_id.unique())
d = df[df.species=='H']
bad = d[(d.oracle_raw < 1e-20) & (d.true>1e-8)]
print('H rows where raw integrator output <1e-20 while true>1e-8:', len(bad), bad.trajectory_id.unique())
