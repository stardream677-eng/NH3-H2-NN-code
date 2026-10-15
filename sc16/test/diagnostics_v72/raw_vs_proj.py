import sys, pandas as pd, numpy as np
df = pd.read_csv(sys.argv[1] + "/v71_qssa_best_oracle_h128/trajectory_states.csv")
pd.set_option('display.width',250)
macro=['H2','H2O','N2','NH3','O2','NO','OH','H']
for tid, steps in [(1,[0,1,2,5,50]),(4,[0,3,5,10,60]),(8,[70,76,77,78,100])]:
    g = df[df.trajectory_id==tid]
    print(f'--- traj {tid}')
    for s in steps:
        r = g[g.step==s].set_index('species')
        print(f' step {s:3d} ' + ' | '.join(f"{sp}: t={r.loc[sp,'true']:.2e} oR={r.loc[sp,'oracle_raw']:.2e} oP={r.loc[sp,'oracle']:.2e}" for sp in ['H2O','NH3','O2','N2']))
    # mass sum of raw vs projected along oracle
    piv_raw = g.pivot(index='step', columns='species', values='oracle_raw')
    piv_p = g.pivot(index='step', columns='species', values='oracle')
    print('  oracle_raw mass sum (min/max):', piv_raw.sum(1).min(), piv_raw.sum(1).max(), ' projected:', piv_p.sum(1).min(), piv_p.sum(1).max())
# H flicker in traj 2
g = df[(df.trajectory_id==2)&(df.species=='H')].set_index('step')
low = g[g.oracle<1e-20]
print('traj2 H: steps with oracle<1e-20:', len(low), ' of', len(g))
print(low[['true','oracle_raw','oracle_clamped','oracle']].head(8).to_string())
g = df[(df.trajectory_id==2)&(df.species=='H2')].set_index('step')
low = g[g.oracle<1e-20]
print('traj2 H2: steps with oracle<1e-20:', len(low))
print(low[['true','oracle_raw','oracle_clamped','oracle']].head(5).to_string())
