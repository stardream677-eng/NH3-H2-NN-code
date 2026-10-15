import sys, pandas as pd, numpy as np
base = sys.argv[1]
L = lambda a: np.log10(np.maximum(a, 1e-30))
for ck in ['v71_qssa_best_oracle_h128','v71_qssa_latest']:
    df = pd.read_csv(f"{base}/{ck}/trajectory_states.csv")
    print('=====', ck, 'n_traj', df.trajectory_id.nunique(), 'steps', df.step.max()+1)
    rows=[]
    for tid, g in df.groupby('trajectory_id'):
        p = g.pivot(index='step', columns='species')
        tr = p['true']; pa = p['pure_ar']; orc = p['oracle']
        r = dict(tid=tid)
        for sp in ['NH3','O2','H2O','OH','NO','H','HO2','NH2','N2O']:
            t = tr[sp].values; a = pa[sp].values; o = orc[sp].values
            act = t > 1e-12
            r[f'{sp}_arlog'] = np.sqrt(np.mean((L(a[act])-L(t[act]))**2)) if act.any() else np.nan
            r[f'{sp}_orlog'] = np.sqrt(np.mean((L(o[act])-L(t[act]))**2)) if act.any() else np.nan
        # end-state relative error
        for sp in ['NH3','O2','NO','OH']:
            r[f'{sp}_end_true']=tr[sp].values[-1]; r[f'{sp}_end_ar']=pa[sp].values[-1]
        # OH peak
        r['OHpk_true']=tr['OH'].max(); r['OHpk_ar']=pa['OH'].max(); r['OHpk_or']=orc['OH'].max()
        r['NH3_0']=tr['NH3'].values[0]; r['NH3_end']=tr['NH3'].values[-1]
        rows.append(r)
    R = pd.DataFrame(rows)
    pd.set_option('display.width',250); pd.set_option('display.max_columns',60)
    print(R[['tid']+[c for c in R.columns if c.endswith('_orlog')]].round(2).to_string(index=False))
    print(R[['tid']+[c for c in R.columns if c.endswith('_arlog')]].round(2).to_string(index=False))
    print(R[['tid','NH3_0','NH3_end','NH3_end_ar','NO_end_true','NO_end_ar','OH_end_true','OH_end_ar','OHpk_true','OHpk_or','OHpk_ar']].to_string(index=False, float_format=lambda x:f'{x:.3e}'))
