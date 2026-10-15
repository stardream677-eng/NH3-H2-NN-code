import sys, pandas as pd, numpy as np
base = sys.argv[1]
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500); pd.set_option('display.max_columns', 30)
for ck in ['v71_qssa_best_oracle_h128','v71_qssa_latest','v71_qssa_best_robust_raw','v71_qssa_best_total']:
    df = pd.read_csv(f"{base}/{ck}/physical_metrics.csv")
    print('=====', ck, df['scope'].unique(), df['mode'].unique())
    for scope in ['constrained']:
        d = df[(df.scope==scope)]
        if d.empty: d = df[df.scope==df.scope.unique()[0]]
        piv = d[d.horizon.isin([1,8,32,128])].pivot_table(index='species', columns=['mode','horizon'], values='active_log_rmse')
        if ck=='v71_qssa_best_oracle_h128':
            print(piv.round(2).to_string())
        sub = d[d.species.isin(['NH3','O2','H2','H2O','OH','H','O','HO2','NH2','NO','N2O','NO2'])]
        print(sub[sub.horizon.isin([1,128])][['mode','horizon','species','rmse','log_rmse','active_log_rmse','floor_hit_ratio']].to_string(index=False))
