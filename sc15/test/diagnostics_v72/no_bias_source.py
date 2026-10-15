import sys, numpy as np, pandas as pd
df = pd.read_csv(sys.argv[1] + "/v71_qssa_best_oracle_h128/trajectory_states.csv")
L = lambda a: np.log10(np.maximum(a, 1e-30))
for s in ['NO', 'OH', 'H', 'O', 'H2O', 'O2', 'NH3', 'N2O']:
    d = df[(df.species == s) & (df.step > 0) & (df.true > 1e-10)]
    raw = L(d.oracle_raw) - L(d.true); prj = L(d.oracle) - L(d.true); ar = L(d.pure_ar) - L(d.true)
    print(f"{s:4s} n={len(d):5d} oracle raw: median {np.median(raw):+.3f} mean {np.mean(raw):+.3f} | after projection: median {np.median(prj):+.3f} mean {np.mean(prj):+.3f} | pure-AR median {np.median(ar):+.3f}")
