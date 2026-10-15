import sys, glob, importlib.util, joblib, numpy as np
spec=importlib.util.spec_from_file_location('tr','train_sc_v68.py'); tr=importlib.util.module_from_spec(spec); sys.modules['tr']=tr; spec.loader.exec_module(tr)
sys.modules['__main__'].FeatureScaler = tr.FeatureScaler
root="output1/phase68_runs/v71_qssa_asymptotic_exact/scalers_v68"
fi=joblib.load(glob.glob(root+"/feature_info_*.pkl")[0]); sy=joblib.load(glob.glob(root+"/scaler_Y_*.pkl")[0]); sx=joblib.load(glob.glob(root+"/scaler_X_*.pkl")[0])
print(type(sy), [a for a in dir(sy) if not a.startswith('_')])
mean=np.asarray(getattr(sy,'mean_',getattr(sy,'mean',None))); std=np.asarray(getattr(sy,'scale_',getattr(sy,'std',None)))
cols=fi['output_cols']
print('n out', len(cols))
for c,m,s in zip(cols,mean,std):
    if any(k in c for k in ['_NH3','_O2','_H2O','_OH','_H2','_NO','_HO2','_NH2','_N2O']) and not c.endswith(('NO2','NO3','H2O2','NH2OH','H2NO','HNO','N2H2','N2H3','N2H4','HONO','H2NN','NNH')):
        print(f'{c:28s} mean={m:8.2f} std={s:6.2f}  -> RMSE(dex) at norm-MSE 0.02: {np.sqrt(0.02)*s:5.2f}  at 0.04: {np.sqrt(0.04)*s:5.2f}')
print('std median', np.median(std), 'min', std.min(), 'max', std.max())
print('input cols', fi['input_cols'][:10], '...', fi['input_cols'][-3:])
mx=np.asarray(getattr(sx,'mean_',getattr(sx,'mean',None))); sxs=np.asarray(getattr(sx,'scale_',getattr(sx,'std',None)))
for c,m,s in zip(fi['input_cols'],mx,sxs):
    if c in ('T','P','log_dt','logY_OH','logY_H','Y_NH3','Y_O2','logY_HO2','Y_H2O'): print(f'  X {c:10s} mean={m:9.3f} std={s:9.3f}')
