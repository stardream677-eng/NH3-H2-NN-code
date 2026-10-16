from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import train_sc_v68 as tr
torch.set_num_threads(4)

@pytest.fixture(scope='session')
def chemistry():
    # The old scaler deliberately reproduces the diagnosed float32 cancellation.
    setattr(sys.modules['__main__'],'FeatureScaler',tr.FeatureScaler)
    old=ROOT.parent/'sc15/phase68_runs/v72_dual_branch/scalers_v68'
    sx=joblib.load(next(old.glob('scaler_X*')))
    sy=joblib.load(next(old.glob('scaler_Y*')))
    fi=joblib.load(next(old.glob('feature_info*')))
    fi.update(mechanism='nuig_2024',mechanism_path=str(ROOT/'NUIG2024.yaml'),
              transform_signature=tr.transform_signature_for('dual_branch'))
    ds=joblib.load(ROOT/'data_pipeline/mini_ds.pkl');rows=ds['rows']
    cols={k:np.array([r[k] for r in rows]) for k in ['T','P','dt']}
    cols['T_next']=np.array([r['Tn'] for r in rows])
    for i,sp in enumerate(ds['names']):
        for k,prefix in [('Y','Y_'),('Yn','Y_next_'),('Pq','P_avg_'),('Dq','D_coef_avg_')]:
            cols[prefix+sp]=np.array([r[k][i] for r in rows])
        cols['dYdt_'+sp]=(cols['Y_next_'+sp]-cols['Y_'+sp])/cols['dt']
    df=pd.DataFrame(cols)
    tools=tr.CoarseOperatorTools(fi,sx,sy,device='cpu')
    inp=tr.transform_input_array(df,fi['inert_species'],fi['macro_species'],fi['micro_species'])
    target=tr.transform_target_array(df,fi['macro_species'],fi['micro_species'],fi['macro_rate_scale'],'dual_branch')
    x=torch.tensor(sx.transform(inp),dtype=torch.float32)
    y=torch.tensor(sy.transform(target),dtype=torch.float32)
    return tools,x,y,df,fi
