import sys, os, importlib.util, glob, numpy as np, torch, joblib
spec=importlib.util.spec_from_file_location('tr','train_sc_v68.py'); tr=importlib.util.module_from_spec(spec); sys.modules['tr']=tr; spec.loader.exec_module(tr)
sys.modules['__main__'].FeatureScaler = tr.FeatureScaler
W=os.environ["W"]; sd=W+"/run_v71/scalers_v68/"
fi=joblib.load(glob.glob(sd+"feature_info_*.pkl")[0]); sx=joblib.load(glob.glob(sd+"scaler_X_*.pkl")[0]); sy=joblib.load(glob.glob(sd+"scaler_Y_*.pkl")[0])
db=W+"/db/trajectories_nuig_2024"; parts=sorted(glob.glob(db+"/part-*.parquet"))
from pathlib import Path
import pandas as pd
train_ids=set(pd.read_csv(W+"/db/trajectory_splits_QSSA/nuig_2024_train_sample_ids.csv")["sample_id"].astype(str))
ds=tr.CoarseOperatorDataset([Path(p) for p in parts],fi,sx,sy,max_horizon=1,shuffle_parts=False,stride=1,seed=0,parts_per_epoch=len(parts),sample_ids=train_ids)
Xs=[];Ys=[]
for x,y in ds:
    Xs.append(x);Ys.append(y)
    if len(Xs)>=60000: break
X=torch.stack(Xs).to(tr.DEVICE); Y=torch.stack(Ys).to(tr.DEVICE)
tools=tr.CoarseOperatorTools(fi,sx,sy,rate_decode_mode="qssa_asymptotic",rate_z_limit=10.0)
torch.manual_seed(0)
def make(): return tr.CoupledKineticsNet(X.shape[-1],Y.shape[-1],output_parameterization="qssa_asymptotic").to(tr.DEVICE)
def losses(model,xb,yb):
    cur=xb[:,0,:]; nxt=xb[:,1,:]; tgt=yb[:,0,:]
    pr=model(cur)
    mask=tr.compute_qssa_rate_mask(tools,tgt)
    rl=tr.weighted_mse_loss(pr,tgt,mask=mask) if 'mask' in tr.weighted_mse_loss.__code__.co_varnames else tr.weighted_mse_loss(pr,tgt)
    ps=tools.predict_next_state(cur,pr,safe_for_rollout=True,state_update_mode="qssa_asymptotic")
    m,lm=tr.select_training_state(ps,"constrained")
    pss=torch.cat([(m-tools.mean_X[tools.macro_idx])/tools.scale_X[tools.macro_idx],(lm-tools.mean_X[tools.micro_idx])/tools.scale_X[tools.micro_idx]],1)
    tss=torch.cat([nxt[:,tools.macro_idx],nxt[:,tools.micro_idx]],1)
    sl=((pss-tss)**2).mean()
    return rl,sl
model=make()
idx=torch.randperm(len(X),device=tr.DEVICE)[:2048]
rl,sl=losses(model,X[idx],Y[idx])
g_r=torch.autograd.grad(rl,list(model.parameters()),retain_graph=True)
g_s=torch.autograd.grad(sl,list(model.parameters()),retain_graph=True,allow_unused=True)
fr=torch.cat([g.flatten() for g in g_r]); fs=torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for g,p in zip(g_s,model.parameters())])
print(f"[init] rate_loss={rl.item():.3f} state_loss={sl.item():.3f} | |grad_rate|={fr.norm():.3e} |grad_state|={fs.norm():.3e} ratio={fs.norm()/fr.norm():.1e} cos={torch.dot(fr,fs)/(fr.norm()*fs.norm()+1e-30):+.3f}")
# train variants for a fixed budget and compare
def train(name,use_state,clip,steps=1500,lr=2e-4):
    torch.manual_seed(0); model=make(); opt=torch.optim.AdamW(model.parameters(),lr,weight_decay=1e-5)
    for it in range(steps):
        b=torch.randint(0,len(X),(1024,),device=tr.DEVICE)
        rl,sl=losses(model,X[b],Y[b]); loss=rl+(sl if use_state else 0.0)
        opt.zero_grad(); loss.backward()
        gn=torch.nn.utils.clip_grad_norm_(model.parameters(),clip if clip else 1e12)
        opt.step()
    with torch.no_grad():
        b=torch.arange(0,min(len(X),20000),device=tr.DEVICE); rl,sl=losses(model,X[b],Y[b])
    print(f"{name:42s} -> rate={rl.item():.3f} state={sl.item():.3f} (last grad-norm before clip {gn:.2e})")
train("rate only, clip=1.0",False,1.0)
train("rate+state (v71), clip=1.0",True,1.0)
train("rate+state, no clip",True,None)
train("rate only, no clip, lr=1e-3",False,None,lr=1e-3)
