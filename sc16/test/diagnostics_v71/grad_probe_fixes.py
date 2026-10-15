exec(open(__import__('os').environ["GP"]).read().split("model=make()")[0])
import torch.nn.functional as F
LN10=float(np.log(10))
def losses_fixed(model,xb,yb,mode):
    cur=xb[:,0,:]; nxt=xb[:,1,:]; tgt=yb[:,0,:]
    pr=model(cur); mask=tr.compute_qssa_rate_mask(tools,tgt)
    rl=tr.weighted_mse_loss(pr,tgt,mask=mask)
    ps=tools.predict_next_state(cur,pr,safe_for_rollout=True,state_update_mode="qssa_asymptotic")
    tss=torch.cat([nxt[:,tools.macro_idx],nxt[:,tools.micro_idx]],1)
    if mode=="raw_log":           # state loss on RAW integrator output (smooth, positive), not the projection
        m=ps["Y_macro_next_raw"]; lm=torch.log10(ps["Y_micro_next_raw"].clamp_min(1e-40))
    elif mode=="proj_floor":       # projected state but log taken with a soft floor (gradient bounded)
        m=ps["Y_macro_next_constrained"]; lm=torch.log10(ps["Y_micro_next_constrained"]+1e-30)
    pss=torch.cat([(m-tools.mean_X[tools.macro_idx])/tools.scale_X[tools.macro_idx],(lm-tools.mean_X[tools.micro_idx])/tools.scale_X[tools.micro_idx]],1)
    sl=((pss-tss)**2).mean()
    return rl,sl
def train(name,mode,w_state=1.0,steps=1500,lr=2e-4):
    torch.manual_seed(0); model=make(); opt=torch.optim.AdamW(model.parameters(),lr,weight_decay=1e-5)
    for it in range(steps):
        b=torch.randint(0,len(X),(1024,),device=tr.DEVICE)
        rl,sl=losses_fixed(model,X[b],Y[b],mode); loss=rl+w_state*sl
        opt.zero_grad(); loss.backward(); gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    with torch.no_grad():
        b=torch.arange(0,min(len(X),20000),device=tr.DEVICE); rl,sl=losses_fixed(model,X[b],Y[b],mode)
    print(f"{name:52s} -> rate={rl.item():.3f} state={sl.item():.3f} last grad-norm={gn:.2e}")
    return model
train("FIX-1: state loss on RAW integrator output",       "raw_log")
train("FIX-2: projected state, log10(Y+1e-30)",           "proj_floor")
