"""Bounded subdivision with fresh model evaluations at each physical substep."""
import math
import torch
from physical_projection_contract import ProjectionConstraintError

MAX_SUBDIVISIONS=4

def recoverable(exc):
    if isinstance(exc,ProjectionConstraintError):
        return exc.reason in ('not_converged','optimizer_failed','fallback_failed')
    message=str(exc).lower()
    return isinstance(exc,RuntimeError) and any(s in message for s in ('energy','temperature','autonomous pressure'))

def advance_autonomous(model,tools,current,closure,state_update_mode,
                       trust_region_fraction=0.,max_delta_temperature=1000.,
                       base_next=None,max_subdivisions=MAX_SUBDIVISIONS):
    if not isinstance(max_subdivisions,int) or not 0<=max_subdivisions<=8:
        raise ValueError('max_subdivisions must be an integer in [0,8]')
    info={'accepted_substeps':0,'model_evaluations':0,'rejected_steps':0}
    original_dt_column=current[:,tools.dt_idx].clone()
    def step(state,depth):
        info['model_evaluations']+=1
        try:
            predicted=model(state)
            pred=tools.predict_next_state(state,predicted,safe_for_rollout=True,
                state_update_mode=state_update_mode,trust_region_fraction=trust_region_fraction)
            nxt=tools.state_to_input_scaled(state,pred['Y_macro_next_constrained'],
                pred['logY_micro_next_constrained'],temperature_mode='autonomous',
                thermo_closure=closure,current_X_scaled=state,
                max_delta_temperature=max_delta_temperature)
            info['accepted_substeps']+=1
            return nxt
        except (RuntimeError,FloatingPointError) as exc:
            if depth>=max_subdivisions or not recoverable(exc):raise
            info['rejected_steps']+=1
            half=state.clone()
            half[:,tools.dt_idx]=state[:,tools.dt_idx]-math.log10(2.)/tools.scale_X[tools.dt_idx]
            middle=step(half,depth+1)
            # The second half starts from the first half's actual T/P/Y.
            end=step(middle,depth+1).clone()
            end[:,tools.dt_idx]=state[:,tools.dt_idx]
            return end
    nxt=step(current,0)
    result=nxt.clone()
    result[:,tools.dt_idx]=(base_next[:,tools.dt_idx] if base_next is not None else original_dt_column)
    return result,info
