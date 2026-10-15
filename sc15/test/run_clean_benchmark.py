import os
import sys
import json
import time
import glob
import numpy as np
from pathlib import Path

# Add current directory and test directory
SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "test"))

import benchmark_cfd_closed_loop as bch

def run_benchmark(target_ckpt_name=None):
    root = "output1/phase68_runs/v72_dual_branch"
    weights_dir = os.path.join(root, "weights_v68")
    
    if target_ckpt_name:
        if os.path.exists(os.path.join(weights_dir, target_ckpt_name)):
            target_ckpt = target_ckpt_name
        elif os.path.exists(target_ckpt_name):
            target_ckpt = os.path.basename(target_ckpt_name)
        else:
            print(f"Error: Specified checkpoint '{target_ckpt_name}' not found in {weights_dir}")
            return
    else:
        # Prioritize oracle avg / composite checkpoints sorted by modification time
        oracle_ckpts = sorted(glob.glob(os.path.join(weights_dir, "best_oracle_avg_*.pth")), key=os.path.getmtime)
        comp_ckpts = sorted(glob.glob(os.path.join(weights_dir, "best_composite_*.pth")), key=os.path.getmtime)
        default_ckpt = os.path.join(weights_dir, "best_v68_net_nuig_2024.pth")
        
        if comp_ckpts:
            target_ckpt = os.path.basename(comp_ckpts[-1])
        elif oracle_ckpts:
            target_ckpt = os.path.basename(oracle_ckpts[-1])
        elif os.path.exists(default_ckpt):
            target_ckpt = os.path.basename(default_ckpt)
        else:
            print("No checkpoint found yet in:", weights_dir)
            return
            
    print(f"Evaluating checkpoint: {target_ckpt} in {root}")
    
    engine = bch.BenchmarkEngine(root, target_ckpt, device="cuda")
    
    results = []
    print(f"\n{'Case':<32} | {'Ref Tend':<8} | {'NN Tend':<8} | {'dT':<8} | {'Ref IDT':<10} | {'NN IDT':<10} | {'IDT Ratio':<10} | {'NO RelErr':<10}")
    print("-" * 115)
    
    ignited_cases = 0
    valid_ratios = []
    abs_delta_tends = []
    no_errors = []
    
    for case in bch.BENCHMARK_CASES:
        res = bch.run_single_case(engine, case, dt=2e-6, closure="UV", projection_mode="kl")
        results.append(res)
        
        cname = case["name"]
        ref_t = f"{res['ref_Tend']:.0f}"
        nn_t = f"{res['nn_Tend']:.0f}"
        dt_val = f"{res['delta_Tend']:+.0f}"
        
        ref_idt = f"{res['ref_idt_ms']:.2f}ms" if res['ref_idt_ms'] is not None else "None"
        nn_idt = f"{res['nn_idt_ms']:.2f}ms" if res['nn_idt_ms'] is not None else "None"
        ratio = f"{res['idt_ratio']:.2f}" if res['idt_ratio'] is not None else "None"
        no_err = f"{res['no_rel_err']*100:.1f}%" if res['no_rel_err'] is not None else "None"
        
        if res['nn_idt_ms'] is not None:
            ignited_cases += 1
        if res['idt_ratio'] is not None:
            valid_ratios.append(res['idt_ratio'])
        abs_delta_tends.append(abs(res['delta_Tend']))
        no_errors.append(res['no_rel_err'])
        
        print(f"{cname:<32} | {ref_t:<8} | {nn_t:<8} | {dt_val:<8} | {ref_idt:<10} | {nn_idt:<10} | {ratio:<10} | {no_err:<10}")
        
    valid_no_errors = [
        float(value) for value in no_errors
        if value is not None and np.isfinite(value)
    ]
    valid_abs_delta_tends = [
        float(value) for value in abs_delta_tends if np.isfinite(value)
    ]
    mean_no_error = float(np.mean(valid_no_errors)) if valid_no_errors else None
    mean_abs_delta_tend = (
        float(np.mean(valid_abs_delta_tends)) if valid_abs_delta_tends else None
    )

    summary = {
        "checkpoint": target_ckpt,
        "closure": "UV",
        "projection": "kl",
        "dt": 2e-6,
        "num_cases": len(bch.BENCHMARK_CASES),
        "ignited_count": ignited_cases,
        "mean_idt_ratio": float(np.mean(valid_ratios)) if valid_ratios else None,
        "median_idt_ratio": float(np.median(valid_ratios)) if valid_ratios else None,
        "mean_abs_delta_Tend": mean_abs_delta_tend,
        "mean_no_rel_err": mean_no_error,
        "cases": results,
    }
    
    out_json = os.path.join(root, "clean_benchmark_24cases.json")
    with open(out_json, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"\n>> Benchmark completed! Saved to {out_json}")
    print(f">> Ignited: {ignited_cases}/24 cases")
    if valid_ratios:
        print(f">> Mean IDT ratio: {np.mean(valid_ratios):.3f}, Median: {np.median(valid_ratios):.3f}")
    if mean_abs_delta_tend is not None:
        print(f">> Mean Abs delta Tend: {mean_abs_delta_tend:.1f} K")
    else:
        print(">> Mean Abs delta Tend: N/A")
    if mean_no_error is not None:
        print(f">> Mean NO error: {mean_no_error*100:.1f}%")
    else:
        print(">> Mean NO error: N/A")

if __name__ == "__main__":
    ckpt_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_benchmark(ckpt_arg)
