import json
import numpy as np

with open('baseline_benchmark_v71.json', 'r', encoding='utf-8') as f:
    b = json.load(f)

with open('retrained_benchmark_v72_audited.json', 'r', encoding='utf-8') as f:
    r = json.load(f)

print("=" * 80)
print("BENCHMARK COMPARISON: Baseline (v71) vs Retrained (v72_cfd_optimized)")
print("=" * 80)
print(f"Metric                      | Baseline (v71) | Retrained (v72) | Change")
print(f"----------------------------|----------------|-----------------|------------")
print(f"Closure Success Rate        | {b['closure_success_rate']:.1%}          | {r['closure_success_rate']:.1%}          | {r['closure_success_rate'] - b['closure_success_rate']:+.1%}")
print(f"Ignited Cases               | {b['ignited_count']}/24           | {r['ignited_count']}/24           | {r['ignited_count'] - b['ignited_count']:+} cases")
print(f"Mean Abs Tend Err           | {b['mean_abs_delta_Tend']:.1f} K        | {r['mean_abs_delta_Tend']:.1f} K        | {r['mean_abs_delta_Tend'] - b['mean_abs_delta_Tend']:+.1f} K")
print(f"Mean NO Relative Err        | {b['mean_no_rel_err']:.1%}          | {r['mean_no_rel_err']:.1%}          | {r['mean_no_rel_err'] - b['mean_no_rel_err']:+.1%}")
print(f"Total Wiped Species         | {b['total_wiped_species']}              | {r['total_wiped_species']}              | 0")
print("=" * 80)

# Per-case comparison
print(f"{'Case Name':<28s} | {'IDT: Base':>9s} {'v72':>7s} {'Ref':>7s} | {'Tend: Base':>10s} {'v72':>7s} {'Ref':>6s} | {'NO Err: Base':>12s} {'v72':>7s}")
print("-" * 105)

b_cases = {c['case']: c for c in b['cases']}
r_cases = {c['case']: c for c in r['cases']}

for name, rc in r_cases.items():
    bc = b_cases.get(name, {})
    b_idt = f"{bc.get('nn_idt_ms'):.3f}" if bc.get('nn_idt_ms') is not None else "nan"
    r_idt = f"{rc.get('nn_idt_ms'):.3f}" if rc.get('nn_idt_ms') is not None else "nan"
    ref_idt = f"{rc.get('ref_idt_ms'):.3f}" if rc.get('ref_idt_ms') is not None else "nan"
    
    b_tend = f"{bc.get('nn_Tend', 0):.0f}"
    r_tend = f"{rc.get('nn_Tend', 0):.0f}"
    ref_tend = f"{rc.get('ref_Tend', 0):.0f}"
    
    b_no = f"{bc.get('no_rel_err', 1.0):.1%}"
    r_no = f"{rc.get('no_rel_err', 1.0):.1%}"
    
    print(f"{name:<28s} | {b_idt:>9s} {r_idt:>7s} {ref_idt:>7s} | {b_tend:>10s} {r_tend:>7s} {ref_tend:>6s} | {b_no:>12s} {r_no:>7s}")
