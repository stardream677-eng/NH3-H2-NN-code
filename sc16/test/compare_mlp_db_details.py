import json

with open('retrained_benchmark_v72_audited.json') as fp:
    b_cfd = json.load(fp)

with open('output1/phase68_runs/v72_dual_branch/benchmark_24cases.json') as fp:
    b_db = json.load(fp)

cases_cfd = {c['case']: c for c in b_cfd['cases']}
cases_db = {c['case']: c for c in b_db['cases']}

print(f"{'Case':<32} | {'Ref Tend':<8} | {'MLP Tend':<8} | {'MLP Ratio':<10} | {'DB Tend':<8} | {'DB Ratio':<10}")
print('-' * 85)
for name, c_cfd in list(cases_cfd.items())[:24]:
    c_db = cases_db.get(name, {})
    ref_t = f"{c_cfd['ref_Tend']:.0f}"
    mlp_t = f"{c_cfd['nn_Tend']:.0f}"
    mlp_idt = f"{c_cfd['idt_ratio']:.2f}" if c_cfd['idt_ratio'] is not None else "None"
    db_t = f"{c_db.get('nn_Tend', 0):.0f}"
    db_idt = f"{c_db.get('idt_ratio'):.2f}" if c_db.get('idt_ratio') is not None else "None"
    print(f"{name:<32} | {ref_t:<8} | {mlp_t:<8} | {mlp_idt:<10} | {db_t:<8} | {db_idt:<10}")
