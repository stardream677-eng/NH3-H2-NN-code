"""Connect independent complete trajectories to training's primary selection."""
from pathlib import Path
from functools import lru_cache
import importlib.util
import hashlib,math
import autonomous_benchmark_protocol as protocol

ROOT=Path(__file__).resolve().parent
DEFAULT_CASES=ROOT/'test/benchmark_autonomous_cases.example.json'

@lru_cache(maxsize=1)
def runner_module():
    path=ROOT/'test/benchmark_autonomous_h128.py'
    spec=importlib.util.spec_from_file_location('sc17_complete_runner',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def evaluate(model,tools,mechanism_path,config,cases_path=None):
    runner=runner_module();path=Path(cases_path or DEFAULT_CASES).resolve()
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if config.get('complete_validation_cases_sha256',digest)!=digest:
        raise ValueError('Complete validation cases changed during the experiment')
    cases=runner.load_cases(path)
    reactor_type=config.get('complete_validation_reactor_type')
    if reactor_type is not None:
        cases=[case for case in cases if case['reactor_type']==reactor_type]
        if not cases:raise ValueError('No fixed validation cases match the training reactor type')
    tools._benchmark_state_update_mode=config['state_update_mode']
    tools._benchmark_trust_region_fraction=config['trust_region_fraction']
    tools._benchmark_max_delta_temperature=config['max_delta_temperature']
    was_training=model.training;model.eval()
    records=[]
    try:
        for case in cases:
            try:
                record,_=runner.run_case(case,model,tools,Path(mechanism_path))
            except Exception as exc:
                record=dict(name=case['name'],status='failed',failure_phase='complete_runner',failure_message=f'{type(exc).__name__}: {exc}')
            records.append(record)
    finally:model.train(was_training)
    score=protocol.selection_score(records)
    return dict(protocol=protocol.BENCHMARK_PROTOCOL,cases_path=str(path),
        cases_sha256=digest,reactor_type_filter=reactor_type,
        role='fixed_model_selection_validation; never an independent test set',
        selection_score=score if math.isfinite(score) else None,
        is_valid_candidate=math.isfinite(score),summary=protocol.summarize_cases(records),cases=records)

def gate_primary(window,complete):
    result=dict(window)
    result['window_composite_score']=window['composite_score']
    result['complete_trajectory_validation']=complete
    result['is_valid_candidate']=bool(window['is_valid_candidate'] and complete['is_valid_candidate'])
    if result['is_valid_candidate']:
        result['composite_score']=window['composite_score']+complete['selection_score']
    return result
