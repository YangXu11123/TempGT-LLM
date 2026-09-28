"""Sequential experiment-level resume. No within-training checkpoint resume."""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from asset_provenance import capture_assets, file_hash
from config import ARTIFACT_ROOT, WORKSPACE_ROOT, load_config
from experiment_suite import aggregate, METRICS

CODE = Path(__file__).resolve().parent


@contextmanager
def idle_stage_guard():
    locks = []
    try:
        for name in ('.new_exe_sequence.lock', '.new_exe_pipeline.lock'):
            handle = (Path(WORKSPACE_ROOT)/name).open('a')
            locks.append(handle)
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('A project stage is still running/waiting; refusing to modify its outputs. Check its log before restarting.')
        yield
    finally:
        for handle in reversed(locks):
            handle.close()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name+'.tmp')
    with temp.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def scoped(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError(f'Path outside expected suite scope: {path}')
    return path


def verify_config(effective, config):
    if any(effective.get(key) != value for key, value in config.items() if key != 'device'):
        raise ValueError('Effective run configuration differs from prepared configuration')


def execution_identity(root, manifest):
    code = {path.name: file_hash(path) for path in sorted(CODE.glob('*.py'))}
    scripts = {path.name: file_hash(path) for path in sorted(CODE.glob('*.sh'))}
    configs = {}
    backbones = {}
    backbone_inventory = {}
    if len(manifest['runs']) != 25:
        raise ValueError('Expected exactly 25 prepared runs')
    for run in manifest['runs']:
        out = scoped(run['output_dir'], root)
        cfg_path = scoped(run['config_file'], out)
        cfg = read_json(cfg_path)
        if Path(cfg['output_dir']).resolve() != out or cfg['training_seed'] != run['training_seed']:
            raise ValueError('Manifest/config mismatch')
        if file_hash(out/'dataset_split.json') != manifest['split_sha256']:
            raise ValueError('Fixed split changed: '+str(out))
        configs[str(cfg_path.relative_to(root))] = file_hash(cfg_path)
        backbones[cfg['llm_model_path']] = file_hash(Path(cfg['llm_model_path'])/'config.json')
        if cfg['llm_model_path'] not in backbone_inventory:
            backbone_inventory[cfg['llm_model_path']] = [
                [file.name, file.stat().st_size, file.stat().st_mtime_ns]
                for file in sorted(Path(cfg['llm_model_path']).iterdir()) if file.is_file()]
    assets = capture_assets(load_config(manifest['runs'][0]['config_file']))
    versions = {'python': platform.python_version()}
    for name in ('torch', 'transformers', 'peft', 'numpy', 'scipy', 'scikit-learn', 'sentencepiece'):
        versions[name] = importlib.metadata.version(name)
    label_path = Path(read_json(manifest['runs'][0]['config_file'])['original_graph_data_path'])
    label_stat = label_path.stat()
    return {'code_sha256': code, 'script_sha256': scripts, 'config_sha256': configs,
            'manifest_sha256': file_hash(root/'suite_manifest.json'),
            'backbone_config_sha256': backbones, 'backbone_file_inventory': backbone_inventory,
            'label_file_stat': [str(label_path), label_stat.st_size, label_stat.st_mtime_ns],
            'versions': versions, 'assets': assets}


def marker(path, out, identity_hash, files):
    atomic_json(path, {'identity_sha256': identity_hash,
                      'completed_at': datetime.now().astimezone().isoformat(),
                      'files': {str(file.relative_to(out)): file_hash(file) for file in files}})


def verify_marker(path, out, identity_hash):
    value = read_json(path)
    if value['identity_sha256'] != identity_hash or not value['files']:
        raise ValueError('Completion marker identity mismatch: '+str(path))
    for relative, digest in value['files'].items():
        file = scoped(out/relative, out)
        if not file.is_file() or file_hash(file) != digest:
            raise ValueError('Completed artifact changed or missing: '+str(file))


def resolve_identity(root, manifest, *, commit):
    """Keep the first 11 completed runs under their original code identity.

    The only permitted migration is the verified low-memory loader used by
    subsequent runs. Existing completion markers are never rewritten.
    """
    current = execution_identity(root, manifest)
    old_path = root/'suite_execution_identity.json'
    if not old_path.exists():
        if commit:
            atomic_json(old_path, current)
        return current, None, 0
    old = read_json(old_path)
    if old == current:
        return current, None, 0
    if {key: value for key, value in old.items() if key != 'code_sha256'} != \
       {key: value for key, value in current.items() if key != 'code_sha256'}:
        raise ValueError('Suite configuration, scripts, assets, or environment changed')
    changed = {name for name in current['code_sha256']
               if current['code_sha256'][name] != old['code_sha256'].get(name)}
    if set(current['code_sha256']) != set(old['code_sha256']) or \
       changed != {'train.py', 'data_loader.py', 'run_suite.py'}:
        raise ValueError(f'Unexpected suite code change: {changed}')
    old_hash = hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest()
    current_hash = hashlib.sha256(json.dumps(current, sort_keys=True).encode()).hexdigest()
    legacy_count = 11
    for index, run in enumerate(manifest['runs'], 1):
        out = scoped(run['output_dir'], root)
        marker_path = out/'.experiment_complete.json'
        if index <= legacy_count:
            if not marker_path.is_file():
                raise ValueError(f'Legacy completed run {index} is missing')
            verify_marker(marker_path, out, old_hash)
            if training_files(out, read_json(run['config_file']), old['code_sha256']) is None:
                raise ValueError(f'Legacy training provenance invalid: {index}')
        elif marker_path.exists():
            verify_marker(marker_path, out, current_hash)
    record = {'legacy_identity_sha256': old_hash, 'current_identity_sha256': current_hash,
              'legacy_completed_count': legacy_count,
              'reason': 'float32 train/test residency and lazy validation augmentation; no model or experiment configuration change'}
    new_path = root/'suite_execution_identity_lowmem.json'
    if new_path.exists():
        if read_json(new_path) != {'identity': current, 'migration': record}:
            raise ValueError('Low-memory suite migration record changed')
    elif commit:
        atomic_json(new_path, {'identity': current, 'migration': record})
    return current, old_hash, legacy_count


def training_files(out, config, code):
    files = [out/'config.json', out/'runtime_metadata.json', out/'training_summary.json',
             out/'fold_1/fold_summary.json', out/'fold_1/fold_1_best_model.pth']
    if any(not file.is_file() or not file.stat().st_size for file in files):
        return None
    try:
        effective, runtime, total, fold = [read_json(file) for file in files[:4]]
    except (ValueError, OSError):
        return None
    verify_config(effective, config)
    verify_config(total['config'], config)
    if runtime['training_seed'] != config['training_seed'] or runtime['code_sha256'] != code:
        raise ValueError('Training provenance mismatch: '+str(out))
    if fold['num_epochs'] != config['num_epochs'] or total['num_runs'] != 1:
        return None
    theta = fold.get('best_thresholds', {}).get('theta')
    if theta is None or not math.isfinite(theta) or not 0 <= theta <= 1:
        return None
    if not math.isfinite(fold['best_val_loss']):
        return None
    if Path(fold['best_model_path']).resolve() != files[-1].resolve():
        raise ValueError('Unexpected checkpoint path')
    return files


def inference_files(out, config):
    summary_path = out/'fold_1/ratio_seed_eval/ratio_seed_aggregate_summary.json'
    if not summary_path.is_file():
        return None
    try:
        summary = read_json(summary_path)
    except (ValueError, OSError):
        return None
    verify_config(summary['config'], config)
    ratios, seeds = config['evaluation_ratios'], config['evaluation_seeds']
    if summary['ratio_list'] != ratios or summary['seed_list'] != seeds:
        raise ValueError('Evaluation settings mismatch')
    expected = {(ratio, seed) for ratio in ratios for seed in seeds}
    records = summary.get('all_run_records', [])
    if len(records) != len(expected) or {(item['ratio'], item['seed']) for item in records} != expected:
        return None
    files = [summary_path]
    for ratio in ratios:
        record = summary['ratio_aggregate'].get(f'1:{ratio}', {})
        if record.get('num_runs') != len(seeds):
            return None
        if any(not math.isfinite(float(record['metrics'][key]['mean'])) for key in METRICS):
            return None
    for record in records:
        folder = out/f"fold_1/ratio_seed_eval/ratio_1to{record['ratio']}/seed_{record['seed']}"
        result_path = folder/'inference_test_users.json'
        required = [result_path, folder/'user_classification_details.csv', folder/'score_statistics.json']
        if any(not file.is_file() or not file.stat().st_size for file in required):
            return None
        try:
            result = read_json(result_path)
            read_json(required[-1])
        except (ValueError, OSError):
            return None
        if result['seed'] != record['seed'] or result['ratio_normal_multiplier'] != record['ratio']:
            raise ValueError('Per-sample evaluation identity mismatch')
        if result['num_normal'] != record['ratio']*result['num_malicious']:
            return None
        size = result['num_normal']+result['num_malicious']
        if any(len(result[key]) != size for key in ('node_ids', 's_gen', 'y_true', 'y_pred', 'cee_raw')):
            return None
        if any(not math.isfinite(float(record[key])) for key in METRICS):
            return None
        files.extend(required)
    full_summary = scoped(summary['cee_full_test_summary_path'], out)
    if not full_summary.is_file():
        return None
    try:
        read_json(full_summary)
    except (ValueError, OSError):
        return None
    files.append(full_summary)
    if config['lambda_cee'] == 0:
        abc, image = out/'fold_1/cee_groups_abc.json', out/'fold_1/cee_groups_abc_cdf.png'
        if not abc.is_file() or not image.is_file() or not image.stat().st_size:
            return None
        try:
            tests = read_json(abc)['pairwise_mann_whitney_u']
        except (ValueError, OSError):
            return None
        if len(tests) != 3 or any('p_value' not in value or 'effect_size_r_approx' not in value for value in tests.values()):
            return None
        files.extend([abc, image])
    return files


def archive_partial(out):
    protected = {'run_config.json', 'dataset_split.json', '.attempts'}
    candidates = [file for file in out.iterdir() if file.name not in protected]
    if not candidates:
        return
    archive = scoped(out/'.attempts'/datetime.now().strftime('%Y%m%d_%H%M%S_%f'), out)
    archive.mkdir(parents=True)
    for file in candidates:
        source, destination = scoped(file, out), scoped(archive/file.name, out)
        source.rename(destination)
    print('Archived interrupted training outputs:', archive, flush=True)


def run_stage(gpu, stage, run, root):
    config = run['config_file']
    log = root/'logs'/f"{run['experiment']}_seed{run['training_seed']}_{stage}_{datetime.now():%Y%m%d_%H%M%S}.log"
    log.parent.mkdir(exist_ok=True)
    environment = dict(os.environ, NEW_EXE_WAIT_FOR_GPU='1', NEW_EXE_CONFIG_FILE=config)
    subprocess.run([str(CODE/'run_sequence.sh'), str(gpu), str(log), stage], env=environment, check=True)


def execute(root, gpu, stage_runner=run_stage):
    with idle_stage_guard():
        pass
    root = Path(root).resolve()
    scoped(root, WORKSPACE_ROOT)
    manifest = read_json(root/'suite_manifest.json')
    identity, legacy_hash, legacy_count = resolve_identity(root, manifest, commit=True)
    identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    for index, run in enumerate(manifest['runs'], 1):
        out = scoped(run['output_dir'], root)
        config = read_json(run['config_file'])
        complete, trained = out/'.experiment_complete.json', out/'.training_complete.json'
        print(f"[{index}/25] {run['experiment']} seed={run['training_seed']}", flush=True)
        if complete.exists():
            verify_marker(complete, out, legacy_hash if index <= legacy_count else identity_hash)
            print('SKIP: completed experiment verified', flush=True)
            continue
        if trained.exists():
            verify_marker(trained, out, identity_hash)
        train = training_files(out, config, identity['code_sha256'])
        if train is None:
            if trained.exists():
                raise ValueError('Marked training is no longer valid')
            with idle_stage_guard():
                archive_partial(out)
            print('TRAIN: starting this run from the beginning', flush=True)
            stage_runner(gpu, 'train', run, root)
            train = training_files(out, config, identity['code_sha256'])
            if train is None:
                raise RuntimeError('Training exited but required outputs are incomplete')
        marker(trained, out, identity_hash, train)
        infer = inference_files(out, config)
        if infer is None:
            print('INFERENCE: reusing completed training', flush=True)
            stage_runner(gpu, 'inference', run, root)
            infer = inference_files(out, config)
            if infer is None:
                raise RuntimeError('Inference exited but required outputs are incomplete')
        marker(complete, out, identity_hash, train+infer)
        print('DONE: experiment completion committed', flush=True)
    aggregate(root)
    print('ALL 25 EXPERIMENTS COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--check', action='store_true', help='Read-only suite/configuration check; never starts stages')
    parser.add_argument('--root', default=str(Path(ARTIFACT_ROOT)/'experiments_method2_v1'))
    args = parser.parse_args()
    if args.check:
        root = Path(args.root).resolve()
        scoped(root, WORKSPACE_ROOT)
        resolve_identity(root, read_json(root/'suite_manifest.json'), commit=False)
        print('SUITE_CHECK_OK: 25 run configurations; no training started.')
        raise SystemExit(0)
    if args.gpu is None or args.gpu < 0:
        parser.error('--gpu must be a nonnegative physical GPU index')
    # Suite lock prevents duplicate batches; stage locks guard ordinary sequences.
    lock_path = Path(WORKSPACE_ROOT)/'.new_exe_suite.lock'
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another experiment suite is already running; refusing duplicate launch.')
        execute(Path(args.root), args.gpu)
