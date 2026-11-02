from __future__ import annotations
import argparse
import json
import math
import multiprocessing as mp
from multiprocessing.util import Finalize
import os
import queue
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import sparse
from validate_pbmc_branch import validate_cache_artifacts
from pbmc_gurobi_session import start_gurobi_environment
_A_SUB: Optional[np.ndarray] = None
_Y_SUB: Optional[np.ndarray] = None
_CFG: Dict[str, Any] = {}
_ENV = None
_GP = None
_GRB = None

def _dispose_environment() -> None:
    global _ENV
    (env, _ENV) = (_ENV, None)
    if env is not None:
        env.dispose()

def _initialize_single_environment() -> Dict[str, Any]:
    global _ENV, _GP, _GRB
    import gurobipy as gp
    from gurobipy import GRB
    (_GP, _GRB) = (gp, GRB)
    (_ENV, metadata) = start_gurobi_environment(gp)
    return metadata

def _worker_init(ready_queue, cfg: Dict[str, Any]) -> None:
    global _CFG, _ENV, _GP, _GRB
    _CFG = cfg
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    env = None
    try:
        import gurobipy as gp
        from gurobipy import GRB
        _GP = gp
        _GRB = GRB
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()
    except Exception as exc:
        if env is not None:
            try:
                env.dispose()
            except Exception:
                pass
        status = {'pid': os.getpid(), 'error_type': type(exc).__name__}
        error_code = getattr(exc, 'errno', None)
        if isinstance(error_code, int):
            status['error_code'] = error_code
        ready_queue.put(status)
        return
    _ENV = env
    Finalize(None, _dispose_environment, exitpriority=10)
    ready_queue.put({'pid': os.getpid()})

def _run_parallel(tasks, workers: int, args: argparse.Namespace, out_dir: Path, pool_callback=None):
    ctx = mp.get_context('fork')
    children_before = {child.pid for child in mp.active_children()}
    ready_queue = None
    pool = None
    ready_pids = set()
    statuses = []
    startup_complete = False
    startup_t0 = time.perf_counter()
    deadline = startup_t0 + float(args.worker_startup_timeout_sec)
    failure_reason = 'pool_construction_failed'
    try:
        ready_queue = ctx.Queue()
        pool = ctx.Pool(processes=workers, initializer=_worker_init, initargs=(ready_queue, _CFG))
        original_workers = list(pool._pool)
        original_pids = {process.pid for process in original_workers}
        while len(ready_pids) < workers:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                failure_reason = 'worker_startup_timeout'
                raise TimeoutError('Parallel Gurobi worker startup exceeded its total deadline')
            if any((process.exitcode is not None for process in original_workers)) or {process.pid for process in pool._pool} != original_pids:
                failure_reason = 'worker_exited_during_startup'
                raise RuntimeError('A parallel Gurobi worker exited during startup')
            try:
                status = ready_queue.get(timeout=min(remaining, 0.2))
            except queue.Empty:
                continue
            statuses.append(status)
            if 'error_type' in status:
                failure_reason = 'worker_environment_startup_failed'
                raise RuntimeError('Parallel Gurobi environment startup failed; see parallel_startup_failure.json')
            ready_pids.add(int(status['pid']))
        worker_startup_sec = time.perf_counter() - startup_t0
        startup_complete = True
        lp_t0 = time.perf_counter()
        if pool_callback is None:
            results = list(pool.imap_unordered(_solve_one_class, tasks, chunksize=1))
        else:
            results = pool_callback(pool, worker_startup_sec, tuple(sorted(ready_pids)))
        lp_pool_wall_sec = time.perf_counter() - lp_t0
        pool.close()
        pool.join()
        return (results, worker_startup_sec, lp_pool_wall_sec)
    except BaseException as exc:
        if not startup_complete:
            failure = {'status': 'failed', 'reason': failure_reason, 'exception_type': type(exc).__name__, 'n_workers_requested': int(args.workers), 'n_workers_effective': int(workers), 'n_workers_ready': len(ready_pids), 'worker_startup_timeout_sec': float(args.worker_startup_timeout_sec), 'startup_elapsed_sec': float(time.perf_counter() - startup_t0), 'worker_statuses': statuses, 'tasks_dispatched': False, 'serial_fallback': False}
            (out_dir / 'parallel_startup_failure.json').write_text(json.dumps(failure, indent=2))
        raise
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
        if pool is None:
            children = [child for child in mp.active_children() if child.pid not in children_before]
            for child in children:
                child.terminate()
            for child in children:
                child.join(timeout=2.0)
                if child.is_alive():
                    child.kill()
                    child.join()
        if ready_queue is not None:
            ready_queue.close()
            ready_queue.join_thread()

def _apply_fast_params(model) -> None:
    cfg = _CFG
    model.Params.OutputFlag = 0
    model.Params.Method = 2
    model.Params.Threads = int(cfg.get('gurobi_threads', 1))
    model.Params.Crossover = 0
    model.Params.BarConvTol = float(cfg['bar_conv_tol'])
    model.Params.FeasibilityTol = float(cfg['feasibility_tol'])
    model.Params.OptimalityTol = float(cfg['optimality_tol'])
    model.Params.Presolve = int(cfg['presolve'])
    model.Params.Aggregate = int(cfg['aggregate'])
    model.Params.PreSparsify = int(cfg['pre_sparsify'])
    model.Params.PreDual = int(cfg['pre_dual'])
    model.Params.BarHomogeneous = 0
    model.Params.BarCorrectors = int(cfg['bar_correctors'])
    if float(cfg['time_limit_sec']) > 0:
        model.Params.TimeLimit = float(cfg['time_limit_sec'])
    if int(cfg['bar_iter_limit']) > 0:
        model.Params.BarIterLimit = int(cfg['bar_iter_limit'])
    if float(cfg['mem_limit_gb']) > 0:
        model.Params.MemLimit = float(cfg['mem_limit_gb'])

def _solve_one_class(task: Tuple[int, str, int, int]) -> Dict[str, Any]:
    global _A_SUB, _Y_SUB, _ENV, _GP, _GRB
    (rank, cls, train_count, n_train) = task
    assert _A_SUB is not None and _Y_SUB is not None
    (gp, GRB) = (_GP, _GRB)
    t_total = time.perf_counter()
    y_pm = np.where(_Y_SUB == cls, 1.0, -1.0)
    pos = int(np.sum(y_pm == 1.0))
    neg = int(len(y_pm) - pos)
    result: Dict[str, Any] = {'class_label': cls, 'positive_label': cls, 'selected_class_rank_by_train_count': int(rank), 'pos_count_train': int(train_count), 'pos_fraction_train': float(train_count / max(n_train, 1)), 'pos_count_mu_subset': pos, 'neg_count_mu_subset': neg, 'pos_fraction_mu_subset': float(pos / max(len(y_pm), 1)), 'mu': np.nan, 'mu_squared': np.nan, 'lp_success': False, 'lp_exact_optimal': False, 'lp_approx_accepted': False, 'lp_status': np.nan, 'lp_message': '', 'lp_runtime_sec': np.nan, 'model_build_runtime_sec': np.nan, 'bar_iter_count': np.nan, 'sol_count': 0, 'worker_pid': os.getpid(), 'gurobi_threads_per_worker': int(_CFG.get('gurobi_threads', 1))}
    if pos < int(_CFG['mu_min_pos']) or neg < int(_CFG['mu_min_neg']):
        result['lp_message'] = f'insufficient rows: pos={pos}, neg={neg}'
        result['total_class_runtime_sec'] = time.perf_counter() - t_total
        return result
    model = None
    try:
        t_build = time.perf_counter()
        B = sparse.csr_matrix(_A_SUB * y_pm[:, None])
        (n, d) = B.shape
        model = gp.Model(f'mu_{rank}', env=_ENV)
        _apply_fast_params(model)
        beta = model.addMVar(d, lb=-GRB.INFINITY, name='beta')
        z_pos = model.addMVar(n, lb=0.0, name='z_pos')
        z_neg = model.addMVar(n, lb=0.0, name='z_neg')
        model.addConstr(B @ beta - z_pos + z_neg == np.zeros(n), name='decomp')
        model.addConstr(z_neg.sum() <= 1.0, name='neg_l1')
        model.setObjective(z_pos.sum(), GRB.MAXIMIZE)
        model.update()
        result['model_build_runtime_sec'] = time.perf_counter() - t_build
        t_solve = time.perf_counter()
        model.optimize()
        result['lp_runtime_sec'] = time.perf_counter() - t_solve
        result['lp_status'] = int(model.Status)
        result['lp_message'] = str(model.Status)
        result['sol_count'] = int(model.SolCount)
        try:
            result['bar_iter_count'] = float(model.BarIterCount)
        except Exception:
            pass
        exact = bool(model.Status == GRB.OPTIMAL and model.SolCount > 0)
        approx_statuses = {GRB.SUBOPTIMAL, GRB.TIME_LIMIT, GRB.ITERATION_LIMIT, GRB.NUMERIC}
        approx = bool(_CFG['accept_approx'] and model.SolCount > 0 and (model.Status in approx_statuses))
        if exact or approx:
            mu = float(model.ObjVal)
            if math.isfinite(mu):
                result['mu'] = mu
                result['mu_squared'] = mu * mu
                result['lp_success'] = True
                result['lp_exact_optimal'] = exact
                result['lp_approx_accepted'] = approx
            else:
                result['lp_message'] += '; non-finite objective'
        else:
            result['lp_message'] += '; no accepted finite solution'
    except Exception as exc:
        result['lp_message'] = f'Gurobi error: {exc}'
    finally:
        if model is not None:
            try:
                model.dispose()
            except Exception:
                pass
    result['total_class_runtime_sec'] = time.perf_counter() - t_total
    return result

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Cached direct OvR LP solver with a persistent parallel class pool.')
    p.add_argument('--cache_dir', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--workers', type=int, default=2, help='Parallel class workers, capped at the number of classes; each uses one persistent Gurobi environment. Use 1 for the serial comparison.')
    p.add_argument('--gurobi_threads', '--gurobi-threads', type=int, default=1, help='Solver threads per worker; workers multiplied by threads must fit the CPU allocation')
    p.add_argument('--worker_startup_timeout_sec', type=float, default=120.0, help='Total deadline for parallel pool/environment startup; fail without a serial fallback if any worker cannot start')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--dataset_name', default='dataset')
    p.add_argument('--mu-centering', '--mu_centering', dest='mu_centering', choices=['centered', 'uncentered'], required=True)
    p.add_argument('--mu_min_pos', type=int, default=5)
    p.add_argument('--mu_min_neg', type=int, default=5)
    p.add_argument('--bar_conv_tol', type=float, default=1.0)
    p.add_argument('--feasibility_tol', type=float, default=0.01)
    p.add_argument('--optimality_tol', type=float, default=0.01)
    p.add_argument('--presolve', type=int, default=2)
    p.add_argument('--aggregate', type=int, default=2)
    p.add_argument('--pre_sparsify', type=int, default=2)
    p.add_argument('--pre_dual', type=int, default=1)
    p.add_argument('--bar_correctors', type=int, default=0)
    p.add_argument('--time_limit_sec', type=float, default=0.0)
    p.add_argument('--bar_iter_limit', type=int, default=0)
    p.add_argument('--accept_approx', action='store_true')
    p.add_argument('--mem_limit_gb', type=float, default=2.0)
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args(argv)

def _load_cached_tasks(args: argparse.Namespace):
    global _A_SUB, _Y_SUB
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_t0 = time.perf_counter()
    _A_SUB = np.load(cache_dir / 'A_sub.npy', allow_pickle=False)
    _Y_SUB = np.load(cache_dir / 'y_sub.npy', allow_pickle=True).astype(str)
    counts_df = pd.read_csv(cache_dir / 'train_counts.csv')
    meta = json.loads((cache_dir / 'cache_meta.json').read_text())
    validate_cache_artifacts(cache_dir, meta)
    if meta['seed'] != args.seed or meta['dataset_name'] != args.dataset_name or meta['mu_centering'] != args.mu_centering:
        raise ValueError('Mu cache seed/dataset/centering mismatch')
    if _A_SUB.shape[0] != len(_Y_SUB) or not np.isfinite(_A_SUB).all():
        raise ValueError('Invalid mu cache design')
    counts_df['class_label'] = counts_df['class_label'].astype(str)
    counts_df['train_count'] = pd.to_numeric(counts_df['train_count'], errors='raise').astype(int)
    counts_df = counts_df.sort_values('train_count', ascending=False).reset_index(drop=True)
    n_train = int(counts_df['train_count'].sum())
    tasks = [(i + 1, row.class_label, int(row.train_count), n_train) for (i, row) in counts_df.iterrows()]
    if not tasks:
        raise ValueError('Mu cache contains no class tasks')
    return (tasks, meta, n_train, time.perf_counter() - cache_t0)

def _solver_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {'bar_conv_tol': args.bar_conv_tol, 'feasibility_tol': args.feasibility_tol, 'optimality_tol': args.optimality_tol, 'presolve': args.presolve, 'aggregate': args.aggregate, 'pre_sparsify': args.pre_sparsify, 'pre_dual': args.pre_dual, 'bar_correctors': args.bar_correctors, 'time_limit_sec': args.time_limit_sec, 'bar_iter_limit': args.bar_iter_limit, 'accept_approx': bool(args.accept_approx), 'mem_limit_gb': args.mem_limit_gb, 'mu_min_pos': args.mu_min_pos, 'mu_min_neg': args.mu_min_neg, 'gurobi_threads': args.gurobi_threads}

def _results_frame(results, args, meta, n_train, workers, license_metadata=None):
    df = pd.DataFrame(results).sort_values(['selected_class_rank_by_train_count', 'class_label'], kind='stable')
    df.insert(0, 'dataset_name', args.dataset_name)
    df.insert(1, 'seed', int(args.seed))
    df['embedding'] = meta.get('embedding', 'svd')
    df['n_pcs'] = int(meta.get('n_pcs_effective', _A_SUB.shape[1]))
    df['n_pcs_requested'] = int(meta.get('n_pcs_requested', df['n_pcs'].iloc[0]))
    df['n_pcs_effective'] = int(meta.get('n_pcs_effective', df['n_pcs'].iloc[0]))
    df['mu_rows_requested'] = int(meta.get('mu_rows_requested', len(_Y_SUB)))
    df['mu_rows_actual'] = int(len(_Y_SUB))
    df['mu_row_sampling'] = meta.get('mu_row_sampling', meta.get('row_sampling', ''))
    df['intercept_added'] = bool(meta.get('intercept_added', False))
    df['n_features_mu'] = int(_A_SUB.shape[1])
    df['range_basis_dim'] = int(meta.get('range_basis_dim', _A_SUB.shape[1]))
    df['n_train'] = int(meta.get('n_train', n_train))
    df['mu_solver'] = 'gurobi'
    df['mu_solver_variant'] = 'barrier_ultrafast_persistent_single_session' if args.workers == 1 else 'barrier_ultrafast_persistent_pool'
    df['gurobi_session_policy'] = license_metadata['session_policy'] if license_metadata else 'persistent_environment_per_worker'
    df['gurobi_workers'] = workers
    df['n_workers_requested'] = int(args.workers)
    df['n_workers_effective'] = workers
    df['gurobi_threads_per_worker'] = int(args.gurobi_threads)
    df['gurobi_method_requested'] = 2
    df['gurobi_crossover_requested'] = 0
    df['bar_conv_tol_requested'] = float(args.bar_conv_tol)
    df['feasibility_tol_requested'] = float(args.feasibility_tol)
    df['optimality_tol_requested'] = float(args.optimality_tol)
    df['bar_iter_limit_requested'] = int(args.bar_iter_limit)
    df['time_limit_sec_requested'] = float(args.time_limit_sec)
    df['mu_source'] = 'solve_mu_5k_cached_pool_ultrafast'
    df['dataset_sha256'] = meta['dataset_sha256']
    df['mu_design_sha256'] = meta['mu_design_sha256']
    df['mu_centering'] = args.mu_centering
    df['input_state'] = 'preprocessed'
    df['normalization_backend'] = 'preprocessed'
    return df

def _run(args: argparse.Namespace, license_metadata: Optional[Dict[str, Any]], single_startup_sec: float=0.0) -> None:
    global _A_SUB, _Y_SUB, _CFG
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    if out_dir.exists() and args.overwrite:
        import shutil
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total_t0 = time.perf_counter()
    (tasks, meta, n_train, cache_load_sec) = _load_cached_tasks(args)
    _CFG = _solver_config(args)
    workers = min(int(args.workers), len(tasks))
    timing_policy = 'cache_load_and_solve_and_mu_csv_write_excluding_environment_startup_and_license_wait' if args.workers == 1 else 'cache_load_pool_environment_startup_solve_and_mu_csv_write_excluding_cache_construction_and_external_license_guard_wait'
    config = {**vars(args), 'cache_dir': str(cache_dir), 'output_dir': str(out_dir), 'n_classes': len(tasks), 'n_workers_requested': int(args.workers), 'n_workers_effective': workers, 'gurobi_threads_per_worker': int(args.gurobi_threads), 'cache_meta': meta, 'license_session': license_metadata, 'timing_policy': timing_policy, 'lp_pool_timing_policy': 'parallel_elapsed_time_includes_class_setup_model_build_optimize_disposal_dispatch_and_result_collection_excludes_cache_load_environment_startup_and_output_writes', 'per_class_lp_timing_policy': 'sum_of_individual_model_optimize_elapsed_times_not_parallel_wall_time_or_cpu_time'}
    (out_dir / 'run_config.json').write_text(json.dumps(config, indent=2, default=str))
    if args.workers == 1:
        worker_startup_sec = single_startup_sec
        lp_t0 = time.perf_counter()
        results = [_solve_one_class(task) for task in tasks]
        lp_pool_wall_sec = time.perf_counter() - lp_t0
    else:
        print(f'[parallel] requested={args.workers} effective={workers}; {args.gurobi_threads} solver threads and one persistent Gurobi environment per worker', flush=True)
        (results, worker_startup_sec, lp_pool_wall_sec) = _run_parallel(tasks, workers, args, out_dir)
    write_t0 = time.perf_counter()
    df = _results_frame(results, args, meta, n_train, workers, license_metadata)
    df.to_csv(out_dir / 'mu_by_class.csv', index=False)
    output_write_sec = time.perf_counter() - write_t0
    finite = np.isfinite(pd.to_numeric(df['mu'], errors='coerce').to_numpy(dtype=float))
    success = df['lp_success'].astype(bool).to_numpy()
    exact = df['lp_exact_optimal'].astype(bool).to_numpy()
    approx = df['lp_approx_accepted'].astype(bool).to_numpy()
    total_solveonly_wall_sec = time.perf_counter() - total_t0
    validity = {'output_dir': str(out_dir), 'n_workers': workers, 'n_workers_requested': int(args.workers), 'n_workers_effective': workers, 'gurobi_threads_per_worker': int(args.gurobi_threads), 'n_classes_expected': len(tasks), 'n_classes_combined': int(len(df)), 'n_classes_finite_mu': int(finite.sum()), 'n_classes_lp_failed': int((~success).sum()), 'n_exact_optimal': int(exact.sum()), 'n_approx_accepted': int(approx.sum()), 'valid_for_all_class_mu': bool(len(df) == len(tasks) and finite.sum() == len(tasks) and success.all()), 'cache_load_sec': float(cache_load_sec), 'worker_startup_sec': float(worker_startup_sec), 'environment_startup_sec': float(license_metadata['environment_startup_sec']) if license_metadata else None, 'license_wait_sec': float(license_metadata['license_wait_sec']) if license_metadata else 0.0, 'license_attempts': int(license_metadata['attempts']) if license_metadata else workers, 'gurobi_session_policy': license_metadata['session_policy'] if license_metadata else 'persistent_environment_per_worker', 'timing_policy': config['timing_policy'], 'lp_pool_timing_policy': config['lp_pool_timing_policy'], 'per_class_lp_timing_policy': config['per_class_lp_timing_policy'], 'worker_startup_included_in_total': bool(args.workers != 1), 'lp_pool_wall_sec': float(lp_pool_wall_sec), 'output_write_sec': float(output_write_sec), 'total_solveonly_wall_sec': float(total_solveonly_wall_sec), 'total_wall_excluding_worker_startup_sec': float(max(0.0, total_solveonly_wall_sec - worker_startup_sec) if args.workers != 1 else total_solveonly_wall_sec), 'sum_per_class_lp_runtime_sec': float(pd.to_numeric(df['lp_runtime_sec'], errors='coerce').sum()), 'sum_model_build_sec': float(pd.to_numeric(df['model_build_runtime_sec'], errors='coerce').sum()), 'max_per_class_lp_runtime_sec': float(pd.to_numeric(df['lp_runtime_sec'], errors='coerce').max())}
    pd.DataFrame([validity]).to_csv(out_dir / 'mu_validity.csv', index=False)
    pd.DataFrame([validity]).to_csv(out_dir / 'mu_walltime.csv', index=False)
    (out_dir / 'mu_validity.json').write_text(json.dumps(validity, indent=2))
    print(json.dumps(validity, indent=2), flush=True)
    print(f"[output] {out_dir / 'mu_by_class.csv'} rows={len(df)}", flush=True)
    if not validity['valid_for_all_class_mu']:
        raise SystemExit(2)

def main() -> None:
    args = _parse_args()
    if args.gurobi_threads < 1:
        raise ValueError('--gurobi_threads must be at least 1')
    if args.workers < 1:
        raise ValueError('--workers must be at least 1')
    if not math.isfinite(args.worker_startup_timeout_sec) or args.worker_startup_timeout_sec <= 0:
        raise ValueError('--worker_startup_timeout_sec must be finite and positive')
    metadata = None
    startup_sec = 0.0
    try:
        if args.workers == 1:
            startup_t0 = time.perf_counter()
            metadata = _initialize_single_environment()
            startup_sec = time.perf_counter() - startup_t0
        _run(args, metadata, startup_sec)
    finally:
        _dispose_environment()
if __name__ == '__main__':
    main()
