"""Deterministically compare exact FAISS Flat with HNSW; never mutates runtime config."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import resource
import time

import faiss
import numpy as np


def _p95(values: list[float]) -> float:
    return sorted(values)[max(0, math.ceil(0.95 * len(values)) - 1)]


def _rss_mib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 * 1024) if platform.system() == "Darwin" else value / 1024


def _search_metrics(index, queries: np.ndarray, k: int) -> tuple[np.ndarray, dict]:
    durations = []
    identifiers = []
    for query in queries:
        started = time.perf_counter()
        _, found = index.search(query.reshape(1, -1), k)
        durations.append((time.perf_counter() - started) * 1000)
        identifiers.append(found[0])
    return np.asarray(identifiers), {
        "latency_ms_p50": float(np.median(durations)),
        "latency_ms_p95": _p95(durations),
        "latency_ms_mean": sum(durations) / len(durations),
    }


def benchmark(*, vector_count: int = 5000, dimension: int = 64, query_count: int = 100,
              k: int = 10, seed: int = 17, hnsw_m: int = 32,
              ef_construction: int = 80, ef_search: int = 64,
              flat_latency_target_ms: float = 5.0, minimum_recall: float = 0.98,
              minimum_speedup: float = 1.5) -> dict:
    if not 100 <= vector_count <= 200_000:
        raise ValueError("vector_count must be between 100 and 200000")
    if not 4 <= dimension <= 4096:
        raise ValueError("dimension must be between 4 and 4096")
    if not 1 <= query_count <= 2000:
        raise ValueError("query_count must be between 1 and 2000")
    if not 1 <= k <= min(100, vector_count):
        raise ValueError("k is out of bounds")
    if not 4 <= hnsw_m <= 64 or ef_search < k or ef_construction < hnsw_m:
        raise ValueError("Invalid HNSW parameters")

    previous_threads = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(1)
    try:
        generator = np.random.default_rng(seed)
        vectors = generator.normal(size=(vector_count, dimension)).astype("float32")
        queries = generator.normal(size=(query_count, dimension)).astype("float32")

        started = time.perf_counter()
        flat = faiss.IndexFlatL2(dimension)
        flat.add(vectors)
        flat_build_ms = (time.perf_counter() - started) * 1000
        exact_ids, flat_search = _search_metrics(flat, queries, k)

        started = time.perf_counter()
        hnsw = faiss.IndexHNSWFlat(dimension, hnsw_m, faiss.METRIC_L2)
        hnsw.hnsw.efConstruction = ef_construction
        hnsw.hnsw.efSearch = ef_search
        hnsw.add(vectors)
        hnsw_build_ms = (time.perf_counter() - started) * 1000
        hnsw_ids, hnsw_search = _search_metrics(hnsw, queries, k)

        recalled = sum(len(set(exact) & set(approximate)) for exact, approximate in zip(exact_ids, hnsw_ids))
        recall_at_k = recalled / (query_count * k)
        speedup = flat_search["latency_ms_p95"] / max(hnsw_search["latency_ms_p95"], 1e-9)
        flat_meets_target = flat_search["latency_ms_p95"] <= flat_latency_target_ms
        ann_meets_target = recall_at_k >= minimum_recall and speedup >= minimum_speedup
        recommendation = "evaluate_hnsw" if not flat_meets_target and ann_meets_target else "retain_flat"
        if flat_meets_target:
            reason = "exact_flat_meets_latency_target"
        elif recall_at_k < minimum_recall:
            reason = "hnsw_recall_below_target"
        elif speedup < minimum_speedup:
            reason = "hnsw_speedup_below_target"
        else:
            reason = "flat_misses_latency_and_hnsw_meets_recall_speedup"

        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "Synthetic single-process CPU microbenchmark; not end-to-end retrieval latency",
            "configuration": {
                "vector_count": vector_count, "dimension": dimension, "query_count": query_count,
                "k": k, "seed": seed, "threads": 1,
                "hnsw_m": hnsw_m, "ef_construction": ef_construction, "ef_search": ef_search,
            },
            "targets": {
                "flat_latency_ms_p95": flat_latency_target_ms,
                "minimum_hnsw_recall_at_k": minimum_recall,
                "minimum_hnsw_speedup": minimum_speedup,
            },
            "flat": {
                "index": "IndexFlatL2", "exact": True, "build_ms": flat_build_ms,
                "serialized_bytes": int(faiss.serialize_index(flat).nbytes), **flat_search,
            },
            "hnsw": {
                "index": "IndexHNSWFlat", "exact": False, "build_ms": hnsw_build_ms,
                "serialized_bytes": int(faiss.serialize_index(hnsw).nbytes),
                "recall_at_k_vs_flat": recall_at_k, **hnsw_search,
            },
            "hnsw_speedup_p95": speedup,
            "peak_process_rss_mib": _rss_mib(),
            "recommendation": recommendation,
            "reason": reason,
            "production_index_changed": False,
        }
    finally:
        faiss.omp_set_num_threads(previous_threads)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--vectors", type=int, default=5000)
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    report = benchmark(vector_count=args.vectors, dimension=args.dimension,
                       query_count=args.queries, k=args.k)
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
