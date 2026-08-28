from eval_pipeline.benchmarks.swe_bench import SWEBenchEvaluator
from eval_pipeline.benchmarks.swe_bench_live import SWEBenchLiveEvaluator

BENCHMARK_REGISTRY: dict[str, type] = {
    "swe_bench": SWEBenchEvaluator,
    "swe_bench_live": SWEBenchLiveEvaluator,
}


def create_benchmark(name: str, **kwargs):
    if name not in BENCHMARK_REGISTRY:
        raise ValueError(f"Unknown benchmark: {name}. Available: {list(BENCHMARK_REGISTRY)}")
    return BENCHMARK_REGISTRY[name](**kwargs)
