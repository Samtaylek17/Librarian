from eval_pipeline.extractors.base import Extractor
from eval_pipeline.extractors.token_usage import TokenUsageExtractor
from eval_pipeline.extractors.joules import JoulesExtractor


def default_extractors() -> list[Extractor]:
    return [
        TokenUsageExtractor(),
        JoulesExtractor(),
    ]
