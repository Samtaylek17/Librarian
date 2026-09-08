from __future__ import annotations

from typing import Any, Callable

from eval_pipeline.agents.bases import build_openhands_agent
from eval_pipeline.agents.mas import (
    BOADMASRunner,
    BOADLibrarianMASRunner,
    HyperAgentMASRunner,
    HyperAgentLibrarianMASRunner,
    LibrarianMASRunner,
)
from eval_pipeline.agents.mas.boad_librarian_rag import BOADLibrarianRAGMASRunner
from eval_pipeline.agents.mas.librarian_ablations import (
    BOADLibrarianNoFreshRunner,
    BOADLibrarianNoPersistRunner,
    BOADLibrarianNoPointerRunner,
    BOADLibrarianNoScopeRunner,
    HyperAgentLibrarianNoFreshRunner,
    HyperAgentLibrarianNoPersistRunner,
    HyperAgentLibrarianNoPointerRunner,
    HyperAgentLibrarianNoScopeRunner,
)
from eval_pipeline.agents.mas.hyperagent_exec_librarian import (
    HyperAgentExecLibrarianMASRunner,
)
from eval_pipeline.agents.mas.hyperagent_librarian_rag import (
    HyperAgentLibrarianRAGMASRunner,
)
from eval_pipeline.agents.mas.librarian_rag import LibrarianRAGMASRunner
from eval_pipeline.agents.vanilla import VanillaRunner


def _make_hyperagent(**kwargs: Any) -> HyperAgentMASRunner:
    return HyperAgentMASRunner(
        base_agent_builder=build_openhands_agent,
        base_name="oh", **kwargs,
    )


def _make_hyperagent_caveman(**kwargs: Any) -> HyperAgentMASRunner:
    kwargs.setdefault("caveman_level", "full")
    return HyperAgentMASRunner(
        base_agent_builder=build_openhands_agent,
        base_name="oh_caveman", **kwargs,
    )


def _make_hyperagent_lastn(**kwargs: Any) -> HyperAgentMASRunner:
    """HyperAgent MAS with SWE-Agent's ``last-N-observations`` context
    policy (N=5 default, applied uniformly to planner + navigator +
    editor + executor). Each sub-agent's request path elides
    ``function_call_output`` items older than the last N, keeping the
    full action trace + last N observation outputs verbatim. Targets
    long-trajectory context blow-up. Override with
    ``last_n_observations=<int>`` or
    ``last_n_observations={"role": N, ...}``."""
    kwargs.setdefault("last_n_observations", 5)
    return HyperAgentMASRunner(
        base_agent_builder=build_openhands_agent,
        base_name="oh_lastn", **kwargs,
    )


def _make_hyperagent_librarian(**kwargs: Any) -> HyperAgentLibrarianMASRunner:
    return HyperAgentLibrarianMASRunner(
        base_agent_builder=build_openhands_agent,
        base_name="oh", **kwargs,
    )


def _make_hyperagent_librarian_caveman(**kwargs: Any) -> HyperAgentLibrarianMASRunner:
    kwargs.setdefault("caveman_level", "full")
    return HyperAgentLibrarianMASRunner(
        base_agent_builder=build_openhands_agent,
        base_name="oh_caveman", **kwargs,
    )


def _make_hyperagent_exec_librarian(**kwargs: Any) -> HyperAgentExecLibrarianMASRunner:
    """HyperAgent with ONLY the Executor swapped for the persistent
    exec-librarian (Navigator / Editor untouched)."""
    return HyperAgentExecLibrarianMASRunner(
        base_agent_builder=build_openhands_agent,
        base_name="oh", **kwargs,
    )


def _make_boad(**kwargs: Any) -> BOADMASRunner:
    return BOADMASRunner(
        base_agent_builder=build_openhands_agent,
        base_name="oh", **kwargs,
    )


def _make_boad_caveman(**kwargs: Any) -> BOADMASRunner:
    kwargs.setdefault("caveman_level", "full")
    return BOADMASRunner(
        base_agent_builder=build_openhands_agent,
        base_name="oh_caveman", **kwargs,
    )


def _make_boad_lastn(**kwargs: Any) -> BOADMASRunner:
    """BOAD MAS with SWE-Agent's ``last-N-observations`` context policy
    (N=5 default, applied uniformly to orchestrator + issue_analyzer +
    code_navigator). Each sub-agent's request path elides
    ``function_call_output`` items older than the last N, keeping the
    full action trace + last N observation outputs verbatim. Targets
    long-trajectory context blow-up. Override with
    ``last_n_observations=<int>`` or
    ``last_n_observations={"role": N, ...}``."""
    kwargs.setdefault("last_n_observations", 5)
    return BOADMASRunner(
        base_agent_builder=build_openhands_agent,
        base_name="oh_lastn", **kwargs,
    )


def _make_boad_librarian(**kwargs: Any) -> BOADLibrarianMASRunner:
    return BOADLibrarianMASRunner(
        base_agent_builder=build_openhands_agent,
        base_name="oh", **kwargs,
    )


def _make_boad_librarian_caveman(**kwargs: Any) -> BOADLibrarianMASRunner:
    kwargs.setdefault("caveman_level", "full")
    return BOADLibrarianMASRunner(
        base_agent_builder=build_openhands_agent,
        base_name="oh_caveman", **kwargs,
    )


def _make_librarian(**kwargs: Any) -> LibrarianMASRunner:
    return LibrarianMASRunner(
        base_agent_builder=build_openhands_agent,
        base_name="oh", **kwargs,
    )


def _make_librarian_caveman(**kwargs: Any) -> LibrarianMASRunner:
    """Librarian variant with the caveman ("full") output-style overlay
    applied to BOTH the main agent and the librarian sub-agent system
    prompts. Persistence + bash-routing policy preserved; only prose
    style changes. Targets out_tok."""
    kwargs.setdefault("caveman_level", "full")
    return LibrarianMASRunner(
        base_agent_builder=build_openhands_agent,
        base_name="oh_caveman", **kwargs,
    )


def _make_librarian_ablation(cls: type, ablation: str, **kwargs: Any):
    """Construct a leave-one-out librarian ablation runner, forcing its
    ``librarian_ablation``. A conflicting caller-supplied value is an
    error — the registry name IS the condition."""
    passed = kwargs.setdefault("librarian_ablation", ablation)
    if passed != ablation:
        raise ValueError(
            f"{cls.__name__} forces librarian_ablation={ablation!r}; "
            f"got conflicting {passed!r}"
        )
    return cls(
        base_agent_builder=build_openhands_agent,
        base_name="oh", **kwargs,
    )


def _make_hyperagent_librarian_nopersist(**kwargs: Any):
    return _make_librarian_ablation(
        HyperAgentLibrarianNoPersistRunner, "persistence", **kwargs)


def _make_hyperagent_librarian_nopointer(**kwargs: Any):
    return _make_librarian_ablation(
        HyperAgentLibrarianNoPointerRunner, "pointer", **kwargs)


def _make_hyperagent_librarian_noscope(**kwargs: Any):
    return _make_librarian_ablation(
        HyperAgentLibrarianNoScopeRunner, "scoped", **kwargs)


def _make_hyperagent_librarian_nofresh(**kwargs: Any):
    return _make_librarian_ablation(
        HyperAgentLibrarianNoFreshRunner, "freshness", **kwargs)


def _make_boad_librarian_nopersist(**kwargs: Any):
    return _make_librarian_ablation(
        BOADLibrarianNoPersistRunner, "persistence", **kwargs)


def _make_boad_librarian_nopointer(**kwargs: Any):
    return _make_librarian_ablation(
        BOADLibrarianNoPointerRunner, "pointer", **kwargs)


def _make_boad_librarian_noscope(**kwargs: Any):
    return _make_librarian_ablation(
        BOADLibrarianNoScopeRunner, "scoped", **kwargs)


def _make_boad_librarian_nofresh(**kwargs: Any):
    return _make_librarian_ablation(
        BOADLibrarianNoFreshRunner, "freshness", **kwargs)


def _make_librarian_rag(**kwargs: Any) -> LibrarianRAGMASRunner:
    return LibrarianRAGMASRunner(
        base_agent_builder=build_openhands_agent,
        base_name="oh", **kwargs,
    )


def _make_boad_librarian_rag(**kwargs: Any) -> BOADLibrarianRAGMASRunner:
    return BOADLibrarianRAGMASRunner(
        base_agent_builder=build_openhands_agent,
        base_name="oh", **kwargs,
    )


def _make_hyperagent_librarian_rag(**kwargs: Any) -> HyperAgentLibrarianRAGMASRunner:
    return HyperAgentLibrarianRAGMASRunner(
        base_agent_builder=build_openhands_agent,
        base_name="oh", **kwargs,
    )


AGENT_REGISTRY: dict[str, Callable | type] = {
    "vanilla": VanillaRunner,
    "librarian": _make_librarian,
    "librarian_caveman": _make_librarian_caveman,
    "librarian_rag": _make_librarian_rag,
    "hyperagent": _make_hyperagent,
    "hyperagent_caveman": _make_hyperagent_caveman,
    "hyperagent_lastn": _make_hyperagent_lastn,
    "hyperagent_librarian": _make_hyperagent_librarian,
    "hyperagent_librarian_caveman": _make_hyperagent_librarian_caveman,
    "hyperagent_librarian_rag": _make_hyperagent_librarian_rag,
    "hyperagent_librarian_nopersist": _make_hyperagent_librarian_nopersist,
    "hyperagent_librarian_nopointer": _make_hyperagent_librarian_nopointer,
    "hyperagent_librarian_noscope": _make_hyperagent_librarian_noscope,
    "hyperagent_librarian_nofresh": _make_hyperagent_librarian_nofresh,
    "hyperagent_exec_librarian": _make_hyperagent_exec_librarian,
    "boad": _make_boad,
    "boad_caveman": _make_boad_caveman,
    "boad_lastn": _make_boad_lastn,
    "boad_librarian": _make_boad_librarian,
    "boad_librarian_caveman": _make_boad_librarian_caveman,
    "boad_librarian_rag": _make_boad_librarian_rag,
    "boad_librarian_nopersist": _make_boad_librarian_nopersist,
    "boad_librarian_nopointer": _make_boad_librarian_nopointer,
    "boad_librarian_noscope": _make_boad_librarian_noscope,
    "boad_librarian_nofresh": _make_boad_librarian_nofresh,
}


def create_agent(name: str, **kwargs):
    if name not in AGENT_REGISTRY:
        raise ValueError(
            f"Unknown agent: {name}. Available: {sorted(AGENT_REGISTRY)}"
        )
    return AGENT_REGISTRY[name](**kwargs)


def _disable_openai_platform_tracing() -> None:
    """Eval runs talk to local vLLM with ``OPENAI_API_KEY=dummy``.

    The Agents SDK still exports traces to api.openai.com by default,
    which 401s and prints ``[non-fatal] Tracing client error 401``.
    Our own call log / energy telemetry is the observability path.
    """
    try:
        from agents import set_tracing_disabled

        set_tracing_disabled(True)
    except Exception:
        pass


_disable_openai_platform_tracing()
