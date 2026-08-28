"""Leave-one-out librarian ablation runners (L3-L6).

Thin subclasses of the two ``*_librarian`` overlays: no behaviour of their
own — the ablation is selected by the ``librarian_ablation`` constructor
kwarg (forced by the registry factories in :mod:`eval_pipeline.agents`).
Distinct CLASS NAMES exist only so episode logs group per condition
(``episode_meta`` keys on the runner class name).

| ablation      | removes                                              |
|---------------|------------------------------------------------------|
| "persistence" | persistent history — librarian ephemeral-per-call    |
| "pointer"     | view_commands side-channel — code inline in result   |
| "scoped"      | lookup-only restriction — Navigator-style gatherer   |
| "freshness"   | per-wake FILE FRESHNESS REPORT                       |
"""

from __future__ import annotations

from eval_pipeline.agents.mas.boad_librarian import BOADLibrarianMASRunner
from eval_pipeline.agents.mas.hyperagent_librarian import (
    HyperAgentLibrarianMASRunner,
)


class HyperAgentLibrarianNoPersistRunner(HyperAgentLibrarianMASRunner):
    LABEL = "HyperAgentLibrarianMAS-nopersist"


class HyperAgentLibrarianNoPointerRunner(HyperAgentLibrarianMASRunner):
    LABEL = "HyperAgentLibrarianMAS-nopointer"


class HyperAgentLibrarianNoScopeRunner(HyperAgentLibrarianMASRunner):
    LABEL = "HyperAgentLibrarianMAS-noscope"


class HyperAgentLibrarianNoFreshRunner(HyperAgentLibrarianMASRunner):
    LABEL = "HyperAgentLibrarianMAS-nofresh"


class BOADLibrarianNoPersistRunner(BOADLibrarianMASRunner):
    LABEL = "BOADLibrarianMAS-nopersist"


class BOADLibrarianNoPointerRunner(BOADLibrarianMASRunner):
    LABEL = "BOADLibrarianMAS-nopointer"


class BOADLibrarianNoScopeRunner(BOADLibrarianMASRunner):
    LABEL = "BOADLibrarianMAS-noscope"


class BOADLibrarianNoFreshRunner(BOADLibrarianMASRunner):
    LABEL = "BOADLibrarianMAS-nofresh"
