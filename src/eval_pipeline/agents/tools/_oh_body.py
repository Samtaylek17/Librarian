"""OH primitive tool combinator + back-compat re-exports.

After the F2 split, the only function defined here is :func:`make_tools` —
a thin combinator over :func:`make_oh_bash` / :func:`make_oh_str_replace_editor`
/ :func:`make_oh_submit` (each in its own module under ``tools/oh/``).

Everything else (parsers, policy variants, contextvars, format helpers)
is re-exported from :mod:`tools._parsers`, :mod:`tools.policies.*`,
:mod:`tools._state`, :mod:`tools._format` so that legacy imports
``from eval_pipeline.agents.tools._oh_body import _check_orchestrator_bash``
keep working.
"""

from __future__ import annotations

from eval_pipeline.agents.tools._format import (  # noqa: F401
    OH_DIR_TRUNCATE_NOTICE,
    OH_MAX_RESPONSE_LEN_CHAR,
    OH_SNIPPET_CONTEXT_WINDOW,
    OH_TEXT_FILE_TRUNCATE_NOTICE,
    _maybe_truncate_oh,
    _oh_make_output,
    _truncate,
    _truncate_bash_output,
)
from eval_pipeline.agents.tools._state import (  # noqa: F401
    _LAST_TOOL_EVENT_PAIR,
    _canonical_args,
)
from eval_pipeline.utils.docker import DockerExecutor


def make_tools(
    docker: DockerExecutor, work_dir: str,
    *,
    max_observation_length: int = OH_MAX_RESPONSE_LEN_CHAR,
    bash_timeout: int = 1800,
):
    from eval_pipeline.agents.tools.oh.bash import make_oh_bash
    from eval_pipeline.agents.tools.oh.editor import make_oh_str_replace_editor
    from eval_pipeline.agents.tools.oh.submit import make_oh_submit
    return [
        make_oh_bash(
            docker, work_dir,
            max_observation_length=max_observation_length,
            timeout=bash_timeout,
        ),
        make_oh_str_replace_editor(docker, work_dir, max_observation_length=max_observation_length),
        make_oh_submit(),
    ]


from eval_pipeline.agents.tools._parsers import (  # noqa: F401, E402
    _AUTO_ROUTE_QUERY_TMPL,
    _BASH_FILE_READ_VERBS,
    _LINE_NUM_PREFIX_RX,
    _SCOPED_SOURCE_EXTS,
    _build_auto_route_query,
    _check_orchestrator_bash,
    _find_bash_target_file,
    _looks_like_file_content,
    _parse_command_range,
)
from eval_pipeline.agents.tools.policies.edit_only_editor import (  # noqa: F401, E402
    make_edit_only_str_replace_editor,
)
from eval_pipeline.agents.tools.policies.view_only_bash import (  # noqa: F401, E402
    _BASH_EDIT_VERB_RE,
    make_view_only_bash,
)
from eval_pipeline.agents.tools.policies.view_only_editor import (  # noqa: F401, E402
    make_view_only_str_replace_editor,
)
