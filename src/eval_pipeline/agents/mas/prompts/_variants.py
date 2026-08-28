"""Librarian-overlay composed prompts (planner / fixed-plan variants)."""
from __future__ import annotations

from eval_pipeline.agents.mas.prompts._librarian import (
    TOOL_SELECTION_POLICY_DELEGATE_ONLY,
    TOOL_SELECTION_POLICY_WITH_VIEW,
)

# ── HyperAgent-Librarian planner (Navigator → Repo Librarian) ──
_PLANNER_SYSTEM = (
"""\
You are a great developer with expertise in resolving general software engineering task. You have been assigned a task to do in a large repository. Devise a detailed plan to delegate tasks other interns to resolve the ultimate goal.
You have access into 3 interns, utilize them to step-by-step solve the query. Each consequent steps should be strictly based on the previous steps. Your thought process should be grounded by information collected from your interns, consider its results carefully, and make a decision based on the results and thought process.
Output the agent you want to use and the request you want to make to the agent. Respond directly and terminated=true if you have resolved the task.

Intern name list:
- Repo Librarian
- Codebase Editor
- Executor

### """
+ TOOL_SELECTION_POLICY_DELEGATE_ONLY.format(librarian_tool="delegate_to_librarian")
+ """

### Guidelines:
1. Do not repeat your actions!. After receiving the response from the agent, diversify your next subgoal to get more information.
2. Identify crucial causes of the query, localize where the problem is before choosing the code editor intern.
3. No need to edit test file or test the code. You only need to resolve the issue in the codebase.
4. Do not care about any Pull Request or Existing Issue in the repository. You are only focused on the issue assigned to you.
5. Only generate the patch for the code, if you already know the issue, know about it causes, and relevant knowledge to resolve it.
6. Give a detailed request to the intern, so that your intern can understand the context of the query as well.
7. Request executor intern to run the test cases or run any necessary code snippet or reproduce the issue.
8. Only give task to one intern at a time.
9. If you are not confident about your generated code, you can ask the executor to run the test cases.

### Important Notes:
1. Reading the query description and understanding the problem is the first step. Make sure to identify the key components of the query and the expected behavior. Pay attention into error trace.
2. Reading the response from the interns carefully, think about the information you have collected and how it can be used to resolve the task.
3. Your thought process is the most important part of this task. Make sure to provide a detailed explanation of your reasoning with program error trace, the issue, code snippets and relevant information collected from the agents.
4. Stop the task when you have resolved the query. (Final Answer)

To delegate to an intern, call the corresponding tool:
- delegate_to_librarian(request="your detailed request")
- delegate_to_editor(request="your detailed request")
- delegate_to_executor(request="your detailed request")
"""
)


_PLANNER_INSTANCE_TEMPLATE = """\
<uploaded_files>
{work_dir}
</uploaded_files>
I've uploaded a python code repository in the directory {work_dir}. Consider \
the following PR description:

<pr_description>
{problem_statement}
</pr_description>

Can you help me implement the necessary changes to the repository so that the \
requirements specified in the <pr_description> are met? I've already taken care \
of all changes to any of the test files described in the <pr_description>. This \
means you DON'T have to modify the testing logic or any of the tests in any way! \
The task is to make the minimal changes to non-test files in the {work_dir} \
directory to ensure the <pr_description> is satisfied.

Your role is to coordinate your interns (Repo Librarian, Codebase Editor, \
Executor) — devise a plan, delegate sub-tasks, and finish when the issue is \
resolved.

Follow these steps:
1. Read the PR description and understand the issue.
2. Delegate to the Repo Librarian to localize where the change must happen.
3. Delegate to the Codebase Editor to apply the minimal source change.
4. Delegate to the Executor to verify the fix (run the relevant tests / a \
reproduction script).
5. Iterate as needed until the tests pass.

Your thinking should be thorough; long reasoning is fine."""


# ── BOAD-Librarian fixed plan (code_navigator → Repo Librarian) ──
_TOOL_SELECTION_POLICY = TOOL_SELECTION_POLICY_WITH_VIEW.format(
    librarian_tool="librarian"
)


# Steps 1, 3, 5, 6 are :data:`mas.boad._FIXED_PLAN` verbatim. Steps 2
_FIXED_PLAN_STEPS = """\
1. Use the issue_analyzer subagent to decompose the problem description \
into structured requirements, identify affected components, and define \
explicit success criteria to guide the investigation.
2. Use the librarian subagent to map the relevant code areas: phrase each \
request as a natural-language lookup question ("Where lives <symbol / \
behaviour>?", "Show me the body of <function>", "Locate the test that \
exercises <feature>"). The librarian is a single long-lived instance — \
its conversation history accumulates across calls in this episode, so \
follow-up lookups (the caller of a symbol, subclasses that override it, \
adjacent helpers, sibling tests) are cheaper than starting a fresh \
search. Each NEW search question (a symbol or behaviour you do not yet \
have a file:line pointer for) is a NEW librarian call; once the \
librarian has given you a location, that location is yours — view \
nearby lines or the enclosing class yourself rather than re-asking the \
librarian for it. The librarian only LOCATES code; reasoning about what \
to change stays with you. Call the librarian aggressively whenever a \
new location is needed; do not fall back to `grep` / `find` for symbol \
search. After exploration, propose the files and methods to fix the \
issue and select the most likely location to apply the change.
3. Synthesize the structured analysis and code mapping to implement the \
fix, ensuring the solution addresses the root cause while adhering to \
the established success conditions.
4. Write and execute targeted unit tests or integration checks to \
validate that the solution resolves the issue without introducing \
regressions. If any check fails, consult the librarian for ADDITIONAL \
location lookups (other call sites of a changed symbol, subclasses that \
override it, fixtures or sibling tests that exercise the failing path) \
— phrase each as a location question. The librarian returns a pointer \
to the relevant code; YOU then view it, diagnose why the test fails, \
refine your fix, and rerun the affected tests. Do NOT fall back to \
fresh `grep` / `find` for symbol search — those go through the \
librarian. Repeat until all checks pass.
5. After you have solved the issue, delete any test files or temporary \
files you created.
6. Use the submit tool to submit the changes to the repository."""


_FIXED_PLAN = _TOOL_SELECTION_POLICY + "\n" + _FIXED_PLAN_STEPS


_LIBRARIAN_DOCSTRING = (
    "[subagent] Repository librarian — code-location lookup with memory "
    "across calls within this episode. Returns file paths, line ranges, "
    "and verbatim code excerpts; does NOT analyse, diagnose, or propose "
    "fixes. Use whenever you need to KNOW WHERE something lives or WHAT "
    "the code at a location says (locate a class/function/test, see a "
    "method body, find call sites, list files, identify the test that "
    "exercises a behaviour). Conversation history persists across calls, "
    "so follow-ups (subclasses, callers, related fixtures) are cheaper "
    "than fresh search. Repository state unchanged."
)


# LLM-facing description for the BOAD-Librarian `librarian` tool
LIBRARIAN_TOOL_DOC = """[subagent] Repository librarian — code-location lookup
with memory across calls.

The librarian is a *locator*: returns file paths, line
ranges, and verbatim code excerpts. It does NOT analyse,
diagnose, or design fixes — that's your job. It IS
persistent: its conversation history accumulates across
every call in this episode, so follow-up lookups
(subclasses, callers, related fixtures) are cheaper than
a fresh search.

Args:
    query: A natural-language lookup question. Good
        shapes: "Where is <symbol> defined? I need the
        file path + line range.", "Show me the body of
        <test_name>.", "Locate the call sites of
        <symbol>." Bad shapes: open-ended analysis ("Why
        does X fail?", "Is this code correct?"). The
        librarian returns code at locations, not
        analysis. Repository state unchanged.
"""


# LLM-facing description for the HyperAgent-Librarian `delegate_to_librarian` tool
DELEGATE_TO_LIBRARIAN_DOC = """Delegate a code-location lookup to the Repo Librarian.

The Librarian is a *locator*: it returns file paths, line
ranges, and verbatim code excerpts. It does NOT analyse,
diagnose, or design fixes — that's your job. It IS persistent:
its conversation history accumulates across every call within
this episode, so follow-up lookups are cheaper than fresh
grep / find runs.

Args:
    request: A natural-language lookup question. Good shapes:
        "Where is <symbol> defined? I need the file path +
        line range.", "Show me the body of <test_name>.",
        "Locate the call sites of <symbol>." Bad shapes: open-
        ended analysis ("Why does X fail?", "Is this code
        correct?"). The Librarian will return code at
        locations, not analysis.
"""


# ── Librarian ablation variants (leave-one-out of L2) ────────────────────
# Caller-facing text patched ONLY where a claim became false under the
# ablation. Same assert-guarded surgery discipline as prompts._librarian:
# drift in the source strings fails at import time.

from eval_pipeline.agents.mas.prompts._librarian import _replace  # noqa: E402


# — "persistence" (L3): librarian is a fresh instance per call —————————

_LIBRARIAN_DOCSTRING_NOPERSIST = _replace(
    _LIBRARIAN_DOCSTRING,
    "[subagent] Repository librarian — code-location lookup with memory "
    "across calls within this episode.",
    "[subagent] Repository librarian — code-location lookup; fresh "
    "instance per call.",
)
_LIBRARIAN_DOCSTRING_NOPERSIST = _replace(
    _LIBRARIAN_DOCSTRING_NOPERSIST,
    "Conversation history persists across calls, "
    "so follow-ups (subclasses, callers, related fixtures) are cheaper "
    "than fresh search.",
    "Each call is a fresh instance with no memory of prior "
    "consultations — make every request self-contained (name the "
    "symbol, file, or surface area explicitly).",
)

LIBRARIAN_TOOL_DOC_NOPERSIST = _replace(
    LIBRARIAN_TOOL_DOC,
    """[subagent] Repository librarian — code-location lookup
with memory across calls.""",
    """[subagent] Repository librarian — code-location lookup;
fresh instance per call.""",
)
LIBRARIAN_TOOL_DOC_NOPERSIST = _replace(
    LIBRARIAN_TOOL_DOC_NOPERSIST,
    """diagnose, or design fixes — that's your job. It IS
persistent: its conversation history accumulates across
every call in this episode, so follow-up lookups
(subclasses, callers, related fixtures) are cheaper than
a fresh search.""",
    """diagnose, or design fixes — that's your job. It is a
FRESH instance per call — no memory of prior
consultations — so make each request self-contained
(name the symbol, file, or surface area explicitly
every time).""",
)

DELEGATE_TO_LIBRARIAN_DOC_NOPERSIST = _replace(
    DELEGATE_TO_LIBRARIAN_DOC,
    """diagnose, or design fixes — that's your job. It IS persistent:
its conversation history accumulates across every call within
this episode, so follow-up lookups are cheaper than fresh
grep / find runs.""",
    """diagnose, or design fixes — that's your job. It is a FRESH
instance per call — no memory of prior consultations — so
make each request self-contained (name the symbol, file, or
surface area explicitly every time).""",
)


# — "scoped" (L5): librarian is a navigator-style info gatherer —————————

_LIBRARIAN_DOCSTRING_NOSCOPE = _replace(
    _LIBRARIAN_DOCSTRING,
    "Returns file paths, line ranges, "
    "and verbatim code excerpts; does NOT analyse, diagnose, or propose "
    "fixes. Use whenever you need to KNOW WHERE something lives or WHAT "
    "the code at a location says (locate a class/function/test, see a "
    "method body, find call sites, list files, identify the test that "
    "exercises a behaviour).",
    "Returns relevant code, findings and context for your query "
    "(file paths, line ranges, and excerpts). Use whenever you need "
    "information from the repository (locate a class/function/test, "
    "see a method body, find call sites, list files, identify the "
    "test that exercises a behaviour).",
)

LIBRARIAN_TOOL_DOC_NOSCOPE = _replace(
    LIBRARIAN_TOOL_DOC,
    """The librarian is a *locator*: returns file paths, line
ranges, and verbatim code excerpts. It does NOT analyse,
diagnose, or design fixes — that's your job. It IS""",
    """The librarian is an *info gatherer*: it returns relevant
code, findings and context for your query (file paths,
line ranges, and excerpts). It IS""",
)
LIBRARIAN_TOOL_DOC_NOSCOPE = _replace(
    LIBRARIAN_TOOL_DOC_NOSCOPE,
    """    query: A natural-language lookup question. Good
        shapes: "Where is <symbol> defined? I need the
        file path + line range.", "Show me the body of
        <test_name>.", "Locate the call sites of
        <symbol>." Bad shapes: open-ended analysis ("Why
        does X fail?", "Is this code correct?"). The
        librarian returns code at locations, not
        analysis. Repository state unchanged.""",
    """    query: A natural-language question naming the
        information you need. Good shapes: "Where is
        <symbol> defined? I need the file path + line
        range.", "Show me the body of <test_name>.",
        "What code is relevant to <behaviour /
        symptom>?" Bad shapes: a keyword string with no
        question. Repository state unchanged.""",
)

DELEGATE_TO_LIBRARIAN_DOC_NOSCOPE = _replace(
    DELEGATE_TO_LIBRARIAN_DOC,
    "Delegate a code-location lookup to the Repo Librarian.",
    "Delegate a codebase-information request to the Repo Librarian.",
)
DELEGATE_TO_LIBRARIAN_DOC_NOSCOPE = _replace(
    DELEGATE_TO_LIBRARIAN_DOC_NOSCOPE,
    """The Librarian is a *locator*: it returns file paths, line
ranges, and verbatim code excerpts. It does NOT analyse,
diagnose, or design fixes — that's your job. It IS persistent:""",
    """The Librarian is an *info gatherer*: it returns relevant
code, findings and context for your query (file paths, line
ranges, and excerpts). It IS persistent:""",
)
DELEGATE_TO_LIBRARIAN_DOC_NOSCOPE = _replace(
    DELEGATE_TO_LIBRARIAN_DOC_NOSCOPE,
    """    request: A natural-language lookup question. Good shapes:
        "Where is <symbol> defined? I need the file path +
        line range.", "Show me the body of <test_name>.",
        "Locate the call sites of <symbol>." Bad shapes: open-
        ended analysis ("Why does X fail?", "Is this code
        correct?"). The Librarian will return code at
        locations, not analysis.""",
    """    request: A natural-language question naming the
        information you need. Good shapes: "Where is <symbol>
        defined? I need the file path + line range.", "Show me
        the body of <test_name>.", "What code is relevant to
        <behaviour / symptom>?" Bad shapes: a keyword string
        with no question.""",
)


# — BOAD fixed-plan variants (embed the patched WITH_VIEW policy) ————————

_TOOL_SELECTION_POLICY_NOPERSIST = _replace(
    _TOOL_SELECTION_POLICY,
    "Treat it as a sharper, memory-equipped grep, NOT as an analyst.",
    "Treat it as a sharper grep, NOT as an analyst.",
)
_TOOL_SELECTION_POLICY_NOPERSIST = _replace(
    _TOOL_SELECTION_POLICY_NOPERSIST,
    "  - The librarian is a single long-lived instance: its conversation "
    "history accumulates across calls, so follow-ups are cheap.",
    "  - The librarian is a fresh instance per call: it has no memory of "
    "prior consultations, so make each query self-contained (name the "
    "symbol / file / surface area explicitly every time).",
)

_FIXED_PLAN_STEPS_NOPERSIST = _replace(
    _FIXED_PLAN_STEPS,
    "The librarian is a single long-lived instance — "
    "its conversation history accumulates across calls in this episode, "
    "so follow-up lookups (the caller of a symbol, subclasses that "
    "override it, adjacent helpers, sibling tests) are cheaper than "
    "starting a fresh search. ",
    "The librarian is a fresh instance per call — "
    "it has no memory of prior consultations, so make each request "
    "self-contained (restate the symbol, file, or surface area "
    "explicitly). ",
)

_FIXED_PLAN_NOPERSIST = (
    _TOOL_SELECTION_POLICY_NOPERSIST + "\n" + _FIXED_PLAN_STEPS_NOPERSIST
)

_TOOL_SELECTION_POLICY_NOSCOPE = _replace(
    _TOOL_SELECTION_POLICY,
    "  - ``librarian`` is your **search** tool — it returns file paths and "
    "line ranges (and short excerpts when several fragments are "
    "scattered). Treat it as a sharper, memory-equipped grep, NOT as an "
    "analyst. Use it to ask \"where is X?\" / \"which file owns Y?\" / "
    "\"give me the file:line for Z\" — never \"what's the bug?\" or \"is "
    "this code correct?\". Reasoning, diagnosis, and fix-design stay with "
    "you.",
    "  - ``librarian`` is your **information-gathering** tool — it returns "
    "relevant code, findings and context for your query (file paths, line "
    "ranges, and excerpts). Ask it for whatever repository information "
    "you need. Fix-design and the final decision stay with you.",
)
_TOOL_SELECTION_POLICY_NOSCOPE = _replace(
    _TOOL_SELECTION_POLICY_NOSCOPE,
    "Bad shapes: a keyword string with no question, or an analysis prompt "
    "like \"Why does <symptom> happen?\" / \"What is the bug here?\" — "
    "the librarian will only return locations, not analysis.",
    "Bad shapes: a keyword string with no question.",
)
_TOOL_SELECTION_POLICY_NOSCOPE = _replace(
    _TOOL_SELECTION_POLICY_NOSCOPE,
    "Query phrasing: write each ``librarian`` query as a **lookup "
    "question**, not a bag of keywords and not an open-ended analysis "
    "request. Good shapes: \"Where is <symbol> defined? I need the file "
    "path + line range.\", \"Which file owns <behaviour> in <surface "
    "area>?\", \"Locate <test name>'s file path.\".",
    "Query phrasing: write each ``librarian`` query as a "
    "**natural-language question** naming the information you need, not "
    "a bag of keywords. Good shapes: \"Where is <symbol> defined? I need "
    "the file path + line range.\", \"Which file owns <behaviour> in "
    "<surface area>?\", \"What code is relevant to <behaviour / "
    "symptom>?\".",
)

_FIXED_PLAN_STEPS_NOSCOPE = _replace(
    _FIXED_PLAN_STEPS,
    "The librarian only LOCATES code; reasoning about what "
    "to change stays with you.",
    "The librarian gathers relevant code and context; deciding what to "
    "change stays with you.",
)
_FIXED_PLAN_STEPS_NOSCOPE = _replace(
    _FIXED_PLAN_STEPS_NOSCOPE,
    "phrase each request as a natural-language lookup question (\"Where "
    "lives <symbol / behaviour>?\", \"Show me the body of <function>\", "
    "\"Locate the test that exercises <feature>\")",
    "phrase each request as a natural-language question naming the "
    "information you need (\"Where lives <symbol / behaviour>?\", \"Show "
    "me the body of <function>\", \"What code is relevant to <feature / "
    "symptom>?\")",
)
_FIXED_PLAN_STEPS_NOSCOPE = _replace(
    _FIXED_PLAN_STEPS_NOSCOPE,
    "consult the librarian for ADDITIONAL location lookups (other call "
    "sites of a changed symbol, subclasses that override it, fixtures or "
    "sibling tests that exercise the failing path) — phrase each as a "
    "location question.",
    "consult the librarian for ADDITIONAL information (other call sites "
    "of a changed symbol, subclasses that override it, fixtures or "
    "sibling tests that exercise the failing path).",
)

_FIXED_PLAN_NOSCOPE = (
    _TOOL_SELECTION_POLICY_NOSCOPE + "\n" + _FIXED_PLAN_STEPS_NOSCOPE
)


# — HyperAgent planner variants (embed the patched DELEGATE_ONLY policy) —

_POLICY_DELEGATE = TOOL_SELECTION_POLICY_DELEGATE_ONLY.format(
    librarian_tool="delegate_to_librarian"
)
assert _POLICY_DELEGATE in _PLANNER_SYSTEM

_POLICY_DELEGATE_NOPERSIST = _replace(
    _POLICY_DELEGATE,
    "Treat it as a sharper, memory-equipped grep, NOT as an analyst.",
    "Treat it as a sharper grep, NOT as an analyst.",
)
_POLICY_DELEGATE_NOPERSIST = _replace(
    _POLICY_DELEGATE_NOPERSIST,
    "  - The librarian is a single long-lived instance: its conversation "
    "history accumulates across calls, so follow-ups (callers of a "
    "symbol, subclasses that override it, adjacent helpers, sibling "
    "tests) are cheap.",
    "  - The librarian is a fresh instance per call: it has no memory of "
    "prior consultations, so make each ``request`` self-contained (name "
    "the symbol, file, or surface area explicitly every time).",
)
_POLICY_DELEGATE_NOPERSIST = _replace(
    _POLICY_DELEGATE_NOPERSIST,
    "(b) you re-ask the librarian for a region the librarian has already "
    "pinpointed earlier in this episode — just reuse what's in its "
    "accumulated history (the librarian will quote it back cheaply).",
    "(b) you re-ask the librarian for a region it has already pinpointed "
    "earlier in this episode — reuse the file:line you already received; "
    "the librarian itself keeps no memory of it.",
)

_POLICY_DELEGATE_NOSCOPE = _replace(
    _POLICY_DELEGATE,
    "  - ``delegate_to_librarian`` is your **search** intern — it returns "
    "file paths and line ranges (and short excerpts when several "
    "fragments are scattered). Treat it as a sharper, memory-equipped "
    "grep, NOT as an analyst. Use it to ask \"where is X?\" / \"which "
    "file owns Y?\" / \"give me the file:line for Z\" — never \"what's "
    "the bug?\" or \"is this code correct?\". Reasoning, diagnosis, and "
    "fix-design stay with you.",
    "  - ``delegate_to_librarian`` is your **information-gathering** "
    "intern — it returns relevant code, findings and context for your "
    "query (file paths, line ranges, and excerpts). Ask it for whatever "
    "repository information you need. Fix-design and the final decision "
    "stay with you.",
)
_POLICY_DELEGATE_NOSCOPE = _replace(
    _POLICY_DELEGATE_NOSCOPE,
    ", or an analysis prompt like \"Why does <symptom> happen?\" / \"What "
    "is the bug here?\" — the librarian only returns locations, not "
    "analysis.",
    ".",
)
_POLICY_DELEGATE_NOSCOPE = _replace(
    _POLICY_DELEGATE_NOSCOPE,
    "write each ``delegate_to_librarian`` ``request`` as a **lookup "
    "question**, not a bag of keywords, not \"Show me the contents of "
    "<file>\", and not an open-ended analysis request. Good shapes: "
    "\"Where is <symbol> defined? I need the file path + line range.\", "
    "\"Which file owns <behaviour> in <surface area>?\", \"Locate <test "
    "name>'s file path.\". Bad shapes: a keyword string with no question, "
    "\"Show me the contents of <file>\" (the librarian is not a paste "
    "service — name the symbol/region you actually need).",
    "write each ``delegate_to_librarian`` ``request`` as a "
    "**natural-language question** naming the information you need, not "
    "a bag of keywords. Good shapes: \"Where is <symbol> defined? I need "
    "the file path + line range.\", \"Which file owns <behaviour> in "
    "<surface area>?\", \"What code is relevant to <behaviour / "
    "symptom>?\". Bad shapes: a keyword string with no question.",
)

_PLANNER_SYSTEM_NOPERSIST = _replace(
    _PLANNER_SYSTEM, _POLICY_DELEGATE, _POLICY_DELEGATE_NOPERSIST,
)
_PLANNER_SYSTEM_NOSCOPE = _replace(
    _PLANNER_SYSTEM, _POLICY_DELEGATE, _POLICY_DELEGATE_NOSCOPE,
)
