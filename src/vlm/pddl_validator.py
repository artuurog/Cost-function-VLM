"""
VLM-based PDDL validation and repair
"""

from __future__ import annotations

import base64
import io
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# User settings — edit these before running
# ─────────────────────────────────────────────────────────────────────────────

# --- The three input PDDL files ----------------------------------------------
DOMAIN_FILE:  str = "pddl/stacking/domain.pddl"
PROBLEM_FILE: str = "pddl/stacking/problem.pddl"
PLAN_FILE:    str = "pddl/stacking/planner.pddl"

# Where the revised files are written (input files are never overwritten).
OUTPUT_DIR: str = "results/pddl_revised"

# --- VLM access (HuggingFace-style: an API key plus a model endpoint) ---------
API_KEY:  str = ""                                  # your access token
BASE_URL: str = "https://router.huggingface.co/v1"  # OpenAI-compatible endpoint
MODEL_NAME: str = "allenai/Molmo2-8B"               # model served at BASE_URL

# Optional workspace image; when set, it is attached to the review so the VLM
# can check the symbolic plan against the real scene. Leave "" for text-only.
SCENE_IMAGE: str = ""

# --- Repair behaviour ---------------------------------------------------------
# Number of validate → repair rounds. 0 disables the VLM (static check only).
MAX_REPAIR_ROUNDS: int = 2

# Call the VLM even when the static checker finds nothing: the VLM is the
# reasoning core and catches semantic faults no parser can see.
ALWAYS_REVIEW: bool = True

# --- Generation settings ------------------------------------------------------
MAX_NEW_TOKENS: int = 4096
TEMPERATURE: float = 0.0

# Longest image side (px) sent to the VLM; keeps token / bandwidth cost bounded.
MAX_IMAGE_SIDE: int = 768


# ─────────────────────────────────────────────────────────────────────────────
# S-expression reader
# ─────────────────────────────────────────────────────────────────────────────

# A parsed PDDL form: either an atom (str) or a nested list of forms.
Sexp = object

_COMMENT_RE = re.compile(r";[^\n]*")
_TOKEN_RE = re.compile(r"\(|\)|[^\s()]+")


def strip_comments(text: str) -> str:
    """Remove PDDL line comments (';' to end of line), keeping line numbering."""
    return _COMMENT_RE.sub("", text)


def parse_sexps(text: str) -> List[Sexp]:
    """
    Parse *text* into a list of top-level s-expressions.

    Raises ValueError on unbalanced parentheses, which is itself a finding the
    checker reports rather than a crash.
    """
    tokens = _TOKEN_RE.findall(strip_comments(text))
    forms: List[Sexp] = []
    stack: List[List[Sexp]] = []

    for token in tokens:
        if token == "(":
            stack.append([])
        elif token == ")":
            if not stack:
                raise ValueError("unbalanced ')': closing parenthesis without opening")
            done = stack.pop()
            (stack[-1] if stack else forms).append(done)
        else:
            if not stack:
                forms.append(token)
            else:
                stack[-1].append(token)

    if stack:
        raise ValueError(f"unbalanced '(': {len(stack)} section(s) never closed")
    return forms


def sexp_to_str(form: Sexp) -> str:
    """Render a parsed form back to PDDL-like text (used in report messages)."""
    if isinstance(form, list):
        return "(" + " ".join(sexp_to_str(f) for f in form) + ")"
    return str(form)


def head(form: Sexp) -> str:
    """Lowercased first atom of a form, or '' if the form has no atom head."""
    if isinstance(form, list) and form and isinstance(form[0], str):
        return form[0].lower()
    return ""


def find_section(forms: Sequence[Sexp], keyword: str) -> Optional[list]:
    """Return the first '(:keyword ...)' sub-form, or None."""
    for form in forms:
        if head(form) == keyword.lower():
            return form  # type: ignore[return-value]
    return None


def parse_typed_list(items: Sequence[Sexp]) -> List[Tuple[str, Sexp]]:
    """
    Parse a PDDL typed list such as 'a b - block c - location' into
    [('a', 'block'), ('b', 'block'), ('c', 'location')].

    Names declared without a '- type' suffix default to 'object'. A type may be
    a list, for the '(either t1 t2)' form.
    """
    result: List[Tuple[str, Sexp]] = []
    pending: List[str] = []
    index = 0

    while index < len(items):
        item = items[index]
        if item == "-":
            type_spec: Sexp = items[index + 1] if index + 1 < len(items) else "object"
            result.extend((name, type_spec) for name in pending)
            pending = []
            index += 2
        else:
            if isinstance(item, str):
                pending.append(item)
            index += 1

    result.extend((name, "object") for name in pending)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Parsed PDDL structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Action:
    """A parametric domain action."""
    name: str
    parameters: List[Tuple[str, Sexp]]          # [(?var, type), ...]
    precondition: Optional[Sexp] = None
    effect: Optional[Sexp] = None
    durative: bool = False

    @property
    def arity(self) -> int:
        return len(self.parameters)


@dataclass
class Domain:
    name: str = ""
    requirements: List[str] = field(default_factory=list)
    parent_of: Dict[str, str] = field(default_factory=dict)   # type -> supertype
    constants: Dict[str, Sexp] = field(default_factory=dict)  # name -> type
    predicates: Dict[str, List[Sexp]] = field(default_factory=dict)  # name -> param types
    functions: Dict[str, List[Sexp]] = field(default_factory=dict)   # numeric fluents
    actions: Dict[str, Action] = field(default_factory=dict)


@dataclass
class Problem:
    name: str = ""
    domain_name: str = ""
    objects: Dict[str, Sexp] = field(default_factory=dict)    # name -> type
    init: List[Sexp] = field(default_factory=list)
    goal: Optional[Sexp] = None


@dataclass
class PlanStep:
    """One grounded action taken from the plan file."""
    line_no: int
    name: str
    args: List[str]
    raw: str

    def __str__(self) -> str:
        return "(" + " ".join([self.name, *self.args]) + ")"


# ─────────────────────────────────────────────────────────────────────────────
# Findings
# ─────────────────────────────────────────────────────────────────────────────

ERROR = "ERROR"
WARNING = "WARNING"


@dataclass
class Finding:
    """One problem detected by the symbolic checker."""
    severity: str      # ERROR | WARNING
    file: str          # domain | problem | plan
    category: str      # syntax | symbol | arity | type | precondition | goal
    message: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.file}/{self.category}: {self.message}"


@dataclass
class Report:
    """The full outcome of one validation pass."""
    findings: List[Finding] = field(default_factory=list)
    simulated: bool = False       # did the feasibility simulation run at all?
    goal_reached: bool = False

    def add(self, severity: str, file: str, category: str, message: str) -> None:
        self.findings.append(Finding(severity, file, category, message))

    @property
    def errors(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == ERROR]

    @property
    def is_clean(self) -> bool:
        return not self.errors and self.simulated and self.goal_reached

    def to_text(self) -> str:
        """Render the report as the evidence block handed to the VLM."""
        lines: List[str] = []
        if not self.findings:
            lines.append("No structural or symbolic problems detected.")
        else:
            for i, finding in enumerate(self.findings, start=1):
                lines.append(f"{i:3d}. {finding}")

        lines.append("")
        if not self.simulated:
            lines.append(
                "PLAN SIMULATION: not run (the files could not be parsed far enough)."
            )
        elif self.goal_reached:
            lines.append("PLAN SIMULATION: the goal state IS reached by the plan.")
        else:
            lines.append("PLAN SIMULATION: the goal state is NOT reached by the plan.")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Domain / problem / plan parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_domain(text: str, report: Report) -> Optional[Domain]:
    """Parse a PDDL domain, recording findings instead of raising."""
    try:
        forms = parse_sexps(text)
    except ValueError as exc:
        report.add(ERROR, "domain", "syntax", str(exc))
        return None

    root = next((f for f in forms if head(f) == "define"), None)
    if root is None:
        report.add(ERROR, "domain", "syntax", "no '(define (domain ...) ...)' form found")
        return None

    domain = Domain()
    body = root[1:]

    header = next((f for f in body if head(f) == "domain"), None)
    if header is None or len(header) < 2:
        report.add(ERROR, "domain", "syntax", "the domain name is missing")
    else:
        domain.name = str(header[1]).lower()

    requirements = find_section(body, ":requirements")
    if requirements:
        domain.requirements = [str(r).lower() for r in requirements[1:]]

    types_section = find_section(body, ":types")
    if types_section:
        for name, parent in parse_typed_list(types_section[1:]):
            domain.parent_of[name.lower()] = str(parent).lower() if isinstance(parent, str) else "object"

    constants = find_section(body, ":constants")
    if constants:
        for name, type_spec in parse_typed_list(constants[1:]):
            domain.constants[name.lower()] = _lower_type(type_spec)

    predicates = find_section(body, ":predicates")
    if predicates:
        for declaration in predicates[1:]:
            if not isinstance(declaration, list) or not declaration:
                report.add(ERROR, "domain", "syntax",
                           f"malformed predicate declaration: {sexp_to_str(declaration)}")
                continue
            name = str(declaration[0]).lower()
            if name in domain.predicates:
                report.add(WARNING, "domain", "symbol",
                           f"predicate '{name}' is declared more than once")
            domain.predicates[name] = [
                _lower_type(t) for _, t in parse_typed_list(declaration[1:])
            ]

    functions = find_section(body, ":functions")
    if functions:
        for declaration in functions[1:]:
            if isinstance(declaration, list) and declaration:
                domain.functions[str(declaration[0]).lower()] = [
                    _lower_type(t) for _, t in parse_typed_list(declaration[1:])
                ]
        if domain.functions:
            report.add(WARNING, "domain", "syntax",
                       "the domain declares numeric functions; numeric conditions "
                       "and effects are checked for symbols only, not simulated")

    for form in body:
        keyword = head(form)
        if keyword not in (":action", ":durative-action"):
            continue
        if len(form) < 2:
            report.add(ERROR, "domain", "syntax", "an ':action' has no name")
            continue

        action = Action(name=str(form[1]).lower(), parameters=[],
                        durative=(keyword == ":durative-action"))
        if action.durative:
            report.add(WARNING, "domain", "syntax",
                       f"action '{action.name}' is durative; temporal semantics are not simulated")

        # Walk the ':key value' pairs that follow the action name.
        index = 2
        while index < len(form) - 1:
            key = str(form[index]).lower()
            value = form[index + 1]
            if key == ":parameters":
                action.parameters = [
                    (str(v).lower(), _lower_type(t))
                    for v, t in parse_typed_list(value if isinstance(value, list) else [])
                ]
            elif key in (":precondition", ":condition"):
                action.precondition = value
            elif key == ":effect":
                action.effect = value
            index += 2

        if action.name in domain.actions:
            report.add(ERROR, "domain", "symbol",
                       f"action '{action.name}' is defined more than once")
        domain.actions[action.name] = action

    if not domain.actions:
        report.add(ERROR, "domain", "syntax", "the domain declares no actions")
    return domain


def parse_problem(text: str, report: Report) -> Optional[Problem]:
    """Parse a PDDL problem, recording findings instead of raising."""
    try:
        forms = parse_sexps(text)
    except ValueError as exc:
        report.add(ERROR, "problem", "syntax", str(exc))
        return None

    root = next((f for f in forms if head(f) == "define"), None)
    if root is None:
        report.add(ERROR, "problem", "syntax", "no '(define (problem ...) ...)' form found")
        return None

    problem = Problem()
    body = root[1:]

    header = next((f for f in body if head(f) == "problem"), None)
    if header is None or len(header) < 2:
        report.add(ERROR, "problem", "syntax", "the problem name is missing")
    else:
        problem.name = str(header[1]).lower()

    domain_ref = find_section(body, ":domain")
    if domain_ref is None or len(domain_ref) < 2:
        report.add(ERROR, "problem", "syntax", "the problem does not declare '(:domain ...)'")
    else:
        problem.domain_name = str(domain_ref[1]).lower()

    objects = find_section(body, ":objects")
    if objects:
        for name, type_spec in parse_typed_list(objects[1:]):
            problem.objects[name.lower()] = _lower_type(type_spec)
    else:
        report.add(WARNING, "problem", "syntax", "the problem declares no ':objects'")

    init = find_section(body, ":init")
    if init is None:
        report.add(ERROR, "problem", "syntax", "the problem has no ':init' section")
    else:
        problem.init = list(init[1:])

    goal = find_section(body, ":goal")
    if goal is None or len(goal) < 2:
        report.add(ERROR, "problem", "syntax", "the problem has no ':goal' section")
    else:
        problem.goal = goal[1]

    return problem


# Grounded action line, optionally in IPC format:  '0.001: (grasp r b) [1.0]'
_PLAN_LINE_RE = re.compile(
    r"^\s*(?:[\d.]+\s*:\s*)?\(\s*(?P<body>[^()]*?)\s*\)\s*(?:\[[\d.]+\])?\s*$"
)


def parse_plan(text: str, report: Report) -> List[PlanStep]:
    """
    Parse a plan file into grounded steps.

    Comment lines and blank lines are skipped; any other unparsable line is
    reported so the VLM can see the malformed text verbatim.
    """
    steps: List[PlanStep] = []
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = strip_comments(raw).strip()
        if not line:
            continue

        match = _PLAN_LINE_RE.match(line)
        if match is None:
            report.add(ERROR, "plan", "syntax",
                       f"line {line_no} is not a grounded action: {raw.strip()!r}")
            continue

        tokens = match.group("body").split()
        if not tokens:
            report.add(ERROR, "plan", "syntax", f"line {line_no} is an empty action '()'")
            continue

        steps.append(
            PlanStep(line_no=line_no,
                     name=tokens[0].lower(),
                     args=[t.lower() for t in tokens[1:]],
                     raw=raw.strip())
        )

    if not steps:
        report.add(ERROR, "plan", "syntax", "the plan file contains no grounded actions")
    return steps


def _lower_type(type_spec: Sexp) -> Sexp:
    """Lowercase a type specification, which may be an '(either ...)' list."""
    if isinstance(type_spec, list):
        return [str(t).lower() for t in type_spec]
    return str(type_spec).lower()


# ─────────────────────────────────────────────────────────────────────────────
# Type system
# ─────────────────────────────────────────────────────────────────────────────

class TypeSystem:
    """Subtype queries over the domain's type hierarchy."""

    def __init__(self, domain: Domain, problem: Problem) -> None:
        self._parent = dict(domain.parent_of)
        self._type_of: Dict[str, Sexp] = {**domain.constants, **problem.objects}

    def type_of(self, obj: str) -> Optional[Sexp]:
        return self._type_of.get(obj.lower())

    def is_known_object(self, obj: str) -> bool:
        return obj.lower() in self._type_of

    def objects_of_type(self, wanted: Sexp) -> List[str]:
        return [name for name, actual in self._type_of.items()
                if self.conforms(actual, wanted)]

    def conforms(self, actual: Sexp, wanted: Sexp) -> bool:
        """True if an object of type *actual* is acceptable where *wanted* is required."""
        if isinstance(wanted, list):        # (either t1 t2 ...)
            return any(self.conforms(actual, w) for w in wanted[1:])
        if isinstance(actual, list):        # object declared as (either ...)
            return any(self.conforms(a, wanted) for a in actual[1:])

        wanted = str(wanted).lower()
        current = str(actual).lower()
        if wanted in ("object", ""):
            return True

        # Walk up the hierarchy; the guard stops cyclic '- type' declarations.
        seen: Set[str] = set()
        while current and current not in seen:
            if current == wanted:
                return True
            seen.add(current)
            current = self._parent.get(current, "object" if current != "object" else "")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Static consistency checks
# ─────────────────────────────────────────────────────────────────────────────

# Logical connectives that are not predicates and must not be looked up as such.
_CONNECTIVES = {"and", "or", "not", "imply", "when", "forall", "exists", "="}

# Numeric comparisons and assignments. Their arguments are arithmetic, not
# predicates, so the whole form is skipped: this checker is STRIPS-level and
# reports numeric parts as unsimulated rather than inventing findings on them.
_NUMERIC_OPS = {"<", "<=", ">", ">=", "increase", "decrease",
                "assign", "scale-up", "scale-down", "+", "-", "*", "/"}


def _iter_atoms(form: Sexp) -> Iterable[list]:
    """Yield every predicate application inside a logical formula."""
    if not isinstance(form, list) or not form:
        return
    name = head(form)
    if name in _NUMERIC_OPS:
        return
    if name in _CONNECTIVES:
        # 'forall'/'exists' carry a variable list as their first argument.
        rest = form[2:] if name in ("forall", "exists") else form[1:]
        for sub in rest:
            yield from _iter_atoms(sub)
    else:
        yield form


def check_predicate_usage(form: Optional[Sexp], domain: Domain, where: str,
                          file: str, report: Report) -> None:
    """Verify that every predicate used in *form* is declared with the right arity."""
    if form is None:
        return
    for atom in _iter_atoms(form):
        name = head(atom)
        if not name:
            continue
        if name in domain.functions:
            continue  # a numeric fluent, not a predicate
        if name not in domain.predicates:
            report.add(ERROR, file, "symbol",
                       f"{where} uses undeclared predicate '{name}': {sexp_to_str(atom)}")
            continue
        expected = len(domain.predicates[name])
        actual = len(atom) - 1
        if expected != actual:
            report.add(ERROR, file, "arity",
                       f"{where}: predicate '{name}' takes {expected} argument(s) "
                       f"but got {actual}: {sexp_to_str(atom)}")


def check_cross_file_consistency(domain: Domain, problem: Problem,
                                 steps: List[PlanStep], types: TypeSystem,
                                 report: Report) -> None:
    """Check that the three files agree on names, types and arities."""
    if domain.name and problem.domain_name and domain.name != problem.domain_name:
        report.add(ERROR, "problem", "symbol",
                   f"the problem targets domain '{problem.domain_name}' but the "
                   f"domain file defines '{domain.name}'")

    # --- Problem objects must have declared types ---
    declared_types = set(domain.parent_of) | {"object"}
    for name, type_spec in problem.objects.items():
        wanted = type_spec if isinstance(type_spec, list) else [type_spec]
        for one in (wanted[1:] if isinstance(type_spec, list) else wanted):
            if str(one).lower() not in declared_types:
                report.add(ERROR, "problem", "type",
                           f"object '{name}' is declared of type '{one}', "
                           f"which the domain does not define")

    # --- Predicate usage across problem and domain ---
    for literal in problem.init:
        check_predicate_usage(literal, domain, "the ':init' state", "problem", report)
    check_predicate_usage(problem.goal, domain, "the ':goal'", "problem", report)

    for action in domain.actions.values():
        check_predicate_usage(action.precondition, domain,
                              f"the precondition of '{action.name}'", "domain", report)
        check_predicate_usage(action.effect, domain,
                              f"the effect of '{action.name}'", "domain", report)

        # Every ?variable used in the body must be bound by :parameters
        # (or by an enclosing forall/exists, collected here too).
        bound = {v for v, _ in action.parameters}
        bound |= _quantified_variables(action.precondition)
        bound |= _quantified_variables(action.effect)
        for used in _used_variables(action.precondition) | _used_variables(action.effect):
            if used not in bound:
                report.add(ERROR, "domain", "symbol",
                           f"action '{action.name}' uses unbound variable '{used}'")

    # --- Init literals must reference known objects ---
    for literal in problem.init:
        for atom in _iter_atoms(literal):
            if head(atom) in domain.functions:
                continue
            for arg in atom[1:]:
                if isinstance(arg, str) and not types.is_known_object(arg):
                    report.add(ERROR, "problem", "symbol",
                               f"':init' references undeclared object '{arg}' "
                               f"in {sexp_to_str(literal)}")

    # --- Plan steps must match the domain action signatures ---
    for step in steps:
        action = domain.actions.get(step.name)
        if action is None:
            report.add(ERROR, "plan", "symbol",
                       f"line {step.line_no}: action '{step.name}' is not defined "
                       f"in the domain: {step.raw}")
            continue
        if action.arity != len(step.args):
            report.add(ERROR, "plan", "arity",
                       f"line {step.line_no}: '{step.name}' takes {action.arity} "
                       f"argument(s) but got {len(step.args)}: {step.raw}")
            continue
        for arg, (var, wanted) in zip(step.args, action.parameters):
            if not types.is_known_object(arg):
                report.add(ERROR, "plan", "symbol",
                           f"line {step.line_no}: '{arg}' is not declared in the "
                           f"problem ':objects': {step.raw}")
            elif not types.conforms(types.type_of(arg), wanted):
                report.add(ERROR, "plan", "type",
                           f"line {step.line_no}: '{step.name}' expects {var} of type "
                           f"'{sexp_to_str(wanted)}' but '{arg}' is of type "
                           f"'{sexp_to_str(types.type_of(arg))}'")

    # --- Objects that never appear anywhere are usually a modelling slip ---
    used_in_plan = {arg for step in steps for arg in step.args}
    used_in_state = {
        str(arg).lower()
        for literal in ([*problem.init] + ([problem.goal] if problem.goal else []))
        for atom in _iter_atoms(literal)
        for arg in atom[1:]
        if isinstance(arg, str)
    }
    for name in problem.objects:
        if name not in used_in_plan and name not in used_in_state:
            report.add(WARNING, "problem", "symbol",
                       f"object '{name}' is declared but never used in ':init', "
                       f"':goal' or the plan")


def _used_variables(form: Sexp) -> Set[str]:
    """Every '?var' token appearing anywhere in a formula."""
    if isinstance(form, list):
        found: Set[str] = set()
        for sub in form:
            found |= _used_variables(sub)
        return found
    text = str(form)
    return {text.lower()} if text.startswith("?") else set()


def _quantified_variables(form: Sexp) -> Set[str]:
    """Variables introduced by forall / exists anywhere in a formula."""
    if not isinstance(form, list) or not form:
        return set()
    found: Set[str] = set()
    if head(form) in ("forall", "exists") and len(form) > 1 and isinstance(form[1], list):
        found |= {v.lower() for v, _ in parse_typed_list(form[1])}
    for sub in form:
        found |= _quantified_variables(sub)
    return found


# ─────────────────────────────────────────────────────────────────────────────
# Feasibility simulation (STRIPS state progression)
# ─────────────────────────────────────────────────────────────────────────────

# A grounded literal, e.g. ('on', 'block_1', 'table_slot_1').
Literal = Tuple[str, ...]
State = Set[Literal]


def _ground(form: Sexp, binding: Dict[str, str]) -> Sexp:
    """Substitute '?variables' in *form* with their bound object names."""
    if isinstance(form, list):
        return [_ground(sub, binding) for sub in form]
    text = str(form).lower()
    return binding.get(text, text)


def _as_literal(atom: Sexp) -> Literal:
    """Turn a fully grounded predicate application into a hashable tuple."""
    if isinstance(atom, list):
        return tuple(str(part).lower() for part in atom)
    return (str(atom).lower(),)


def holds(form: Sexp, state: State, types: TypeSystem,
          binding: Optional[Dict[str, str]] = None) -> bool:
    """Evaluate a grounded (or partially bound) formula against *state*."""
    grounded = _ground(form, binding or {})
    name = head(grounded)

    if name == "and":
        return all(holds(sub, state, types) for sub in grounded[1:])
    if name == "or":
        return any(holds(sub, state, types) for sub in grounded[1:])
    if name == "not":
        return not holds(grounded[1], state, types) if len(grounded) > 1 else True
    if name == "imply":
        return (not holds(grounded[1], state, types)) or holds(grounded[2], state, types)
    if name in _NUMERIC_OPS:
        return True   # numeric conditions are not simulated; never fail on them
    if name == "=":
        if len(grounded) > 2 and any(isinstance(a, list) for a in grounded[1:3]):
            return True   # equality over numeric fluents
        return len(grounded) > 2 and str(grounded[1]) == str(grounded[2])
    if name in ("forall", "exists"):
        variables = parse_typed_list(grounded[1]) if isinstance(grounded[1], list) else []
        body = grounded[2] if len(grounded) > 2 else "true"
        combinations = _expand(variables, types)
        if name == "forall":
            return all(holds(body, state, types, b) for b in combinations)
        return any(holds(body, state, types, b) for b in combinations)

    return _as_literal(grounded) in state


def _expand(variables: List[Tuple[str, Sexp]], types: TypeSystem) -> List[Dict[str, str]]:
    """All bindings of the quantified *variables* over the objects of their types."""
    bindings: List[Dict[str, str]] = [{}]
    for var, wanted in variables:
        candidates = types.objects_of_type(wanted)
        bindings = [{**b, var.lower(): obj} for b in bindings for obj in candidates]
    return bindings


def explain_failure(form: Sexp, state: State, types: TypeSystem) -> List[str]:
    """
    List the conjuncts of *form* that do not hold in *state*.

    Only the top-level 'and' is decomposed: that is the level a plan repair
    actually acts on, and deeper decomposition produces noise.
    """
    if head(form) == "and":
        return [sexp_to_str(sub) for sub in form[1:]
                if not holds(sub, state, types)]
    return [] if holds(form, state, types) else [sexp_to_str(form)]


def apply_effect(form: Sexp, state: State, types: TypeSystem,
                 binding: Dict[str, str],
                 add: Set[Literal], delete: Set[Literal]) -> None:
    """
    Collect the add and delete literals of an effect formula.

    Conditional effects are evaluated against the state *before* the action,
    which is the standard PDDL semantics.
    """
    grounded = _ground(form, binding)
    name = head(grounded)

    if name in _NUMERIC_OPS:
        return   # numeric effects do not change the symbolic state
    if name == "and":
        for sub in grounded[1:]:
            apply_effect(sub, state, types, {}, add, delete)
    elif name == "not":
        if len(grounded) > 1:
            delete.add(_as_literal(grounded[1]))
    elif name == "when":
        if len(grounded) > 2 and holds(grounded[1], state, types):
            apply_effect(grounded[2], state, types, {}, add, delete)
    elif name == "forall":
        variables = parse_typed_list(grounded[1]) if isinstance(grounded[1], list) else []
        body = grounded[2] if len(grounded) > 2 else None
        if body is not None:
            for sub_binding in _expand(variables, types):
                apply_effect(body, state, types, sub_binding, add, delete)
    elif name:
        add.add(_as_literal(grounded))


def simulate_plan(domain: Domain, problem: Problem, steps: List[PlanStep],
                  types: TypeSystem, report: Report) -> None:
    """
    Execute the plan symbolically from the initial state and record every
    unsatisfied precondition, then check whether the goal is reached.

    A failing action is still applied so that the remaining steps can also be
    checked; this gives the VLM the full picture in a single pass rather than
    one fault per round.
    """
    # Numeric fluent initialisations carry no symbolic truth and are dropped.
    state: State = {_as_literal(literal) for literal in problem.init
                    if head(literal) not in _NUMERIC_OPS
                    and head(literal) not in domain.functions}
    report.simulated = True
    failures = 0

    for position, step in enumerate(steps, start=1):
        action = domain.actions.get(step.name)
        if action is None or action.arity != len(step.args):
            continue  # already reported by the static checks
        if action.durative:
            continue  # temporal semantics are out of scope

        binding = {var: arg for (var, _), arg in zip(action.parameters, step.args)}

        if action.precondition is not None:
            unsatisfied = explain_failure(
                _ground(action.precondition, binding), state, types
            )
            if unsatisfied:
                failures += 1
                report.add(ERROR, "plan", "precondition",
                           f"step {position} (line {step.line_no}) {step} is not "
                           f"executable: unsatisfied precondition(s) "
                           f"{', '.join(unsatisfied)}")

        if action.effect is not None:
            add: Set[Literal] = set()
            delete: Set[Literal] = set()
            apply_effect(action.effect, state, types, binding, add, delete)
            state -= delete
            state |= add

    if problem.goal is None:
        report.goal_reached = False
        return

    unmet = explain_failure(problem.goal, state, types)
    report.goal_reached = not unmet
    if unmet:
        report.add(ERROR, "plan", "goal",
                   "after the last action the goal is not satisfied; unmet goal "
                   f"condition(s): {', '.join(unmet)}")
    elif failures:
        report.add(WARNING, "plan", "goal",
                   "the goal literals hold at the end, but the plan passed through "
                   f"{failures} inexecutable action(s), so the run is not valid")


# ─────────────────────────────────────────────────────────────────────────────
# Validation entry point
# ─────────────────────────────────────────────────────────────────────────────

def validate(domain_text: str, problem_text: str, plan_text: str) -> Report:
    """Run the complete symbolic validation of one domain/problem/plan triple."""
    report = Report()

    domain = parse_domain(domain_text, report)
    problem = parse_problem(problem_text, report)
    steps = parse_plan(plan_text, report)

    if domain is None or problem is None:
        return report

    types = TypeSystem(domain, problem)
    check_cross_file_consistency(domain, problem, steps, types, report)

    # Simulation is only meaningful once the symbols themselves line up.
    blocking = {"syntax", "symbol", "arity"}
    if any(f.severity == ERROR and f.category in blocking for f in report.findings):
        return report

    simulate_plan(domain, problem, steps, types, report)
    return report


# ─────────────────────────────────────────────────────────────────────────────
# VLM backend
# ─────────────────────────────────────────────────────────────────────────────

Message = Dict[str, object]


def _image_to_data_url(path: str) -> str:
    """Encode an image file as a base64 JPEG data URL."""
    from PIL import Image

    image = Image.open(path).convert("RGB")
    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def call_vlm(messages: List[Message]) -> str:
    """
    Query the VLM through the OpenAI-compatible endpoint at BASE_URL.

    Same access pattern as the rest of the pipeline: an API key set by the user
    plus the model served at the configured router URL.
    """
    from openai import OpenAI  # imported lazily so static checks need no SDK

    if not API_KEY:
        raise ValueError("API_KEY is empty; set it to your access token.")

    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
    )
    return completion.choices[0].message.content or ""


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_INSTRUCTION = (
    "You are the reasoning core of a PDDL validation and repair module for a "
    "robotic manipulator. You are given a PDDL domain, a PDDL problem and a "
    "plan, together with the output of a symbolic checker that parsed and "
    "simulated them.\n\n"
    "Your job has three parts:\n"
    "1. VALIDITY — every action in the plan must be declared in the domain, "
    "with the right number of arguments and argument types, and every symbol "
    "must be declared in the problem.\n"
    "2. CONSISTENCY — the domain, the problem and the plan must agree on types, "
    "predicates, object names and the intent of the task. Predicates must be "
    "used with a single, coherent meaning across preconditions and effects.\n"
    "3. FEASIBILITY — executing the plan from the initial state must satisfy "
    "every action precondition in turn and must end in a state that entails the "
    "goal. Check the physical plausibility of the sequence too: a support must "
    "exist before something is placed on it, the gripper must be free before it "
    "grasps, an object must be released before another is picked up.\n\n"
    "The checker's findings are evidence, not instructions: it can only see "
    "structure, so it may miss a semantic fault and it may flag something that "
    "is actually a modelling choice. Decide for yourself what is wrong.\n\n"
    "Repair the smallest thing that fixes the fault. Prefer correcting the plan "
    "over changing the problem, and the problem over changing the domain; change "
    "the domain only when the fault is genuinely in the action model. Never "
    "change the goal to make a broken plan pass. Preserve the comments and the "
    "formatting of the parts you do not need to touch.\n\n"
    "Return all three components, complete and self-contained, in exactly this "
    "format, with valid PDDL inside the tags and nothing else between them:\n"
    "<DOMAIN>\n(define (domain ...) ...)\n</DOMAIN>\n"
    "<PROBLEM>\n(define (problem ...) (:domain ...) ...)\n</PROBLEM>\n"
    "<PLAN>\n(one grounded action per line, in execution order)\n</PLAN>\n"
    "<NOTES>\none short line per change you made, or 'no changes'\n</NOTES>"
)


def build_review_prompt(domain_text: str, problem_text: str, plan_text: str,
                        report: Report, round_index: int) -> str:
    """Assemble the text prompt for one review round."""
    if round_index == 1:
        opening = "Validate and, where needed, correct the following PDDL task."
    else:
        opening = (
            f"This is repair round {round_index}. The files below are YOUR previous "
            "correction, re-checked by the symbolic checker. The findings that "
            "remain are listed after them; fix what is still wrong."
        )

    return (
        f"{opening}\n\n"
        "=== PDDL DOMAIN ===\n"
        f"{domain_text.strip()}\n\n"
        "=== PDDL PROBLEM ===\n"
        f"{problem_text.strip()}\n\n"
        "=== PLAN ===\n"
        f"{plan_text.strip()}\n\n"
        "=== SYMBOLIC CHECKER REPORT ===\n"
        f"{report.to_text()}\n\n"
        "=== YOUR TASK ===\n"
        "Reason about validity, consistency and feasibility as instructed, then "
        "output the revised DOMAIN, PROBLEM, PLAN and NOTES in the required "
        "delimited format. If a component needs no change, return it unchanged."
    )


def build_messages(prompt: str) -> List[Message]:
    """Build the chat messages, attaching the workspace image when configured."""
    content: List[Dict[str, object]] = [{"type": "text", "text": prompt}]
    if SCENE_IMAGE:
        content.append({
            "type": "image_url",
            "image_url": {"url": _image_to_data_url(SCENE_IMAGE)},
        })
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": content},
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Response parsing
# ─────────────────────────────────────────────────────────────────────────────

_SECTION_RE = {
    "domain":  re.compile(r"<DOMAIN>\s*(.*?)\s*</DOMAIN>", re.DOTALL | re.IGNORECASE),
    "problem": re.compile(r"<PROBLEM>\s*(.*?)\s*</PROBLEM>", re.DOTALL | re.IGNORECASE),
    # Accept <PLANNER> too: that is the tag used by vlm_learning.py.
    "plan":    re.compile(r"<PLANN?E?R?>\s*(.*?)\s*</PLANN?E?R?>", re.DOTALL | re.IGNORECASE),
    "notes":   re.compile(r"<NOTES>\s*(.*?)\s*</NOTES>", re.DOTALL | re.IGNORECASE),
}

# Strip any markdown code fences the model may wrap the PDDL in.
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$", re.MULTILINE)


def parse_revision(text: str) -> Dict[str, str]:
    """
    Extract the revised components from the VLM response.

    The three PDDL sections are required; NOTES is optional.
    """
    sections: Dict[str, str] = {}
    for name, pattern in _SECTION_RE.items():
        match = pattern.search(text)
        if match is None:
            if name == "notes":
                continue
            raise ValueError(
                f"the VLM response has no <{name.upper()}> ... </{name.upper()}> "
                f"section. Raw response:\n{text}"
            )
        sections[name] = _FENCE_RE.sub("", match.group(1)).strip()
    return sections


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

def write_outputs(domain_text: str, problem_text: str, plan_text: str,
                  report: Report, output_dir: str,
                  source: Dict[str, Path]) -> Dict[str, Path]:
    """
    Write the revised triple plus the final report to *output_dir*.

    The revised files keep the names of the originals, so the output directory
    is a drop-in replacement for the input one.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    revised = {"domain": domain_text, "problem": problem_text, "plan": plan_text}
    written: Dict[str, Path] = {}
    for name, content in revised.items():
        path = out / source[name].name
        path.write_text(content.rstrip() + "\n", encoding="utf-8")
        written[name] = path

    report_path = out / "validation_report.txt"
    status = "VALID" if report.is_clean else "ISSUES REMAIN"
    report_path.write_text(
        "PDDL validation report\n"
        "======================\n"
        f"model  : {MODEL_NAME}\n"
        f"domain : {source['domain']}\n"
        f"problem: {source['problem']}\n"
        f"plan   : {source['plan']}\n"
        f"status : {status}\n\n"
        f"{report.to_text()}\n",
        encoding="utf-8",
    )
    written["report"] = report_path
    return written


# ─────────────────────────────────────────────────────────────────────────────
# Top-level pipeline
# ─────────────────────────────────────────────────────────────────────────────

def say(text: str = "") -> None:
    """
    Print a line without ever aborting the run.

    Console output goes through the terminal codepage (cp1252 on a default
    Windows shell), so a stray non-ASCII character in a PDDL comment or in the
    VLM's notes would otherwise raise and lose the repair before the files are
    written. The written files are always UTF-8 and keep such characters intact.
    """
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "ascii"
        print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))


def _print_report(report: Report, indent: str = "    ") -> None:
    for finding in report.findings:
        say(f"{indent}{finding}")
    if report.simulated and not report.findings:
        say(f"{indent}no findings")


def review_and_correct(domain_file: str = DOMAIN_FILE,
                       problem_file: str = PROBLEM_FILE,
                       plan_file: str = PLAN_FILE,
                       output_dir: str = OUTPUT_DIR) -> Dict[str, Path]:
    """
    Validate the three PDDL files, repair them with the VLM and write the
    revised triple. Returns the paths of the files written.
    """
    source = {"domain": Path(domain_file),
              "problem": Path(problem_file),
              "plan": Path(plan_file)}
    for name, path in source.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} file not found: '{path}'")

    say("=== VLM PDDL validator ===")
    say(f"  Model   : {MODEL_NAME}  ({BASE_URL})")
    say(f"  Domain  : {source['domain']}")
    say(f"  Problem : {source['problem']}")
    say(f"  Plan    : {source['plan']}")

    domain_text  = source["domain"].read_text(encoding="utf-8")
    problem_text = source["problem"].read_text(encoding="utf-8")
    plan_text    = source["plan"].read_text(encoding="utf-8")

    say("[INFO] Pass 0 - symbolic validation of the input files ...")
    report = validate(domain_text, problem_text, plan_text)
    _print_report(report)

    for round_index in range(1, MAX_REPAIR_ROUNDS + 1):
        if report.is_clean and not (ALWAYS_REVIEW and round_index == 1):
            break

        reason = "reviewing" if report.is_clean else "repairing"
        say(f"[INFO] Round {round_index} - {reason} with the VLM ...")
        prompt = build_review_prompt(domain_text, problem_text, plan_text,
                                     report, round_index)
        response = call_vlm(build_messages(prompt))

        try:
            sections = parse_revision(response)
        except ValueError as exc:
            say(f"  [WARNING] {exc}")
            say("  [WARNING] keeping the previous version of the files.")
            break

        candidate = validate(sections["domain"], sections["problem"], sections["plan"])

        # Guard against a revision that is worse than what we already have.
        if not candidate.is_clean and report.simulated and \
                len(candidate.errors) > len(report.errors):
            say(f"  [WARNING] the revision has {len(candidate.errors)} error(s) "
                  f"against {len(report.errors)} before; discarding it.")
            break

        domain_text  = sections["domain"]
        problem_text = sections["problem"]
        plan_text    = sections["plan"]
        report = candidate

        if "notes" in sections and sections["notes"]:
            say("  VLM notes:")
            for line in sections["notes"].splitlines():
                if line.strip():
                    say(f"    - {line.strip()}")

        say(f"  Symbolic re-check after round {round_index}:")
        _print_report(report, indent="      ")

        if report.is_clean:
            break

    written = write_outputs(domain_text, problem_text, plan_text,
                            report, output_dir, source)

    say("[DONE] Revised PDDL written:")
    for name in ("domain", "problem", "plan", "report"):
        say(f"    {name:8s} -> {written[name]}")
    say(f"    status   : {'VALID' if report.is_clean else 'ISSUES REMAIN'}")
    return written


if __name__ == "__main__":
    review_and_correct()
