"""
pine_parser.py — Lightweight Pine Script analyzer (common subset).

This is NOT a full Pine compiler. It does just enough to:
  * detect indicator vs strategy and the declared title
  * pull out `input.*(...)` declarations (tunable params + defaults)
  * pull out `strategy.*(title=...)` / `indicator(title=...)` lines
  * report the constructs we don't transpile (so we never silently fake a result)

The transpiler (pine_transpiler.py) consumes the output of parse().
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ── Regexes ────────────────────────────────────────────────────────────────
RE_TITLE = re.compile(
    r"(?:indicator|strategy)\s*\(\s*[\"']([^\"']+)[\"']"
    r"|(?:indicator|strategy)\s*\([^)]*?title\s*=\s*[\"']([^\"']+)[\"']"
)
RE_KIND = re.compile(r"^\s*(indicator|strategy)\s*\(")
RE_INPUT = re.compile(
    r"^\s*(?:\[[^\]]*\]\s*)?(\w+)\s*=\s*input(?:\.\w+)?\s*\("
    r"(?:title\s*=\s*[\"']([^\"']*)[\"'])?"
    r"(?=[^)]*\))"          # lookahead until the close paren of the input() call
)
RE_INPUT_FULL = re.compile(
    r"input(?:\.\w+)?\s*\((.*?)\)", re.DOTALL
)
# match a single `tok = input.xxx(...)` assignment, capturing the whole RHS call
RE_INPUT_ASSIGN = re.compile(
    r"^\s*(\w+)\s*=\s*input(?:\.\w+)?\s*\((?P<args>.*?)\)\s*$",
    re.DOTALL | re.MULTILINE,
)

# keywords / functions we deliberately do NOT support
UNSUPPORTED_KEYWORDS = [
    "security(", "request.security(", "var ", "varip ", "matrix.", "array.",
    "line.new", "box.new", "label.new", "table.new", "plotchar", "barcolor",
    "bgcolor", "fill(", "hline(", "alertcondition", "strategy.risk",
    "strategy.initial_capital", "for ", "while ", "switch ", "ta.cum",
    "math.sum", "ta.valuewhen", "ta.barssince", "ta.highestbars",
    "ta.lowestbars", "ta.crossed", "ta.pivothigh", "ta.pivotlow",
    "iff(", "na", "nz(", "ta.change", "ta.rising", "ta.falling",
]
# lighter-weight "smell" markers
SMELL_MARKERS = ["security", "request.security", "array", "matrix", "var "]


@dataclass
class ParseResult:
    kind: str = "unknown"          # 'indicator' | 'strategy' | 'unknown'
    title: str = "Untitled Pine"
    inputs: List[Dict] = field(default_factory=list)   # [{name, type, default, title, min, max, step}]
    unsupported: List[str] = field(default_factory=list)
    smells: List[str] = field(default_factory=list)
    line_count: int = 0
    has_strategy_entry: bool = False
    raw: str = ""

    @property
    def is_strategy(self) -> bool:
        return self.kind == "strategy" or self.has_strategy_entry


def _split_top_level_args(argstr: str) -> List[str]:
    """Split an input(...) arg string on commas that are NOT inside brackets/parens."""
    out, depth, cur = [], 0, []
    for ch in argstr:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        out.append("".join(cur).strip())
    return out


def _parse_one_input(name: str, inner: str) -> Dict:
    """Parse the body of one input()/input.int()/... call.

    Supports: title=, defval=, minval=, maxval=, step=, options=[...], and a
    bare default as the first positional argument.
    """
    args = _split_top_level_args(inner)
    d = {"name": name, "type": "float", "default": 0.0,
         "title": name, "min": None, "max": None, "step": None,
         "options": None}

    # positional first arg (default value) if present and no defval given
    positional_default = None
    if args and "=" not in args[0]:
        positional_default = args[0]
        args = args[1:]

    for a in args:
        if "=" not in a:
            continue
        k, v = a.split("=", 1)
        k, v = k.strip(), v.strip()
        if k == "title":
            d["title"] = v.strip("\"'")
        elif k == "defval":
            d["default"] = _coerce(v)
        elif k == "minval":
            d["min"] = _coerce(v)
        elif k == "maxval":
            d["max"] = _coerce(v)
        elif k == "step":
            d["step"] = _coerce(v)
        elif k == "options":
            d["options"] = v

    if positional_default is not None and d["default"] == 0.0:
        d["default"] = _coerce(positional_default)

    # infer type from default
    if isinstance(d["default"], bool):
        d["type"] = "bool"
    elif isinstance(d["default"], int):
        d["type"] = "int"
    else:
        d["type"] = "float"
    return d


def _coerce(v: str):
    v = v.strip()
    if v is None:
        return 0.0
    if v.lower() in ("true",):
        return True
    if v.lower() in ("false",):
        return False
    try:
        if "." in v or "e" in v.lower():
            return float(v)
        return int(v)
    except (TypeError, ValueError):
        # strip quotes → string, but for numeric inputs this is unusual
        return v.strip("\"'")


def parse(source: str) -> ParseResult:
    """Analyze Pine source. Returns ParseResult with metadata + warnings."""
    res = ParseResult()
    res.raw = source
    lines = source.splitlines()
    res.line_count = len(lines)

    first = ""
    for ln in lines:
        m = RE_KIND.match(ln)
        if m and not first:
            first = m.group(1)
            res.kind = m.group(1)
            break
    # also detect strategy on the title line regardless of indentation
    tm = RE_TITLE.search(source)
    if tm:
        res.title = tm.group(1) or tm.group(2) or res.title

    # inputs
    for m in RE_INPUT_ASSIGN.finditer(source):
        name, inner = m.group(1), m.group("args")
        # inner may contain nested parens from options=[...]; extract only the
        # first-balanced input(...) by re-searching from the 'input' keyword.
        full = re.search(r"input(?:\.\w+)?\s*\((.*)\)\s*$", m.group(0), re.DOTALL)
        inner_args = full.group(1) if full else inner
        d = _parse_one_input(name, inner_args)
        # avoid duplicates
        if not any(x["name"] == d["name"] for x in res.inputs):
            res.inputs.append(d)

    # strategy entry/exit detection
    if re.search(r"strategy\.(entry|close|exit|order)\s*\(", source):
        res.has_strategy_entry = True

    # unsupported constructs
    seen_unsup = set()
    for marker in UNSUPPORTED_KEYWORDS:
        if marker in source:
            seen_unsup.add(marker)
    res.unsupported = sorted(seen_unsup)

    # smells (heavy cross-series / stateful stuff)
    seen_smell = set()
    for marker in SMELL_MARKERS:
        if marker in source:
            seen_smell.add(marker)
    res.smells = sorted(seen_smell)

    return res


def summarize(res: ParseResult) -> str:
    """Human-readable summary of what we found / can't handle."""
    lines = [
        f"Kind   : {res.kind}",
        f"Title  : {res.title}",
        f"Lines  : {res.line_count}",
        f"Inputs : {len(res.inputs)}  "
        + (", ".join(f"{i['name']}={i['default']}" for i in res.inputs) or "-"),
        f"Strategy entry/exit detected: {res.has_strategy_entry}",
    ]
    if res.unsupported:
        lines.append("UNSUPPORTED constructs (won't be transpiled faithfully):")
        lines += [f"   - {u}" for u in res.unsupported]
    if res.smells:
        lines.append("Heavier constructs that may degrade fidelity:")
        lines += [f"   ~ {s}" for s in res.smells]
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    src = sys.stdin.read()
    r = parse(src)
    print(summarize(r))
    import json
    print(json.dumps({"inputs": r.inputs}, indent=2))
