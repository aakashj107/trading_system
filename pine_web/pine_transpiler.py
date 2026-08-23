"""
pine_transpiler.py — common-subset Pine → Python Strategy transpiler.

Scope (honest): it transpiles the *common indicator subset* of Pine v5:
  * variable assignments using OHLCV, ta.* math, comparisons, boolean ops
  * series operators: + - * /, comparisons, and/or, ?: ternary, parentheses
  * helper calls: sma/ema/rsi/atr/adx/macd/cross/donchian/stoch/roc/cci/mfi/
                   supertrend/hma/alma/kama/vwma/wma/highest/lowest/linreg
  * signal emission from: strategy.entry("Long"/"Short"), strategy.close(...),
                           strategy.order(...), and crossover style rules
  * input.* tunables (exposed as strategy params)

It does NOT (and cannot reliably) support: security()/request.security(),
var/varip stateful vars, arrays/matrices, loops, labels/lines/boxes, custom
functions, and most `strategy.*` order-management nuance. These are reported
up front by the parser so the user is never handed a silently-wrong backtest.

Output: a subclass of strategies.Strategy with a generate_signals(df) method
that reuses the SAME signal contract your Backtester expects (df["Signal"]).
"""

import ast
import os
import sys
import re
from typing import Dict, List, Optional

# make project root importable (strategies.py, strategies_pine.py live there)
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import numpy as np
import pandas as pd

from strategies import Strategy, Signal
from .pine_parser import parse as _parse_pine, ParseResult
from .pine_indicators import (
    ema, sma, rsi, atr, adx, linreg, hma, alma, kama, vwma, wma,
    highest, lowest, donchian, cross, crossunder, macd, stoch,
    roc, cci, mfi, supertrend,
)

# OHLCV + common builtins available inside generated signal code
COMPILE_NS = {
    "open": "df['Open']", "high": "df['High']", "low": "df['Low']",
    "close": "df['Close']", "volume": "df['Volume']",
    "hl2": "(df['High']+df['Low'])/2", "ohlc4": "(df['Open']+df['High']+df['Low']+df['Close'])/4",
    "hlc3": "(df['High']+df['Low']+df['Close'])/3",
    # helpers
    "sma": "sma", "ema": "ema", "rsi": "rsi", "atr": "atr", "adx": "adx",
    "linreg": "linreg", "hma": "hma", "alma": "alma", "kama": "kama",
    "vwma": "vwma", "wma": "wma", "highest": "highest", "lowest": "lowest",
    "donchian": "donchian", "cross": "cross", "crossunder": "crossunder",
    "macd": "macd", "stoch": "stoch", "roc": "roc", "cci": "cci",
    "mfi": "mfi", "supertrend": "supertrend", "abs": "abs", "min": "np.minimum",
    "max": "np.maximum", "sqrt": "np.sqrt", "pow": "np.power",
    "math_abs": "abs", "math_max": "np.maximum", "math_min": "np.minimum",
    "math_sqrt": "np.sqrt", "math_pow": "np.power", "math_round": "np.round",
    "np": "np", "pd": "pd", "nan": "np.nan",
}

TA_FUNCS = {"sma", "ema", "rsi", "atr", "adx", "linreg", "hma", "alma", "kama",
            "vwma", "wma", "highest", "lowest", "donchian", "cross",
            "crossunder", "macd", "stoch", "roc", "cci", "mfi", "supertrend"}


class TranspileError(Exception):
    pass


# ── Expression compiler (Pine expr → Python source string) ─────────────────
def compile_expr(node, params: Dict[str, str], warns: List[str]) -> str:
    """Recursively turn a Python ast node into a Python source string.

    `params` maps the user's input variable names → their parameter accessor
    (e.g. 'self._p["len"]'). OHLCV/helpers come from COMPILE_NS.
    """
    t = type(node).__name__

    if t == "Expression":
        return compile_expr(node.body, params, warns)
    if t == "Module":
        # single expression module
        stmts = [s for s in node.body if isinstance(s, ast.Expr)]
        if len(stmts) == 1:
            return compile_expr(stmts[0].value, params, warns)
        raise TranspileError("Expected a single expression")

    if t == "Constant":
        if isinstance(node.value, bool):
            return "True" if node.value else "False"
        if isinstance(node.value, (int, float)):
            return repr(node.value)
        return repr(str(node.value))

    if t == "Name":
        name = node.id
        if name in params:
            return params[name]
        if name in COMPILE_NS:
            return COMPILE_NS[name]
        if name in ("na", "NaN"):
            return "np.nan"
        if name == "true":
            return "True"
        if name == "false":
            return "False"
        raise TranspileError(f"Unknown identifier '{name}'")

    if t == "BinOp":
        L = compile_expr(node.left, params, warns)
        R = compile_expr(node.right, params, warns)
        op = type(node.op).__name__
        sym = {"Add": "+", "Sub": "-", "Mult": "*", "Div": "/",
               "Mod": "%", "Pow": "**", "FloorDiv": "//"}.get(op)
        if sym is None:
            raise TranspileError(f"Unsupported operator {op}")
        return f"({L} {sym} {R})"

    if t == "UnaryOp":
        op = type(node.op).__name__
        V = compile_expr(node.operand, params, warns)
        if op == "USub":
            return f"(-{V})"
        if op == "UAdd":
            return f"(+{V})"
        if op == "Not":
            return f"(not {V})"
        raise TranspileError(f"Unsupported unary op {op}")

    if t == "BoolOp":
        vals = [compile_expr(v, params, warns) for v in node.values]
        op = " and " if type(node.op).__name__ == "And" else " or "
        return "(" + op.join(vals) + ")"

    if t == "Compare":
        L = compile_expr(node.left, params, warns)
        out = [L]
        for op, comp in zip(node.ops, node.comparators):
            R = compile_expr(comp, params, warns)
            o = type(op).__name__
            sym = {"Gt": ">", "Lt": "<", "GtE": ">=", "LtE": "<=",
                   "Eq": "==", "NotEq": "!="}.get(o)
            if sym is None:
                raise TranspileError(f"Op {o} unsupported")
            out.append(f" {sym} {R}")
        return "(" + "".join(out) + ")"

    if t == "Call":
        fname = None
        if isinstance(node.func, ast.Name):
            fname = node.func.id
        elif isinstance(node.func, ast.Attribute):
            # ta.sma -> sma ; ta.cross -> cross ; math.xxx -> math_xxx
            base = node.func.value.id if isinstance(node.func.value, ast.Name) else ""
            attr = node.func.attr
            if base in ("ta", "math"):
                fname = attr if base == "ta" else "math_" + attr
            else:
                raise TranspileError(f"Unsupported call {base}.{attr}")
        if fname is None:
            raise TranspileError("Unsupported call expression")

        # rename a few ta.* spelling variants
        rename = {"change": "roc", "rising": "_rising", "falling": "_falling",
                  "highest": "highest", "lowest": "lowest", "cross": "cross",
                  "crossover": "cross", "crossunder": "crossunder"}
        fname = rename.get(fname, fname)

        args = [compile_expr(a, params, warns) for a in node.args]
        if fname in ("cross", "crossunder", "crossover"):
            if len(args) != 2:
                raise TranspileError(f"{fname} needs 2 args")
            return f"{fname}({args[0]}, {args[1]})"
        if fname in ("supertrend",):
            # supertrend(high, low, close, atr_p, mult) -> returns (series, dir)
            if len(args) < 5:
                raise TranspileError("supertrend needs 5 args")
            return f"{fname}({args[0]}, {args[1]}, {args[2]}, {args[3]}, {args[4]})[0]"
        if fname not in TA_FUNCS and fname not in ("_rising", "_falling",):
            raise TranspileError(f"Unsupported function '{fname}'")
        if fname in ("_rising", "_falling"):
            if len(args) != 2:
                raise TranspileError(f"{fname} needs 2 args")
            src = "roc" if fname == "_rising" else "roc"
            # rising(x,n): x > x.shift(n)
            return f"({args[0]} > {args[0]}.shift({args[1]}))"
        return f"{fname}({', '.join(args)})"

    if t == "IfExp":  # a ? b : c
        C = compile_expr(node.test, params, warns)
        A = compile_expr(node.body, params, warns)
        B = compile_expr(node.orelse, params, warns)
        return f"({A} if {C} else {B})"

    if t == "Attribute":
        # e.g. close, open — handled by Name normally; ignore
        raise TranspileError("Unsupported attribute access")

    raise TranspileError(f"Unsupported expression node: {t}")


# ── Pre-processing helpers ─────────────────────────────────────────────────
def _strip_comments(line: str) -> str:
    """Remove Pine `//` comments, ignoring those inside quotes."""
    out, in_str, q = [], False, ""
    i = 0
    while i < len(line):
        ch = line[i]
        if in_str:
            out.append(ch)
            if ch == q:
                in_str = False
        else:
            if ch in "\"'":
                in_str = True
                q = ch
                out.append(ch)
            elif ch == "/" and i + 1 < len(line) and line[i + 1] == "/":
                break
            else:
                out.append(ch)
        i += 1
    return "".join(out).rstrip()


def _clean_source(source: str) -> str:
    lines = []
    for ln in source.splitlines():
        s = _strip_comments(ln)
        if s.strip():
            lines.append(s)
    return "\n".join(lines)


# ── Extraction of user variable assignments ───────────────────────────────
_ASSIGN_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*=(?!=)(?![<>])\s*(.+?)\s*$")


def _extract_assignments(src: str):
    """Return list of (varname, expr_str) for top-level `var = expr`."""
    out = []
    for ln in src.splitlines():
        stripped = ln.strip()
        if not stripped or stripped.startswith(("if", "else", "for", "while",
                "strategy.", "plot", "alert", "indicator(", "study(")):
            continue
        m = _ASSIGN_RE.match(ln)
        if m:
            name, expr = m.group(1), m.group(2).strip()
            if expr.startswith("input") or name in ("ta", "math"):
                continue
            if ": " in expr and "(" not in expr:
                continue
            out.append((name, expr))
    return out


# ── Signal extraction (entry / exit conditions) ───────────────────────────
_RE_IF_ENTRY = re.compile(
    r"if\s*\(([^()]*)\)\s*strategy\.entry\s*\([^)]*?(long|short)",
    re.IGNORECASE | re.DOTALL)
_RE_IF_CLOSE = re.compile(
    r"if\s*\(([^()]*)\)\s*strategy\.close\s*\(", re.IGNORECASE | re.DOTALL)


def _extract_signals(src: str):
    sig = {"entry_long": None, "entry_short": None,
           "exit_long": None, "exit_short": None}
    for m in re.finditer(
            r"strategy\.entry\s*\(([^)]*)\)", src, re.IGNORECASE | re.DOTALL):
        body = m.group(1)
        when = re.search(r"when\s*=\s*([^\),]+)", body)
        side_m = re.search(r"(?:strategy\.)?(long|short)", body, re.IGNORECASE)
        if not side_m:
            continue
        side = side_m.group(1).lower()
        cond = when.group(1).strip() if when else "True"
        sig["entry_long" if side == "long" else "entry_short"] = cond
    for m in _RE_IF_ENTRY.finditer(src):
        cond, side = m.group(1).strip(), m.group(2).lower()
        sig["entry_long" if side == "long" else "entry_short"] = cond
    for m in _RE_IF_CLOSE.finditer(src):
        sig["exit_long"] = m.group(1).strip()
    for m in re.finditer(
            r"strategy\.close\s*\(([^)]*)\)", src, re.IGNORECASE | re.DOTALL):
        when = re.search(r"when\s*=\s*([^\),]+)", m.group(1))
        if when:
            sig["exit_long"] = when.group(1).strip()
    return sig


def _build_method(var_code, el, es, xl, xs, has_short):
    ind = "        "
    lines = list(var_code)
    lines.append(f"        _el = {el}")
    lines.append(f"        _es = {es}")
    lines.append(f"        _xl = {xl}")
    lines.append(f"        _xs = {xs}")
    # normalize the four condition series to boolean arrays
    for v in ("_el", "_es", "_xl", "_xs"):
        lines.append(f"        {v} = pd.Series({v}).fillna(False).astype(bool)")
    # build signals state machine
    lines.append("        signals = np.zeros(len(df), dtype=int)")
    lines.append("        in_pos = 0  # 0 flat, 1 long, -1 short")
    lines.append("        for i in range(1, len(df)):")
    lines.append("            if in_pos == 1:")
    lines.append("                if _xl.iloc[i]:")
    lines.append("                    signals[i] = Signal.SELL.value")
    lines.append("                    in_pos = 0")
    lines.append("            elif in_pos == -1:")
    lines.append("                if _xs.iloc[i]:")
    lines.append("                    signals[i] = Signal.COVER.value")
    lines.append("                    in_pos = 0")
    lines.append("            if in_pos == 0:")
    lines.append("                if _el.iloc[i]:")
    lines.append("                    signals[i] = Signal.BUY.value")
    lines.append("                    in_pos = 1")
    lines.append("                elif _es.iloc[i]:")
    lines.append("                    signals[i] = Signal.SHORT.value")
    lines.append("                    in_pos = -1")
    lines.append("        df['Signal'] = signals")
    lines.append("        return df")
    body = "\n".join(l for l in lines if l != "")
    return body


def transpile(source: str, param_overrides: Optional[Dict] = None):
    parsed = _parse_pine(source)
    param_overrides = param_overrides or {}
    hard = [u for u in parsed.unsupported
            if u in ("security(", "request.security(", "var ",
                     "array.", "matrix.", "for ", "while ", "switch ")]
    if hard:
        raise TranspileError(
            "Cannot transpile: contains unsupported constructs -> "
            + ", ".join(hard))
    clean = _clean_source(source)
    assigns = _extract_assignments(clean)
    sigs = _extract_signals(clean)
    params = {inp["name"]: f"self._p['{inp['name']}']"
              for inp in parsed.inputs}
    # every user-assigned var becomes a bare local name inside generate_signals
    for name, _ in assigns:
        if name not in params:
            params[name] = name
    defaults = {inp["name"]: param_overrides.get(inp["name"], inp["default"])
                for inp in parsed.inputs}
    warns: List[str] = list(parsed.smells)
    var_code = []
    for name, expr in assigns:
        try:
            py = compile_expr(ast.parse(expr, mode="eval"), params, warns)
        except (SyntaxError, TranspileError) as e:
            raise TranspileError(f"Could not compile `{name} = {expr}`: {e}")
        var_code.append(f"        {name} = {py}")
    el = _comp_cond(sigs["entry_long"], params, warns)
    es = _comp_cond(sigs["entry_short"], params, warns)
    xl = _comp_cond(sigs["exit_long"], params, warns)
    xs = _comp_cond(sigs["exit_short"], params, warns)
    has_short = sigs["entry_short"] is not None
    method_src = _build_method(var_code, el, es, xl, xs, has_short)
    cls_name = "PineTranspiled"
    ns = {"Strategy": Strategy, "Signal": Signal, "np": np, "pd": pd,
          "sma": sma, "ema": ema, "rsi": rsi, "atr": atr, "adx": adx,
          "linreg": linreg, "hma": hma, "alma": alma, "kama": kama,
          "vwma": vwma, "wma": wma, "highest": highest, "lowest": lowest,
          "donchian": donchian, "cross": cross, "crossunder": crossunder,
          "macd": macd, "stoch": stoch, "roc": roc, "cci": cci,
          "mfi": mfi, "supertrend": supertrend}
    cls_body = (
        f"class {cls_name}(Strategy):\n"
        f"    def __init__(self, **kwargs):\n"
        f"        self._p = {defaults!r}\n"
        f"        for k, v in kwargs.items():\n"
        f"            self._p[k] = v\n"
        f"        super().__init__({parsed.title!r})\n\n"
        f"    def generate_signals(self, df):\n"
        f"        df = df.copy()\n"
        f"        if len(df) < 2:\n"
        f"            df['Signal'] = 0\n"
        f"            return df\n"
        f"{method_src}\n"
    )
    try:
        exec(cls_body, ns)
    except Exception as e:
        raise TranspileError(f"Generated code failed: {e}\n{cls_body}")
    return ns[cls_name](), parsed


def _comp_cond(opt, params, warns):
    # None  → no rule found → never fire (let Backtester SL manage exits)
    # "True"→ explicit always-on (strategy.entry without a `when`)
    if opt is None:
        return "pd.Series(False, index=df.index)"
    if opt.strip() == "True":
        return "pd.Series(True, index=df.index)"
    try:
        return compile_expr(ast.parse(opt, mode="eval"), params, warns)
    except (SyntaxError, TranspileError) as e:
        raise TranspileError(f"Could not compile condition `{opt}`: {e}")
