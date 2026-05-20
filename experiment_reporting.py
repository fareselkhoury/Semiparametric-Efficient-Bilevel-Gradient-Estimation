import math
from pathlib import Path


def finite_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def format_number(value, digits=4, sci_cutoff=1e-3):
    value = finite_float(value)
    if value is None:
        return "--"
    if value != 0.0 and abs(value) < sci_cutoff:
        return f"{value:.2e}".replace("e-0", "e-").replace("e+0", "e")
    return f"{value:.{digits}f}"


def format_pm(value, se=None, digits=4):
    text = format_number(value, digits=digits)
    se = finite_float(se)
    if se is None:
        return text
    return f"{text} ({format_number(1.96 * se, digits=digits)})"


def format_lambda(value):
    value = finite_float(value)
    if value is None:
        return "--"
    exponent = int(math.floor(math.log10(value))) if value > 0 else 0
    mantissa = value / (10**exponent) if value > 0 else value
    if value > 0 and abs(mantissa - 1.0) < 1e-10:
        return f"$10^{{{exponent}}}$"
    if value > 0:
        mantissa_text = f"{mantissa:.0f}" if abs(mantissa - round(mantissa)) < 1e-10 else f"{mantissa:.1f}"
        return rf"${mantissa_text}\times 10^{{{exponent}}}$"
    return format_number(value)


def latex_escape(text):
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in str(text))


def write_latex_table(path, caption, label, columns, rows, align=None, notes=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if align is None:
        align = "l" + "c" * (len(columns) - 1)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(str(cell) for cell in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    if notes:
        lines.append(r"\vspace{2pt}")
        lines.append(r"\footnotesize{" + notes + "}")
    lines.append(r"\end{table}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
