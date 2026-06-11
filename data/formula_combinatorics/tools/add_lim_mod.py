#!/usr/bin/env python3
"""Add \\limits variants to integral/sum/product templates across all domain files.

STATUS: Applied migration — this script has already been run against all domain
files. The ``_LIM_MOD`` slot (``S(("", r"\\limits"))``) is now present in every
relevant template. Do NOT re-run without resetting domain files first.

For each template whose latex contains \\int_{{, \\sum_{{, \\prod_{{, or \\oint_{{
(explicit bounded operators), this script:
  1. Inserts {lim_mod} between the operator and the subscript in the latex string.
  2. Adds "lim_mod": S(("", r"\\limits")), as the first entry in the template's
     slots dict.

The lim_mod slot draws "" (no change) or "\\limits" with equal probability,
producing both display styles in the training corpus.
"""

import re
import sys
from pathlib import Path

OPERATORS = [r"\int", r"\sum", r"\prod", r"\oint"]
LIM_MOD_LINE = r'    "lim_mod": S(("", r"\limits")),' + "\n"


def transform(content: str) -> str:
    lines = content.splitlines(keepends=True)

    # Pass 1: replace \op_{{ with \op{lim_mod}_{{ in every line.
    # These patterns only occur inside latex= string literals in the templates.
    pass1 = []
    for line in lines:
        for op in OPERATORS:
            line = line.replace(op + "_{{", op + "{lim_mod}_{{")
        pass1.append(line)

    # Pass 2: insert the lim_mod slot into the slots dict of each template
    # that now has {lim_mod} in its latex string.
    #
    # Algorithm: scan line-by-line; when we see {lim_mod} (indicating we're
    # inside a template that needs the slot), set pending=True.  The next
    # "slots={" line is that template's slots dict — insert there.
    pass2 = []
    pending = False

    for line in pass1:
        if "{lim_mod}" in line:
            pending = True

        if pending:
            m = re.match(r"^(\s+)slots=\{", line)
            if m:
                indent = m.group(1)
                slot_line = indent + LIM_MOD_LINE.lstrip()
                if '"lim_mod"' not in line:
                    stripped = line.rstrip()
                    # Detect whether the entire dict is on this one line.
                    # A one-liner ends with "}" or "}," (possibly with trailing comma/paren).
                    # A multi-liner ends with just "{".
                    if stripped.endswith("{"):
                        # Multi-line dict — insert after opening brace
                        pass2.append(line)
                        pass2.append(slot_line)
                    else:
                        # Entire dict is on one line: slots={...},
                        # Expand to multi-line with lim_mod first.
                        brace_idx = line.index("slots={") + len("slots={")
                        prefix = line[:brace_idx]  # everything up to and including "{"
                        rest = line[brace_idx:].rstrip()  # existing content + closing
                        # rest looks like  "n": S(...)}  or  }  etc.
                        # Strip the trailing closing brace(s) and comma
                        # Find the matching closing brace (last "}" in rest)
                        close_match = re.search(r"\}([,)]?\s*)$", rest)
                        if close_match:
                            inner = rest[: close_match.start()]  # content between { and }
                            trailing = close_match.group(1)  # comma or empty
                            pass2.append(prefix.rstrip() + "\n")
                            pass2.append(slot_line)
                            if inner.strip():
                                pass2.append(indent + "    " + inner.strip() + "\n")
                            pass2.append(indent + "}" + trailing + "\n")
                        else:
                            # Fallback: just append and warn
                            pass2.append(line)
                            pass2.append(slot_line)
                    pending = False
                    continue
                else:
                    # lim_mod already present (e.g. multiple operators in one template)
                    pending = False

        pass2.append(line)

    return "".join(pass2)


def main() -> None:
    domains_dir = Path(__file__).parent.parent / "formula_combinatorics" / "domains"
    files = sorted(domains_dir.glob("*.py"))
    if not files:
        print("No domain files found.", file=sys.stderr)
        sys.exit(1)

    total_changed = 0
    for path in files:
        if path.name == "__init__.py":
            continue
        original = path.read_text()
        transformed = transform(original)
        if transformed != original:
            path.write_text(transformed)
            total_changed += 1
            print(f"  modified  {path.name}")
        else:
            print(f"  unchanged {path.name}")

    print(f"\nDone: {total_changed}/{len(files) - 1} domain files modified.")


if __name__ == "__main__":
    main()
