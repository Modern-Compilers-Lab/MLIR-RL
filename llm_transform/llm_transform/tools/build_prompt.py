"""Render the initial Claude prompt.

Reads `resources/prompt/prompt.txt` and substitutes its `${SCOPE}` placeholder
with a scope description derived from the `INSTANCE_FILTER` environment variable
(space- or comma-separated benchmark names and/or full `{name}_{instance}` IDs).
Empty value selects `resources/prompt/scope_all.txt`; otherwise the tokens are
rendered into `resources/prompt/scope_subset.txt` via its `${TOKENS}` placeholder.

Writes the rendered prompt to stdout for the launch script to capture.
"""
import sys
from pathlib import Path
from string import Template

from llm_transform.utils.instance_filter import parse_instance_filter

PROMPT_DIR = Path(__file__).resolve().parents[2] / "resources" / "prompt"
PROMPT_FILE = PROMPT_DIR / "prompt.txt"
SCOPE_ALL = PROMPT_DIR / "scope_all.txt"
SCOPE_SUBSET = PROMPT_DIR / "scope_subset.txt"


def _build_scope() -> str:
    tokens = parse_instance_filter()
    if tokens is None:
        return SCOPE_ALL.read_text().rstrip("\n")

    rendered_tokens = ", ".join(f"`{t}`" for t in sorted(tokens))
    template = Template(SCOPE_SUBSET.read_text().rstrip("\n"))
    return template.substitute(TOKENS=rendered_tokens)


def main() -> None:
    prompt = Template(PROMPT_FILE.read_text()).substitute(SCOPE=_build_scope())
    sys.stdout.write(prompt)


if __name__ == "__main__":
    main()
