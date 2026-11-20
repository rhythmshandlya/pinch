---
name: pinch-decision
description: Record a technical decision in docs/DECISIONS.md the way this project requires - the change, the measurement behind it, the sources (papers or well-maintained, widely starred projects), what was rejected and why. Use whenever a design, tooling, training or behaviour choice is made or a candidate is adopted or rejected.
---

# Write a decision entry

## Before deciding

Search for evidence: arXiv or conference papers, and established open-source projects. For a project, check it
is alive and widely used: `gh api repos/<owner>/<repo> --jq '"\(.stargazers_count) \(.pushed_at)"'`. Prefer
adopting an established tool to writing new code; write new code only for a gap the tools leave, and say what
the gap is.

## The entry

Append to `docs/DECISIONS.md` with the next number (in a parallel task branch, put it in the PR body instead and
let the integrator number it). Follow the existing entries' shape:

```markdown
90. **One-line decision.**
    - **Why:** the problem, with numbers.
    - **Evidence:** sources (paper or repo with stars), what they show.
    - **Change:** what was done, where (paths).
    - **Measured:** before -> after on the same bench, seeds and layouts.
    - **Rejected:** alternatives and why.
```

Keep it plain and short; numbers over adjectives. Never quote the private requirements document.
