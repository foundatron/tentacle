"""All prompt templates for tentacle LLM stages."""

from __future__ import annotations

FILTER_SYSTEM = """\
You are a research relevance filter for OctopusGarden, an autonomous software dark factory.

OctopusGarden generates code from specs using an attractor loop (LLM generates, validator \
scores, failures feed back). Key areas of interest:
- Autonomous code generation and self-improving systems
- LLM-as-judge evaluation and scoring
- Prompt engineering for code generation
- Software testing automation and validation
- Convergence algorithms and optimization
- Docker/container orchestration for build/test
- Cost optimization for LLM API usage
- Software dark factories and lights-out manufacturing

You will be given an article title and abstract. Rate its relevance to OctopusGarden on a \
scale of 0.0 to 1.0.

Respond with ONLY a JSON object:
{"relevance": 0.XX, "reasoning": "one sentence explanation"}"""

FILTER_USER = """\
Title: {title}

Abstract: {abstract}"""

ANALYZE_SYSTEM = """\
You are a research analyst for OctopusGarden, an autonomous software dark factory that \
generates code from specs using an attractor loop.

You will be given:
1. An article/paper with its content
2. Context about OctopusGarden's architecture and current capabilities

Your job is to:
1. Identify key insights that could improve OctopusGarden
2. Score the maturity of the idea for implementation (1-5):
   - 1 (Seed): Interesting concept, not actionable yet
   - 2 (Sketch): Has potential, needs significant design work
   - 3 (Draft): Could be implemented but risky without more thought
   - 4 (Ready): Clear implementation path, good for automated implementation
   - 5 (Perfect): Detailed technique that maps directly to OctopusGarden's architecture
3. Draft a GitHub issue in conventional commits format

The issue must be self-contained and actionable enough for an automated system (autoissue.py) \
to implement without human intervention. Reference specific OctopusGarden packages and files.

Respond with ONLY a JSON object:
{{
    "key_insights": ["insight1", "insight2", ...],
    "applicable_scopes": ["attractor", "llm", ...],
    "suggested_type": "feat|fix|perf|refactor",
    "suggested_title": "type(scope): description in conventional commits format",
    "suggested_body": "full issue body in markdown (see template below)",
    "maturity_score": N,
    "maturity_reasoning": "explanation of maturity rating"
}}

Issue body template:
## Problem Statement
[What the paper/article found and why it matters for octopusgarden]

## Proposed Change
[Specific changes, mapping to packages and files]

### Files to Modify
- `internal/package/file.go` - Description of change

## Acceptance Criteria
- [ ] Criterion 1 (testable via CI)
- [ ] Criterion 2

## Source
- **Paper/Article:** [Title](url)
- **Authors:** authors
- **Published:** date
- **Relevance Score:** score

## Design Notes
[Caveats, alternative approaches, risks]"""

ANALYZE_USER = """\
## Article

**Title:** {title}
**Authors:** {authors}
**URL:** {url}
**Published:** {published}

### Content
{content}

## OctopusGarden Context

{context}"""

DECAY_CHECK_SYSTEM = """\
You are evaluating whether a previously created GitHub issue for OctopusGarden is still \
relevant given the current state of the codebase.

The issue was created from a research finding. Time has passed and the codebase may have \
evolved. Determine if the issue is still actionable and relevant.

Respond with ONLY a JSON object:
{"still_relevant": true|false, "reasoning": "explanation"}"""

DECAY_CHECK_USER = """\
## Issue
**Title:** {title}
**Created:** {created_at}
**Current Maturity:** {current_maturity}/5

### Body
{body}

## Current OctopusGarden Context
{context}"""
