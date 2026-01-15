# Research Mode
You are in RESEARCH mode. Your task is to gather up-to-date information and document findings with clear attribution.

## Phase 0: Orient
0a. Study `specs/*` with up to 250 parallel Sonnet subagents to understand what the application needs.
0bb. if @IMPLEMENTATION_PLAN.md is not present, create it with RESEARCH section.
0b. Study @IMPLEMENTATION_PLAN.md (if present) to identify areas requiring research. 
0c. Study `research/*` (if present) to avoid duplication.

## Phase 1: Research
1. Identify topics from `specs/*` and @IMPLEMENTATION_PLAN.md benefiting from research. Use up to 500 Sonnet subagents with WebSearch to find academic papers, best practices, and standards (prefer 2026 sources). Use 'WebSearch' regularly.
2. Create research documents in `research/` with: summary, sources table with URLs, key findings with citations and confidence levels, and actionable recommendations.
3. Use an Opus subagent with 'ultrathink' to update `specs/*.md` and @IMPLEMENTATION_PLAN.md with findings.

## Guardrails
- **Attribution is mandatory**: Every finding must cite its source with URL
- **No speculation**: If you cannot find authoritative sources, document the gap