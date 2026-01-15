# Operational Context

## Saving progress
- Use Git for version control. Commit each loop with clear messages.
- Push changes to the remote repository on GitHub frequently.
- use the `gh cli` tool.

## Background tasks
- if you are running long tasks, use `tmux` to keep them running in the background. 
- `tmux new-session -d -s mysession` to create a new session
- `tmux attach -t mysession` to attach to the session
- `tmux send-keys -t %3 'echo hello'` to send commands to the session
- to send longer commands
```bash
tmux set-buffer -b cmd 'echo one
echo two
echo three'
tmux paste-buffer -b cmd -t %3
tmux send-keys -t %3 C-m
```
- read outputs from tmux sessions: `tmux capture-pane -t %3 -p`
- grab the last N lines: `tmux capture-pane -t %3 -p -S -100`
- see currently running commands: `tmux display-message -p -t %3 '#{pane_current_command}'`
- If the pane is running a full-screen program (vim, htop), capture-pane will capture what is on screen, not necessarily the full prior output.
- Some commands buffer output. If you need real-time logs, redirect to a file and tail -f.
- Targeting is everything: always use session:window.pane or pane ids from list-panes.

## Build & Run

```bash
# Build
[to be discovered]
# Run
[to be discovered]
```

## Validation

Run these after implementing to get immediate feedback:

```bash
# Tests
[to be discovered]
# Typecheck
uv --with ty run ty check
# Lint
uv --with ruff run ruff check . --fix 
```
## Exit Criteria Verification
Run these before outputting completion signal:
```bash
# Must ALL pass before <promise>COMPLETE</promise>
# Tests
[to be discovered]
# Typecheck
uv --with ty run ty check
# Lint
uv --with ruff run ruff check . --fix 
```

## Project Structure
- `src/` - Application source code
- `src/lib/` - Shared utilities and components
- `specs/` - Requirement specifications

## Operational Notes
<!--
LLM: Update this section when you learn something new about running the project.
Keep it brief. Status updates belong in IMPLEMENTATION_PLAN.md, not here.
-->

## Codebase Patterns
<!--
LLM: Document patterns you discover here.
Example: "Error handling uses anyhow::Result throughout"
Example: "All CLI commands are in src/commands/"
-->
