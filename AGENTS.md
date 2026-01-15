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
# Install dependencies
uv sync

# Run weather pipeline scripts in order
uv run python scripts/01_inspect_and_standardise.py
uv run python scripts/02_areal_aggregation.py
uv run python scripts/03_climatology.py
uv run python scripts/04_indices_and_anomalies.py
uv run python scripts/05_visualise.py
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
- `scripts/` - Weather pipeline scripts (01-05)
- `configs/` - YAML configuration files
- `artefacts/` - Generated visualisations
- `data/` - Local data files

## Operational Notes
- zarr v2 required (v3 incompatible with existing data)
- Drop non-numeric variables before computing climatology/indices
- Convert geoid to str before writing zarr

## Codebase Patterns
- Scripts use xarray for data handling
- Outputs go to `artefacts/weather_risk/`
- Config YAML in `configs/vietnam_coffee_indices.yaml`
