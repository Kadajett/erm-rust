#!/usr/bin/env python3
"""
AIM Dashboard Setup — programmatically creates dashboards and bookmarks
for tracking active ERM training runs.

Creates dashboards with pre-configured explorer states that filter on
`run.active == True` so they always show whichever training run is
currently live.

Usage:
    python scripts/aim-setup-dashboards.py [--aim-url URL] [--reset]

Options:
    --aim-url URL   AIM server base URL (default: http://aim-dash.tailf93a13.ts.net:43800)
    --reset         Delete all existing dashboards before creating new ones
"""

import argparse
import json
import sys
import urllib.parse

import requests

DEFAULT_AIM_URL = "http://aim-dash.tailf93a13.ts.net:43800"


# ---------------------------------------------------------------------------
# Explorer state templates
# ---------------------------------------------------------------------------
# The `state` dict is the serialized UI config for each explorer type.
# The AIM frontend reads this and configures the explorer accordingly.
# Keys not present will use frontend defaults.

def _metrics_state(query: str, metrics: list[dict]) -> dict:
    """Metrics explorer state: line charts of selected metrics."""
    return {
        "select": {
            "advancedMode": True,
            "advancedQuery": query,
            "options": metrics,
            "query": query,
        },
        "liveUpdate": {
            "delay": 7000,
            "enabled": True,
        },
        "chart": {
            "ignoreOutliers": False,
            "zoom": {"active": False, "mode": 0, "history": []},
            "axesScaleType": {"xAxis": "linear", "yAxis": "linear"},
            "smoothing": {
                "algorithm": "EMA",
                "factor": 0,
                "curveInterpolation": "linear",
                "isApplied": False,
            },
            "aggregationConfig": {
                "methods": {"area": "min_max", "line": "mean"},
                "isApplied": False,
                "isEnabled": False,
            },
            "densityType": 500,
            "tooltip": {
                "appearance": "auto",
                "display": True,
                "selectedFields": [],
            },
            "highlightMode": "run",
            "focusedState": {"key": None, "xValue": None, "yValue": None, "active": False},
        },
        "grouping": {
            "color": ["run.hash"],
            "stroke": [],
            "chart": ["name"],
            "reverseMode": {"color": False, "stroke": False, "chart": False},
            "isApplied": {"color": True, "stroke": True, "chart": True},
            "persistence": {"color": False, "stroke": False},
            "seed": {"color": 10, "stroke": 10},
            "paletteIndex": 0,
        },
    }


def _scatters_state(query: str) -> dict:
    """Scatters explorer state: param vs metric scatter plots."""
    return {
        "select": {
            "advancedMode": True,
            "advancedQuery": query,
            "options": [],
            "query": query,
        },
        "liveUpdate": {
            "delay": 7000,
            "enabled": True,
        },
    }


def _params_state(query: str) -> dict:
    """Params explorer state: parallel coordinates for hyperparams."""
    return {
        "select": {
            "advancedMode": True,
            "advancedQuery": query,
            "options": [],
            "query": query,
        },
        "liveUpdate": {
            "delay": 7000,
            "enabled": True,
        },
    }


def _text_state(query: str) -> dict:
    """Text explorer state: IO examples and training events."""
    return {
        "select": {
            "advancedMode": True,
            "advancedQuery": query,
            "options": [],
            "query": query,
        },
        "liveUpdate": {
            "delay": 7000,
            "enabled": True,
        },
    }


# ---------------------------------------------------------------------------
# Dashboard definitions
# ---------------------------------------------------------------------------

ACTIVE_QUERY = "run.active == True"

DASHBOARDS = [
    {
        "name": "Training Loss & Accuracy",
        "description": (
            "Core training metrics: loss, loss_per_edit, edit_rate, edits, deaths. "
            "Auto-filters to active run."
        ),
        "app_type": "metrics",
        "state_fn": lambda: _metrics_state(
            query=(
                '(metric.name == "loss" or metric.name == "loss_per_edit" '
                'or metric.name == "edit_rate" or metric.name == "edits" '
                'or metric.name == "deaths") and run.active == True'
            ),
            metrics=[
                {"label": "loss", "value": {"option_name": "loss", "context": {"subset": "train"}}},
                {"label": "loss_per_edit", "value": {"option_name": "loss_per_edit", "context": {"subset": "train"}}},
                {"label": "edit_rate", "value": {"option_name": "edit_rate", "context": {"subset": "train"}}},
                {"label": "edits", "value": {"option_name": "edits", "context": {"subset": "train"}}},
                {"label": "deaths", "value": {"option_name": "deaths", "context": {"subset": "train"}}},
            ],
        ),
    },
    {
        "name": "Learning Rate & Temperature",
        "description": (
            "Optimizer and sampling dynamics: learning rate, leader/follower temps. "
            "Auto-filters to active run."
        ),
        "app_type": "metrics",
        "state_fn": lambda: _metrics_state(
            query=(
                '(metric.name == "lr" or metric.name == "leader_temp" '
                'or metric.name == "follower_temp") and run.active == True'
            ),
            metrics=[
                {"label": "lr", "value": {"option_name": "lr", "context": {"subset": "train"}}},
                {"label": "leader_temp", "value": {"option_name": "leader_temp", "context": {"subset": "train"}}},
                {"label": "follower_temp", "value": {"option_name": "follower_temp", "context": {"subset": "train"}}},
            ],
        ),
    },
    {
        "name": "Pheromone & Colony Health",
        "description": (
            "Ant colony dynamics: mean_phi, leader_fraction, total_ants, streak stats. "
            "Auto-filters to active run."
        ),
        "app_type": "metrics",
        "state_fn": lambda: _metrics_state(
            query=(
                '(metric.name == "mean_phi" or metric.name == "leader_fraction" '
                'or metric.name == "total_ants" or metric.name == "mean_streak" '
                'or metric.name == "max_streak" or metric.name == "zero_streak_ants") '
                'and run.active == True'
            ),
            metrics=[
                {"label": "mean_phi", "value": {"option_name": "mean_phi", "context": {"subset": "train"}}},
                {"label": "leader_fraction", "value": {"option_name": "leader_fraction", "context": {"type": "ant_colony"}}},
                {"label": "total_ants", "value": {"option_name": "total_ants", "context": {"type": "ant_colony"}}},
                {"label": "mean_streak", "value": {"option_name": "mean_streak", "context": {"type": "ant_colony"}}},
                {"label": "max_streak", "value": {"option_name": "max_streak", "context": {"type": "ant_colony"}}},
                {"label": "zero_streak_ants", "value": {"option_name": "zero_streak_ants", "context": {"type": "ant_colony"}}},
            ],
        ),
    },
    {
        "name": "GPU & System Health",
        "description": (
            "System monitoring: GPU utilization, GPU memory, GPU temp, GPU power, "
            "CPU, disk, memory. Auto-filters to active run."
        ),
        "app_type": "metrics",
        "state_fn": lambda: _metrics_state(
            query=(
                '(metric.name == "__system__gpu" or metric.name == "__system__gpu_memory_percent" '
                'or metric.name == "__system__gpu_temp" or metric.name == "__system__gpu_power_watts" '
                'or metric.name == "__system__cpu" or metric.name == "__system__memory_percent") '
                'and run.active == True'
            ),
            metrics=[
                {"label": "gpu_util", "value": {"option_name": "__system__gpu", "context": {"gpu": 0}}},
                {"label": "gpu_mem", "value": {"option_name": "__system__gpu_memory_percent", "context": {"gpu": 0}}},
                {"label": "gpu_temp", "value": {"option_name": "__system__gpu_temp", "context": {"gpu": 0}}},
                {"label": "gpu_power", "value": {"option_name": "__system__gpu_power_watts", "context": {"gpu": 0}}},
                {"label": "cpu", "value": {"option_name": "__system__cpu", "context": {}}},
                {"label": "memory", "value": {"option_name": "__system__memory_percent", "context": {}}},
            ],
        ),
    },
    {
        "name": "IO Examples (Text)",
        "description": (
            "Text explorer showing model input/output samples from training. "
            "Auto-filters to active run."
        ),
        "app_type": "runs",
        "state_fn": lambda: _text_state(
            query='run.active == True'
        ),
    },
    {
        "name": "Hyperparameter Scatter",
        "description": (
            "Scatter plots: correlate hyperparameters with final metrics. "
            "Shows all runs (not just active) for comparison."
        ),
        "app_type": "scatters",
        "state_fn": lambda: _scatters_state(
            query='run.experiment == "erm-training"'
        ),
    },
    {
        "name": "Run Comparison (Params)",
        "description": (
            "Parallel coordinates view of all erm-training runs. "
            "Compare hyperparameters across experiments."
        ),
        "app_type": "params",
        "state_fn": lambda: _params_state(
            query='run.experiment == "erm-training"'
        ),
    },
]


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

class AimAPI:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def _url(self, path: str) -> str:
        return f"{self.base_url}/api/{path.lstrip('/')}"

    def list_dashboards(self) -> list[dict]:
        resp = self.session.get(self._url("dashboards/"))
        resp.raise_for_status()
        return resp.json()

    def create_app(self, app_type: str, state: dict) -> dict:
        resp = self.session.post(
            self._url("apps/"),
            json={"type": app_type, "state": state},
        )
        resp.raise_for_status()
        return resp.json()

    def create_dashboard(self, name: str, description: str, app_id: str = None) -> dict:
        payload = {"name": name, "description": description}
        if app_id:
            payload["app_id"] = app_id
        resp = self.session.post(self._url("dashboards/"), json=payload)
        resp.raise_for_status()
        return resp.json()

    def delete_dashboard(self, dashboard_id: str):
        resp = self.session.delete(self._url(f"dashboards/{dashboard_id}/"))
        resp.raise_for_status()

    def delete_app(self, app_id: str):
        resp = self.session.delete(self._url(f"apps/{app_id}/"))
        resp.raise_for_status()

    def list_apps(self) -> list[dict]:
        resp = self.session.get(self._url("apps/"))
        resp.raise_for_status()
        return resp.json()

    def get_project_activity(self) -> dict:
        resp = self.session.get(self._url("projects/activity"))
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# URL builder for direct explorer links
# ---------------------------------------------------------------------------

def build_explorer_url(base_url: str, explorer_type: str, query: str) -> str:
    """Build a direct URL to an AIM explorer with a pre-configured query."""
    # `runs` apps are used for text traces in this setup.
    page = "text" if explorer_type == "runs" else explorer_type
    # AIM uses URL search params for some state
    encoded_query = urllib.parse.quote(query)
    return f"{base_url}/{page}?q={encoded_query}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Set up AIM dashboards for ERM training")
    parser.add_argument("--aim-url", default=DEFAULT_AIM_URL, help="AIM server base URL")
    parser.add_argument("--reset", action="store_true", help="Delete all existing dashboards first")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be created without creating")
    args = parser.parse_args()

    api = AimAPI(args.aim_url)

    # Check connectivity
    try:
        activity = api.get_project_activity()
        print(f"Connected to AIM server at {args.aim_url}")
        print(f"  Active runs: {activity.get('num_active_runs', 0)}")
        print(f"  Total runs:  {activity.get('num_runs', 0)}")
        print()
    except Exception as e:
        print(f"ERROR: Cannot connect to AIM server at {args.aim_url}: {e}", file=sys.stderr)
        sys.exit(1)

    # Reset if requested
    if args.reset:
        print("Resetting: deleting all existing dashboards and apps...")
        for dash in api.list_dashboards():
            print(f"  Deleting dashboard: {dash['name']} ({dash['id']})")
            if not args.dry_run:
                api.delete_dashboard(dash["id"])
        for app in api.list_apps():
            print(f"  Deleting app: {app['type']} ({app['id']})")
            if not args.dry_run:
                api.delete_app(app["id"])
        print()

    # Create dashboards
    print(f"Creating {len(DASHBOARDS)} dashboards...\n")

    created = []
    for defn in DASHBOARDS:
        name = defn["name"]
        desc = defn["description"]
        app_type = defn["app_type"]
        state = defn["state_fn"]()

        if args.dry_run:
            print(f"  [DRY RUN] Would create: {name}")
            print(f"            Type: {app_type}")
            print(f"            Query: {state.get('select', {}).get('advancedQuery', 'N/A')}")
            print()
            continue

        # 1. Create the explore state app
        app = api.create_app(app_type, state)
        app_id = app["id"]

        # 2. Create the dashboard linked to the app
        dash = api.create_dashboard(name, desc, app_id)
        dash_id = dash["id"]

        # Build direct URL
        query = state.get("select", {}).get("advancedQuery", "")
        url = build_explorer_url(args.aim_url, app_type, query)

        created.append({
            "name": name,
            "dashboard_id": dash_id,
            "app_id": app_id,
            "app_type": app_type,
            "url": url,
        })

        print(f"  Created: {name}")
        print(f"    Dashboard ID: {dash_id}")
        print(f"    App ID:       {app_id}")
        print(f"    Type:         {app_type}")
        print(f"    Direct URL:   {url}")
        print()

    if args.dry_run:
        print("Dry run complete. No changes made.")
        return

    # Print summary
    print("=" * 70)
    print("Dashboard Setup Complete!")
    print("=" * 70)
    print()
    print(f"AIM UI: {args.aim_url}")
    print()
    print("Dashboards page (all dashboards):")
    print(f"  {args.aim_url}/dashboard")
    print()
    print("Quick links:")
    for item in created:
        print(f"  {item['name']}")
        print(f"    Dashboard: {args.aim_url}/dashboard/{item['dashboard_id']}")
        print()

    # Also print direct explorer links with query params
    print("Direct explorer links (with pre-configured queries):")
    print()
    print(f"  Metrics (active run, training):")
    print(f"    {args.aim_url}/metrics")
    print()
    print(f"  IO Examples (active run):")
    print(f"    {args.aim_url}/text")
    print()
    print(f"  Scatters (all erm-training runs):")
    print(f"    {args.aim_url}/scatters")
    print()
    print(f"  Runs overview:")
    print(f"    {args.aim_url}/runs")
    print()
    print("Tip: The dashboards auto-filter to `run.active == True`.")
    print("     When you start a new training run, dashboards will")
    print("     automatically switch to show the new run's data.")


if __name__ == "__main__":
    main()
