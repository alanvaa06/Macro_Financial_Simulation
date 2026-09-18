"""Keep web/observables.js consistent with the sliders it annotates.

Every observed value must map to a control in ``web/index.html`` and sit inside
that control's range, and every control must either have an observed value or
an explicit "not applicable" reason. Skipped when Node.js is not on PATH.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def _load_observables() -> dict:
    script = "process.stdout.write(JSON.stringify(require(process.argv[1])))"
    out = subprocess.run(
        ["node", "-e", script, str(WEB / "observables.js")],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return json.loads(out)


def _controls() -> dict[str, tuple[float, float]]:
    """id -> (min, max) for every numeric control in index.html."""
    html = (WEB / "index.html").read_text(encoding="utf-8")
    controls: dict[str, tuple[float, float]] = {}
    for tag in re.findall(r"<input[^>]*type=\"(?:range|number)\"[^>]*>", html):
        attrs = dict(re.findall(r"(\w+)=\"([^\"]*)\"", tag))
        if "id" in attrs and "min" in attrs and "max" in attrs:
            controls[attrs["id"]] = (float(attrs["min"]), float(attrs["max"]))
    return controls


def test_every_observable_maps_to_a_control_and_is_in_range() -> None:
    obs = _load_observables()
    controls = _controls()
    assert controls, "no controls parsed from index.html"
    for pid, entry in obs["params"].items():
        assert pid in controls, f"{pid} has an observable but no control"
        lo, hi = controls[pid]
        assert lo <= entry["value"] <= hi, f"{pid}: {entry['value']} outside [{lo}, {hi}]"
        for key in ("label", "source", "asOf", "note"):
            assert entry.get(key), f"{pid} is missing {key}"


def test_every_control_is_annotated() -> None:
    obs = _load_observables()
    annotated = set(obs["params"]) | set(obs["notApplicable"])
    for pid in _controls():
        assert pid in annotated, f"{pid} has neither an observable nor a not-applicable reason"


def test_kpi_annotations_target_existing_cards() -> None:
    obs = _load_observables()
    html = (WEB / "index.html").read_text(encoding="utf-8")
    for kpi in obs["kpis"]:
        assert f'data-kpi="{kpi}"' in html, f"{kpi} has no KPI card slot"
