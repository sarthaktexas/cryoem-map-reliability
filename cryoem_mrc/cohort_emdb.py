"""EMDB metadata helpers for the thesis cohort."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

EMDB_ENTRY_API = "https://www.ebi.ac.uk/emdb/api/entry/{emdb_id}"


def parse_emdb_global_resolution_a(entry_json: dict[str, Any]) -> float | None:
    """
    Extract author-reported global resolution (Å) from an EMDB entry JSON blob.

    Uses ``final_reconstruction.resolution`` from the first structure determination.
    """
    try:
        sd = entry_json["structure_determination_list"]["structure_determination"]
        if not sd:
            return None
        ip = sd[0]["image_processing"]
        if not ip:
            return None
        res = ip[0]["final_reconstruction"]["resolution"]
        val = res.get("valueOf_")
        return float(val) if val is not None else None
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def fetch_emdb_global_resolution_a(
    emdb_id: str | int,
    *,
    timeout_s: float = 30.0,
    retries: int = 2,
    retry_delay_s: float = 0.5,
) -> float | None:
    """Query the EMDB REST API for one entry's global resolution (Å)."""
    eid = str(emdb_id).strip().removeprefix("EMD-").removeprefix("emd-")
    url = EMDB_ENTRY_API.format(emdb_id=eid)
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout_s) as resp:
                data = json.load(resp)
            return parse_emdb_global_resolution_a(data)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(retry_delay_s)
    if last_err is not None:
        raise RuntimeError(f"EMDB API failed for EMD-{eid}: {last_err}") from last_err
    return None
