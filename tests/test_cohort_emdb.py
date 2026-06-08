"""Tests for EMDB cohort metadata helpers."""

from cryoem_mrc.cohort_emdb import parse_emdb_global_resolution_a


def test_parse_emdb_global_resolution_a() -> None:
    entry = {
        "structure_determination_list": {
            "structure_determination": [
                {
                    "image_processing": [
                        {
                            "final_reconstruction": {
                                "resolution": {"valueOf_": "2.73", "units": "Å"},
                            }
                        }
                    ]
                }
            ]
        }
    }
    assert parse_emdb_global_resolution_a(entry) == 2.73
    assert parse_emdb_global_resolution_a({}) is None
