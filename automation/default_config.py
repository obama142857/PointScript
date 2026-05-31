import copy
from typing import Any, Dict


DEFAULT_BATCH_CONFIG: Dict[str, Any] = {
    "enable_report": True,
    "show_child_windows": False,
    "pcot": {
        "downsample_length": 0.001,
        "semantic_group_export_enabled": True,
        "semantic_groups": [
            {"name": "\u7ba1\u9053\u76f8\u5173", "semantic_ids": [1, 10, 11, 12, 13, 14, 15, 24]},
            {"name": "\u50a8\u7f50\u76f8\u5173", "semantic_ids": [3, 17, 18, 19, 20]},
        ],
    },
    "fact": {
        "enabled": True,
        "mode": "auto",
        "manual": {
            "merge_rate": 30,
            "group_downsample_rate": 80,
            "auto_connect": {
                "tolerance_parallel_cos": 0.8,
                "mini_length": 0.03,
                "radius_ratio_tolerance": 0.4,
                "radius_diff_tolerance": 0.2,
                "patching_threshold_range": 15.0,
                "tolerance_angle": 18.0,
            },
        },
    },
}


def get_default_batch_config() -> Dict[str, Any]:
    return copy.deepcopy(DEFAULT_BATCH_CONFIG)
