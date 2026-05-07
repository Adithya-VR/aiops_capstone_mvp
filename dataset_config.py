from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def project_path(path: str) -> Path:
    return PROJECT_ROOT / path


DATASETS = {
    "bgl": {
        "name": "bgl",
        "display_name": "BGL",
        "description": "BlueGene/L supercomputer logs",
        "raw_path": project_path("data/BGL.log"),
        "output_dir": project_path("output/bgl"),
        "parser": "bgl",
        "has_labels": True,
        "evaluation_mode": "supervised",
        "window": 3600,
        "step": 1800,
        "post_filter": "evidence",
    },
    "openssh": {
        "name": "openssh",
        "display_name": "OpenSSH",
        "description": "OpenSSH authentication logs",
        "raw_path": project_path("data/OpenSSH/SSH.log"),
        "output_dir": project_path("output/openssh"),
        "parser": "openssh",
        "has_labels": False,
        "evaluation_mode": "unlabeled",
        "window": 300,
        "step": 150,
        "contamination": 0.03,
        "contamination_note": (
            "Unlabeled dataset assumption: flag the most unusual 3% of "
            "sliding windows. Tune this to control alert volume."
        ),
        "post_filter": "predicted_only",
        "post_filter_note": (
            "OpenSSH levels are mostly security-relevant by design, so the "
            "generic evidence filter is equivalent to raw model predictions."
        ),
    }
}

DEFAULT_DATASET = "bgl"


def get_dataset(dataset: str = DEFAULT_DATASET) -> dict:
    key = dataset.lower()
    if key not in DATASETS:
        raise KeyError(f"Unknown dataset: {dataset}")
    return DATASETS[key]


def dataset_paths(dataset: str = DEFAULT_DATASET) -> dict:
    cfg = get_dataset(dataset)
    out = cfg["output_dir"]
    return {
        "raw": cfg["raw_path"],
        "output_dir": out,
        "parsed": out / "parsed.parquet",
        "features": out / "features.parquet",
        "scores": out / "scores.parquet",
        "model": out / "model.pkl",
        "metrics": out / "metrics.json",
        "alerts": out / "alerts.parquet",
        "alerts_minilm": out / "alerts_minilm.parquet",
        "clustering_comparison": out / "clustering_comparison.json",
    }


def available_datasets() -> list[dict]:
    rows = []
    for key, cfg in DATASETS.items():
        paths = dataset_paths(key)
        ready = paths["parsed"].exists() and paths["scores"].exists()
        rows.append({
            "name": key,
            "display_name": cfg["display_name"],
            "description": cfg["description"],
            "status": "ready" if ready else "not_ready",
            "parser": cfg["parser"],
            "has_labels": cfg["has_labels"],
            "evaluation_mode": cfg["evaluation_mode"],
            "window": cfg["window"],
            "step": cfg["step"],
            "contamination": cfg.get("contamination"),
            "contamination_note": cfg.get("contamination_note"),
            "files": {
                "parsed": str(paths["parsed"]),
                "features": str(paths["features"]),
                "scores": str(paths["scores"]),
                "alerts": str(paths["alerts"]),
                "metrics": str(paths["metrics"]),
            },
        })
    return rows
