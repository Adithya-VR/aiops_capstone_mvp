from pathlib import Path


DATASETS = {
    "bgl": {
        "name": "bgl",
        "display_name": "BGL",
        "description": "BlueGene/L supercomputer logs",
        "raw_path": Path("data/BGL.log"),
        "output_dir": Path("output/bgl"),
        "parser": "bgl",
        "window": 3600,
        "step": 1800,
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
            "window": cfg["window"],
            "step": cfg["step"],
            "files": {
                "parsed": str(paths["parsed"]),
                "features": str(paths["features"]),
                "scores": str(paths["scores"]),
                "alerts": str(paths["alerts"]),
                "metrics": str(paths["metrics"]),
            },
        })
    return rows
