"""
CLI: export OD event data from SQLite to JSON or CSV.

Usage:
    python scripts/export_od.py --db data/od_events.db --format json
    python scripts/export_od.py --db data/od_events.db --format csv --out od_matrix.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from storage.db import Database


def export_json(db: Database, out_path: str | None) -> None:
    matrix = db.get_od_matrix()
    data = json.dumps(matrix, ensure_ascii=False, indent=2)
    if out_path:
        Path(out_path).write_text(data, encoding="utf-8")
        print(f"Exported {len(matrix)} OD pairs to {out_path}")
    else:
        print(data)


def export_csv(db: Database, out_path: str | None) -> None:
    matrix = db.get_od_matrix()
    rows = [{"board_stop": r["board_stop"], "alight_stop": r["alight_stop"], "count": r["count"]}
            for r in matrix]

    if out_path:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["board_stop", "alight_stop", "count"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Exported {len(rows)} OD pairs to {out_path}")
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=["board_stop", "alight_stop", "count"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export OD event data")
    parser.add_argument("--db", required=True, help="Path to od_events.db")
    parser.add_argument("--format", choices=["json", "csv"], default="json")
    parser.add_argument("--out", default=None, help="Output file path (stdout if omitted)")
    parser.add_argument("--route", default=None, help="Filter by route_id")
    parser.add_argument("--vehicle", default=None, help="Filter by vehicle_id")
    args = parser.parse_args()

    db = Database(args.db)

    if args.format == "json":
        export_json(db, args.out)
    else:
        export_csv(db, args.out)


if __name__ == "__main__":
    main()
