import json
import logging
import shutil
from datetime import timedelta
from pathlib import Path

import pandas as pd

from typing import Any

from ..utils import BIN_FOLDER, TEST_FOLDER, TRAIN_FOLDER, VALIDATION_FOLDER


def read_from_file(path: Path | str) -> dict[Any, Any]:
    with open(path, "r", encoding="UTF-8") as file:
        return json.load(file)


def _clean_status(status: str) -> str:
    lower = status.lower().strip()
    return "FINISHED" if lower in ("finished", "lapped") or lower.startswith("+") else "DNF"


def parse_qualifying(data: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for race in data:
        season_int = int(race["season"])
        round_int = int(race["round"])
        for result in race["QualifyingResults"]:
            rows.append(
                {
                    "season": season_int,
                    "round": round_int,
                    "car_number": result["number"],
                    "position": result["position"],
                    "driver_id": result["Driver"]["driverId"],
                    "constructor_id": result["Constructor"]["constructorId"],
                    "q1_time": result.get("Q1"),
                    "q2_time": result.get("Q2"),
                    "q3_time": result.get("Q3"),
                }
            )

    return pd.DataFrame(rows)


def parse_laps(data: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for race_id, block in data.items():
        for segment in block["laps"]:
            for lap in segment["Laps"]:
                lap_num = int(lap["number"])
                for timing in lap["Timings"]:
                    rows.append(
                        {
                            "season": int(segment["season"]),
                            "round": int(segment["round"]),
                            "race_id": race_id,
                            "driver_id": timing["driverId"],
                            "lap_number": lap_num,
                            "position": timing["position"],
                            "time": timing["time"],
                        }
                    )

    return pd.DataFrame(rows)


def parse_pitstops(data: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for race_id, block in data.items():
        for segment in block["pitStops"]:
            for stop in segment["PitStops"]:
                rows.append(
                    {
                        "season": int(segment["season"]),
                        "round": int(segment["round"]),
                        "race_id": race_id,
                        "driver_id": stop["driverId"],
                        "stop_number": stop["stop"],
                        "lap_number": stop["lap"],
                        "time": stop["time"],
                        "duration": stop["duration"],
                    }
                )

    return pd.DataFrame(rows)


def parse_results(raw_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, race_row in raw_df.iterrows():
        base_time = None
        for result in race_row["Results"]:
            rec = {
                "season": int(race_row["season"]),
                "round": int(race_row["round"]),
                "car_number": result["number"],
                "position": result["position"],
                "points": result["points"],
                "driver_id": result["Driver"]["driverId"],
                "constructor_id": result["Constructor"]["constructorId"],
                "grid": result["grid"],
                "laps": result["laps"],
                "status": _clean_status(result["status"]),
                "fastest_lap": result.get("FastestLap", {}).get("Time", {}).get("time"),
            }
            t = result.get("Time", {}).get("time")
            if t:
                if t.startswith("+") and base_time:
                    b = list(map(float, base_time.split(":")))
                    b_sec = (
                        b[0] * 3600 + b[1] * 60 + b[2]
                        if len(b) == 3
                        else b[0] * 60 + b[1]
                        if len(b) == 2
                        else b[0]
                    )
                    p = t[1:].split(":")
                    add_sec = float(p[-1])
                    add_min = int(p[0]) if len(p) > 1 else 0
                    delta = timedelta(seconds=b_sec + add_min * 60 + add_sec)
                    rec["time"] = f"{delta.seconds // 3600}:{(delta.seconds // 60) % 60:02d}:{delta.seconds % 60:06.3f}"
                else:
                    rec["time"] = t
                    base_time = t if not t.startswith("+") else base_time
            else:
                rec["time"] = None
            rows.append(rec)

    return pd.DataFrame(rows)[
        [
            "season",
            "round",
            "car_number",
            "position",
            "points",
            "driver_id",
            "constructor_id",
            "grid",
            "laps",
            "status",
            "time",
            "fastest_lap",
        ]
    ]


def parse_circuits(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, race_row in data.iterrows():
        rows.append(
            {
                "season": int(race_row["season"]),
                "round": int(race_row["round"]),
                "url": race_row["url"],
                "race_name": race_row["raceName"],
                "circuit_id": race_row["Circuit.circuitId"],
                "circuit_url": race_row["Circuit.url"],
                "circuit_name": race_row["Circuit.circuitName"],
                "latitude": race_row["Circuit.Location.lat"],
                "longitude": race_row["Circuit.Location.long"],
                "locality": race_row["Circuit.Location.locality"],
                "country": race_row["Circuit.Location.country"],
                "date": race_row["date"],
                "time": race_row["time"],
            }
        )

    return pd.DataFrame(rows)


def parse_drivers(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, race_row in data.iterrows():
        for result in race_row["Results"]:
            driver = result["Driver"]
            rows.append(
                {
                    "season": int(race_row["season"]),
                    "driver_id": driver["driverId"],
                    "permanent_number": driver.get("permanentNumber"),
                    "code": driver["code"],
                    "url": driver["url"],
                    "given_name": driver["givenName"],
                    "family_name": driver["familyName"],
                    "date_of_birth": driver["dateOfBirth"],
                    "nationality": driver["nationality"],
                    "constructor_id": result["Constructor"]["constructorId"],
                }
            )

    return pd.DataFrame(rows).drop_duplicates(subset=["season", "driver_id"])


def parse_constructors(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, race_row in data.iterrows():
        for result in race_row["Results"]:
            cons = result["Constructor"]
            rows.append(
                {
                    "season": int(race_row["season"]),
                    "constructor_id": cons["constructorId"],
                    "url": cons["url"],
                    "name": cons["name"],
                    "nationality": cons["nationality"],
                }
            )

    return pd.DataFrame(rows).drop_duplicates(subset=["constructor_id"])


def parse_data() -> tuple[tuple[pd.DataFrame, ...], tuple[pd.DataFrame, ...], tuple[pd.DataFrame, ...]]:
    for folder in (TRAIN_FOLDER, VALIDATION_FOLDER, TEST_FOLDER):
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)

    res_chunks, circ_chunks, drv_chunks, cons_chunks = [], [], [], []
    qual_chunks, lap_chunks, pit_chunks = [], [], []

    for yr in sorted((p for p in BIN_FOLDER.iterdir() if p.is_dir()), key=lambda p: p.name):
        raw = pd.json_normalize(read_from_file(yr / "results.json")["results"])

        res_chunks.append(parse_results(raw))
        circ_chunks.append(parse_circuits(raw))
        drv_chunks.append(parse_drivers(raw))
        cons_chunks.append(parse_constructors(raw))

        qual_chunks.append(parse_qualifying(read_from_file(yr / "qualifying.json")["qualifying"]))
        lp = read_from_file(yr / "laps_pitstops.json")
        lap_chunks.append(parse_laps(lp))
        pit_chunks.append(parse_pitstops(lp))

    results = pd.concat(res_chunks, ignore_index=True)
    circuits = pd.concat(circ_chunks, ignore_index=True)
    drivers = pd.concat(drv_chunks, ignore_index=True)
    constructors = pd.concat(cons_chunks, ignore_index=True)
    qualifying = pd.concat(qual_chunks, ignore_index=True)
    laps = pd.concat(lap_chunks, ignore_index=True)
    pitstops = pd.concat(pit_chunks, ignore_index=True)

    results["season"] = pd.to_numeric(results["season"], errors="coerce")
    results["round"] = pd.to_numeric(results["round"], errors="coerce")

    last_year = int(results["season"].max())
    last_round = int(results.loc[results["season"] == last_year, "round"].max())

    train_res = results[results["season"] < last_year]
    val_res = results[(results["season"] == last_year) & (results["round"] < last_round)]
    test_res = results[(results["season"] == last_year) & (results["round"] == last_round)]

    def subset(tbl: pd.DataFrame, mask_df: pd.DataFrame) -> pd.DataFrame:
        if "round" in tbl.columns:
            key = pd.MultiIndex.from_frame(mask_df[["season", "round"]].drop_duplicates())
            return tbl[pd.MultiIndex.from_frame(tbl[["season", "round"]]).isin(key)]
        return tbl[tbl["season"].isin(mask_df["season"].unique())]

    for df, name in [
        (train_res, "results.csv"),
        (subset(circuits, train_res), "circuits.csv"),
        (subset(drivers, train_res), "drivers.csv"),
        (subset(constructors, train_res), "constructors.csv"),
        (subset(qualifying, train_res), "qualifying.csv"),
        (subset(laps, train_res), "laps.csv"),
        (subset(pitstops, train_res), "pitstops.csv"),
    ]:
        df.to_csv(TRAIN_FOLDER / name, sep=";", index=False, encoding="UTF-8")

    val_feature_mask = results["season"] == last_year

    for df, name in [
        (val_res, "results.csv"),
        (subset(circuits, results[val_feature_mask]), "circuits.csv"),
        (subset(drivers, results[val_feature_mask]), "drivers.csv"),
        (subset(constructors, results[val_feature_mask]), "constructors.csv"),
        (subset(qualifying, results[val_feature_mask & (results['round'] < last_round)]), "qualifying.csv"),
        (subset(laps, results[val_feature_mask & (results['round'] < last_round)]), "laps.csv"),
        (subset(pitstops, results[val_feature_mask & (results['round'] < last_round)]), "pitstops.csv"),
    ]:
        df.to_csv(VALIDATION_FOLDER / name, sep=";", index=False, encoding="UTF-8")

    test_res.to_csv(TEST_FOLDER / "results.csv", sep=";", index=False, encoding="UTF-8")
    subset(qualifying, test_res).to_csv(TEST_FOLDER / "qualifying.csv", sep=";", index=False, encoding="UTF-8")

    logging.info(f"Training Races: {train_res[['season', 'round']].drop_duplicates().shape[0]}")
    logging.info(f"Validation Races: {val_res[['season', 'round']].drop_duplicates().shape[0]}")
    logging.info(f"Test Race: Season {last_year}, Round {last_round}")

    train = (
        pd.read_csv(TRAIN_FOLDER / "circuits.csv", sep=";", encoding="UTF-8"),
        pd.read_csv(TRAIN_FOLDER / "constructors.csv", sep=";", encoding="UTF-8"),
        pd.read_csv(TRAIN_FOLDER / "drivers.csv", sep=";", encoding="UTF-8"),
        pd.read_csv(TRAIN_FOLDER / "laps.csv", sep=";", encoding="UTF-8"),
        pd.read_csv(TRAIN_FOLDER / "pitstops.csv", sep=";", encoding="UTF-8"),
        pd.read_csv(TRAIN_FOLDER / "qualifying.csv", sep=";", encoding="UTF-8"),
        pd.read_csv(TRAIN_FOLDER / "results.csv", sep=";", encoding="UTF-8"),
    )
    val = (
        pd.read_csv(VALIDATION_FOLDER / "circuits.csv", sep=";", encoding="UTF-8"),
        pd.read_csv(VALIDATION_FOLDER / "constructors.csv", sep=";", encoding="UTF-8"),
        pd.read_csv(VALIDATION_FOLDER / "drivers.csv", sep=";", encoding="UTF-8"),
        pd.read_csv(VALIDATION_FOLDER / "laps.csv", sep=";", encoding="UTF-8"),
        pd.read_csv(VALIDATION_FOLDER / "pitstops.csv", sep=";", encoding="UTF-8"),
        pd.read_csv(VALIDATION_FOLDER / "qualifying.csv", sep=";", encoding="UTF-8"),
        pd.read_csv(VALIDATION_FOLDER / "results.csv", sep=";", encoding="UTF-8"),
    )
    test = (
        pd.read_csv(TEST_FOLDER / "qualifying.csv", sep=";", encoding="UTF-8"),
        pd.read_csv(TEST_FOLDER / "results.csv", sep=";", encoding="UTF-8"),
    )

    return train, val, test
