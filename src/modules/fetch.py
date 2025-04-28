import itertools
import json
import logging
import os
import requests

from typing import Any, Iterator

from ..utils import API_URL, BIN_FOLDER, DATA_FOLDER

proxies_file = DATA_FOLDER / 'proxies.txt'
INVALID_PROXIES: set[str] = set()
CURRENT_PROXY: str | None = None
USE_PROXIES: bool = False
PROXIES: Iterator[str] = itertools.cycle([])

if proxies_file.exists():
    with open(proxies_file, 'r') as F:
        PROXIES = itertools.cycle(F.read().splitlines())

    CURRENT_PROXY = next(PROXIES)
    USE_PROXIES = True


def write_to_file(data: dict[Any, Any], filename: str) -> None:
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def read_from_file(file_path: str) -> dict[Any, Any]:
    with open(file_path, "r") as file:
        return json.load(file)


def fetch(session: requests.Session, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
    global CURRENT_PROXY

    params = kwargs.pop('params', {})
    params['limit'] = 100
    params['offset'] = 0

    data = []
    while True:
        if USE_PROXIES:
            if CURRENT_PROXY in INVALID_PROXIES:
                raise Exception("Tried All Proxies - Exiting . . .")

            proxy = {
                'http': f'http://{CURRENT_PROXY}',
                'https': f'http://{CURRENT_PROXY}'
            }
        else:
            proxy = None

        kwargs['params'] = params
        try:
            if USE_PROXIES and proxy is not None:
                response = session.get(*args, **kwargs, proxies=proxy, timeout=5)
            else:
                response = session.get(*args, **kwargs, timeout=5)
        except (requests.exceptions.ProxyError, requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            logging.warning(f"Connection Error: {CURRENT_PROXY}")
            if USE_PROXIES and CURRENT_PROXY is not None:
                logging.warning(f"Proxy Error: {CURRENT_PROXY}")
                INVALID_PROXIES.add(CURRENT_PROXY)
                CURRENT_PROXY = next(PROXIES)
                continue
            else:
                raise

        if response.status_code == 200:
            result = response.json()

            if "RaceTable" in result['MRData']:
                data.extend(result['MRData']['RaceTable']['Races'])
            elif "DriverTable" in result['MRData']:
                data.extend(result['MRData']['DriverTable']['Drivers'])
            else:
                break

            total = int(result['MRData']['total'])
            params['offset'] += params['limit']
            if params['offset'] >= total:
                break
        elif response.status_code == 429:
            if USE_PROXIES:
                CURRENT_PROXY = next(PROXIES)
            else:
                raise Exception("Rate Limit Exceeded - Not Using Proxies.")
        elif response.status_code == 407:
            if USE_PROXIES and CURRENT_PROXY is not None:
                logging.warning(f"Proxy Authentication Error: {CURRENT_PROXY}")
                INVALID_PROXIES.add(CURRENT_PROXY)
                CURRENT_PROXY = next(PROXIES)
                continue
            else:
                raise Exception("Proxy Authentication Error - Not Using Proxies.")
        else:
            raise Exception(f"Failed to Fetch Data [{response.status_code}]: {response.text}")

    return data


def fetch_drivers(session: requests.Session, *, year: int) -> dict[str, Any]:
    data = fetch(session, f"{API_URL}/{year}/drivers")
    return {'drivers': data}


def fetch_qualifying(session: requests.Session, *, year: int) -> dict[str, Any]:
    data = fetch(session, f"{API_URL}/{year}/qualifying")
    return {'qualifying': data}


def fetch_results(session: requests.Session, *, year: int) -> dict[str, Any]:
    data = fetch(session, f"{API_URL}/{year}/results")
    return {'results': data}


def fetch_laps(session: requests.Session, *, year: int, race: int) -> dict[str, Any]:
    data = fetch(session, f"{API_URL}/{year}/{race}/laps")
    return {'laps': data}


def fetch_pitstops(session: requests.Session, *, year: int, race: int) -> dict[str, Any]:
    data = fetch(session, f"{API_URL}/{year}/{race}/pitstops")
    return {'pitStops': data}

def fetch_data() -> None:
    logging.info("Fetching Data . . .")
    os.makedirs(BIN_FOLDER, exist_ok=True)

    headers = {'Content-Type': 'application/json'}
    session = requests.Session()
    session.headers.update(headers)

    for year in range(2017, 2024 + 1):
        BIN_YEAR_FOLDER = BIN_FOLDER / str(year)
        os.makedirs(BIN_YEAR_FOLDER, exist_ok=True)

        if not os.path.exists(BIN_YEAR_FOLDER / 'drivers.json'):
            drivers = fetch_drivers(session, year=year)
            write_to_file(drivers, str(BIN_YEAR_FOLDER / 'drivers.json'))
        logging.info(f"- Fetched Driver Data for {year}")

        if not os.path.exists(BIN_YEAR_FOLDER / 'qualifying.json'):
            qualifying = fetch_qualifying(session, year=year)
            write_to_file(qualifying, str(BIN_YEAR_FOLDER / 'qualifying.json'))
        logging.info(f"- Fetched Qualifying Data for {year}")

        if not os.path.exists(BIN_YEAR_FOLDER / 'results.json'):
            results = fetch_results(session, year=year)
            write_to_file(results, str(BIN_YEAR_FOLDER / 'results.json'))

            if not os.path.exists(BIN_YEAR_FOLDER / 'laps_pitstops.json'):
                data: dict[int, Any] = {}
                for race in range(1, len(results['results'])):
                    data[race] = {}

                    laps = fetch_laps(session, year=year, race=race)
                    data[race].update(laps)

                    pitstops = fetch_pitstops(session, year=year, race=race)
                    data[race].update(pitstops)

                write_to_file(data, str(BIN_YEAR_FOLDER / 'laps_pitstops.json'))
            logging.info(f"- Fetched Laps & Pitstops Data for {year}")

        logging.info(f"- Fetched Results Data for {year}")

        logging.info(f"Data Successfully Fetched for {year}!\n")
