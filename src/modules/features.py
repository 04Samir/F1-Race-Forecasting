import logging
import re

import numpy as np
import pandas as pd

import torch
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from typing import Any

FEATURE_SET = [
    'grid', 'q1_seconds', 'q2_seconds', 'q3_seconds', 'best_qualifying_time',
    'qualifying_delta', 'qualifying_delta_pct', 'driver_age', 'driver_experience',
    'avg_last_3_positions', 'avg_last_5_positions', 'avg_last_10_positions',
    'position_momentum', 'avg_constructor_last_3_positions', 'constructor_momentum',
    'avg_position_at_circuit', 'best_position_at_circuit', 'races_at_circuit',
    'constructor_avg_position_at_circuit', 'constructor_best_position_at_circuit',
    'constructor_races_at_circuit', 'pitstop_count', 'avg_pitstop_duration',
    'position_std', 'year_weight', 'position_consistency', 'constructor_consistency',
    'trend_direction', 'ewm_avg_position', 'constructor_ewm_avg'
]


def get_weight_by_year(year: int, max_year: int) -> float:
    return 0.8 ** (max_year - year)


def convert_qualifying_time(time_str: str) -> float:
    if pd.isna(time_str):
        return 0.0

    try:
        parts = time_str.split(':')
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
        elif len(parts) == 1:
            return float(parts[0])
        else:
            return 0.0
    except:
        return 0.0


def clean_pitstop_duration(pitstops_df: pd.DataFrame) -> pd.DataFrame:
    if pitstops_df.empty:
        return pitstops_df

    cleaned_df = pitstops_df.copy()

    if 'duration' in cleaned_df.columns:
        cleaned_df['duration'] = cleaned_df['duration'].astype(str)

        def extract_first_number(val: str) -> float:
            try:
                return float(val)
            except ValueError:
                matches = re.findall(r'\d+\.\d+', val)
                if matches:
                    return float(matches[0])
                return np.nan

        cleaned_df['duration'] = cleaned_df['duration'].apply(extract_first_number)
        mask = (cleaned_df['duration'] < 1) | (cleaned_df['duration'] > 60)
        median_duration = cleaned_df.loc[~mask, 'duration'].median() if not mask.all() else 25.0
        cleaned_df.loc[mask, 'duration'] = median_duration

    return cleaned_df


def calculate_driver_position_boundaries(results_df: pd.DataFrame, min_races: int = 5) -> dict[str, tuple[float, float]]:
    max_season = results_df['season'].max()
    results = results_df.copy()
    results['position_num'] = pd.to_numeric(results['position'], errors='coerce')

    results['recency_weight'] = results['season'].apply(
        lambda s: get_weight_by_year(s, max_season)
    )

    position_boundaries = {}
    for driver in results['driver_id'].unique():
        driver_results = results[results['driver_id'] == driver]

        if len(driver_results) < min_races:
            continue

        driver_positions = driver_results['position_num'].dropna().tolist()
        weights = driver_results['recency_weight'].dropna().tolist()

        if len(driver_positions) < min_races or len(weights) < min_races:
            continue

        combined = sorted(zip(driver_positions, weights))
        positions = [p for p, w in combined]
        weights = [w for p, w in combined]

        cumulative_weights = []
        cumulative_sum = 0
        for w in weights:
            cumulative_sum += w
            cumulative_weights.append(cumulative_sum)

        total_weight = cumulative_weights[-1]
        cumulative_weights = [w / total_weight for w in cumulative_weights]

        q1_idx = next((i for i, w in enumerate(cumulative_weights) if w >= 0.25), 0)
        q3_idx = next((i for i, w in enumerate(cumulative_weights) if w >= 0.75), len(positions) - 1)

        q1 = positions[q1_idx]
        q3 = positions[q3_idx]

        position_boundaries[driver] = (q1, q3)

    return position_boundaries


def find_top_drivers(results_df: pd.DataFrame, n: int = 3, recent_seasons_count: int = 3) -> list[str]:
    available_seasons = results_df['season'].nunique()
    recent_seasons_count = min(recent_seasons_count, available_seasons - 1) if available_seasons > 1 else 1

    max_season = results_df['season'].max()
    recent_seasons = [max_season - i for i in range(recent_seasons_count)]
    recent_results = results_df[results_df['season'].isin(recent_seasons)].copy()

    recent_results['position'] = recent_results['position'].astype(str)
    recent_results['position_num'] = pd.to_numeric(recent_results['position'], errors='coerce')

    recent_results['recency_weight'] = recent_results['season'].apply(
        lambda s: get_weight_by_year(s, max_season + recent_seasons_count - min(recent_seasons))
    )

    driver_wins = recent_results[recent_results['position'] == '1'].groupby('driver_id').apply(
        lambda x: (x['recency_weight'].sum())
    ).reset_index(name='weighted_wins')

    podium_positions = ['1', '2', '3']
    driver_podiums = recent_results[recent_results['position'].isin(podium_positions)].groupby('driver_id').apply(
        lambda x: (x['recency_weight'].sum())
    ).reset_index(name='weighted_podiums')

    driver_avg_pos = recent_results.groupby('driver_id').apply(
        lambda x: (x['position_num'] * x['recency_weight']).sum() / x['recency_weight'].sum()
    ).reset_index(name='weighted_avg_position')

    if driver_wins.empty:
        if driver_podiums.empty:
            if driver_avg_pos.empty:
                top_positions = recent_results.groupby('driver_id')['position_num'].min().reset_index()
                top_positions.columns = ['driver_id', 'best_position']
                return top_positions.sort_values('best_position').head(n)['driver_id'].tolist()
            else:
                return driver_avg_pos.sort_values('weighted_avg_position').head(n)['driver_id'].tolist()
        else:
            driver_stats = pd.merge(driver_podiums, driver_avg_pos, on='driver_id', how='outer').fillna(0)
            driver_stats = driver_stats.sort_values(['weighted_podiums', 'weighted_avg_position'])
            return driver_stats.head(n)['driver_id'].tolist()

    driver_stats = pd.merge(driver_wins, driver_podiums, on='driver_id', how='outer').fillna(0)
    driver_stats = pd.merge(driver_stats, driver_avg_pos, on='driver_id', how='outer').fillna(0)
    driver_stats = driver_stats.sort_values(
        ['weighted_wins', 'weighted_podiums', 'weighted_avg_position'],
        ascending=[False, False, True]
    )

    top_drivers = driver_stats.head(n)['driver_id'].tolist()
    return top_drivers


def find_mid_tier_drivers(results_df: pd.DataFrame, top_drivers: list, n: int = 3, recent_seasons_count: int = 3) -> list[str]:
    available_seasons = results_df['season'].nunique()
    recent_seasons_count = min(recent_seasons_count, available_seasons - 1) if available_seasons > 1 else 1

    max_season = results_df['season'].max()
    recent_seasons = [max_season - i for i in range(recent_seasons_count)]
    recent_results = results_df[results_df['season'].isin(recent_seasons)].copy()

    recent_results['position'] = recent_results['position'].astype(str)
    recent_results['position_num'] = pd.to_numeric(recent_results['position'], errors='coerce')

    recent_results['recency_weight'] = recent_results['season'].apply(
        lambda s: get_weight_by_year(s, max_season + recent_seasons_count - min(recent_seasons))
    )

    filtered_results = recent_results
    if top_drivers:
        filtered_results = recent_results[~recent_results['driver_id'].isin(top_drivers)]

    points_positions = [str(i) for i in range(1, 11)]
    points_finishes = filtered_results[filtered_results['position'].isin(points_positions)]

    if points_finishes.empty:
        avg_positions = filtered_results.groupby('driver_id').apply(
            lambda x: (x['position_num'] * x['recency_weight']).sum() / x['recency_weight'].sum()
        ).reset_index(name='weighted_avg_position')

        return avg_positions.sort_values('weighted_avg_position').head(n)['driver_id'].tolist()

    points_freq = points_finishes.groupby('driver_id').apply(
        lambda x: x['recency_weight'].sum()
    ).reset_index(name='weighted_points_finishes')

    avg_points_pos = points_finishes.groupby('driver_id').apply(
        lambda x: (x['position_num'] * x['recency_weight']).sum() / x['recency_weight'].sum()
    ).reset_index(name='weighted_avg_points_position')

    overall_avg_pos = filtered_results.groupby('driver_id').apply(
        lambda x: (x['position_num'] * x['recency_weight']).sum() / x['recency_weight'].sum()
    ).reset_index(name='weighted_avg_position')

    driver_stats = pd.merge(points_freq, avg_points_pos, on='driver_id', how='outer').fillna(0)
    driver_stats = pd.merge(driver_stats, overall_avg_pos, on='driver_id', how='outer').fillna(0)

    driver_stats = driver_stats.sort_values(
        ['weighted_points_finishes', 'weighted_avg_points_position', 'weighted_avg_position'],
        ascending=[False, True, True]
    )

    mid_tier_drivers = driver_stats.head(n)['driver_id'].tolist()
    return mid_tier_drivers


def get_combined_domain_knowledge(
    train_results_df: pd.DataFrame,
    val_results_df: pd.DataFrame,
    train_knowledge: tuple[dict, list, list],
    val_knowledge: tuple[dict, list, list],
) -> tuple[dict[str, tuple[float, float]], list[str], list[str]]:
    train_position_boundaries, train_top_drivers, train_mid_tier_drivers = train_knowledge
    val_position_boundaries, val_top_drivers, val_mid_tier_drivers = val_knowledge

    all_results_df = pd.concat([train_results_df, val_results_df], ignore_index=True)

    combined_position_boundaries = calculate_driver_position_boundaries(all_results_df)

    shared_top_drivers = set(train_top_drivers) & set(val_top_drivers)
    val_unique_top = set(val_top_drivers) - shared_top_drivers
    train_unique_top = set(train_top_drivers) - shared_top_drivers

    combined_top_drivers = list(shared_top_drivers) + list(val_unique_top)

    max_top_drivers = max(len(train_top_drivers), len(val_top_drivers))
    if len(combined_top_drivers) < max_top_drivers:
        remaining = max_top_drivers - len(combined_top_drivers)
        combined_top_drivers.extend(list(train_unique_top)[:remaining])

    all_mid_drivers = set(train_mid_tier_drivers) | set(val_mid_tier_drivers)

    all_mid_drivers = all_mid_drivers - set(combined_top_drivers)

    shared_mid_drivers = (set(train_mid_tier_drivers) & set(val_mid_tier_drivers)) - set(combined_top_drivers)
    val_unique_mid = set(val_mid_tier_drivers) - shared_mid_drivers - set(combined_top_drivers)
    train_unique_mid = set(train_mid_tier_drivers) - shared_mid_drivers - set(combined_top_drivers)

    combined_mid_drivers = list(shared_mid_drivers) + list(val_unique_mid)

    max_mid_drivers = max(len(train_mid_tier_drivers), len(val_mid_tier_drivers))
    if len(combined_mid_drivers) < max_mid_drivers:
        remaining = max_mid_drivers - len(combined_mid_drivers)
        combined_mid_drivers.extend(list(train_unique_mid)[:remaining])

    return combined_position_boundaries, combined_top_drivers, combined_mid_drivers


def engineer_features(
    results_df: pd.DataFrame,
    qualifying_df: pd.DataFrame,
    drivers_df: pd.DataFrame,
    constructors_df: pd.DataFrame,
    circuits_df: pd.DataFrame,
    laps_df: pd.DataFrame,
    pitstops_df: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]], list[str], list[str]]:

    features = pd.merge(
        results_df, qualifying_df,
        on=['season', 'round', 'driver_id', 'constructor_id'],
        how='left',
        suffixes=('_res', '_qual')
    )

    if 'position_res' in features.columns:
        features.rename(columns={'position_res': 'position'}, inplace=True)

    features['position_numeric'] = pd.to_numeric(features['position'], errors='coerce')

    features['full_data_available'] = True

    features = pd.merge(features, drivers_df, on=['season', 'driver_id', 'constructor_id'], how='left')
    features = pd.merge(features, constructors_df, on=['season', 'constructor_id'], how='left')
    features = pd.merge(features, circuits_df, on=['season', 'round'], how='left')

    features['q1_seconds'] = features['q1_time'].apply(convert_qualifying_time)
    features['q2_seconds'] = features['q2_time'].apply(convert_qualifying_time)
    features['q3_seconds'] = features['q3_time'].apply(convert_qualifying_time)
    features['best_qualifying_time'] = features[['q1_seconds', 'q2_seconds', 'q3_seconds']].min(axis=1)

    q_best_per_session = features.groupby(['season', 'round'])['best_qualifying_time'].min().reset_index()
    q_best_per_session.rename(columns={'best_qualifying_time': 'session_best_time'}, inplace=True)
    features = pd.merge(features, q_best_per_session, on=['season', 'round'], how='left')
    features['qualifying_delta'] = features['best_qualifying_time'] - features['session_best_time']
    features['qualifying_delta_pct'] = features['qualifying_delta'] / features['session_best_time'] * 100

    features['grid'] = pd.to_numeric(features['grid'], errors='coerce')

    features['date_of_birth'] = pd.to_datetime(features['date_of_birth'])
    features['race_date'] = pd.to_datetime(features['date'])
    features['driver_age'] = (features['race_date'] - features['date_of_birth']).dt.days / 365.25

    driver_exp = results_df.groupby(['driver_id', 'season', 'round']).size().reset_index(name='race_count')
    driver_exp['race_count'] = driver_exp.groupby('driver_id')['race_count'].cumsum()
    features = pd.merge(features, driver_exp, on=['driver_id', 'season', 'round'], how='left')
    features.rename(columns={'race_count': 'driver_experience'}, inplace=True)

    driver_positions = results_df[['driver_id', 'season', 'round', 'position']].copy()
    driver_positions['position'] = pd.to_numeric(driver_positions['position'], errors='coerce')
    driver_positions['season_round'] = driver_positions['season'].astype(str) + '_' + driver_positions['round'].astype(str)
    driver_positions = driver_positions.sort_values(['driver_id', 'season', 'round'])

    driver_positions['avg_last_3_positions'] = driver_positions.groupby(
        'driver_id'
    )['position'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())

    driver_positions['avg_last_5_positions'] = driver_positions.groupby(
        'driver_id'
    )['position'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())

    driver_positions['avg_last_10_positions'] = driver_positions.groupby(
        'driver_id'
    )['position'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())

    driver_positions['position_momentum'] = driver_positions.groupby('driver_id')['position'].diff(-3)

    driver_positions['position_consistency'] = driver_positions.groupby('driver_id')['position'].transform(
        lambda x: x.rolling(window=5, min_periods=3).std()
    )

    driver_positions['trend_direction'] = driver_positions.groupby('driver_id')['position'].diff(-1).apply(
        lambda x: 1 if pd.notnull(x) and x > 0 else (-1 if pd.notnull(x) and x < 0 else 0)
    )

    driver_positions_by_driver = driver_positions.sort_values(['driver_id', 'season', 'round'])
    driver_positions_by_driver['ewm_avg_position'] = driver_positions_by_driver.groupby('driver_id')['position'].transform(
        lambda x: x.ewm(span=5, min_periods=1).mean()
    )

    features = pd.merge(
        features,
        driver_positions[['driver_id', 'season', 'round', 'avg_last_3_positions',
                          'avg_last_5_positions', 'avg_last_10_positions',
                          'position_momentum', 'position_consistency', 'trend_direction']],
        on=['driver_id', 'season', 'round'],
        how='left'
    )

    features = pd.merge(
        features,
        driver_positions_by_driver[['driver_id', 'season', 'round', 'ewm_avg_position']],
        on=['driver_id', 'season', 'round'],
        how='left'
    )

    constructor_positions = results_df[['constructor_id', 'season', 'round', 'position']].copy()
    constructor_positions['position'] = pd.to_numeric(constructor_positions['position'], errors='coerce')
    constructor_positions['season_round'] = constructor_positions['season'].astype(str) + '_' + constructor_positions['round'].astype(str)
    constructor_positions = constructor_positions.sort_values(['constructor_id', 'season_round'])

    constructor_positions['avg_constructor_last_3_positions'] = constructor_positions.groupby(
        'constructor_id'
    )['position'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())

    constructor_positions['constructor_momentum'] = constructor_positions.groupby('constructor_id')['position'].diff(-3)

    constructor_positions['constructor_consistency'] = constructor_positions.groupby(
        'constructor_id'
    )['position'].transform(lambda x: x.rolling(window=5, min_periods=3).std())

    constructor_positions_by_constructor = constructor_positions.sort_values(['constructor_id', 'season', 'round'])
    constructor_positions_by_constructor['constructor_ewm_avg'] = constructor_positions_by_constructor.groupby('constructor_id')['position'].transform(
        lambda x: x.ewm(span=5, min_periods=1).mean()
    )

    features = pd.merge(
        features,
        constructor_positions[['constructor_id', 'season', 'round',
                               'avg_constructor_last_3_positions',
                               'constructor_momentum', 'constructor_consistency']],
        on=['constructor_id', 'season', 'round'],
        how='left'
    )

    features = pd.merge(
        features,
        constructor_positions_by_constructor[['constructor_id', 'season', 'round', 'constructor_ewm_avg']],
        on=['constructor_id', 'season', 'round'],
        how='left'
    )

    circuit_driver_perf = features[['driver_id', 'circuit_id', 'position_numeric']].copy()
    circuit_driver_perf = circuit_driver_perf.dropna(subset=['position_numeric'])
    circuit_driver_perf = circuit_driver_perf.groupby(['driver_id', 'circuit_id']).agg(
        avg_position_at_circuit=('position_numeric', 'mean'),
        best_position_at_circuit=('position_numeric', 'min'),
        races_at_circuit=('position_numeric', 'count')
    ).reset_index()
    features = pd.merge(features, circuit_driver_perf, on=['driver_id', 'circuit_id'], how='left')

    circuit_constructor_perf = features[['constructor_id', 'circuit_id', 'position_numeric']].copy()
    circuit_constructor_perf = circuit_constructor_perf.dropna(subset=['position_numeric'])
    circuit_constructor_perf = circuit_constructor_perf.groupby(['constructor_id', 'circuit_id']).agg(
        constructor_avg_position_at_circuit=('position_numeric', 'mean'),
        constructor_best_position_at_circuit=('position_numeric', 'min'),
        constructor_races_at_circuit=('position_numeric', 'count')
    ).reset_index()
    features = pd.merge(features, circuit_constructor_perf, on=['constructor_id', 'circuit_id'], how='left')

    cleaned_pitstops_df = clean_pitstop_duration(pitstops_df)

    pitstop_counts = cleaned_pitstops_df.groupby(['driver_id', 'season', 'round']).size().reset_index(name='pitstop_count')
    features = pd.merge(features, pitstop_counts, on=['driver_id', 'season', 'round'], how='left')
    features['pitstop_count'] = features['pitstop_count'].fillna(0)

    pitstop_duration = cleaned_pitstops_df.groupby(['driver_id', 'season', 'round'])['duration'].mean().reset_index(name='avg_pitstop_duration')
    features = pd.merge(features, pitstop_duration, on=['driver_id', 'season', 'round'], how='left')
    features['avg_pitstop_duration'] = features['avg_pitstop_duration'].fillna(25.0)

    position_std = results_df.groupby('driver_id')['position'].apply(
        lambda x: pd.to_numeric(x, errors='coerce').std()
    ).reset_index()
    position_std.rename(columns={'position': 'position_std'}, inplace=True)
    features = pd.merge(features, position_std, on='driver_id', how='left')

    max_year = features['season'].max()
    features['year_weight'] = features['season'].apply(lambda x: get_weight_by_year(x, max_year))

    numerical_features = FEATURE_SET

    for col in numerical_features:
        if col in features.columns:
            if features[col].isna().any():
                if col in ['position_momentum', 'constructor_momentum', 'trend_direction']:

                    features[col] = features[col].fillna(0)
                elif col in ['driver_experience', 'races_at_circuit', 'constructor_races_at_circuit']:

                    features[col] = features[col].fillna(0)
                elif col.startswith('pitstop'):

                    default_value = 0 if col == 'pitstop_count' else 25.0
                    features[col] = features[col].fillna(default_value)
                else:

                    features[col] = features[col].fillna(features[col].median())
        else:

            features[col] = 0

    features.drop(
        ['q1_time', 'q2_time', 'q3_time', 'date_of_birth', 'race_date'],
        axis=1, errors='ignore', inplace=True
    )

    position_boundaries = calculate_driver_position_boundaries(results_df)
    top_drivers = find_top_drivers(results_df)
    mid_tier_drivers = find_mid_tier_drivers(results_df, top_drivers)

    return features, position_boundaries, top_drivers, mid_tier_drivers


class F1FeatureProcessor:
    def __init__(self) -> None:
        self.driver_encoder = None
        self.constructor_encoder = None
        self.circuit_encoder = None
        self.feature_scaler = None
        self.position_boundaries = {}
        self.sequence_length = 5
        self.top_drivers = []
        self.mid_tier_drivers = []
        self.features_used_in_fitting: list[str] = []

    def fit_transform(self, features_df: pd.DataFrame) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        return self._prepare_sequences(features_df, is_training=True)

    def transform(self, features_df: pd.DataFrame) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        return self._prepare_sequences(features_df, is_training=False)

    def transform_test_data(
        self,
        qualifying_df: pd.DataFrame,
        drivers_df: pd.DataFrame,
        constructors_df: pd.DataFrame,
        circuits_df: pd.DataFrame,
        laps_df: pd.DataFrame,
        pitstops_df: pd.DataFrame,
        results_train_df: pd.DataFrame,
        test_season: int,
        test_round: int
    ) -> tuple[torch.FloatTensor, list[str], list[int]]:
        test_df, test_features = self._prepare_test_dataset(
            qualifying_df, drivers_df, constructors_df, circuits_df,
            laps_df, pitstops_df, results_train_df, test_season, test_round
        )

        return self._process_test_features(test_df, test_features, qualifying_df, test_season, test_round)

    def _prepare_sequences(self, features: pd.DataFrame, is_training: bool = False) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        driver_encoded, constructor_encoded, circuit_encoded = self._encode_categorical_features(features, is_training)
        numerical_scaled = self._process_numerical_features(features, is_training)

        X_combined = np.hstack([
            np.array(driver_encoded),
            np.array(constructor_encoded),
            np.array(circuit_encoded),
            np.array(numerical_scaled)
        ])

        return self._create_sequences(features, X_combined)

    def _encode_categorical_features(self, features: pd.DataFrame, is_training: bool) -> tuple:
        if is_training or self.driver_encoder is None:
            self.driver_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            driver_encoded = self.driver_encoder.fit_transform(features[['driver_id']])
        else:
            driver_encoded = self.driver_encoder.transform(features[['driver_id']])

        if is_training or self.constructor_encoder is None:
            self.constructor_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            constructor_encoded = self.constructor_encoder.fit_transform(features[['constructor_id']])
        else:
            constructor_encoded = self.constructor_encoder.transform(features[['constructor_id']])

        if is_training or self.circuit_encoder is None:
            self.circuit_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            circuit_encoded = self.circuit_encoder.fit_transform(features[['circuit_id']])
        else:
            circuit_encoded = self.circuit_encoder.transform(features[['circuit_id']])

        return driver_encoded, constructor_encoded, circuit_encoded

    def _process_numerical_features(self, features: pd.DataFrame, is_training: bool) -> np.ndarray:
        base_numerical_features = FEATURE_SET
        optional_features = ['position_consistency', 'constructor_consistency']

        all_potential_features = base_numerical_features + optional_features
        available_features = [f for f in all_potential_features if f in features.columns]

        if is_training:
            self.features_used_in_fitting = available_features

        features_to_use = available_features if is_training else self.features_used_in_fitting
        safe_features = features[features_to_use].fillna(0).replace([np.inf, -np.inf], 0)

        if is_training or self.feature_scaler is None:
            self.feature_scaler = StandardScaler()
            numerical_scaled = self.feature_scaler.fit_transform(safe_features)
        else:
            numerical_scaled = self.feature_scaler.transform(safe_features)

        numerical_scaled = np.nan_to_num(numerical_scaled, nan=0.0)

        return numerical_scaled

    def _create_sequences(self, features: pd.DataFrame, X_combined: np.ndarray) -> tuple:
        drivers = features['driver_id'].unique()
        X_sequences = []
        y_positions = []
        sample_weights = []

        max_year = features['season'].max()
        position_column = 'position_numeric' if 'position_numeric' in features.columns else 'position'

        for driver in drivers:
            driver_data = features[features['driver_id'] == driver].sort_values(['season', 'round'])

            for i in range(len(driver_data) - self.sequence_length):
                target_idx = i + self.sequence_length
                X_seq = X_combined[driver_data.index[i:i + self.sequence_length]]
                y_pos = driver_data.iloc[target_idx][position_column]

                if not pd.isna(y_pos):
                    year_weight = get_weight_by_year(driver_data.iloc[target_idx]['season'], max_year)
                    X_sequences.append(X_seq)
                    y_positions.append(float(y_pos))
                    sample_weights.append(year_weight)

        if X_sequences:
            X_tensor = torch.FloatTensor(np.array(X_sequences))
            y_tensor = torch.FloatTensor(np.array(y_positions))
            weights_tensor = torch.FloatTensor(np.array(sample_weights))
        else:
            X_tensor = torch.FloatTensor()
            y_tensor = torch.FloatTensor()
            weights_tensor = torch.FloatTensor()

        logging.info(f'Input Sequences Shape: {X_tensor.shape}')
        logging.info(f'Target Values Shape: {y_tensor.shape}')
        logging.info(f'Sample Weights Shape: {weights_tensor.shape}')
        return X_tensor, y_tensor, weights_tensor

    def _prepare_test_dataset(
        self,
        qualifying_df: pd.DataFrame,
        drivers_df: pd.DataFrame,
        constructors_df: pd.DataFrame,
        circuits_df: pd.DataFrame,
        laps_df: pd.DataFrame,
        pitstops_df: pd.DataFrame,
        results_train_df: pd.DataFrame,
        test_season: int,
        test_round: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        race_qualifying = qualifying_df[(qualifying_df['season'] == test_season) & (qualifying_df['round'] == test_round)]
        race_circuit = circuits_df[(circuits_df['season'] == test_season) & (circuits_df['round'] == test_round)]

        if race_qualifying.empty or race_circuit.empty:
            raise ValueError(f'No Qualifying or Circuit Data Found for Season {test_season}, Round {test_round}')

        test_data = []
        for _, quali_row in race_qualifying.iterrows():
            driver_id = quali_row['driver_id']
            constructor_id = quali_row['constructor_id']

            driver_info = drivers_df[
                (drivers_df['driver_id'] == driver_id) &
                (drivers_df['season'] <= test_season)
            ].sort_values('season')

            if driver_info.empty:
                continue

            driver_info = driver_info.iloc[-1]

            constructor_info = constructors_df[constructors_df['constructor_id'] == constructor_id]
            if constructor_info.empty:
                continue

            constructor_info = constructor_info.iloc[0]

            driver_row = {
                'season': test_season,
                'round': test_round,
                'car_number': quali_row['car_number'],
                'driver_id': driver_id,
                'constructor_id': constructor_id,
                'q1_time': quali_row['q1_time'],
                'q2_time': quali_row['q2_time'],
                'q3_time': quali_row['q3_time'],
                'grid': quali_row['position'],
                'circuit_id': race_circuit['circuit_id'].iloc[0],
                'date': race_circuit['date'].iloc[0]
            }

            for col in ['given_name', 'family_name', 'date_of_birth', 'nationality']:
                if col in driver_info:
                    driver_row[col] = driver_info[col]

            for col in ['name', 'nationality']:
                if col in constructor_info:
                    driver_row[f'constructor_{col}'] = constructor_info[col]

            test_data.append(driver_row)

        test_df = pd.DataFrame(test_data)

        if test_df.empty:
            raise ValueError(f'Could NOT Create Test Data for Season {test_season}, Round {test_round}')

        test_features, position_boundaries, top_drivers, mid_tier_drivers = engineer_features(
            results_df=results_train_df,
            qualifying_df=qualifying_df,
            drivers_df=drivers_df,
            constructors_df=constructors_df,
            circuits_df=circuits_df,
            laps_df=laps_df,
            pitstops_df=pitstops_df
        )

        self.position_boundaries = position_boundaries
        self.top_drivers = top_drivers
        self.mid_tier_drivers = mid_tier_drivers

        if 'position' in test_features.columns and 'position_numeric' not in test_features.columns:
            test_features['position_numeric'] = pd.to_numeric(test_features['position'], errors='coerce')

        return test_df, test_features

    def _create_dummy_driver(self, driver_id: str, row: pd.Series, test_season: int) -> dict[str, Any]:
        q1 = convert_qualifying_time(row['q1_time'])
        q2 = convert_qualifying_time(row['q2_time'])
        q3 = convert_qualifying_time(row['q3_time'])
        best_qual = min([q1, q2, q3]) if q1 > 0 or q2 > 0 or q3 > 0 else 0

        driver_age = 0
        if 'date_of_birth' in row and 'date' in row:
            dob = pd.to_datetime(row['date_of_birth'])
            race_date = pd.to_datetime(row['date'])
            driver_age = (race_date - dob).days / 365.25

        grid_pos = pd.to_numeric(row['grid'], errors='coerce')

        return {
            'driver_id': driver_id,
            'constructor_id': row['constructor_id'],
            'circuit_id': row['circuit_id'],
            'grid': grid_pos,
            'position': 10.0,
            'position_numeric': 10.0,
            'q1_seconds': q1,
            'q2_seconds': q2,
            'q3_seconds': q3,
            'best_qualifying_time': best_qual,
            'driver_age': driver_age,
            'driver_experience': 0,
            'avg_last_3_positions': 10,
            'avg_last_5_positions': 10,
            'avg_last_10_positions': 10,
            'position_momentum': 0,
            'avg_constructor_last_3_positions': 10,
            'constructor_momentum': 0,
            'position_consistency': 3,
            'constructor_consistency': 3,
            'avg_position_at_circuit': 10,
            'best_position_at_circuit': 10,
            'races_at_circuit': 0,
            'constructor_avg_position_at_circuit': 10,
            'constructor_best_position_at_circuit': 10,
            'constructor_races_at_circuit': 0,
            'pitstop_count': 1,
            'avg_pitstop_duration': 25,
            'position_std': 3,
            'trend_direction': 0,
            'ewm_avg_position': 10,
            'constructor_ewm_avg': 10,
            'year_weight': get_weight_by_year(test_season, test_season)
        }

    def _process_test_features(
        self,
        test_df: pd.DataFrame,
        test_features: pd.DataFrame,
        qualifying_df: pd.DataFrame,
        test_season: int,
        test_round: int
    ) -> tuple[torch.FloatTensor, list[str], list[int]]:
        missing_drivers = set(test_df['driver_id']) - set(test_features['driver_id'])
        if missing_drivers:
            dummy_rows = []
            for d in missing_drivers:
                row = test_df[test_df['driver_id'] == d]
                if row.empty:
                    continue

                dummy = self._create_dummy_driver(d, row.iloc[0], test_season)

                if self.features_used_in_fitting:
                    for feature in self.features_used_in_fitting:
                        if feature not in dummy:
                            dummy[feature] = 0

                dummy_rows.append(dummy)

            if dummy_rows:
                dummy_df = pd.DataFrame(dummy_rows)
                test_features = pd.concat([test_features, dummy_df], ignore_index=True)

        if self.features_used_in_fitting:
            for feature in self.features_used_in_fitting:
                if feature not in test_features.columns:
                    test_features[feature] = 0

        X_test_sequences = []
        test_drivers = []
        qualifying_positions = []

        race_qualifying = qualifying_df[(qualifying_df['season'] == test_season) & (qualifying_df['round'] == test_round)]

        for driver in test_df['driver_id'].unique():
            driver_features = test_features[test_features['driver_id'] == driver].sort_values(['season', 'round'])

            if len(driver_features) < self.sequence_length:
                if len(driver_features) == 0:
                    continue

                num_needed = self.sequence_length - len(driver_features)
                replicated = pd.concat([driver_features.tail(1)] * num_needed, ignore_index=True)
                driver_features = pd.concat([driver_features, replicated], ignore_index=True)
            else:
                driver_features = driver_features.tail(self.sequence_length)

            assert self.driver_encoder is not None, 'Driver Encoder NOT Initialised'
            assert self.constructor_encoder is not None, 'Constructor Encoder NOT Initialised'
            assert self.circuit_encoder is not None, 'Circuit Encoder NOT Initialised'
            assert self.feature_scaler is not None, 'Feature Scaler NOT Initialised'
            assert len(self.features_used_in_fitting) > 0, 'Features NOT Fitted'

            driver_encoded = self.driver_encoder.transform(driver_features[['driver_id']])
            constructor_encoded = self.constructor_encoder.transform(driver_features[['constructor_id']])
            circuit_encoded = self.circuit_encoder.transform(driver_features[['circuit_id']])

            safe_features = driver_features[self.features_used_in_fitting].fillna(0).replace([np.inf, -np.inf], 0)
            for col in safe_features.columns:
                max_val = 1e6
                min_val = -1e6
                safe_features[col] = safe_features[col].clip(lower=min_val, upper=max_val)

            numerical_scaled = self.feature_scaler.transform(safe_features)
            numerical_scaled = np.nan_to_num(numerical_scaled, nan=0.0)

            X_combined = np.hstack([
                np.array(driver_encoded),
                np.array(constructor_encoded),
                np.array(circuit_encoded),
                np.array(numerical_scaled)
            ])

            X_test_sequences.append(X_combined)
            test_drivers.append(driver)

            driver_quali = race_qualifying[race_qualifying['driver_id'] == driver]
            if not driver_quali.empty:
                quali_pos = pd.to_numeric(driver_quali['position'].iloc[0], errors='coerce')
                qualifying_positions.append(int(quali_pos) if not pd.isna(quali_pos) else 10)
            else:
                qualifying_positions.append(10)

        if X_test_sequences:
            X_test_tensor = torch.FloatTensor(np.array(X_test_sequences))
            return X_test_tensor, test_drivers, qualifying_positions
        else:
            return torch.FloatTensor(), [], []
