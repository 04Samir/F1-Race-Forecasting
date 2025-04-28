import logging
import os
import warnings
from tabulate import tabulate

import pandas as pd

from .modules import (
    F1FeatureProcessor,
    F1RacePredictor,
    engineer_features,
    evaluate_race_predictions,
    fetch_data,
    format_evaluation_results,
    get_combined_domain_knowledge,
    parse_data,
    save_metrics,
    set_seeds,
)

warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')

def predict_race() -> None:
    set_seeds(42)

    logging.info("Parsing Data . . .")
    train_data, val_data, test_data = parse_data()
    logging.info("Data Parsed Successfully!")

    circuits_train, constructors_train, drivers_train, laps_train, pitstops_train, qualifying_train, results_train = train_data
    circuits_val, constructors_val, drivers_val, laps_val, pitstops_val, qualifying_val, results_val = val_data
    qualifying_test, results_test = test_data

    test_season = int(qualifying_test['season'].iloc[0])
    test_round = int(qualifying_test['round'].iloc[0])

    logging.info(f"Engineering Training Features . . .")
    train_features, train_position_boundaries, train_top_drivers, train_mid_tier_drivers = engineer_features(
        results_df=results_train,
        qualifying_df=qualifying_train,
        drivers_df=drivers_train,
        constructors_df=constructors_train,
        circuits_df=circuits_train,
        laps_df=laps_train,
        pitstops_df=pitstops_train
    )

    feature_processor = F1FeatureProcessor()
    feature_processor.position_boundaries = train_position_boundaries
    feature_processor.top_drivers = train_top_drivers
    feature_processor.mid_tier_drivers = train_mid_tier_drivers

    X_train, y_train, sample_weights_train = feature_processor.fit_transform(train_features)

    logging.info(f"Engineering Validation Features . . .")
    val_features, val_position_boundaries, val_top_drivers, val_mid_tier_drivers = engineer_features(
        results_df=results_val,
        qualifying_df=qualifying_val,
        drivers_df=drivers_val,
        constructors_df=constructors_val,
        circuits_df=circuits_val,
        laps_df=laps_val,
        pitstops_df=pitstops_val
    )

    X_val, y_val, sample_weights_val = feature_processor.transform(val_features)

    all_drivers_df = pd.concat([drivers_train, drivers_val], ignore_index=True)
    all_constructors_df = pd.concat([constructors_train, constructors_val], ignore_index=True)
    all_circuits_df = pd.concat([circuits_train, circuits_val], ignore_index=True)
    all_laps_df = pd.concat([laps_train, laps_val], ignore_index=True)
    all_pitstops_df = pd.concat([pitstops_train, pitstops_val], ignore_index=True)
    all_results_df = pd.concat([results_train, results_val], ignore_index=True)
    all_qualifying_df = pd.concat([qualifying_train, qualifying_val], ignore_index=True)

    logging.info(f"Combining Domain Knowledge from Training & Validation Sets . . .")
    train_knowledge = (train_position_boundaries, train_top_drivers, train_mid_tier_drivers)
    val_knowledge = (val_position_boundaries, val_top_drivers, val_mid_tier_drivers)

    combined_position_boundaries, combined_top_drivers, combined_mid_tier_drivers = get_combined_domain_knowledge(
        train_results_df=results_train,
        val_results_df=results_val,
        train_knowledge=train_knowledge,
        val_knowledge=val_knowledge,
    )

    feature_processor.position_boundaries = combined_position_boundaries
    feature_processor.top_drivers = combined_top_drivers
    feature_processor.mid_tier_drivers = combined_mid_tier_drivers

    logging.info(f"Position Boundaries Updated with Combined Knowledge: {len(combined_position_boundaries)} drivers")
    logging.info(f"Top Drivers Updated: {combined_top_drivers}")
    logging.info(f"Mid-Tier Drivers Updated: {combined_mid_tier_drivers}")
    logging.info("Model Features Engineered Successfully!")

    logging.info("Training Model . . .")
    predictor = F1RacePredictor(use_saved_model=False)
    predictor.train(
        feature_processor,
        X_train, y_train, sample_weights_train,
        X_val, y_val, sample_weights_val
    )
    logging.info("Model Trained Successfully!")

    logging.info(f"Preparing Test Data for Forecast -> {test_season} Season, Round {test_round} . . .")

    X_test, test_drivers, qualifying_results = feature_processor.transform_test_data(
        qualifying_df=pd.concat([qualifying_test, all_qualifying_df], ignore_index=True),
        drivers_df=all_drivers_df,
        constructors_df=all_constructors_df,
        circuits_df=all_circuits_df,
        laps_df=all_laps_df,
        pitstops_df=all_pitstops_df,
        results_train_df=all_results_df,
        test_season=test_season,
        test_round=test_round
    )

    logging.info("Generating Predictions . . .")
    raw_predictions, adjusted_predictions = predictor.predict(
        X_test, test_drivers, qualifying_results
    )

    predictions_df = pd.DataFrame({
        'driver_id': test_drivers,
        'predicted_position': adjusted_predictions,
        'raw_prediction': raw_predictions
    })

    predictions_df['final_position'] = predictions_df['predicted_position'].rank(method='first')

    driver_names = all_drivers_df[['driver_id', 'given_name', 'family_name']].drop_duplicates()
    predictions_df = pd.merge(predictions_df, driver_names, on='driver_id', how='left')
    predictions_df['driver_name'] = predictions_df['given_name'] + ' ' + predictions_df['family_name']

    race_qualifying_subset = qualifying_test[['driver_id', 'constructor_id']]
    predictions_df = pd.merge(predictions_df, race_qualifying_subset, on='driver_id', how='left')

    constructor_names = all_constructors_df[['constructor_id', 'name']].drop_duplicates()
    predictions_df = pd.merge(predictions_df, constructor_names, on='constructor_id', how='left')
    predictions_df.rename(columns={'name': 'constructor'}, inplace=True)

    predictions_df = predictions_df.sort_values('final_position')

    race_info = all_circuits_df[(all_circuits_df['season'] == test_season) & (all_circuits_df['round'] == test_round)]
    race_name = race_info['race_name'].iloc[0] if not race_info.empty else f"Race {test_round}"

    logging.info(f"Predictions Generated:")
    print(f"\nPredictions for {race_name} (Season {test_season}, Round {test_round}):")

    display_df = predictions_df[['driver_name', 'constructor', 'final_position']].copy()
    display_df['final_position'] = display_df['final_position'].astype(int)

    if not results_test.empty:
        actual_results = results_test[['driver_id', 'position']].copy()
        actual_results['actual_position'] = pd.to_numeric(actual_results['position'], errors='coerce').astype(int)

        qualifying_results_df = qualifying_test[['driver_id', 'position']].copy()
        qualifying_results_df['qualifying_result'] = pd.to_numeric(qualifying_results_df['position'], errors='coerce').astype(int)

        display_with_ids = pd.merge(
            display_df,
            predictions_df[['driver_id', 'driver_name']],
            on='driver_name',
            how='left'
        )

        display_with_actual = pd.merge(
            display_with_ids,
            actual_results[['driver_id', 'actual_position']],
            on='driver_id',
            how='left'
        )

        display_with_qualifying = pd.merge(
            display_with_actual,
            qualifying_results_df[['driver_id', 'qualifying_result']],
            on='driver_id',
            how='left'
        )

        display_df = display_with_qualifying[['driver_name', 'constructor', 'qualifying_result', 'final_position', 'actual_position']]
        display_df['diff'] = display_df['final_position'] - display_df['actual_position']
        display_df['diff'] = display_df['diff'].apply(lambda x: f"+{x}" if pd.notnull(x) and x > 0 else str(x) if pd.notnull(x) else "N/A")
        display_df.columns = ['Driver', 'Constructor', 'Qualifying Result', 'Predicted Position', 'Actual Position', 'Difference']
        display_df = display_df[['Driver', 'Constructor', 'Actual Position', 'Predicted Position', 'Difference', 'Qualifying Result']]
        display_df = display_df.sort_values('Actual Position')
    else:
        qualifying_results_df = qualifying_test[['driver_id', 'position']].copy()
        qualifying_results_df['qualifying_result'] = pd.to_numeric(qualifying_results_df['position'], errors='coerce').astype(int)

        display_with_ids = pd.merge(
            display_df,
            predictions_df[['driver_id', 'driver_name']],
            on='driver_name',
            how='left'
        )

        display_with_qualifying = pd.merge(
            display_with_ids,
            qualifying_results_df[['driver_id', 'qualifying_result']],
            on='driver_id',
            how='left'
        )

        display_df = display_with_qualifying[['driver_name', 'constructor', 'qualifying_result', 'final_position']]
        display_df.columns = ['Driver', 'Constructor', 'Qualifying Result', 'Predicted Position']
        display_df = display_df[['Driver', 'Constructor', 'Predicted Position', 'Qualifying Result']]
        display_df = display_df.sort_values('Predicted Position')

    headers = display_df.columns.tolist()
    table = tabulate(display_df.values.tolist(), headers=headers, tablefmt="pretty")
    print(table)

    if not results_test.empty:
        actual_results = results_test[['driver_id', 'position']].copy()
        actual_results['position'] = pd.to_numeric(actual_results['position'], errors='coerce')

        evaluation_df = pd.merge(
            actual_results,
            predictions_df[['driver_id', 'final_position']],
            on='driver_id',
            how='inner'
        )

        if not evaluation_df.empty:
            y_true = evaluation_df['position'].tolist()
            y_pred = evaluation_df['final_position'].tolist()

            metrics = evaluate_race_predictions(y_true, y_pred, evaluation_df['driver_id'].tolist())
            print("\n" + format_evaluation_results(metrics))
            save_metrics(
                (test_season, test_round, race_name),
                metrics=metrics,
                table=table
            )
        else:
            logging.warning("No Matching Results Found for Evaluation")
    else:
        logging.info("No Test Results Available for Evaluation")


def start(initial_choice: int | None = None) -> None:
    os.system('cls' if os.name == 'nt' else 'clear')

    while True:
        print()

        header = "F1 Race Forecasting"
        logging.info(header)

        logging.info("[1] Fetch from Ergast API")
        logging.info("[2] Parse the RAW Data")
        logging.info("[3] Predict Race Results")
        logging.info("[4] Exit")

        if initial_choice is not None:
            print(f"\nEnter your Choice: {initial_choice}")
            choice = str(initial_choice)
            initial_choice = None
        else:
            choice = input("\nEnter your Choice: ")

        print()
        if choice == '1':
            fetch_data()
        elif choice == '2':
            parse_data()
        elif choice == '3':
            predict_race()
        elif choice == '4':
            logging.info("Exiting . . .")
            break
        else:
            logging.error("Invalid Choice - Please Try Again!")
