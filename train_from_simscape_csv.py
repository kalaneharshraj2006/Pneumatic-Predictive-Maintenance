"""Train a fault classifier from a labelled, normalized Simscape CSV.

    Required columns: supply_pressure_bar, chamber_pressure_bar, pressure_drop_bar,
    air_flow_l_min, actuation_time_s, temperature_c, vibration_rms, load_factor,
    fault_type. Column names must match exactly. This script deliberately stops
    if labels or required sensor features are missing; it will not fabricate them.
    """
    from pathlib import Path
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import joblib

    ROOT = Path(__file__).resolve().parents[1]
    DATA = ROOT / "data" / "simscape_exports" / "labelled_simscape_data.csv"
    OUT = ROOT / "outputs"
    MODELS = ROOT / "models"
    FEATURES = ["supply_pressure_bar", "chamber_pressure_bar", "pressure_drop_bar",
                "air_flow_l_min", "actuation_time_s", "temperature_c",
                "vibration_rms", "load_factor"]
    if not DATA.exists():
        raise SystemExit(f"Place your labelled, normalized Simscape dataset at: {DATA}
"
                         "The raw logged signal export needs unit conversion, feature mapping, "
                         "and a fault_type label before training.")
    df = pd.read_csv(DATA)
    required = FEATURES + ["fault_type", "run_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit("Missing required columns: " + ", ".join(missing))
    if df[required].isna().any().any():
        raise SystemExit("Required columns contain missing values; clean the data first.")
    split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr, te = next(split.split(df, df.fault_type, groups=df.run_id))
    model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    model.fit(df.iloc[tr][FEATURES], df.iloc[tr].fault_type)
    pred = model.predict(df.iloc[te][FEATURES])
    OUT.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)
    joblib.dump(model, MODELS / "simscape_fault_classifier.joblib")
    (OUT / "simscape_classification_report.txt").write_text(
        f"Accuracy: {accuracy_score(df.iloc[te].fault_type, pred):.4f}\n\n" +
        classification_report(df.iloc[te].fault_type, pred, zero_division=0))
    pd.DataFrame(confusion_matrix(df.iloc[te].fault_type, pred,
                 labels=model.classes_), index=model.classes_, columns=model.classes_).to_csv(
                 OUT / "simscape_confusion_matrix.csv")
    print((OUT / "simscape_classification_report.txt").read_text())
