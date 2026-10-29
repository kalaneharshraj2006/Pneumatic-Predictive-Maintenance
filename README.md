# Pneumatic Predictive Maintenance — Submission Package

A proof-of-concept workflow combining a MathWorks Simscape pneumatic example reference with machine-learning experiments for anomaly detection, fault classification, and remaining useful life (RUL).

## Contents
- `main.py`: reproducible equation-based synthetic-data baseline; generates datasets, trains models, and writes plots/metrics.
- `models/`, `outputs/`: baseline fitted models and example outputs from synthetic data.
- `matlab/export_logsout_to_csv.m`: starter utility to export logged Simulink timeseries.
- `python/train_from_simscape_csv.py`: classifier training script for a properly mapped and labelled Simscape dataset.
- `report.pdf`: baseline report.
- `REFERENCES.md`: MathWorks example attribution and scope notes.

## Run baseline on Windows
1. Install Python 3.11 or 3.12.
2. In this folder run:
   ```bat
   py -m venv .venv
   .venv\Scripts\activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   python main.py
   ```

## Simulink workflow
1. Open the MathWorks **Pneumatic Actuation Circuit** example: https://www.mathworks.com/help/simscape/ug/pneumatic-actuation-circuit.html
2. Save a separate working copy locally. Do not upload the original MathWorks model file unless its applicable license permits redistribution.
3. Enable signal logging for the pressure, flow, actuator-position/response signals available in your model. Run with `simOut = sim('YOUR_MODEL_NAME');`.
4. Run `matlab/export_logsout_to_csv.m` in the model's working folder. Inspect exported column names and units.
5. To train from physical-model data, prepare `data/simscape_exports/labelled_simscape_data.csv` with one row per sample and these columns: `run_id`, `fault_type`, `supply_pressure_bar`, `chamber_pressure_bar`, `pressure_drop_bar`, `air_flow_l_min`, `actuation_time_s`, `temperature_c`, `vibration_rms`, `load_factor`. Map the real signals and units carefully; do not fill missing signals with invented measurements.
6. Create separate runs for normal and fault conditions by changing supported model parameters. Record each run's fault label. The supplied example model has not been modified here to implement these faults automatically.
7. Run `python python/train_from_simscape_csv.py` after the labelled dataset is ready.

## Important scope statement
The included metrics and saved models are based on **illustrative synthetic data**, not data exported from Simscape. The included Simulink exporter and training adapter are integration scaffolding; end-to-end Simscape-to-ML integration is complete only after you run the model, export real simulation signals, label fault runs, map features/units, and train/evaluate on that dataset. No real-world predictive maintenance performance is claimed.

## Attribution and licensing
The original pneumatic example is authored by MathWorks; see `REFERENCES.md`. The MIT license in this repository applies to the original project code provided here, not to MathWorks materials or third-party dependencies.
