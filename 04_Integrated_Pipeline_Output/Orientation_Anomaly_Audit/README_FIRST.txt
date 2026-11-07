CAPPIMU ORIENTATION ANOMALY AUDIT
=================================

Files:
- orientation_anomaly_subject_summary.csv: per-subject angle and timestamp statistics
- ey_outlier_local_context.csv: +/- context rows around every ey value outside provisional +/-90 deg
- orientation_anomaly_inventory.csv: processed/missing/error inventory
- orientation_anomaly_summary.json: run totals and assumptions

The +/-180, +/-90, +/-180 degree bounds are provisional. This script does not establish axis mapping or rotation convention and does not rotate any vector. Timestamp differences are reported in raw timestamp units, not assumed milliseconds.
