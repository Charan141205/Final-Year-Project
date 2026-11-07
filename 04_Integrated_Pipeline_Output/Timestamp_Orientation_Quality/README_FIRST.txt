CAPPIMU TIMESTAMP + ORIENTATION QUALITY AUDIT
==============================================

This program separates dt == 0 duplicate timestamps from dt < 0 backwards timestamps.
It also counts positive gaps above the selected threshold (default 500 raw units).
Timestamp units are not assumed. No source data is changed.

Files:
- timestamp_orientation_quality_by_subject.csv: detailed per-subject counts and intervals
- timestamp_audit_inventory.csv: processed/missing/error inventory
- timestamp_event_examples.csv: largest backward steps, largest positive gaps, and first duplicate examples
- timestamp_orientation_audit_summary.json: aggregate totals

The 500 threshold is expressed in raw timestamp units, not automatically milliseconds.
