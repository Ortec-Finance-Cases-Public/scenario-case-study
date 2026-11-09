# Synthetic macro-financial dataset: case_study_example

Synthetic monthly histories of US, euro area, Japanese and global macroeconomic and financial variables. Dates are fictional (the calendar is only meaningful for seasonality).

| file | content |
|---|---|
| `<split>.parquet` | long table: `history_id`, `period` (YYYY-MM), `variable`, `value` |
| `variables.csv` | variable descriptions, frequency (M monthly, Q quarterly: only at quarter-end months), value type |

Each history covers 360 months. Splits: `train`, `validation` (24 months held out after each history for evaluation), `test` (24 months held out after each history for evaluation).
