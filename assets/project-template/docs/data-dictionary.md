# Data dictionary

Document the scientific meaning and coding of each column before fitting.

| Column | Type | Unit/coding | Meaning |
|---|---|---|---|
| `subj_idx` | integer/string | stable ID | Participant identifier |
| `rt` | float | seconds | Reaction time |
| `response` | integer | 0/1 or model-specific | Boundary/choice code |
| `condition` | category | project-specific | Experimental condition |

State whether `response` means accuracy or physical choice. For stimulus coding,
also document the two-level stimulus column and which response is correct.
