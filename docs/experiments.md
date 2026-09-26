# Experiments Tracking Log

**Amazon ML Challenge 2026 – Business Entity Resolution**

Use this log to record experiments across all three team members.

---

## Experiment Registry

| Experiment | Person | Date | Changes | Candidate Count | Precision | Recall | Macro F0.5 | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| EXP-001 | Person 1 | 2026-09-25 | Baseline text and address normalization | N/A | N/A | N/A | N/A | Cleaned legal suffixes & punctuation |
| EXP-002 | Person 3 | 2026-09-25 | Inverted index name token blocking | Top 50 / entity | TBD | TBD | TBD | Candidate generation baseline |
| EXP-003 | Person 2 | 2026-09-25 | Logistic regression baseline with similarity features | Top 50 / entity | TBD | TBD | TBD | Evaluated with threshold 0.5 |
| EXP-004 | Team | - | - | - | - | - | - | - |
| EXP-005 | Team | - | - | - | - | - | - | - |

---

## Guidelines for Logging Experiments

1. **Tag your branch/commit**: Reference the Git commit hash when logging a result.
2. **Validation Setup**: Always test on the same held-out validation split to ensure comparability.
3. **Metric Focus**: Track both raw Precision/Recall and Macro $F_{0.5}$. Because $\beta=0.5$, check whether changes improved precision.
4. **Never Commit Artifacts**: Store model weights and huge TSVs locally or in designated scratch storage, not in Git.
