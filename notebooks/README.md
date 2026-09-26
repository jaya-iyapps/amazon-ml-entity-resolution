# Notebooks Directory

This folder is reserved for exploratory data analysis (EDA), prototype models, error analysis, and visualization.

---

## Guidelines for Team Members

1. **Naming Convention**:
   - `01_eda_profiling_<person>.ipynb` (Data exploration)
   - `02_blocking_prototypes_<person>.ipynb` (Blocking experiments)
   - `03_matching_models_<person>.ipynb` (Feature engineering & training)
   - `04_error_analysis_<person>.ipynb` (Inspecting false positives/negatives)

2. **Git Hygiene**:
   - **Never commit outputs containing raw dataset rows.**
   - Clear all outputs before committing (`Kernel -> Restart & Clear Output`), or use a lightweight git filter.
   - Do not load full 26M dataset records in interactive notebook cells without sample limiting.

3. **Reproducibility**:
   - Keep notebooks self-contained or import reusable logic directly from `src/`.
   - Promote validated logic from exploratory notebooks into `src/` modules.
