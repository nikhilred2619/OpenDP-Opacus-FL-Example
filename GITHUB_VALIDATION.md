# GitHub Actions validation

This project is intended to be uploaded with the **contents of `fl/` at the
repository root**. The workflow lives at `.github/workflows/validate.yml`.

Each push and pull request runs:

1. Python 3.13 dependency installation and `pip check`.
2. All automated pytest tests.
3. Every notebook cell via `nbconvert`.
4. A downloadable Actions artifact containing test and notebook logs,
   package versions, and the executed notebook (when created).

Successful CI is evidence that tests ran in a separate runner. It is **not**
proof of the custom privacy map, ideal-noise assumptions, or production security.

## Viewing evidence

After pushing, open the repository's **Actions** tab, select
**OpenDP FL Stage 1 validation**, and inspect the latest run.
Download the `opendp-fl-stage1-validation` artifact for the actual outputs.

Do not report tests as independently re-run until a green check appears.
