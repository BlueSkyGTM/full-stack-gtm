## Ship It

The pipeline DAG and remote storage mechanism integrate cleanly into CI. On every push to `main`, the CI runner executes `dvc pull` to restore cached data from the remote, `dvc repro` to rebuild any stages whose dependencies changed, and `dvc push` to upload new artifacts. Reviewers see `dvc metrics show` output in the CI log, so a regression is visible before merge — not after a model ships to the scoring API.

Here is a GitHub Actions workflow that implements this:

```yaml
name: DVC Pipeline

on:
  push:
    branches: [main]

jobs:
  reproduce:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install dvc pandas pyarrow numpy pyyaml

      - name: Configure DVC remote
        run: |
          dvc remote add -d storage s3://my-dvc-bucket/dvc-storage
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}

      - name: Pull data
        run: dvc pull

      - name: Reproduce pipeline
        run: dvc repro

      - name: Show metrics
        run: dvc metrics show --all

      - name: Compare with previous commit
        run: dvc metrics diff HEAD~1

      - name: Push updated artifacts
        run: dvc push
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
```

For a local CI equivalent (e.g., a pre-push hook or a self-hosted runner), the same sequence works as a shell script. The observable output is the `dvc metrics diff` block, which prints a table showing whether each metric increased or decreased relative to the previous commit. A reviewer reading the CI log sees the metric delta before approving the merge.

One caveat worth stating plainly: DVC remotes require credentials in CI. For S3, this means `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` as repository secrets. For SSH remotes, it means deploying a key pair. The `dvc remote add` command stores the remote URL in `.dvc/config`, which is committed to Git — so never put credentials in the remote URL itself. Use environment variables or IAM roles.