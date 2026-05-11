# Dataset Splits (DEV v1)

This DEV split is used for iterative pipeline improvement only.
Selection rule: include all keys with html_found=True from tables coverage; then add
pdf_found=True and html_found=False keys until N is reached.
Policy: all future TEST sets must exclude keys listed in dev_keys_v1.tsv to prevent leakage.
