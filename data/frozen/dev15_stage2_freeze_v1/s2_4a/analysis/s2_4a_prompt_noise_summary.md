# S2-4a Prompt Noise Summary

## Overall averages

- Papers audited: 15
- Mean total_characters: 16553.9
- Mean valid_signal_ratio: 0.3181
- Mean weak_signal_ratio: 0.6584
- Mean noise_ratio: 0.0234
- Papers with reference_section_detected=yes: 3 of 15
- Mean duplicate_block_count: 0.07

## Success vs failure comparison

- Success papers: 6
- Failure papers: 9
- Mean noise_ratio, success: 0.0510
- Mean noise_ratio, failure: 0.0050
- Median noise_ratio, success: 0.0000
- Median noise_ratio, failure: 0.0000
- Noise-ratio range, success: 0.0000 to 0.2731
- Noise-ratio range, failure: 0.0000 to 0.0453
- Distribution overlap fraction on noise_ratio ranges: 0.1660
- Top-noise quartile size: 4
- Failures in top-noise quartile: 1 of 4
- Overall failure rate: 9/15 = 0.6000
- Failure rate in top-noise quartile: 1/4 = 0.2500
- Failure types:
  - DeadlineExceeded: 8
  - ValueError: 1

## Top 3 highest-noise papers

- L3H2RS2H: noise_ratio=0.2731, success_or_failure=success, total_characters=16546
- WFDTQ4VX: noise_ratio=0.0453, success_or_failure=failure, total_characters=16829
- RHMJWZX8: noise_ratio=0.0327, success_or_failure=success, total_characters=15561

## Top 3 cleanest papers

- 5GIF3D8W: noise_ratio=0.0000, success_or_failure=success, total_characters=17934
- 5ZXYABSU: noise_ratio=0.0000, success_or_failure=failure, total_characters=18628
- 7ZS858NS: noise_ratio=0.0000, success_or_failure=success, total_characters=14176

## Correlation interpretation

- Noise ratio does not show failure overrepresentation in this frozen set.
- Mean noise-ratio difference (failure - success): -0.0459
- This is correlation only; the frozen audit does not identify causality.
