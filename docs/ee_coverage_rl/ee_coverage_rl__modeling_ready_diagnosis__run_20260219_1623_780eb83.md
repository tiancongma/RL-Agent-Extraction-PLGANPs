# Modeling-Ready Diagnosis (run_20260219_1623_780eb83)

## 1) EE-supported intersection counts
- N_EE_supported: **28**
- N_EE_and_loading: **2**
- N_EE_and_polymer: **0**
- N_EE_and_loading_and_polymer: **0**

## 2) Field-level support in ee_local_support == 1 subset
| field                   |   supported |   unsupported |
|:------------------------|------------:|--------------:|
| loading_content_percent |           0 |            28 |
| drug_feed_amount_text   |           5 |            23 |
| plga_mass_mg            |           8 |            20 |
| la_ga_ratio             |           0 |            28 |
| plga_mw_kDa             |           0 |            28 |

## 3) Bottleneck diagnosis
- loading_proxy_supported failures in EE-supported subset: **26**
- polymer_identity_supported failures in EE-supported subset: **28**

Loading sub-condition failures:
| sub_condition                                    |   count |
|:-------------------------------------------------|--------:|
| a_loading_content_percent_missing_or_unsupported |      28 |
| b_drug_feed_amount_text_unsupported              |      23 |
| c_plga_mass_mg_unsupported                       |      20 |

Polymer identity missing/unsupported:
| sub_condition                      |   count |
|:-----------------------------------|--------:|
| la_ga_ratio_missing_or_unsupported |      28 |
| plga_mw_kDa_missing_or_unsupported |      28 |

- Primary bottleneck: **polymer_identity_supported** (worst sub-condition: `la_ga_ratio_missing_or_unsupported` = 28 failures)
