# EE Coverage RL Metrics Summary (run_20260219_1623_780eb83_goren18_weaklabels_v1)

## 1) TSV Schema
- row_count: 217
- columns (27):
  - `key`
  - `formulation_id`
  - `group_key`
  - `condition_instance_key`
  - `condition_instance_source`
  - `has_explicit_formulation_id`
  - `n_core_fields_supported`
  - `n_fields_local_evidence`
  - `n_fields_inherited_base`
  - `ee_local_support`
  - `fingerprint_field_count_present`
  - `fingerprint_field_count_total`
  - `fingerprint_completeness`
  - `has_explicit_constant_parameter_claim`
  - `inheritance_without_constant_claim`
  - `evidence_source_type_by_field_json`
  - `inherited_base_donor_by_field_json`
  - `shared_span_risk_fields_count`
  - `confidence_tier`
  - `evidence_source_type__encapsulation_efficiency_percent`
  - `evidence_source_type__loading_content_percent`
  - `evidence_source_type__drug_feed_amount_text`
  - `evidence_source_type__plga_mass_mg`
  - `evidence_source_type__la_ga_ratio`
  - `evidence_source_type__plga_mw_kDa`
  - `evidence_source_type__size_nm`
  - `evidence_source_type__pva_conc_percent`

First 3 rows (selected key columns):
| key      | group_key    | condition_instance_key       | condition_instance_source   | confidence_tier   |   ee_local_support |   fingerprint_completeness |   n_fields_local_evidence |   n_fields_inherited_base |   has_explicit_constant_parameter_claim |   inheritance_without_constant_claim |
|:---------|:-------------|:-----------------------------|:----------------------------|:------------------|-------------------:|---------------------------:|--------------------------:|--------------------------:|----------------------------------------:|-------------------------------------:|
| 5GIF3D8W | 5GIF3D8W::1  | 5GIF3D8W::cond::0e5601e5d97b | table_row_conditions        | C                 |                  0 |                   0.888889 |                         0 |                         0 |                                       0 |                                    0 |
| 5GIF3D8W | 5GIF3D8W::10 | 5GIF3D8W::cond::2546aeefb39c | table_row_conditions        | C                 |                  0 |                   0.555556 |                         0 |                         0 |                                       0 |                                    0 |
| 5GIF3D8W | 5GIF3D8W::2  | 5GIF3D8W::cond::7ae8a232cc59 | table_row_conditions        | C                 |                  0 |                   0.888889 |                         0 |                         0 |                                       0 |                                    0 |

## 2) Distributions
### confidence_tier counts
| category   |   count |
|:-----------|--------:|
| A          |       1 |
| B          |       1 |
| C          |     215 |

### condition_instance_source counts
| category                |   count |
|:------------------------|--------:|
| table_row_conditions    |     118 |
| explicit_formulation_id |      99 |

### ee_local_support counts
|   category |   count |
|-----------:|--------:|
|          0 |     210 |
|          1 |       7 |

### ee_local_support rate by confidence_tier
| category   |   rate |
|:-----------|-------:|
| A          | 1      |
| B          | 1      |
| C          | 0.0233 |

### fingerprint_completeness summary (overall)
|      min |   median |   max |
|---------:|---------:|------:|
| 0.111111 | 0.666667 |     1 |

### fingerprint_completeness summary by condition_instance_source
| condition_instance_source   |      min |   median |      max |
|:----------------------------|---------:|---------:|---------:|
| explicit_formulation_id     | 0.111111 | 0.555556 | 0.888889 |
| table_row_conditions        | 0.222222 | 0.777778 | 1        |

### inheritance_without_constant_claim rate
- overall_rate: 0.1475
- by confidence_tier:
| category   |   rate |
|:-----------|-------:|
| A          | 0      |
| B          | 0      |
| C          | 0.1488 |

## 3) Modeling-Intake Counts
Derived flags:
- `has_EE_supported = (ee_local_support == 1)`
- `has_loading_proxy_supported` from field-level gate checks: `loading_content_percent` supported OR (`drug_feed_amount_text` and `plga_mass_mg` both supported)
- `has_polymer_identity_supported` from field-level gate checks: `la_ga_ratio` OR `plga_mw_kDa` supported

- N_total_modeling_ready: **0**
- Breakdown by confidence_tier:
| category   |   N_modeling_ready |
|:-----------|-------------------:|
| A          |                  0 |
| B          |                  0 |
| C          |                  0 |
