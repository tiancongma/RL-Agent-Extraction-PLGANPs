# Shared Carrythrough v1 Notes

Goal: replace paper-local shared method/solvent/material carry-through used by 5GIF3D8W and YGA8VQKU with a generalized mechanism.

Observed remaining paper-local carry-through shapes:
- 5GIF3D8W:
  - row label token selects family (PCL vs PLGA)
  - family determines method_type, solvent_name, stabilizer_name
- YGA8VQKU:
  - direct surface alias/canonicalization (`cP188` -> `Poloxamer 188`)
  - direct solvent carry-through from `organic_solvent_value_text`
  - guarded blank/suppression for method_type and LA/GA
  - normalized polymer_grade fallback to `PLGA`

Generalizable buckets:
1. label-conditioned family carry-through
2. direct shared surface carry-through
3. safe alias normalization on shared material name fields
4. safe blank/suppression for fields explicitly governed as over-strong for current GT

Guardrail:
- no new semantics; only propagate existing shared value surfaces or bounded family token classes.
