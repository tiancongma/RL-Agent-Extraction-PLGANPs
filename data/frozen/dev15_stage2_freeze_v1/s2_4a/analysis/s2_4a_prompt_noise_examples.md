# S2-4a Prompt Noise Examples

These examples are representative spans from the frozen prompt text, labeled by the surface-heuristic audit used for the metrics file.

## L3H2RS2H

- success_or_failure: success
- noise_ratio: 0.2731
- largest_noise_block_size: 1428
- reference_section_detected: yes

Representative noise spans:
- [FORMULATION_RESULT_BLOCK | paragraph] Alonsoc, Madalena M.M. Pintoa, Carlos M. Barbosad,e,* aCentro de Estudos de Quı´mica Orgaˆnica, Fitoquı´mica e Farmacologia da Universidade do Porto-Faculdade de Farma´cia do Porto, Porto, Portugal bInstituto Superior de Cieˆncias da Sau´de-Norte, Gandra PR...
- [OPTIMIZATION_RESULT_BLOCK | paragraph] Optimization result: best/highest/efficiency/loading outcome. 1.5G7.6%) corresponding to a theoretical concentration of 60 mg/mL. For concentrations above 70 mg/mL both xanthones precipitated in the form of crystals, indicating that the maximum loading capa...

Representative valid-signal spans:
- [SYNTHESIS_METHOD_BLOCK | paragraph] Preparation method: Then, acetone was removed under vacuum and the colloidal dispersion of nanocapsules was concentrated to 5–10 mL by evaporation under reduced pressure. The amount of non-encapsulated xanthones (either XAN or 3-MeOXAN) was separated by ult...
- [MATERIALS_PROCUREMENT_BLOCK | paragraph] Formulation table: drug/polymer ratio, loading, EE, DL. As can be observed in both Table 5 Mean diameter, polydispersity index (PI), zeta potential (z) and incorporation parameters of various nanocapsule formulations: empty nanocapsules (0.6 mL Myritol 318 ...

Typical failure pattern note: noisy tail content is present near the end of the prompt.

## 5GIF3D8W

- success_or_failure: success
- noise_ratio: 0.0000
- largest_noise_block_size: 0
- reference_section_detected: no

Representative noise spans:

Representative valid-signal spans:
- [OPTIMIZATION_RESULT_BLOCK | paragraph] Optimization result: best/highest/efficiency/loading outcome. 1997). Ab- sence of drug peak can be attributed to complete encapsulation of etoposide or there might be very less adsorbed drug on the surface of nanoparticles. Effect of Formulation Variables S...
- [SYNTHESIS_METHOD_BLOCK | paragraph] Preparation method: The optimized formulation was prepared using nanoprecipitation method as follows: polymer (50 mg) and etoposide (5 mg) were dissolved in acetone. Dichloromethane was used to dissolve PCL and etoposide in case of emulsion solvent evaporat...

Typical failure pattern note: noise is dominated by parsing or encoding artifacts rather than a reference tail.

## WFDTQ4VX

- success_or_failure: failure
- noise_ratio: 0.0453
- largest_noise_block_size: 763
- reference_section_detected: yes

Representative noise spans:
- [PARAGRAPH_BLOCK | paragraph] Preparation method: 6.1199605 Bioavailability enhancement, Caco-2 cells uptake and intestinal transport of orally administered lopinavir-loaded PLGA nanoparticles Garima Joshi, Abhinesh Kumar, and Krutika Sawant Pharmacy Department, TIFAC Centre of Relevanc...

Representative valid-signal spans:
- [OPTIMIZATION_RESULT_BLOCK | paragraph] Optimization result: best/highest/efficiency/loading outcome. In all, Contour and response plots could explain the relationship of all the factors, in all possible combinations on both the responses and help us in identifying the factor combinations for ach...
- [SYNTHESIS_METHOD_BLOCK | paragraph] Materials: All other chemicals used were of analytical grade. The 24-well Transwell inserts were purchased from Nunc, Roskilde, Denmark. The 6-, 24- and 96-well plates were purchased from Costar Corning, NY, USA. MTT assay dye was purchased from Himedia, Mu...

Typical failure pattern note: noisy tail content is present near the end of the prompt.

