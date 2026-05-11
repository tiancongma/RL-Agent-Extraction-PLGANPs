# Integrated Capability Preservation Check

## UFXX9WXE
- old baseline: anchor final count `1`
- expected repaired behavior: preserve DOE authority table and trigger DOE expansion
- new baseline: rebuilt `S2-2` evidence includes `Table 1` and `Table 2`; no fresh final baseline was produced
- status: partial

## 5GIF3D8W
- old baseline: anchor final count `3`
- expected repaired behavior: preserve table plus sweep evidence and avoid collapse
- new baseline: rebuilt `S2-2` evidence includes the optimized formulation table excerpt; fresh live execution blocked before downstream validation
- status: partial

## QLYKLPKT
- old baseline: anchor final count `3`
- expected repaired behavior: preserve both tables, keep sequential structure, move toward `7`
- new baseline: rebuilt `S2-2` evidence includes multiple table excerpts instead of the anchor collapse; fresh downstream execution did not complete
- status: partial

## WFDTQ4VX
- old baseline: anchor final count `27`
- expected repaired behavior: no regression
- new baseline: rebuilt `S2-2` evidence retains multiple table excerpts and no pre-LLM drop was observed; fresh downstream execution did not complete
- status: partial
