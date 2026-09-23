# W1 constructive baseline

Run `20260923T084933Z_eb2603cd79f7`, source `2965502`.

All18 records verified from registered hashes. Ordered cell additions independently replayed; legality, face adjacency, unchanged guide/envelope/allowance, all nine terms and binary metrics independently recomputed.17 feasible scenes have zero-loss connected witnesses; the sealed reference remains incompatible.

| Scene | Status | Original guide cells | Added cells | Final cells | Volume (m3) | Witness |
|---|---|---:|---:|---:|---:|---|
| legacy-easy-seed-000 | constructed | 22 | 1 | 23 | 11.776 | True |
| legacy-easy-seed-001 | constructed | 23 | 0 | 23 | 11.776 | True |
| legacy-easy-seed-002 | constructed | 38 | 0 | 38 | 19.456 | True |
| legacy-easy-seed-003 | constructed | 34 | 11 | 45 | 23.040 | True |
| legacy-easy-seed-004 | constructed | 32 | 0 | 32 | 16.384 | True |
| legacy-easy-seed-005 | constructed | 36 | 0 | 36 | 18.432 | True |
| legacy-easy-seed-006 | constructed | 33 | 24 | 57 | 29.184 | True |
| legacy-easy-seed-007 | constructed | 22 | 18 | 40 | 20.480 | True |
| legacy-easy-seed-008 | constructed | 17 | 17 | 34 | 17.408 | True |
| legacy-easy-seed-009 | constructed | 37 | 6 | 43 | 22.016 | True |
| legacy-easy-seed-010 | constructed | 36 | 15 | 51 | 26.112 | True |
| legacy-easy-seed-011 | constructed | 31 | 7 | 38 | 19.456 | True |
| ref-01-ground-pair | constructed | 36 | 0 | 36 | 18.432 | True |
| ref-02-facade-pair-and-ground | constructed | 55 | 6 | 61 | 31.232 | True |
| ref-03-wide-gap | constructed | 51 | 11 | 62 | 31.744 | True |
| ref-04-asymmetric-heights | constructed | 69 | 13 | 82 | 41.984 | True |
| ref-05-sealed-partition | incompatible | 16 | 0 | 16 | 8.192 | False |
| ref-06-minimal-smoke | constructed | 32 | 0 | 32 | 16.384 | True |

The fixed radius6/envelope budget and facade_endpoint_v1 accounting are an experimental contract, not production defaults. The construction starts from a full procedural guide, adds only uncharged legal neighboring cells to meet mass/contact bounds, and refuses new radius2 eroded cores. Lexicographic order is deliberately simple and directionally biased.

This proves numerical consistency with a constructive example for each feasible scene, not architectural quality. Some added material serves only to meet a mass floor or dilute the facade ratio. Simple strands remain zero-loss outcomes. NCA training must be compared against this baseline and demonstrate value beyond matching these losses.

No model training, paid compute, cloud operation or deployment. All18 fields and ordered additions remain in the run archive.
