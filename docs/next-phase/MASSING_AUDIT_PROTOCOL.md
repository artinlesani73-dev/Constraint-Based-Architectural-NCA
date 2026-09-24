# MA1 protocol: original and completed building mass

2026-09-24. Implements D058's next comparison, not a learned-model experiment.
Frozen recipe: experiments/configs/MA1-massing.json. Parent evidence: VA1 run
20260924T085504Z_36c834017633. Keep its bytes and material interpretation unchanged.

Use nine saved VA1 fields in the same scene plus two gap counterexamples:
two detached plates and two separated blocks. Run identity, source-only sealed
cavity filling, and vertical axis-span completion on all eleven (33 records).
The latter fills each empty gap of at most 6.4 m between consecutive occupied
samples on a vertical line, in a single pass. No horizontal or iterative closure.
It can erase meaningful gaps; this is a control, not the selected generation rule.

Version massing_v1: new occupied cells mean building volume, interiors deferred.
Use cubic metres, not material quantity or floor area. Domain is the physical
bounding box of declared interface blocks, padded (Z,Y,X)=(6.4,6.4,0) m, sampled
at voxel centers, clipped to grid and intersected with historical legality and
non-context space. It is scene-derived and candidate-independent. On this scene
it contains 3492 cells, including 36 historical anchor-allowed cells below the
street band. This diagnostic denominator is not a new legal ceiling or
budget target; do not transplant the old 3-12% threshold onto it. The padding is
an explicit study parameter, not a claim of universal suitability. At finer
resolution resample the same physical scene and legality; arbitrary boundaries
will incur center-sampling discretization. An aligned doubled-resolution test
checks exact physical-volume preservation, not larger-site generation.

Enclosure uses all six full-grid faces and six-neighbor empty connectivity.
Existing buildings and domain boundaries cannot provide cavity enclosure. Filter
proposed additions against context and the domain, preserving requested/rejected
masks and any original violations. Never delete source cells or conceal clipping.
Record every original/result/addition/rejection field, operation, seed, config,
source snapshot and parent study hash. Originals must reproduce every old score.
Check closed-shell addition686 and final volume1296; open tube cavity-fill adds0;
vertical completion gives1296 for tube/aperture/plates/separated blocks. The last
equivalence exposes loss of design intent. Slabs/paths may remain thin: filling
cannot invent missing depth without an additional explicit rule.

Evaluate unchanged nine-family scores on every result and retain old denominator.
Report a separate new-domain occupancy fraction with no pass/fail budget. No new
loss, tenth family, trained NCA, generalization, structural safety or habitability
claim. Use complete foundation regression plus exact saved-record replay. Verify
actual viewer controls and mobile layout. Document failures and resume state;
archive locally. No Drive operation, paid Colab run, push or publication.

## Recorded correction after first attempt

Attempt20260924T094943Z_1d6ed517bc60 failed only its expected domain-count check:
the initial recipe expected3456, overlooking36 historical anchor allowances.
All other82 checks, including every original legacy-score comparison, passed.
The domain implementation correctly used actual historical legality. Correct the
expected count to3492 and link a fresh attempt; preserve the original recipe and
all33 records in the failed run. No algorithm, padding, field or budget changed.
The synthetic refinement unit test intentionally supplies a no-anchor legality
mask, so its3456-cell expectation remains correct for that separate fixture.
