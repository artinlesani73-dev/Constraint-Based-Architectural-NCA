# MG3 proposal: preserve the contact budget during cube growth

Proposed after MG2 on2026-09-24; not implemented or executed. Keep the same nine
MT1 families, domains, cube size, seeds, requests and contact cost12. Preserve MG1,
MD1 and MG2 outputs. The change is a feasibility check within the existing facade
family, not a new category or relaxed acceptance threshold.

1. Create a separately versioned generator. Retain MG2 routing and frozen cost
   ordering; assemble its complete initial route. If the completed route already
   exceeds the global15% contact budget, record an explicit unsupported start.
   Do not reject individual route prefixes solely for their temporary ratio.
2. For growth, count the unique new occupied cells and unique new contact cells
   of each proposed cube. Admit it only if the resulting whole-mass contact ratio
   remains within MT1's limit. Never tighten the physical domain or silently clip.
3. Keep temporarily inadmissible frontier cubes available for reconsideration
   after accepted growth changes the denominator. Expand frontier only through
   accepted connected origins. Bound reconsideration by actual accepted progress;
   stop explicitly if no candidate is admissible. Preserve deferred/rejected
   choices and request shortfall. Empty/no-new-cell proposals must not create an
   infinite retry loop or falsely count as geometric progress.
4. Test exact contact accounting with overlapping cubes and finite stalled growth.
   Show whether feasible alternatives reach the requested count; do not equate
   contact validity by construction with all-nine validity or optimality.
5. Freeze one full45-case comparison before execution, pairing MG1/MG2/MG3. Use
   the same36/36 nonblocked admission rule, preserved baseline successes, request
   fidelity and caps. Retain all9 blocked outcomes. Compare valid-only diversity.

This repair hypothesis addresses the observed gap between additive cube costs
and the global final ratio. Greedy feasibility can stall even when another path
could work. Do not automatically retune cost, add steps or soften thresholds on
failure. If admitted, implement a separately labeled experimental Studio mass
workflow with durable jobs, failure states, parameter-bound records and replay.
Legacy material records and their interpretation remain unchanged.

No paid Colab, new NCA architecture or Drive access is needed for this proposal.
