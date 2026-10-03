# Final delivery audit — 4 October 2026

## Authoritative outputs

- `Resilient_Task_Offloading_IIoT_FINAL.pdf`: 8-page IEEE two-column paper.
- `Resilient_Task_Offloading_IIoT_FINAL.tex`: standalone source with tables, vector figures, and verified bibliography inlined.
- `REVIEWER_RESPONSE.md`: all nineteen requirements, corrections, and scientific boundaries.
- `Reproduce_Paper_Tables.ipynb`: reproduces final tables and figures without retraining.

## Evidence and reruns

- Frozen main branch: 300 newly executed records (five independent workload seeds × ten episodes × three stresses × two policies). Checkpoints are fixed, not independently retrained QECO seeds.
- Corrected EH ablation: 300 newly executed records (five training seeds × ten episodes × three stresses × enabled/disabled). Static energy-preference observations, battery initialization, workload, and trace window are paired.
- Corrected combined EH/selector experiment: 150 newly executed records on the same conditions.
- Matched-task audit: 20 paired episodes per condition and policy; original executed-action replay reproduces saved delays/transmitted volumes. Actual original outcomes are not inferred from selector proposals.
- Comparator baselines: existing completed 1,000-episode summaries retained and reported; 90 fresh verification episodes (three rules × three stresses × ten episodes) publish raw per-episode records. A second complete 1,000-episode suite was started but stopped in favor of this deterministic subset check; no partial run is passed off as a complete rerun.
- Battery sensitivity: fresh fixed-demand ledger replays at 1,000/2,000/3,000 mAh, harvesting enabled/disabled. Reports first-device zero-charge time and mean-SOC 0.01% threshold time separately.
- UCLM timestamps and model hashes verified from files; actual training/evaluation coverage disclosed separately from partition allocation.
- Seventeen cited scholarly references verified against DOI registration metadata; three additional citations identify software/data. All twenty references are cited. The inherited full registry retains explicit unresolved statuses.

## Main scientific corrections

EH energy differences are approximately −0.528, −1.072, and −1.432 J per 20-UE episode, with exact paired seed-level p = 0.25. Terminal SOC gains after 110 s are 0.1001, 0.1011, and 0.1017 percentage points, with exact p = 0.0625. Parametric 95% intervals and seed SDs accompany these estimates. These are small observed effects, not conventional exact-test significance at 0.05.

The original added counterfactual calculations were invalid because they used proposed original actions on selector trajectories. Earlier 28.25%/−8.54%/−26.54% figures also do not support equal-service efficiency claims when full local work is compared with incomplete offload expenditure. The final paper withdraws those interpretations and reports corrected requirements, actual expenditure, feasibility, and both-feasible completion times.

The inherited EH environment's battery clipping and local fallback do not establish strict execution-level energy causality. Short evaluated rollouts stay away from depletion; post-depletion task-service claims are excluded. The new battery capacity replay is a deterministic demand/charge analysis, not a hardware lifetime or successful service after empty-battery claim.

## Verification

An isolated export containing only committed files executed the notebook successfully and reproduced every manuscript table and the standalone source exactly. Untracked working files were not required for this check.

`check_evidence.py` passes completeness, pairing metadata, charge conservation, seed-unit, model/checkpoint integrity, trace-allocation, matched deadline, citation-coverage, and standalone-source checks. The notebook's actual code cells were executed successfully. The final document was compiled using the existing Tectonic executable and rendered with Poppler for visual inspection. There are no overfull boxes, undefined citations/references, or question-mark reference placeholders. Harmless underfull spacing/font-package warnings are not represented as citation errors.

The native Codex LaTeX editor was opened for the saved source; its compiler returned `Unable to find standard directories for platform`. The verified exported PDF was therefore produced with the existing research-folder compiler. No LaTeX plugin was installed.

## Scope still requiring further research

The experiments are simulations, the effective-capacity mapping/feature weights are author-defined and uncalibrated, and EH evaluation samples a brief daytime window. Original-policy training-seed generalization, weight sensitivity, independently selected weather/day windows, separately retrained non-EH policies, causal post-depletion service, and hardware validation are not claimed complete. No target venue or page limit was specified. These limits are disclosed in the paper rather than filled with invented results.

## Public version

The publication package is versioned on `codex/audited-paper-20261004` in `Sanya06C/Resilient_Task_Offloading`, with a draft pull request for review. It can be reproduced from that public branch; it is not described as merged into the default branch. The final delivery message links the verified public revision.

Public upload verified through the GitHub contents API: the final PDF and reproduction notebook blob hashes exactly match the local committed files. Draft review: https://github.com/Sanya06C/Resilient_Task_Offloading/pull/1.
