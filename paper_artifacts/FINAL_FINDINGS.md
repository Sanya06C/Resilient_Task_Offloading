# Final findings for the Results section

- Across 1,000 matched episodes per condition, Stress-Aware QECO reduced mean UE energy by 8.46% at Normal stress, 7.02% at Medium stress, and 7.65% at High stress relative to Original QECO.
- Total-system energy decreased by 6.77%, 6.64%, and 8.30% at Normal, Medium, and High stress, respectively; transmission energy decreased by 6.47%, 29.74%, and 37.53%.
- At Medium stress, the selector restored positive matched offloading energy savings: the same-task UE comparison improved from -8.54% for Original QECO to +4.91% for Stress-Aware QECO, equal to 12,759.04 J across the evaluated episodes.
- At High stress, the selector reduced overall UE energy and lowered the deadline violation rate from 36.90% to 24.99%, but matched per-task offloading savings remained negative at -21.75%.
- Positive High-stress per-task offloading savings were physically infeasible under the locked assumptions: the effective capacity of 6.3269 was below the transmission-only break-even capacities of 15.1777, 10.0673, and 7.5315 for the three task densities.
- The selector reduced the Medium-stress deadline violation rate from 23.76% to 12.97% while reducing time-to-terminal from 0.7247 s to 0.6975 s; at High stress, time-to-terminal increased slightly from 0.7695 s to 0.7775 s despite the large reduction in violations.
- Normal-stress offloading changed only from 63.96% to 63.54% because 116,415 overrides switched between edge servers and the 117,697 Local-to-Edge transitions substantially offset 133,666 Edge-to-Local transitions.
