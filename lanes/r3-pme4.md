DEAD for the STMV and cellulose order-four cost lever. LJPME is an analysis candidate, pending the liaison's resolution of the brief.

Base: r3/pme4 at c21cad1e7. No platform source, default behavior, FFT kernel, or direct-space code changed. No GPU job, platform build, or whole-benchmark timing ran. The analysis uses the existing r2/pmefft Reference library through a small C ABI bridge.

The requested large-system kill fires under the consult's post-spread cost model: the cheapest screened order-four meshes that match all three force metrics lose 213.8 us on cellulose and 454.0 us on STMV. A separate joint Coulomb/dispersion search finds a 103.3 us predicted LJPME saving. This is not a measured speedup or a validated production policy.

The brief contains conflicting next steps. Outcome 2 says "If a better pair exists for any test, implement it". Acceptance 2 says to report DEAD and "don't build part 2" when cellulose and STMV lose their stencil saving. I sent this conflict through cdx ask, together with the LJPME exception. No answer had arrived when this report was prepared. I followed the explicit do-not-build condition. Metal property work, Metal gates, determinism, and NVE drift remain unperformed. The LJPME candidate is not ready for integration.

Numerical results and policy

Every force comparison uses reciprocal mesh forces against order 8 on a mesh twice as fine as the current mesh. RMS means ||F-Fref||/||Fref||. Max-atom means max_i ||dF_i||/||Fref_i||. The separate global-normalized maximum is also required to improve. Every criterion must pass in every configuration, rather than only in the table's worst-case aggregate. Energy columns are maximum absolute reciprocal energy errors in kJ/mol. Direct, self, and exclusion terms cancel between meshes.

Five Coulomb-only cases use three native benchmark configurations: input, 100 CPU MD steps, and 200 CPU MD steps. LJPME adds nine rigid translations to test mesh phase, for 12 configurations. Table entries are worst errors over the configurations for that row.

| Test | Current order-5 grid | Current RMS / max-atom | Numerical choice | Choice RMS / max-atom | Current / choice abs energy error | Predicted us saved |
|---|---|---:|---|---:|---:|---:|
| pme | 56^3 | 0.000879541 / 0.0248785 | retain order 5, 56^3 | 0.000879541 / 0.0248785 | 0.724525 / 0.724525 | 0.0 |
| amber20-dhfr | 56^3 | 0.000848537 / 0.0510721 | retain order 5, 56^3 | 0.000848537 / 0.0510721 | 0.582856 / 0.582856 | 0.0 |
| apoa1pme | 98^3 | 0.000807431 / 0.0616787 | retain order 5, 98^3 | 0.000807431 / 0.0616787 | 2.00381 / 2.00381 | 0.0 |
| apoa1ljpme | 98^3 + 49^3 | 0.00509547 / 7.56726 | order 4, 99^3 + 40^3 | 0.00464455 / 3.05936 | 176.654 / 100.609 | 103.3 |
| amber20-cellulose | 231x112x110 | 0.000943345 / 0.119359 | retain order 5, 231x112x110 | 0.000943345 / 0.119359 | 15.9062 / 15.9062 | 0.0 |
| amber20-stmv | 198x200x200 | 0.00088406 / 0.117796 | retain order 5, 198x200x200 | 0.00088406 / 0.117796 | 32.8336 / 32.8336 | 0.0 |

The DHFR comparison uses this branch's 56^3 grid. The upstream requested grid is 55^3; r2/pmefft already rounded it to 56^3 and reduced error. The analysis does not take credit for that prior change.

The cheapest screened order-four Coulomb-only alternatives are:

| Test | Order-four grid | RMS error | Global-normalized max error | Max-atom error | Abs energy error | Predicted us saved |
|---|---|---:|---:|---:|---:|---:|
| pme | 88^3 | 0.000623551 | 0.000584555 | 0.0233294 | 0.539878 | -116.2 |
| amber20-dhfr | 90^3 | 0.000564676 | 0.00055284 | 0.0261584 | 0.4445 | -118.0 |
| apoa1pme | 160^3 | 0.0004969 | 0.000423866 | 0.03411 | 1.40956 | -177.5 |
| amber20-cellulose | 352x175x168 | 0.000702449 | 0.000722538 | 0.0515158 | 14.7094 | -213.8 |
| amber20-stmv | 312^3 | 0.000633027 | 0.000635086 | 0.101833 | 26.588 | -454.0 |

The kill does not rely on the stricter per-atom maximum. Using only RMS and the global-normalized maximum gives cellulose 343x165x162 and a 142.3 us predicted loss. STMV still needs 312^3 and loses 454.0 us.

Cost derivation

Let S be post-r2 spread time, I interpolation, and F the full FFT/convolution block. Let V be candidate grid volume and V0 baseline volume. The model is:

    saving = [1-(order/5)^3]*(S+I) - F*[V log(V)/(V0 log(V0))-1]

For pme and DHFR, a mesh above 63 per dimension loses r2's fused FFT. The candidate then uses the old VkFFT census block, while the baseline uses its estimated fused cost. The model deliberately gives order four the full 48.8% stencil-work saving. Real spread costs include fixed work and changes in atom sharing, so the saving is a ceiling rather than a performance promise.

| Test | S, us | I, us | F, us |
|---|---:|---:|---:|
| pme | 36.00 | 14.06 | 20.10 |
| amber20-dhfr | 37.50 | 14.58 | 19.00 |
| apoa1pme | 100.00 | 37.53 | 64.09 |
| apoa1ljpme | 136.00 | 62.60 | 81.56 |
| amber20-cellulose | 330.00 | 138.96 | 149.94 |
| amber20-stmv | 1400.00 | 396.14 | 421.15 |

Evidence: lanes/results/m0/census/*.kern supplies the kernel times. lanes/r2-spread.md measures apoa1pme post-spread S=100 us and the small-grid FFT microbenchmark ratio near 0.54. research/consults/astra-pme.md assumes S=330 us for cellulose and S=1400 us for STMV; those remain assumptions. The pme, DHFR, and LJPME spread estimates subtract r2's reported whole-step saving from the earlier census, rounded to 36, 37.5, and 136 us. These are proxies, not new kernel measurements. No speedup is claimed from a mixed-precision cost model.

STMV census sampling is corrected from 5.75 FFT dispatches to six, 0.96 convolution calls to one, and 0.86 interpolation calls to one: F=369.37*6/5.75+34.29/0.96, I=340.68/0.86. Cellulose F=135.35*6/5.97+13.91. LJPME splits the old 96.45 us block into Coulomb 64.09 and dispersion 32.36; r2 fusion estimates dispersion at 0.54*32.36. Its stencil saving is 0.488*(136+62.60)=96.9168 us. The selected 99^3/40^3 meshes add 6.3358 us from the modeled FFT reduction, giving 103.2526 us. The fused eligibility calculation includes all 7-smooth dimensions through 63, not just the initially sampled 49..63 sizes.

Sensitivity: cellulose would need post-spread S above 768.1 us to break even at its strict qualifying grid. STMV would need S above 2330.3 us. Even substituting the corrected pre-spread STMV census S=970.86/0.47=2065.7 predicts a 129.2 us loss. Cellulose's pre-spread S=630.26/0.73=863.4 would instead predict a 46.5 us gain, so its kill is conditional on the consult's post-spread estimate. I asked the liaison for measured post-spread large-system kernel times; no reply arrived. Census scaling is a sensitivity estimate, not a statistical upper confidence bound.

Within this stencil model, every order of six or higher loses even if its FFT becomes free: 0.728*(S+I) exceeds F for all six cases. The order-three Coulomb screen fails accuracy on the sampled larger meshes. Coarser order-five meshes in the 0.9..1.0 uniform scaling interval fail the per-configuration accuracy rule. These are screened families, not a proof of an optimum over arbitrary independent anisotropic dimensions, all orders, or all future conformations.

LJPME finding

At the default grids the dispersion mesh dominates combined reciprocal error. It is therefore wrong to require separate Coulomb and dispersion error budgets when choosing the total-force policy. The Reference Context cross-check gives exactly zero force difference from the bridge at order five, including the dispersion normalization. After adding self and neutralizing-background constants, energy differs by 8.76e-6 kJ/mol from a different reduction of the large self sum.

The joint search considered 2923 positive-cost candidates with orders 4 and 5, legal cubic Coulomb dimensions 52..176, and dispersion dimensions 26..91. It rejected 724 more-profitable candidates before accepting the common-order policy. Independent orders give the same selected pair. The initial 96^3/49^3 choice passed three MD configurations but failed an added mesh phase by 4.76% RMS. The final 99^3/40^3 pair passes all 12 configurations; its worst per-configuration error ratios to order five are 0.993615 RMS, 0.550146 global maximum, 0.561066 per-atom maximum.

A simpler alternative is order four at the unchanged 98^3/49^3 grids. It passes the nine added mesh phases with worst RMS ratio 0.975978 and predicts 96.9168 us saved. It also has lower energy and per-atom errors than the predicted minimum-cost 99^3/40^3 pair. The extra 6.3 us forecast for the latter is smaller than plausible FFT radix-model error: 99 introduces radix 11, and the model has no measured radix penalty. The head should compare both if it authorizes an LJPME implementation. Neither is a measured speed claim.

Validation, remaining work, and resource use

The C++17 bridge compiled successfully. The required explicit clang++ type/syntax check passed, with one existing unused-function warning in OpenMM Vec3.h. The direct reciprocal Ewald check used 32 neutral random charges and wave vectors -14..14: order eight at 96^3 reached 2.64e-10 relative RMS force error. Order-four, order-five, and order-eight forces also matched energy finite differences. Reference convergence was checked at order seven versus order eight on the fine mesh; the largest dispersion RMS difference was 1.76e-5, far below its 0.15 baseline mesh error. Full convergence rows are in the logs.

Three validation invocations were used. The first direct-Ewald validation passed. The first full-Context energy cross-check omitted the neutralizing-background constant and failed; adding that constant fixed it, and the third invocation passed. The force comparison was exact before and after that accounting fix. Two setup attempts stopped before generating data, first because benchmark.py parses arguments when imported, then because box vectors are a list of unit quantities. The final loader extracts only imports and function definitions and converts each box vector explicitly.

No NVE result is claimed. The ROUND2 protocol requires 2 fs leapfrog Verlet, HBonds and rigid water, constraint tolerance 1e-8, and ewaldErrorTolerance 1e-5. The current sweep uses 5e-4 as requested, and CPU Langevin samples are not a drift test. An implementation decision must be followed by that separate drift comparison, property-on/off quick gates, high-accuracy Reference comparisons, and bitwise DeterministicForces checks. The lane used its three validation invocations; the liaison must allocate any further validation round.

There are 795 recorded single-mesh candidate evaluations, plus the joint LJPME search and convergence calculations. Their single-mesh evaluation calls total 267.1 seconds of wall time, excluding setup, numpy comparisons, convergence references, and the joint search. This is not CPU core time. GPU hours: zero. Platform builds: zero. One small analysis bridge was compiled. No subagents or child lanes were spawned. All numerical background jobs completed; only unanswered cdx ask requests remained pending at handoff.

Changed files and report paths

- Source report: lanes/r3-pme4.md in this worktree, mirrored in the lab experiment's lanes directory.
- Analysis scripts, bridge source, README, JSON/JSONL measurements, exact input snapshots, reference arrays, and logs: /Users/mas/code/metal/lab/experiments/029-2x-moonshots/lanes/results/r3-pme4/.
- provenance.json records Python, numpy, OpenMM, input hashes, and library hashes. manifest.json lists the archived files and hashes.
- No product C++, Metal kernels, platform properties, or platform tests changed.

## Scope extensions

- /Users/mas/code/wt/r3-pme4/lanes/r3-pme4.md: stores the acceptance report on the requested branch so the numerical outcome can be reviewed and committed without a product change.
- /Users/mas/code/metal/lab/experiments/029-2x-moonshots/lanes/r3-pme4.md: required lane handoff outside the owned results subdirectory.
