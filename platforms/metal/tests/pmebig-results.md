# Large-grid fused PME experiment

On 2026-09-27, a five-kernel prototype based on c21cad1e7 failed its 25% complete-block speedup requirement on the M3 Ultra. Production code is unchanged. The large Coulomb grids and the 49-cubed dispersion grid retain VkFFT. The existing small-plane force-only optimization remains unchanged.

The prototype packed pairs of real z lines into complex FFTs, transposed through y loads and stores, and fused x FFT, reciprocal convolution and inverse x FFT. It supported prime radices through 13. The host selected it using device thread and local-memory limits. It reused the two existing grids. Energy accumulated from the spectrum before convolution, including the mirrored triclinic reciprocal factor and the dispersion zero mode.

## Daytime kill test

Complete forward, convolution and inverse sequences, host steady clock, microseconds per sequence. Each arm ran five sequences per synchronized batch. Twenty measured batches followed two warmup batches. Arm order alternated. The reported median is the upper middle sample. Uploads and initial compilation were outside the timed interval. An exclusive GPU lease serialized the run; other daytime CPU/build activity remained possible. These are provisional microbenchmark results, not GPU timestamps or whole-benchmark scores.

| Grid | VkFFT plus convolution | Fused | Time reduction |
| --- | ---: | ---: | ---: |
| 98 x 98 x 98 | 165.833 | 148.208 | 10.6% |
| 231 x 112 x 110 | 271.267 | 319.192 | -17.7% |
| 198 x 200 x 200 | 561.142 | 690.592 | -23.1% |
| 49 x 49 x 49, dispersion | 116.317 | 109.183 | 6.1% |

No grid met the threshold. Do not compare these host-clock values directly with the earlier 56-cubed GPU-timestamp measurements.

## Correctness evidence

A synthetic 1024-particle triclinic system used each actual grid size, with a separate LJPME case using 98-cubed Coulomb and 49-cubed dispersion grids. The control was the energy-step VkFFT path in the c21cad1e7 build. Both precisions used deterministic force accumulation. These were grid coverage checks, not runs of the full cellulose or STMV atom sets.

| Case | Fused/VkFFT relative force difference | Fused/Reference force error | VkFFT/Reference force error |
| --- | ---: | ---: | ---: |
| 98-cubed Coulomb | 9.58e-7 | 1.724e-6 | 1.753e-6 |
| LJPME | 9.58e-7 | 1.724e-6 | 1.753e-6 |
| Cellulose grid | 1.44e-6 | 2.726e-6 | 2.731e-6 |
| STMV grid | 1.65e-6 | 3.184e-6 | 3.186e-6 |

These force figures were identical in single and mixed. Repeats were bitwise identical within each Context and across three fresh processes. Force-only and energy-step forces agreed within 2e-6 relative.

Energy failed the no-regression check. For example, single-precision STMV-grid absolute energy error against Reference was 0.04050 kJ/mol for the prototype and 0.01359 for VkFFT. Mixed errors were 0.03502 and 0.02780. Single LJPME error was 0.01222 versus 0.00512. The prototype must not ship on this evidence.

The first C++ correctness invocation stopped because its extracted control kernel lacked the M_PI definition. The harness definition was fixed and rebuilt, then used in timing mode. Its numerical assertions were not rerun. The separate Python matrix completed all samples, then failed its energy assertion. No correctness pass is claimed for the prototype.

The rejected production patch, C++ and Python probes, logs and raw force/energy arrays are archived in the lab at experiments/029-2x-moonshots/lanes/results/r3-pmebig/. The runtime prototype and its tests were removed from this tree after rejection.

## Restored-tree validation

The complete Release build passed after reverting the prototype. `ultra-tools/gate.sh --quick /tmp/openmm-metal-bench/ultra-r3-pmebig` passed all six force cases in single and mixed precision. The lane used three correctness invocations: the C++ harness failure, the Python energy failure, and this restored-tree gate. No full test suite or whole-benchmark timing screen ran.
