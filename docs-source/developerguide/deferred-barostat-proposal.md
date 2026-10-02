# Proposal: defer the isotropic Metal barostat energies

This proposal targets the FAH mission stack at d670532ac80f29100b35da7bbe66c37b737140af.
It implements phase 1 only. Metal submits the initial energy, volume proposal and
trial energy before downloading either energy. The host still makes the Metropolis
decision. This needs a rebuilt libOpenMM and platform plugins through the FAH core
team. It is not a plugin-only change or a measured speedup.

## Core-team surface

The openmmapi diff is 38 added and 26 removed lines across three files:
`MonteCarloBarostatImpl.cpp`, `ContextImpl.cpp` and `ContextImpl.h`.
The kernel interface in `olla/include/openmm/kernels.h` adds 14 lines.
Together that is 52 added and 26 removed lines, 78 changed lines.
No new ContextImpl fields or MonteCarloBarostatImpl fields are needed.

There is one new virtual hook, appended to `ApplyMonteCarloBarostatKernel`:

```cpp
virtual bool computeTrialEnergies(ContextImpl& context, int groups,
    const std::function<void()>& propose,
    double& initialEnergy, double& finalEnergy);
```

Its inline default returns false without changing any state or invoking `propose`.
The caller then executes the existing pair of `getState(State::Energy)` calls.
An implementation returning true must evaluate the initial energy, invoke the
proposal exactly once, then evaluate the trial energy. It returns both numerical
results before the caller makes the decision. CUDA, OpenCL, CPU and Reference
inherit the default. Common kernels are unchanged.

A nonvirtual `ContextImpl::calcForcesAndEnergy` overload adds a `double& forceEnergy`
output. It returns the platform finish result separately. The original method
adds the two pieces in the original order and retains its signature. The retry
loop and `lastForceGroups` handling are unchanged. This split is needed because
reassociating the host and device energy terms can change the last bits.

Adding a virtual method changes the plugin ABI contract. Appending it preserves
existing slot positions but does not make old plugin binaries safe: they lack the
new entry. Rebuild the core and all platform plugins together. The unchanged
behavior guarantee for other platforms applies to those rebuilt implementations.

## Sequence and numerical identity

Before, in `MonteCarloBarostatImpl::updateContextState`:

1. Advance the frequency counter. On a trial, obtain the integrator force groups.
2. `getState(Energy)` runs the initial force/energy evaluation, waits for its
   reduction, reads potential energy E0 and computes an unused kinetic energy.
3. Read the old box and volume V. Consume one host uniform U and form
   `deltaVolume = volumeScale*2*(U-0.5)`, then Vnew and `pow(Vnew/V, 1.0/3.0)`.
4. Save coordinates, velocities, force buffers, atom order and cell offsets.
   Set the scaled box and call the Common coordinate scaling kernel. Determine
   the molecule count, or particle count when rigid molecule scaling is disabled.
5. `getState(Energy)` evaluates the trial, waits for its reduction and obtains E1
   and another unused kinetic energy.
6. Compute `w = E1-E0 + pressure*deltaVolume - N*kT*log(Vnew/V)` on the host.
   The expression `w > 0 && Uaccept > exp(-w/kT)` consumes a second host uniform
   only when `w > 0`. There is no unconditional acceptance draw.
7. Reject: restore the old box, saved coordinates, velocities and forces, cell
   offsets, and atom order when required. Accept: increment `numAccepted` and
   set `forcesInvalid = true`. Both branches increment `numAttempted`.
8. At ten or more attempts, apply the existing 25%/75% adaptation thresholds,
   division/multiplication by 1.1, maximum `0.3*volume`, and counter resets.

After, for eligible Metal contexts, steps 1 and 3 through 8 retain the same
operations and ordering. Step 2 submits the existing GPU reduction to a private
E0 buffer and returns its separate host terms. Step 5 submits to a private E1
buffer. Only then does the first download finish the queue, including both
reductions. The second download has no intervening GPU submission to wait for.
There is no extra copy kernel or GPU reduction. Two buffers have the same size,
element type, launch geometry and reduction order as the existing `energySum`.
The original host summation loop reads each buffer in index order.

Let F be the sum from ForceImpl calls, P the sum of platform post computations,
and R the GPU reduction. The old arithmetic is `F + (P + R)`. Deferral retains F
and P separately and reconstructs exactly `F + (P + R)`, not `(F + P) + R`.
This includes volume-dependent NonbondedForce host terms such as dispersion and
self energy. An invalid force evaluation repeats the original retry loop and
rewrites the same private slot; only the final valid attempt's host terms survive.
An exception resets the context's temporary reduction destination before leaving.
The second evaluation cannot overwrite the first reduction buffer.

The proposal lambda encloses the unchanged proposal arithmetic and its RNG call.
The decision, reject/accept branches, counters and adaptation code after it are
unchanged. Thus equal E0/E1 bits imply equal w, the same conditional RNG draw,
the same decision, and the same state and next volumeScale. Induction across
trials then preserves the trajectory, provided the underlying force computation
is reproducible. No new code consumes random numbers.

Only an exact `LangevinMiddleIntegrator` takes this path. Its
`kineticEnergyRequiresForce()` is false, and its kinetic energy is a read-only
reduction with time shift zero. Skipping the unused kinetic energy result changes
no positions, velocities, forces or RNG state. The existing prefetch can still
run; neither its kernels nor force-pass ordering is changed. Force-shifted and
custom integrators retain `getState` and its full side effects.

Metal enables this for one device, GPU PME, and systems containing only
HarmonicBondForce, HarmonicAngleForce, PeriodicTorsionForce, NonbondedForce,
CMMotionRemover and MonteCarloBarostat. The allowlist excludes host callbacks and
nested energy evaluations. `OPENMM_METAL_DEFER_BAROSTAT_ENERGIES=0` selects the old
path; unset or nonzero enables deferral when eligible. Eligibility is set at
context initialization. Anisotropic, membrane and flexible barostats retain their
separate existing update paths. Their shared coordinate kernel alone does not
justify changing those decision paths in this proposal.

## Identity test and pending evidence

`TestMetalDeferredBarostat.cpp` builds a 64-molecule constrained four-site water
box with PME, virtual sites, dispersion correction, separate direct/reciprocal
force groups, and a fixed-seed LangevinMiddle integrator. It constructs the off
and on contexts sequentially so each starts with the same barostat RNG seed.
Across 400 steps and 200 trials it compares every scalar's object representation
in positions, velocities, forces, potential and kinetic energies, all nine box
components, and time, plus the step counter. It requires at least one acceptance
and one rejection and crosses the ten-attempt threshold for proposal-size adaptation.
The test enables DeterministicForces to make a bitwise comparison meaningful;
a fixed seed alone cannot promise identity on a nondeterministic force backend.

The existing CMake glob registers `TestMetalDeferredBarostatSingle` and
`TestMetalDeferredBarostatMixed`. Neither test has been executed during the GPU
embargo. Source reasoning is not a passing runtime result. Once the embargo ends,
run both registered cases in one invocation against the gate-built tree:

```sh
ctest --test-dir /tmp/openmm-metal-bench/ultra-m029-baro-defer/build \
  --output-on-failure --no-tests=error \
  -R '^TestMetalDeferredBarostat(Single|Mixed)$'
```

The separate closing-brace repair in `MetalContext.cpp` fixes the reported
inherited d670532ac merge defect before `executeAfterForcePass`. There are no
changes to nonbonded, PME, integrator kernels, or force-pass order.

## Cost and timing screen

The consult report item 2 and r3-fah job 394 price the two remaining drain holes
at 19 to 20 microseconds per integration step after polling and asynchronous
uploads. At barostat frequency 25, one trial per 25 steps means 0.04 trials per
step. The phase 1 estimate is about 200 microseconds saved per trial:
`200 * 0.04 = 8 us/step`. On the consult's 484.43 us reference clock,
`484.43 / (484.43 - 8) = 1.01679`, or 1.68% throughput improvement.
That reference clock is job 404, not a new timing of d670532ac. Measure the latter.
No phase 1 replacement kernel is added; any remaining host overhead must be
included in the screen. Initialization uploads, atom reorders and neighbor-list
retries can still synchronize. This removes the first *energy* readback dependency,
not every possible synchronization on a trial.

Phase 2 is not implemented. Its nominal model removes 20 us and adds 2 to 5 us of
device decision/restore work: net 15 to 18 us, yielding
`484.43/469.43 = 1.03195` to `484.43/466.43 = 1.03859`.
Using the lower 19 us drain estimate broadens the low end to 14 us net,
`484.43/470.43 = 1.02976`. Phase 1 is already included in those totals.
Phase 2 also requires authoritative device box state and exact acceptance/RNG
semantics; copying a host decision into a device restore kernel does not remove
the remaining drain. Host `log`/`exp` cannot simply be replaced by Metal math with
a bitwise-identity claim.

After the GPU is free, have the liaison prepare these two isolated build trees:
`ultra-m029-baro-base` at d670532ac plus only the inherited brace repair, and
`ultra-m029-baro-defer` with this proposal including the same repair. Their venvs
must come from those builds. The base repair must be present in both arms.
Use a fresh output directory. The command is two interleaved 15-second rounds
of tip4pew mixed on the host clock; it is not run during the embargo:

```sh
set -eu
bench=/tmp/openmm-metal-bench
out="$bench/ultra-m029-baro-defer/screen-tip4pew-mixed-2x15"
test "$(cat "$bench/ultra-m029-baro-base/src/.commit")" = d670532ac80f29100b35da7bbe66c37b737140af
"$bench/ultra-tools/ab.sh" "$out" 2 15 tip4pew \
  "base=$bench/ultra-m029-baro-base/venv/bin/python:Metal:mixed:OPENMM_METAL_DEFER_BAROSTAT_ENERGIES=0" \
  "defer=$bench/ultra-m029-baro-defer/venv/bin/python:Metal:mixed:OPENMM_METAL_DEFER_BAROSTAT_ENERGIES=1"
python3 "$bench/ultra-tools/summarize.py" "$out" defer/base
python3 - "$out/result.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
rows = [r for r in report['ratios'] if r['test'] == 'tip4pew'
        and r['num'] == 'defer' and r['den'] == 'base']
assert len(rows) == 1, 'missing tip4pew ratio'
ratio = rows[0]['ratio']
print(f"{'KEEP' if ratio >= 1.01 else 'KILL'}: {ratio:.6f}x, {(ratio-1)*100:.3f}%")
raise SystemExit(0 if ratio >= 1.01 else 1)
PY
```

Kill below 1.0% whole-step throughput gain. At the reference clock, 1.01x requires
`484.43 - 484.43/1.01 = 4.79634 us/step` saved. Inspect each round and the recorded
build/load hygiene before funding a longer confirmation. This screen does not
replace the identity test or core-team review.

Evidence read: `/Users/mas/.cdx/reports/m029-tip4pew-next-r1.md` item 2;
`/Users/mas/code/metal/lab/experiments/029-2x-moonshots/lanes/r3-fah.md` around 663;
the barostat and cache entries in `lanes/r3-step.md`; and
`/tmp/openmm-metal-bench/ultra-lead/STATE.md` core-team list. The third-force-pass
cache, keep-alive and integrator/PME changes are not part of this proposal.
