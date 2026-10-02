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
Phase 2 also requires authoritative device box state and a defined acceptance/RNG
contract; copying a host decision into a device restore kernel does not remove
the remaining drain. Host `log`/`exp` cannot simply be replaced by Metal math with
a bitwise-identity claim. The phase 2 investigation below records the revised
contract and the prerequisites. These cost figures remain a model.

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

## Phase 2 investigation at c0365efdc

Phase 2 is not implemented. The investigation leaves phase 1 unchanged. There is
no device-path engagement claim, new switch, or new core API. The phase 2 core
API delta is zero added and zero removed lines.

The initial brief required bitwise agreement with the host under
DeterministicForces, including the conditional host RNG draw. In its lane reply
on 2026-10-02, the liaison changed that requirement to a statistically correct
device decision with an independent unbiased RNG and energy error well below
kT. It prohibited host-RNG mirroring, lifted the PME file restriction, and
authorized implementation only if the Metal device-box plumbing fit about 300
changed lines without changing CUDA or OpenCL behavior. Otherwise it requested
this prerequisite design. No statistically correct device path has been proved
or tested here. The original bitwise acceptance criteria remain unmet.

### Why a direct decision port cannot claim host identity

`MonteCarloBarostatImpl::updateContextState` consumes one SFMT uniform for the
volume proposal. It computes the following in host double precision:

```cpp
w = finalEnergy-initialEnergy + pressure*deltaVolume
    - numberOfScaledParticles*kT*log(newVolume/volume);
reject = w > 0 && uniform() > exp(-w/kT);
```

The second draw occurs whenever w is positive, including accepted uphill moves.
It does not occur for nonpositive w. `SimTKOpenMMUtilities` owns a static SFMT
state; `genrand_real2` converts a 32-bit integer to `integer * 2^-32`. Drawing two
uniforms unconditionally shifts the next proposal whenever w is nonpositive.
A separate device RNG can implement the revised statistical contract but cannot
prove the original stream identity. DeterministicForces does not change this.

Host precomputation can preserve the volume log term, but the usual threshold
`-kT*log(uniform)` is not an exact inversion of the rounded host exp predicate.
A CPU-only check with the local host libm produced this valid SFMT-lattice input:

```text
kT = 1
uniform integer = 2
u = 0x1.0000000000000p-31
w = -log(u) = 0x1.57cd0e7026820p+4
exp(-w) = 0x1.ffffffffffff7p-32
u > exp(-w): true
w > -log(u): false
```

This refutes that algebraic replacement, not every possible host-precomputed
threshold. An exact threshold design would need to locate the boundary of the
actual host predicate over representable values, establish its monotonicity on
the supported domain, preserve strict comparisons and special cases such as
u=0, and reproduce the rounded energy arithmetic feeding the comparison. It
would also need an unconsumed lookahead draw and conditional ownership of the
shared SFMT cursor. That design is not implemented and the liaison ruled out
host-RNG mirroring for this round.

`MetalContext::readEnergy` adds each downloaded reduction element, in index
order, into a host double. Phase 1 then reconstructs
`hostForce + (hostPlatform + reduction)`. The device must reproduce these
roundings as well as the work expression to retain exact host decisions.
`df64.metal` represents a value with two floats and implements its own exp/log;
it is not a general IEEE binary64 arithmetic replacement. Reusing it is not an
exactness proof. Host exp/log precomputation alone does not fix the energy sum.

One-trial-late host evaluation also needs more than coordinate restoration.
After a wrong speculative decision, subsequent integration has used the wrong
box, forces and random draws. Correct replay needs snapshots of integrator RNG,
velocities, force validity, atom order, cell offsets, neighbor data and adaptation
counters. An observation before validation must wait. No complete speculative
checkpoint or replay protocol is present in the phase 1 hooks.

### Device box is a prerequisite even under the revised contract

The current call chain has multiple host dependencies:

* `CommonCalcNonbondedForceKernel::execute` reads host box vectors and computes
  the reciprocal vectors in double precision. It supplies them by value to PME
  grid indexing and spreading. `computeReciprocalFromGrid` supplies them to
  convolution, energy evaluation and interpolation, including the FFT hook.
* The same force implementation adds `dispersionCoefficient/(a[0]*b[1]*c[2])`
  to the host energy. Updating a device volume does not update that contribution.
* `MetalContext::executeKernelFlat` copies scalar arguments with `setBytes`, or
  packs them with memcpy before `setBytes`. Updating a buffer after submission
  cannot change those encoded values.
* Direct-space and neighbor kernels receive box-size, inverse-size and vector
  arguments too. `MetalNonbondedUtilities::prepareInteractions` also checks the
  box size on the host. A PME-only change would leave these consumers stale.
* Common barostat restoration copies saved device buffers but also restores
  host cell offsets and, if necessary, atom ordering. A conditional position
  copy alone is not a complete rejection.

Changing the host box getter to download the result would restore correctness
but insert the prohibited dependency at the next force submission. The liaison
confirmed that no existing device-box mechanism was missed. I did not identify
a complete implementation within its approximately 300-line limit. This is a
scope assessment, not a proof that no compact design is possible.

### Prerequisite design and validation

First introduce a Metal-owned box-state buffer with current and trial vectors,
reciprocal vectors, size/inverse size and volume. Give every periodic Metal
kernel an explicit binding to that state. Extend the Metal argument path to
support device-selected values in both packed and unpacked kernels. Cover the
FFT convolution hook as well as standalone PME kernels. Preserve the existing
argument path by default on all other platforms. Select the dispersion and
plasma correction volumes from the same state, with each energy contribution
included exactly once.

Then add a device trial-state buffer containing the proposal scale, attempt and
accept counters, outcome and RNG state. Under the revised contract, seed an
independent stream and define its checkpoint/reinitialization behavior. Move
proposal generation, volume adaptation, decision and conditional restore into
an ordered device transaction. The next force submission must consume its
selected box without first obtaining a host scale or decision. Handle atom
reordering, cell offsets and force validity explicitly. Host box getters,
checkpoint operations and counter queries can resolve the pending transaction;
internal force submission must not use those resolving getters.

The statistical decision still needs an error budget. Reduce initial and trial
energies with a documented precision and order, bound the error in their
difference relative to kT on the target systems, and verify the RNG distribution
and acceptance rule. If a threshold is used, document its rounding contract.
Do not describe this path as trajectory-identical to phase 1. Keep
`OPENMM_METAL_DEFER_BAROSTAT_ENERGIES=0` as a host-path fallback. Any additional
device switch must default off outside the established phase 1 eligibility.

Before integration, test both packing modes, rejection followed by a force
evaluation, current-box reporting, dispersion correction, checkpoint/resume,
reordering and counter adaptation. Require an engagement counter and both
accepts and rejects. Keep the existing phase 1 bitwise test. A separate phase 2
test must reflect the revised statistical contract, including single and mixed
precision and at least 100 mixed trials on fah-tip4pew and fah-tip3p-hmr.
Those tests and TestMetalMonteCarloBarostat were not run in this investigation.
All future GPU correctness commands must use the lane's correctness lease.

The original cost model omits this prerequisite machinery. At frequency 25,
write the modeled total phase 2 saving as `(D-R-B)/25` per step, where D is the
removed drain cost per trial, R is decision/restore cost and B is all additional
box, RNG, reduction and bookkeeping cost amortized per trial. The old estimates
used D=500 microseconds, R=50 to 125 and B=0, giving 15 to 18 microseconds per
step in total, including phase 1. B is unknown, so that range is not established
for this design. With no phase 2 implementation, incremental saving over phase
1 is zero. No timing command or GPU execution ran for this investigation.

Kill below 1.0% whole-step throughput gain. At the reference clock, 1.01x requires
`484.43 - 484.43/1.01 = 4.79634 us/step` saved. Inspect each round and the recorded
build/load hygiene before funding a longer confirmation. This screen does not
replace the identity test or core-team review.

Evidence read: `/Users/mas/.cdx/reports/m029-tip4pew-next-r1.md` item 2;
`/Users/mas/code/metal/lab/experiments/029-2x-moonshots/lanes/r3-fah.md` around 663;
the barostat and cache entries in `lanes/r3-step.md`; and
`/tmp/openmm-metal-bench/ultra-lead/STATE.md` core-team list. The third-force-pass
cache, keep-alive and integrator/PME changes are not part of this proposal.

### Phase 2a device box

`OPENMM_METAL_DEVICE_BOX=1` selects a Metal buffer as the periodic box authority.
The flag defaults off. Unsupported force types, CPU PME and non-PME
NonbondedForce methods disable it for the whole context. CUDA and OpenCL keep
their existing behavior. The host barostat still proposes, accepts and rejects
moves, including the phase-1 deferred energy transaction. Device Metropolis and
RNG remain phase 2b.

The buffer stores float box vectors, size and inverse size, reciprocal vectors,
reorder high/low words and bin parameters, IEEE double vectors, and volume.
Argument reflection binds periodic kernel parameters to this buffer, including
packed arguments. Forces, PME, virtual sites, bonded terms, reordering and
volume corrections therefore consume the same queued box state.

`MetalContext::scaleDeviceBox(MetalArray&)` accepts one IEEE double scale factor
in a device buffer. The caller must supply a positive finite isotropic scale.
The entry point prepares the molecule tables, unwraps positions using the old
box and device cell offsets, clears those offsets, translates each rigid
molecule by its center times `(scale-1)`, scales the box and rebuilds every
force consumer's derived field. It marks the host box dirty, invalidates forces
and the neighbor list, and notifies the integrator's cached force state. These
operations enter the same command stream with no explicit host wait. The
coordinate translation matches `scaleCoordinates`: it modifies float position
words and preserves mixed corrections. Large molecules use a serial sum in the
new kernel, so rounding can differ from the host barostat's parallel reduction.
Layouts outside GPU reorder's group and molecule limits reject scaling before
positions change. CPU reorder fallback resolves the box through its existing
box getter on that rare path.

Reorder bin widths and counts are computed on the device after each box change.
A zigzag range that cannot fit the packed key selects the existing bounded
Hilbert curve on the GPU. Cutoff validation also runs on the device after each
box update. It records an error for the next host observation, with no wait in
plain stepping or scaling. The error survives intervening box updates until
an observation reports it.

Host `setPeriodicBoxVectors()` unwraps positions and clears cell offsets on the
device, then stages one ordered 320-byte box copy. The live buffer is never
written from the CPU while queued consumers can read it. Reorder owns the
current device cell offsets; explicit host changes upload a fresh allocation.
Box getters lazily resolve the IEEE vectors and pending cutoff error. Kernel
pointer getters never resolve. `getState` and checkpoints use the existing
common observation paths, which now reach the lazy box getters and synchronized
atom order and offsets. Loading a checkpoint restores those mirrors and uploads
the restored box. The host barostat save and restore implementation is unchanged.

The flag-on path skips host periodic argument calculations in the common
kernel helper, PME setup, bonded dispatch, virtual-site position updates and
virtual-site force redistribution. These skips prevent dirty box resolution
inside normal stepping. The source audit finds zero waits in the enabled GPU
reorder path and zero in the scale entry point. Host observations and the
existing host barostat transaction may wait.

`TestMetalDeviceBox [single|mixed|all [system.xml state.xml integrator.xml]]`
retains the round-2 host-update, binding, charged PME, parameter-offset,
checkpoint and 1000-step host-barostat coverage. It adds bin overflow, device
scale versus host box update plus `scaleCoordinates`, reported coordinate
rounding deltas, bitwise Metal forces on identical copied inputs, Reference
forces and energy, checkpoint restoration into a fresh context, deferred cutoff
errors, and 200 LangevinMiddle steps with eight device scales and no intervening
host observations. Group energy tolerance is `1e-7` of group magnitude; total
Metal energy tolerance is `1e-8` of summed group magnitudes. Reference tolerance
is `1e-4` force relative RMS in both precisions, keeping automatic PME grids.
Device scaling must be no worse than host scaling against Reference on the same
box; the bitwise Metal force comparison establishes that condition. Energy tolerance is
`1e-5` of summed group magnitudes. The round-3 handoff records runtime results
and any remaining limitations; the implementation alone is not validation.
