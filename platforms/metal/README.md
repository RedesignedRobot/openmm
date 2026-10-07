# Metal platform

A native OpenMM platform for Apple GPUs. It runs the Common Compute kernels as Metal Shading Language through a macro prelude, uses metal-cpp for the host side, and adds Metal kernels where Apple GPUs need a different design (neighbor list, bonded forces, PME convolution, the fused LangevinMiddle step, device atom reordering).

## Requirements

- macOS 15 or later. The kernels compile as MSL 3.2 with `MTLCompileOptions.mathMode`, both new in macOS 15.
- An Apple GPU of family Apple7 or later (M1 and later).
- Apple clang with C++17, from Xcode or the Command Line Tools. Kernels compile at run time, so the build needs no offline Metal compiler.

On a Mac that fails either check the platform does not register, and OpenMM picks OpenCL or CPU instead (`src/MetalPlatform.cpp`, `isPlatformSupported()`).

## Build

On an arm64 Mac, `OPENMM_BUILD_METAL_LIB` is on by default (top-level `CMakeLists.txt`).

```sh
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8 install
```

This builds `libOpenMMMetal` and the Metal versions of the AMOEBA, Drude and RPMD plugins. metal-cpp is vendored in `libraries/metal-cpp`. The bundled VkFFT 1.2.33 carries four fixes to its Metal backend, listed in `libraries/vkfft/METAL_PATCHES.txt`.

`OPENMM_BUILD_METAL_TESTS` (on by default) builds the tests, and it also defines `OPENMM_METAL_TEST_HOOKS` for the library. The hooks are static members that tests use to inject faults and to switch off two optimizations for bitwise comparisons. They change no class layout and do nothing unless a test sets them. A release build can turn the tests off.

## Precision

Apple GPUs have no 64-bit floating point, so `Precision=double` is refused when the Context is created.

- `single` computes in float. Forces accumulate as 64-bit fixed point, as on CUDA and HIP.
- `mixed` keeps forces and most arithmetic in float and does the integration and energy sums in df64. A df64 is a pair of floats (hi, lo) whose sum is the value. `src/kernels/df64.metal` defines it and maps `double`, `double2`, `double3` and `double4` in kernel source onto it, so most Common kernels compile for mixed precision unchanged. Addition keeps an error of about 2^-47 of |x|+|y| (the comment on `operator+` in `df64.metal`), against 2^-53 for IEEE double.

In mixed precision, velocities and position deltas live on the device as float pairs. `MetalArray` converts on upload and download, so the host API still sees doubles. Positions use the usual mixed layout of `posq` plus `posqCorrection`.

df64 has no tan, asin, acos, atan, atan2, sinh, cosh, tanh, erf, erfc or non-integer power. A CustomIntegrator whose per-DOF or sum expressions use one of them is refused in mixed precision when the Context is created. Global expressions run on the host in double and may use them.

## Properties

| Property | Default | Meaning |
| --- | --- | --- |
| `DeviceIndex` | | The platform uses the system default GPU and reports it as `0`, the only valid index. |
| `DeviceName` | | Set by the platform to the device name. |
| `Precision` | `single` | `single` or `mixed`. |
| `UseCpuPme` | `false` | Compute PME reciprocal space on the CPU, if the CPU PME plugin is loaded. |
| `DisablePmeStream` | `auto` | `true` keeps PME on the main queue. `false` gives it a second queue. `auto` uses the second queue on Apple9 GPUs (M3 and later), where its work runs beside the force kernels, and the main queue on older ones. |
| `DeterministicForces` | `false` | `true` spreads PME charge with fixed point atomics and keeps the neighbor list reorder on a fixed interval, so repeated runs give the same forces. |
| `TempDirectory` | `$TMPDIR` | Directory for temporary files. |
| `UseBlockingSync` | `true` | Accepted for compatibility with CUDA and HIP. It has no effect. |

The names with a `Metal` prefix (`MetalPrecision` and so on) are accepted as deprecated aliases, like the `Cuda` and `Hip` names on those platforms.

## Differences between GPU families

Apple9 GPUs (M3 and later) add floats atomically in hardware. On them the platform spreads PME charge with float atomics, interpolates PME forces with PME_ORDER lanes per atom, runs the PME convolution in its own kernels instead of VkFFT, runs some kernels beside the bonded and nonbonded pass, and orders the neighbor list by atom role when a water model has LJ-only and charge-only sites (TIP4P-Ew, TIP5P, OPC). Apple7 and Apple8 GPUs (M1, M2) keep the fixed point and serial paths. Each choice is a device family check in the code, with the measurement behind it in a comment.

## Tests

From the build directory:

```sh
ctest -R TestMetal                      # every Metal test, single and mixed
ctest -R TestMetalNonbondedForceMixed   # one test in one precision
ctest -R 'TestMetal(Amoeba|Hippo|Wca)'  # AMOEBA plugin
ctest -R 'TestMetalDrude|TestMetalRpmd' # Drude and RPMD plugins
```

Each test program runs twice, as `<name>Single` and `<name>Mixed`. Most are OpenMM's shared tests. The Metal-only ones cover command batching, deferred barostat energies, the ordered force fold, checkpoint replay, PME energy-only steps, mixed precision edge cases and recovery from host-side faults.

- `TestMetalOrderedForceFold` needs an Apple9 GPU and reports a ctest skip (exit code 77) on older ones.
- `TestMetalHostRobustness` builds a 19,652-site water box and has a 30 minute timeout.
- `TestMetalNonbondedForce` skips its huge system when the GPU's recommended working set is under 8 GB.

## Known limits

- No double precision, as above.
- One GPU per Context. A list of devices in `DeviceIndex` is refused.
- Metal allows 31 buffer bindings per kernel. When a kernel has more arguments, the platform packs its values into one struct, but a kernel with too many arrays still fails to compile, with an error that names the limit. A CustomNonbondedForce with 32 per-particle parameters is one case (`tests/TestMetalCustomNonbondedForce.cpp`).
- With float atomic PME spreading (Apple9, `DeterministicForces=false`), the grid sums in a different order on every evaluation, so forces vary in the last bits from run to run.
