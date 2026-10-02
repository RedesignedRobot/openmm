#!/usr/bin/env python3
"""Export captured PME positions and charges without importing OpenMM.

The input is a serialized System/State pair, not a new simulation.  Coordinates
include the captured virtual sites.  The replay resolves the minimum mesh with
MetalContext::findLegalFFTDimension on the device being measured.
"""
import argparse
import bz2
import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def read_xml(directory, name):
    path = directory / (name + ".xml")
    if path.exists():
        raw = path.read_bytes()
    else:
        path = directory / (name + ".xml.bz2")
        raw = bz2.decompress(path.read_bytes())
    return ET.fromstring(raw), hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("unit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    system, system_hash = read_xml(args.unit, "system")
    state, state_hash = read_xml(args.unit, "state")
    nb = [f for f in system.find("Forces") if f.get("type") == "NonbondedForce"]
    if len(nb) != 1 or nb[0].get("method") != "4":
        raise ValueError("Expected one PME NonbondedForce")
    nb = nb[0]
    if len(nb.find("ParticleOffsets")):
        raise ValueError("Captured charge offsets need explicit evaluation")
    positions = list(state.find("Positions"))
    particles = list(nb.find("Particles"))
    if len(positions) != len(particles):
        raise ValueError("Position/charge count mismatch")
    box = [[float(v.get(axis)) for axis in "xyz"] for v in state.find("PeriodicBoxVectors")]
    mesh = [int(nb.get(axis)) for axis in ("nx", "ny", "nz")]
    alpha = float(nb.get("alpha"))
    if alpha == 0:
        tol = float(nb.get("ewaldTolerance"))
        alpha = math.sqrt(-math.log(2*tol))/float(nb.get("cutoff"))
        # NonbondedForceImpl::calcPMEParameters, Coulomb branch.
        mesh = [max(6, math.ceil(2*alpha*box[i][i]/(3*tol**0.2))) for i in range(3)]
    lines = [" ".join(map(str, [len(positions), *mesh]))]
    lines += [" ".join(format(x, ".17g") for x in v) for v in box]
    lines += [" ".join([*(p.get(axis) for axis in "xyz"), q.get("q")]) for p, q in zip(positions, particles)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(f"capture={args.output} atoms={len(positions)} minimum_mesh={mesh} alpha={alpha:.9g}")
    print(f"system_xml_sha256={system_hash} state_xml_sha256={state_hash}")


if __name__ == "__main__":
    main()
