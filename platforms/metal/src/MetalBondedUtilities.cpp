/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2011-2026 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * This program is free software: you can redistribute it and/or modify       *
 * it under the terms of the GNU Lesser General Public License as published   *
 * by the Free Software Foundation, either version 3 of the License, or       *
 * (at your option) any later version.                                        *
 *                                                                            *
 * This program is distributed in the hope that it will be useful,            *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of             *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
 * GNU Lesser General Public License for more details.                        *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public License   *
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.      *
 * -------------------------------------------------------------------------- */

#include "MetalBondedUtilities.h"
#include "MetalContext.h"
#include <algorithm>
#include <sstream>

using namespace OpenMM;
using namespace std;

// A chunk is one term per thread of a threadgroup, and BondedUtilities::computeInteractions() launches
// threadgroups of ThreadBlockSize.  Only forces whose terms touch up to MaxChunkedAtoms atoms are chunked.
static const int MaxChunkedAtoms = 4;

// Forces with fewer terms keep the per-term atomics.  On the M3 Ultra, chunking every force slowed the
// bonded kernel of the rf and gbsa dhfr systems (at most 7,310 terms per force) by 10 to 15%, and sped it
// up 2.8 to 3.4 times on apoa1 and cellulose, where no force has fewer than 11,428 terms.  dhfr's PME
// exclusions (34,709 terms) stay chunked.
static const int MinChunkedTerms = 10000;

// Metal's limit on a kernel's buffer and value arguments (see MetalContext::createProgram()), and the
// arguments that BondedUtilities::initialize() gives the kernel before any force's.
static const int MaxKernelArguments = 31;
static const int NumFixedArguments = 9;

void MetalBondedUtilities::initialize(const System& system) {
    // For each chunk, the plan lists the atoms the chunk touches, and for each atom the pad slots that hold
    // its forces.  A force's plan is the offset of each chunk's first entry, then one entry per atom
    // (the atom, and its first and end slot in the low and high 16 bits), then numAtoms*chunkSize slots
    // per chunk as ushorts.  A force keeps the per-term atomics of BondedUtilities if it has too few terms,
    // if its terms are too large, if its code could leave the loop over its terms early and so skip a
    // barrier, if its chunks save less than a quarter of its atomic adds, too few to pay for the barriers,
    // or if its plan would take the whole plan past 1% of the device's working set.  So does every force
    // when the plan would take the kernel past the argument limit.

    const int chunkSize = MetalContext::ThreadBlockSize;
    int numForces = forceAtoms.size();
    entryStartOffset.assign(numForces, -1);
    entryOffset.assign(numForces, -1);
    slotOffset.assign(numForces, -1);
    firstChunkedForce = -1;
    maxChunkedAtoms = 0;
    int numArguments = NumFixedArguments+arguments.size()+(energyParameterDerivatives.size() > 0 ? 1 : 0);
    for (int force = 0; force < numForces; force++)
        numArguments += (forceAtoms[force][0].size()+3)/4;
    long long maxPlanBytes = dynamic_cast<MetalContext&>(context).getDevice()->recommendedMaxWorkingSetSize()/100;
    vector<unsigned int> planData;
    for (int force = 0; force < numForces && numArguments < MaxKernelArguments; force++) {
        int numBonds = forceAtoms[force].size();
        int numAtoms = forceAtoms[force][0].size();
        const string& source = forceSource[force];
        if (numBonds < MinChunkedTerms || numAtoms > MaxChunkedAtoms || source.find("continue") != string::npos || source.find("break") != string::npos || source.find("return") != string::npos)
            continue;
        int numChunks = (numBonds+chunkSize-1)/chunkSize;
        vector<unsigned int> entryStart(1, 0), entries;
        vector<unsigned short> slots(numChunks*numAtoms*chunkSize, 0);
        for (int chunk = 0; chunk < numChunks; chunk++) {
            vector<pair<int, int> > atomSlots;
            for (int term = 0; term < chunkSize && chunk*chunkSize+term < numBonds; term++)
                for (int atom = 0; atom < numAtoms; atom++)
                    atomSlots.push_back(make_pair(forceAtoms[force][chunk*chunkSize+term][atom], atom*chunkSize+term));
            sort(atomSlots.begin(), atomSlots.end());
            int last;
            for (int first = 0; first < atomSlots.size(); first = last) {
                for (last = first; last < atomSlots.size() && atomSlots[last].first == atomSlots[first].first; last++)
                    slots[chunk*numAtoms*chunkSize+last] = atomSlots[last].second;
                entries.push_back(atomSlots[first].first);
                entries.push_back(first | (last<<16));
            }
            entryStart.push_back(entries.size()/2);
        }
        if (4*(entries.size()/2) > 3*numBonds*numAtoms)
            continue;
        long long planSize = planData.size()+entryStart.size()+1+entries.size()+slots.size()/2;
        if (planSize*sizeof(unsigned int) > maxPlanBytes)
            continue;
        if (firstChunkedForce == -1)
            firstChunkedForce = force;
        maxChunkedAtoms = max(maxChunkedAtoms, numAtoms);

        // Entries are read as uint2, so they start at an even offset.

        entryStartOffset[force] = planData.size();
        planData.insert(planData.end(), entryStart.begin(), entryStart.end());
        if (planData.size()%2 == 1)
            planData.push_back(0);
        entryOffset[force] = planData.size();
        planData.insert(planData.end(), entries.begin(), entries.end());
        slotOffset[force] = planData.size();
        for (int i = 0; i < slots.size(); i += 2)
            planData.push_back(slots[i] | (slots[i+1]<<16));
    }
    if (firstChunkedForce != -1) {
        plan.initialize<unsigned int>(context, planData.size(), "bondedPlan");
        plan.upload(planData);
        planArgument = addArgument(plan, "unsigned int");
    }
    BondedUtilities::initialize(system);
}

string MetalBondedUtilities::createForceSource(int forceIndex, int numBonds, int numAtoms, int group, const string& computeForce) {
    if (entryStartOffset[forceIndex] == -1)
        return BondedUtilities::createForceSource(forceIndex, numBonds, numAtoms, group, computeForce);
    maxBonds = max(maxBonds, numBonds);
    const int chunkSize = MetalContext::ThreadBlockSize;
    const int padSize = numAtoms*chunkSize;
    string suffix1[] = {""};
    string suffix4[] = {".x", ".y", ".z", ".w"};
    stringstream s;

    // The pad is declared at kernel scope, in the source of the first force that uses it.  It holds
    // each component of the forces as [component][atom][term].

    if (forceIndex == firstChunkedForce)
        s<<"LOCAL real bondedPad["<<3*maxChunkedAtoms*chunkSize<<"];\n";
    s<<"if ((groups&"<<(1<<group)<<") != 0)\n";
    s<<"for (unsigned int chunk = GROUP_ID; chunk < "<<(numBonds+chunkSize-1)/chunkSize<<"; chunk += NUM_GROUPS) {\n";
    s<<"    unsigned int index = chunk*"<<chunkSize<<"+LOCAL_ID;\n";
    s<<"    if (index < "<<numBonds<<") {\n";
    int startAtom = 0;
    for (int i = 0; i < (int) atomIndices[forceIndex].size(); i++) {
        int indexWidth = atomIndices[forceIndex][i].getElementSize()/4;
        string* suffix = (indexWidth == 1 ? suffix1 : suffix4);
        string indexType = (indexWidth == 1 ? "unsigned int" : "uint"+context.intToString(indexWidth));
        s<<"    "<<indexType<<" atoms"<<i<<" = atomIndices"<<forceIndex<<"_"<<i<<"[index];\n";
        int atomsToLoad = min(indexWidth, numAtoms-startAtom);
        for (int j = 0; j < atomsToLoad; j++) {
            s<<"    unsigned int atom"<<(startAtom+j+1)<<" = atoms"<<i<<suffix[j]<<";\n";
            s<<"    real4 pos"<<(startAtom+j+1)<<" = posq[atom"<<(startAtom+j+1)<<"];\n";
        }
        startAtom += indexWidth;
    }
    s<<computeForce<<"\n";
    for (int i = 0; i < numAtoms; i++) {
        s<<"    bondedPad["<<i*chunkSize<<"+LOCAL_ID] = force"<<(i+1)<<".x;\n";
        s<<"    bondedPad["<<padSize+i*chunkSize<<"+LOCAL_ID] = force"<<(i+1)<<".y;\n";
        s<<"    bondedPad["<<2*padSize+i*chunkSize<<"+LOCAL_ID] = force"<<(i+1)<<".z;\n";
    }
    s<<"    }\n";
    s<<"    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    s<<"    GLOBAL const unsigned int* entryStart = "<<planArgument<<"+"<<entryStartOffset[forceIndex]<<";\n";
    s<<"    GLOBAL const uint2* entries = (GLOBAL const uint2*) ("<<planArgument<<"+"<<entryOffset[forceIndex]<<");\n";
    s<<"    GLOBAL const ushort* slots = (GLOBAL const ushort*) ("<<planArgument<<"+"<<slotOffset[forceIndex]<<")+chunk*"<<padSize<<";\n";
    s<<"    for (unsigned int i = entryStart[chunk]+LOCAL_ID; i < entryStart[chunk+1]; i += LOCAL_SIZE) {\n";
    s<<"        uint2 entry = entries[i];\n";
    s<<"        mm_ulong fx = 0, fy = 0, fz = 0;\n";
    s<<"        for (unsigned int j = (entry.y&0xFFFF); j < (entry.y>>16); j++) {\n";
    s<<"            unsigned int slot = slots[j];\n";
    s<<"            fx += (mm_ulong) realToFixedPoint(bondedPad[slot]);\n";
    s<<"            fy += (mm_ulong) realToFixedPoint(bondedPad["<<padSize<<"+slot]);\n";
    s<<"            fz += (mm_ulong) realToFixedPoint(bondedPad["<<2*padSize<<"+slot]);\n";
    s<<"        }\n";
    s<<"        ATOMIC_ADD(&forceBuffer[entry.x], fx);\n";
    s<<"        ATOMIC_ADD(&forceBuffer[entry.x+PADDED_NUM_ATOMS], fy);\n";
    s<<"        ATOMIC_ADD(&forceBuffer[entry.x+PADDED_NUM_ATOMS*2], fz);\n";
    s<<"    }\n";
    s<<"    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    s<<"}\n";
    return s.str();
}
