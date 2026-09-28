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

// Blocks are 64 to 256 atoms.  Larger blocks evaluate fewer terms twice (terms that cross a block edge:
// 3.5-6.1% extra at 128 atoms on the benchmark systems, 6.8-12.2% at 64), but give fewer threadgroups.
// The block doubles while that still leaves MinBlocksPerCore blocks for every GPU core.
static const int MinBlockSize = 64;
static const int MaxBlockSize = 256;
static const int MinBlocksPerCore = 16;

// Metal's limit on a kernel's buffer and value arguments (see MetalContext::createProgram()), and the
// arguments that BondedUtilities::initialize() gives the kernel before any force's.
static const int MaxKernelArguments = 31;
static const int NumFixedArguments = 9;

// Terms in a block are listed by their index, with this bit set when the block is not their home.
static const unsigned int NotHomeBit = 0x80000000u;

void MetalBondedUtilities::initialize(const System& system) {
    // A force keeps the per-term atomics of BondedUtilities if its code could leave the loop over its terms
    // early or return, which would skip the block's flush, or if it accumulates energy derivatives, which
    // a term evaluated in two blocks would count twice.  So does every force when the plan would take the
    // kernel past the argument limit or the plan past 1% of the device's working set.

    MetalContext& cu = dynamic_cast<MetalContext&>(context);
    int numForces = forceAtoms.size();
    int numAtoms = cu.getNumAtoms();
    blockStartOffset.assign(numForces, -1);
    termOffset.assign(numForces, -1);
    blockSource.clear();
    lastBlockForce = -1;
    blockSize = MinBlockSize;
    while (blockSize < MaxBlockSize && numAtoms/(2*blockSize) >= MinBlocksPerCore*cu.getMultiprocessors())
        blockSize *= 2;
    numBlocks = (numAtoms+blockSize-1)/blockSize;
    int numArguments = NumFixedArguments+arguments.size()+(energyParameterDerivatives.size() > 0 ? 1 : 0);
    for (int force = 0; force < numForces; force++)
        numArguments += (forceAtoms[force][0].size()+3)/4;
    long long maxPlanBytes = cu.getDevice()->recommendedMaxWorkingSetSize()/100;
    vector<unsigned int> planData;
    for (int force = 0; force < numForces && numArguments < MaxKernelArguments; force++) {
        const string& source = forceSource[force];
        if (source.find("continue") != string::npos || source.find("break") != string::npos || source.find("return") != string::npos || source.find("energyParamDeriv") != string::npos)
            continue;

        // List the terms touching each block, block by block, in their original order.

        const vector<vector<int> >& atoms = forceAtoms[force];
        int numTerms = atoms.size();
        vector<unsigned int> blockStart(numBlocks+1, 0);
        vector<int> termBlocks, termFirst(numTerms+1, 0);
        for (int term = 0; term < numTerms; term++) {
            int first = termBlocks.size();
            for (int atom : atoms[term]) {
                int block = atom/blockSize;
                if (find(termBlocks.begin()+first, termBlocks.end(), block) == termBlocks.end())
                    termBlocks.push_back(block);
            }
            for (int i = first; i < termBlocks.size(); i++)
                blockStart[termBlocks[i]+1]++;
            termFirst[term+1] = termBlocks.size();
        }
        for (int block = 0; block < numBlocks; block++)
            blockStart[block+1] += blockStart[block];
        int numEvaluations = blockStart[numBlocks];
        long long planSize = planData.size()+numBlocks+1+numEvaluations;
        if (planSize*sizeof(unsigned int) > maxPlanBytes)
            continue;
        vector<unsigned int> terms(numEvaluations);
        vector<vector<int> > evaluationAtoms(numEvaluations);
        vector<unsigned int> next(blockStart.begin(), blockStart.end()-1);
        for (int term = 0; term < numTerms; term++) {
            int home = atoms[term][0]/blockSize;
            for (int i = termFirst[term]; i < termFirst[term+1]; i++) {
                int block = termBlocks[i];
                int position = next[block]++;
                terms[position] = term | (block == home ? 0 : NotHomeBit);
                evaluationAtoms[position] = atoms[term];
            }
        }
        blockStartOffset[force] = planData.size();
        planData.insert(planData.end(), blockStart.begin(), blockStart.end());
        termOffset[force] = planData.size();
        planData.insert(planData.end(), terms.begin(), terms.end());
        forceAtoms[force] = evaluationAtoms;
        lastBlockForce = force;
    }
    if (lastBlockForce != -1) {
        plan.initialize<unsigned int>(context, planData.size(), "bondedPlan");
        plan.upload(planData);
        planArgument = addArgument(plan, "unsigned int");
        addPrefixCode(
            "inline void addBlockForce(threadgroup atomic_uint* sum, mm_long value) {\n"
            "    unsigned int lower = (unsigned int) value;\n"
            "    unsigned int upper = (unsigned int) (((mm_ulong) value) >> 32);\n"
            "    unsigned int oldLower = atomic_fetch_add_explicit(&sum[0], lower, memory_order_relaxed);\n"
            "    if (oldLower+lower < oldLower)\n"
            "        upper++;\n"
            "    if (upper != 0)\n"
            "        atomic_fetch_add_explicit(&sum[1], upper, memory_order_relaxed);\n"
            "}\n");
    }
    BondedUtilities::initialize(system);
}

string MetalBondedUtilities::createForceSource(int forceIndex, int numBonds, int numAtoms, int group, const string& computeForce) {
    if (blockStartOffset[forceIndex] == -1)
        return BondedUtilities::createForceSource(forceIndex, numBonds, numAtoms, group, computeForce);
    maxBonds = max(maxBonds, numBlocks*MetalContext::ThreadBlockSize);
    blockSource.push_back(createBlockSource(forceIndex, numAtoms, group, computeForce));
    if (forceIndex != lastBlockForce)
        return "";

    // All forces that use blocks share one loop over the blocks, placed where the last of them would go.
    // The sums hold each owned atom's x, y and z force as the low and high words of 64-bit fixed point.

    stringstream s;
    s<<"LOCAL atomic_uint blockSums["<<6*blockSize<<"];\n";
    s<<"for (unsigned int i = LOCAL_ID; i < "<<6*blockSize<<"; i += LOCAL_SIZE)\n";
    s<<"    atomic_store_explicit(&blockSums[i], 0u, memory_order_relaxed);\n";
    s<<"threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    s<<"for (unsigned int block = GROUP_ID; block < "<<numBlocks<<"; block += NUM_GROUPS) {\n";
    s<<"    const unsigned int firstAtom = block*"<<blockSize<<";\n";
    for (const string& source : blockSource)
        s<<source;
    s<<"    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    s<<"    for (unsigned int i = LOCAL_ID; i < "<<blockSize<<"; i += LOCAL_SIZE) {\n";
    s<<"        for (int component = 0; component < 3; component++) {\n";
    s<<"            threadgroup atomic_uint* sum = &blockSums[6*i+2*component];\n";
    // Read and clear with an exchange: the Apple8 (G14) compiler service crashes on atomic_load_explicit
    // of a threadgroup atomic_uint.
    s<<"            mm_ulong total = atomic_exchange_explicit(&sum[0], 0u, memory_order_relaxed) | (((mm_ulong) atomic_exchange_explicit(&sum[1], 0u, memory_order_relaxed)) << 32);\n";
    s<<"            if (total != 0)\n";
    s<<"                ATOMIC_ADD(&forceBuffer[firstAtom+i+component*PADDED_NUM_ATOMS], total);\n";
    s<<"        }\n";
    s<<"    }\n";
    s<<"    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    s<<"}\n";
    return s.str();
}

string MetalBondedUtilities::createBlockSource(int forceIndex, int numAtoms, int group, const string& computeForce) {
    string suffix1[] = {""};
    string suffix4[] = {".x", ".y", ".z", ".w"};
    stringstream s;
    s<<"    if ((groups&"<<(1<<group)<<") != 0)\n";
    s<<"    for (unsigned int position = "<<planArgument<<"["<<blockStartOffset[forceIndex]<<"+block]+LOCAL_ID; position < "<<planArgument<<"["<<blockStartOffset[forceIndex]<<"+block+1]; position += LOCAL_SIZE) {\n";
    s<<"    unsigned int term = "<<planArgument<<"["<<termOffset[forceIndex]<<"+position];\n";
    s<<"    unsigned int index = term & "<<(~NotHomeBit)<<"u;\n";
    int startAtom = 0;
    for (int i = 0; i < (int) atomIndices[forceIndex].size(); i++) {
        int indexWidth = atomIndices[forceIndex][i].getElementSize()/4;
        string* suffix = (indexWidth == 1 ? suffix1 : suffix4);
        string indexType = (indexWidth == 1 ? "unsigned int" : "uint"+context.intToString(indexWidth));
        s<<"    "<<indexType<<" atoms"<<i<<" = atomIndices"<<forceIndex<<"_"<<i<<"[position];\n";
        int atomsToLoad = min(indexWidth, numAtoms-startAtom);
        for (int j = 0; j < atomsToLoad; j++) {
            s<<"    unsigned int atom"<<(startAtom+j+1)<<" = atoms"<<i<<suffix[j]<<";\n";
            s<<"    real4 pos"<<(startAtom+j+1)<<" = posq[atom"<<(startAtom+j+1)<<"];\n";
        }
        startAtom += indexWidth;
    }
    s<<"    mixed energyBeforeTerm = energy;\n";
    s<<computeForce<<"\n";
    s<<"    if (term >= "<<NotHomeBit<<"u)\n";
    s<<"        energy = energyBeforeTerm;\n";
    for (int i = 0; i < numAtoms; i++) {
        s<<"    if (atom"<<(i+1)<<"-firstAtom < "<<blockSize<<") {\n";
        s<<"        threadgroup atomic_uint* sum = &blockSums[6*(atom"<<(i+1)<<"-firstAtom)];\n";
        s<<"        addBlockForce(&sum[0], realToFixedPoint(force"<<(i+1)<<".x));\n";
        s<<"        addBlockForce(&sum[2], realToFixedPoint(force"<<(i+1)<<".y));\n";
        s<<"        addBlockForce(&sum[4], realToFixedPoint(force"<<(i+1)<<".z));\n";
        s<<"    }\n";
    }
    s<<"    }\n";
    return s.str();
}
