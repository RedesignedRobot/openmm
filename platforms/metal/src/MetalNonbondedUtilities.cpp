/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2009-2026 Stanford University and the Authors.      *
 * Portions copyright (c) 2026 the Authors.                                   *
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

#include "openmm/OpenMMException.h"
#include "MetalNonbondedUtilities.h"
#include "MetalArray.h"
#include "MetalContext.h"
#include "MetalKernelSources.h"
#include "MetalQueue.h"
#include "openmm/common/CommonKernelUtilities.h"
#include <Metal/Metal.hpp>
#include <algorithm>
#include <map>
#include <set>
#include <utility>

using namespace OpenMM;
using namespace std;

class MetalNonbondedUtilities::BlockSortTrait : public ComputeSortImpl::SortTrait {
public:
    BlockSortTrait() {}
    int getDataSize() const {return sizeof(int);}
    int getKeySize() const {return sizeof(int);}
    const char* getDataType() const {return "unsigned int";}
    const char* getKeyType() const {return "unsigned int";}
    const char* getMinKey() const {return "0";}
    const char* getMaxKey() const {return "0xFFFFFFFFu";}
    const char* getMaxValue() const {return "0xFFFFFFFFu";}
    const char* getSortKey() const {return "value";}
};

MetalNonbondedUtilities::MetalNonbondedUtilities(MetalContext& context) : context(context), useCutoff(false), usePeriodic(false), anyExclusions(false), usePadding(true), useNeighborList(false),
        forceRebuildNeighborList(true), groupFlags(0), tilesAfterReorder(0) {
    // Decide how many thread blocks to use.  Each GPU core can hold 1536 threads.

    numForceThreadBlocks = 6*context.getMultiprocessors();
    forceThreadBlockSize = 256;
    downloadCountEvent = context.createEvent();

    // When building the neighbor list, we can optionally use large blocks (1024 atoms) to
    // accelerate the process.  This makes building the neighbor list faster, but it prevents
    // us from sorting atom blocks by size, which leads to a slightly less efficient neighbor
    // list.  We guess based on system size which will be faster.

    useLargeBlocks = (context.getNumAtoms() > 100000);
    setKernelSource(MetalKernelSources::nonbonded);
}

MetalNonbondedUtilities::~MetalNonbondedUtilities() {
}

void MetalNonbondedUtilities::addInteraction(bool usesCutoff, bool usesPeriodic, bool usesExclusions, double cutoffDistance,
            const vector<vector<int> >& exclusionList, const string& kernel, int forceGroup, bool useNeighborList, bool supportsPairList) {
    if (groupCutoff.size() > 0) {
        if (usesCutoff != useCutoff)
            throw OpenMMException("All Forces must agree on whether to use a cutoff");
        if (usesPeriodic != usePeriodic)
            throw OpenMMException("All Forces must agree on whether to use periodic boundary conditions");
        if (usesCutoff && groupCutoff.find(forceGroup) != groupCutoff.end() && groupCutoff[forceGroup] != cutoffDistance)
            throw OpenMMException("All Forces in a single force group must use the same cutoff distance");
    }
    if (usesExclusions)
        requestExclusions(exclusionList);
    useCutoff = usesCutoff;
    usePeriodic = usesPeriodic;
    this->useNeighborList |= (useNeighborList && useCutoff);
    groupCutoff[forceGroup] = cutoffDistance;
    groupFlags |= 1<<forceGroup;
    if (kernel.size() > 0) {
        if (groupKernelSource.find(forceGroup) == groupKernelSource.end())
            groupKernelSource[forceGroup] = "";
        map<string, string> replacements;
        replacements["CUTOFF"] = "CUTOFF_"+context.intToString(forceGroup);
        replacements["CUTOFF_SQUARED"] = "CUTOFF_"+context.intToString(forceGroup)+"_SQUARED";
        groupKernelSource[forceGroup] += context.replaceStrings(kernel, replacements)+"\n";
    }
}

void MetalNonbondedUtilities::addParameter(ComputeParameterInfo parameter) {
    parameters.push_back(parameter);
}

void MetalNonbondedUtilities::addArgument(ComputeParameterInfo parameter) {
    arguments.push_back(parameter);
}

string MetalNonbondedUtilities::addEnergyParameterDerivative(const string& param) {
    // See if the parameter has already been added.
    
    int index;
    for (index = 0; index < energyParameterDerivatives.size(); index++)
        if (param == energyParameterDerivatives[index])
            break;
    if (index == energyParameterDerivatives.size())
        energyParameterDerivatives.push_back(param);
    context.addEnergyParameterDerivative(param);
    return string("energyParamDeriv")+context.intToString(index);
}

void MetalNonbondedUtilities::requestExclusions(const vector<vector<int> >& exclusionList) {
    if (anyExclusions) {
        bool sameExclusions = (exclusionList.size() == atomExclusions.size());
        for (int i = 0; i < (int) exclusionList.size() && sameExclusions; i++) {
            if (exclusionList[i].size() != atomExclusions[i].size())
                sameExclusions = false;
            set<int> expectedExclusions;
            expectedExclusions.insert(atomExclusions[i].begin(), atomExclusions[i].end());
            for (int j = 0; j < (int) exclusionList[i].size(); j++)
                if (expectedExclusions.find(exclusionList[i][j]) == expectedExclusions.end())
                    sameExclusions = false;
        }
        if (!sameExclusions)
            throw OpenMMException("All Forces must have identical exceptions");
    }
    else {
        atomExclusions = exclusionList;
        anyExclusions = true;
    }
}

static bool compareInt2(mm_int2 a, mm_int2 b) {
    // Sort tiles to improve cache efficiency.

    return ((a.y < b.y) || (a.y == b.y && a.x < b.x));
}

void MetalNonbondedUtilities::initialize(const System& system) {
    if (atomExclusions.size() == 0) {
        // No exclusions were specifically requested, so just mark every atom as not interacting with itself.

        atomExclusions.resize(context.getNumAtoms());
        for (int i = 0; i < (int) atomExclusions.size(); i++)
            atomExclusions[i].push_back(i);
    }

    // Create the list of tiles.

    int numAtomBlocks = context.getNumAtomBlocks();
    int numContexts = context.getPlatformData().contexts.size();
    setAtomBlockRange(context.getContextIndex()/(double) numContexts, (context.getContextIndex()+1)/(double) numContexts);

    // Build a list of tiles that contain exclusions.

    set<pair<int, int> > tilesWithExclusions;
    for (int atom1 = 0; atom1 < (int) atomExclusions.size(); ++atom1) {
        int x = atom1/MetalContext::TileSize;
        for (int j = 0; j < (int) atomExclusions[atom1].size(); ++j) {
            int atom2 = atomExclusions[atom1][j];
            int y = atom2/MetalContext::TileSize;
            tilesWithExclusions.insert(make_pair(max(x, y), min(x, y)));
        }
    }
    vector<mm_int2> exclusionTilesVec;
    for (set<pair<int, int> >::const_iterator iter = tilesWithExclusions.begin(); iter != tilesWithExclusions.end(); ++iter)
        exclusionTilesVec.push_back(mm_int2(iter->first, iter->second));
    sort(exclusionTilesVec.begin(), exclusionTilesVec.end(), compareInt2);
    exclusionTiles.initialize<mm_int2>(context, exclusionTilesVec.size(), "exclusionTiles");
    exclusionTiles.upload(exclusionTilesVec);
    map<pair<int, int>, int> exclusionTileMap;
    for (int i = 0; i < (int) exclusionTilesVec.size(); i++) {
        mm_int2 tile = exclusionTilesVec[i];
        exclusionTileMap[make_pair(tile.x, tile.y)] = i;
    }
    vector<vector<int> > exclusionBlocksForBlock(numAtomBlocks);
    for (set<pair<int, int> >::const_iterator iter = tilesWithExclusions.begin(); iter != tilesWithExclusions.end(); ++iter) {
        exclusionBlocksForBlock[iter->first].push_back(iter->second);
        if (iter->first != iter->second)
            exclusionBlocksForBlock[iter->second].push_back(iter->first);
    }
    vector<unsigned int> exclusionRowIndicesVec(numAtomBlocks+1, 0);
    vector<unsigned int> exclusionIndicesVec;
    for (int i = 0; i < numAtomBlocks; i++) {
        exclusionIndicesVec.insert(exclusionIndicesVec.end(), exclusionBlocksForBlock[i].begin(), exclusionBlocksForBlock[i].end());
        exclusionRowIndicesVec[i+1] = exclusionIndicesVec.size();
    }
    maxExclusions = 0;
    for (int i = 0; i < (int) exclusionBlocksForBlock.size(); i++)
        maxExclusions = (maxExclusions > exclusionBlocksForBlock[i].size() ? maxExclusions : exclusionBlocksForBlock[i].size());
    exclusionIndices.initialize<unsigned int>(context, exclusionIndicesVec.size(), "exclusionIndices");
    exclusionRowIndices.initialize<unsigned int>(context, exclusionRowIndicesVec.size(), "exclusionRowIndices");
    exclusionIndices.upload(exclusionIndicesVec);
    exclusionRowIndices.upload(exclusionRowIndicesVec);

    // Record the exclusion data.

    exclusions.initialize<unsigned int>(context, tilesWithExclusions.size()*MetalContext::TileSize, "exclusions");
    unsigned int allFlags = (unsigned int) -1;
    vector<unsigned int> exclusionVec(exclusions.getSize(), allFlags);
    for (int i = 0; i < exclusions.getSize(); ++i)
        exclusionVec[i] = 0xFFFFFFFF;
    for (int atom1 = 0; atom1 < (int) atomExclusions.size(); ++atom1) {
        int x = atom1/MetalContext::TileSize;
        int offset1 = atom1-x*MetalContext::TileSize;
        for (int j = 0; j < (int) atomExclusions[atom1].size(); ++j) {
            int atom2 = atomExclusions[atom1][j];
            int y = atom2/MetalContext::TileSize;
            int offset2 = atom2-y*MetalContext::TileSize;
            if (x > y) {
                int index = exclusionTileMap[make_pair(x, y)]*MetalContext::TileSize;
                exclusionVec[index+offset1] &= allFlags-(1<<offset2);
            }
            else {
                int index = exclusionTileMap[make_pair(y, x)]*MetalContext::TileSize;
                exclusionVec[index+offset2] &= allFlags-(1<<offset1);
            }
        }
    }
    atomExclusions.clear(); // We won't use this again, so free the memory it used
    exclusions.upload(exclusionVec);

    // Create data structures for the neighbor list.

    maxCutoff = getMaxCutoffDistance();
    if (useCutoff) {
        // Select a size for the arrays that hold the neighbor list.  We have to make a fairly
        // arbitrary guess, but if this turns out to be too small we'll increase it later.

        int maxTiles = 20*numAtomBlocks;
        if (maxTiles > numTiles)
            maxTiles = numTiles;
        if (maxTiles < 1)
            maxTiles = 1;
        int numAtoms = context.getNumAtoms();
        interactingTiles.initialize<int>(context, maxTiles, "interactingTiles");
        interactingAtoms.initialize<int>(context, MetalContext::TileSize*maxTiles, "interactingAtoms");
        interactionCount.initialize<unsigned int>(context, 1, "interactionCount");
        int elementSize = sizeof(float);
        blockCenter.initialize(context, numAtomBlocks, 4*elementSize, "blockCenter");
        blockBoundingBox.initialize(context, numAtomBlocks, 4*elementSize, "blockBoundingBox");
        sortedBlocks.initialize<unsigned int>(context, numAtomBlocks, "sortedBlocks");
        sortedBlockCenter.initialize(context, numAtomBlocks+1, 4*elementSize, "sortedBlockCenter");
        sortedBlockBoundingBox.initialize(context, numAtomBlocks+1, 4*elementSize, "sortedBlockBoundingBox");
        numBlockSizes = min((context.getNumAtomBlocks()+63)/64, context.getNumThreadBlocks());
        blockSizeRange.initialize(context, numBlockSizes, 2*elementSize, "blockSizeRange");
        largeBlockCenter.initialize(context, numAtomBlocks, 4*elementSize, "largeBlockCenter");
        largeBlockBoundingBox.initialize(context, numAtomBlocks, 4*elementSize, "largeBlockBoundingBox");
        oldPositions.initialize(context, numAtoms, 4*elementSize, "oldPositions");
        rebuildNeighborList.initialize<int>(context, 1, "rebuildNeighborList");
        blockSorter = context.createSort(new BlockSortTrait(), numAtomBlocks, false);
        vector<unsigned int> count(1, 0);
        interactionCount.upload(count);
        rebuildNeighborList.upload(count);
    }
}

double MetalNonbondedUtilities::getMaxCutoffDistance() {
    double cutoff = 0.0;
    for (map<int, double>::const_iterator iter = groupCutoff.begin(); iter != groupCutoff.end(); ++iter)
        cutoff = max(cutoff, iter->second);
    return cutoff;
}

double MetalNonbondedUtilities::padCutoff(double cutoff) {
    double padding = (usePadding ? 0.1*cutoff : 0.0);
    return cutoff+padding;
}

void MetalNonbondedUtilities::prepareInteractions(int forceGroups) {
    if ((forceGroups&groupFlags) == 0)
        return;
    if (groupKernels.find(forceGroups) == groupKernels.end())
        createKernelsForGroups(forceGroups);
    KernelSet& kernels = groupKernels[forceGroups];
    if (useCutoff && usePeriodic) {
        Vec3 a, b, c;
        context.getPeriodicBoxVectors(a, b, c);
        double minAllowedSize = 1.999999*maxCutoff;
        if (a[0] < minAllowedSize || b[1] < minAllowedSize || c[2] < minAllowedSize)
            throw OpenMMException("The periodic box size has decreased to less than twice the nonbonded cutoff.");
    }
    if (!useNeighborList)
        return;
    if (numTiles == 0)
        return;

    // Compute the neighbor list.

    setPeriodicBoxArgs(context, kernels.findBlockBoundsKernel, 1);
    kernels.findBlockBoundsKernel->execute(context.getNumAtomBlocks());
    kernels.computeSortKeysKernel->execute(context.getNumAtomBlocks());
    if (useLargeBlocks)
        setPeriodicBoxArgs(context, kernels.sortBoxDataKernel, 12);
    blockSorter->sort(sortedBlocks);
    kernels.sortBoxDataKernel->setArg(9, (int) forceRebuildNeighborList);
    kernels.sortBoxDataKernel->execute(context.getNumAtoms());
    setPeriodicBoxArgs(context, kernels.findInteractingBlocksKernel, 0);
    kernels.findInteractingBlocksKernel->execute(context.getNumAtoms(), interactingBlocksThreadBlockSize);
    forceRebuildNeighborList = false;

    // Submitting the event also submits the neighbor list kernels, so the GPU starts on them
    // while the host encodes the rest of the step.

    downloadCountEvent->enqueue();
}

void MetalNonbondedUtilities::computeInteractions(int forceGroups, bool includeForces, bool includeEnergy) {
    if ((forceGroups&groupFlags) == 0)
        return;
    KernelSet& kernels = groupKernels[forceGroups];
    if (kernels.hasForces && (includeForces || includeEnergy)) {
        ComputeKernel& kernel = (includeForces ? (includeEnergy ? kernels.forceEnergyKernel : kernels.forceKernel) : kernels.energyKernel);
        if (kernel.get() == NULL)
            kernel = createInteractionKernel(kernels.source, parameters, arguments, true, true, forceGroups, includeForces, includeEnergy);
        if (useCutoff)
            setPeriodicBoxArgs(context, kernel, 9);
        kernel->execute(numForceThreadBlocks*forceThreadBlockSize, forceThreadBlockSize);
    }
    if (useNeighborList && numTiles > 0) {
        // Submit the nonbonded kernel so the GPU works on it while we wait for the count.

        if (kernels.hasForces)
            context.getCurrentStream().commit();
        downloadCountEvent->wait();
        updateNeighborListSize();
    }
}

bool MetalNonbondedUtilities::updateNeighborListSize() {
    if (!useCutoff)
        return false;

    // The count is in shared memory, and the event guarantees the kernel that wrote it has finished.

    unsigned int count = *(unsigned int*) interactionCount.getDevicePointer()->contents();
    if (context.getStepsSinceReorder() == 0 || tilesAfterReorder == 0)
        tilesAfterReorder = count;
    else if (context.getStepsSinceReorder() > 25 && count > 1.1*tilesAfterReorder)
        context.forceReorder();
    if (count <= interactingTiles.getSize())
        return false;

    // The most recent timestep had too many interactions to fit in the arrays.  Make the arrays bigger to prevent
    // this from happening in the future.

    unsigned int maxTiles = (unsigned int) (1.2*count);
    unsigned int numBlocks = context.getNumAtomBlocks();
    int totalTiles = numBlocks*(numBlocks+1)/2;
    if (maxTiles > totalTiles)
        maxTiles = totalTiles;
    interactingTiles.resize(maxTiles);
    interactingAtoms.resize(MetalContext::TileSize*(size_t) maxTiles);
    for (map<int, KernelSet>::iterator iter = groupKernels.begin(); iter != groupKernels.end(); ++iter) {
        KernelSet& kernels = iter->second;
        if (kernels.forceKernel.get() != NULL) {
            kernels.forceKernel->setArg(7, interactingTiles);
            kernels.forceKernel->setArg(14, maxTiles);
            kernels.forceKernel->setArg(17, interactingAtoms);
        }
        if (kernels.energyKernel.get() != NULL) {
            kernels.energyKernel->setArg(7, interactingTiles);
            kernels.energyKernel->setArg(14, maxTiles);
            kernels.energyKernel->setArg(17, interactingAtoms);
        }
        if (kernels.forceEnergyKernel.get() != NULL) {
            kernels.forceEnergyKernel->setArg(7, interactingTiles);
            kernels.forceEnergyKernel->setArg(14, maxTiles);
            kernels.forceEnergyKernel->setArg(17, interactingAtoms);
        }
        kernels.findInteractingBlocksKernel->setArg(6, interactingTiles);
        kernels.findInteractingBlocksKernel->setArg(7, interactingAtoms);
        kernels.findInteractingBlocksKernel->setArg(9, maxTiles);
    }
    forceRebuildNeighborList = true;
    context.setForcesValid(false);
    return true;
}

void MetalNonbondedUtilities::setUsePadding(bool padding) {
    usePadding = padding;
}

void MetalNonbondedUtilities::setAtomBlockRange(double startFraction, double endFraction) {
    int numAtomBlocks = context.getNumAtomBlocks();
    startBlockIndex = (int) (startFraction*numAtomBlocks);
    numBlocks = (int) (endFraction*numAtomBlocks)-startBlockIndex;
    long long totalTiles = context.getNumAtomBlocks()*((long long)context.getNumAtomBlocks()+1)/2;
    startTileIndex = (int) (startFraction*totalTiles);;
    numTiles = (long long) (endFraction*totalTiles)-startTileIndex;
    if (useCutoff) {
        // We are using a cutoff, and the kernels have already been created.

        for (map<int, KernelSet>::iterator iter = groupKernels.begin(); iter != groupKernels.end(); ++iter) {
            KernelSet& kernels = iter->second;
            if (kernels.forceKernel.get() != NULL) {
                kernels.forceKernel->setArg(5, (unsigned int) startTileIndex);
                kernels.forceKernel->setArg(6, (unsigned long long) numTiles);
            }
            if (kernels.energyKernel.get() != NULL) {
                kernels.energyKernel->setArg(5, (unsigned int) startTileIndex);
                kernels.energyKernel->setArg(6, (unsigned long long) numTiles);
            }
            if (kernels.forceEnergyKernel.get() != NULL) {
                kernels.forceEnergyKernel->setArg(5, (unsigned int) startTileIndex);
                kernels.forceEnergyKernel->setArg(6, (unsigned long long) numTiles);
            }
            kernels.findInteractingBlocksKernel->setArg(10, (unsigned int) startBlockIndex);
            kernels.findInteractingBlocksKernel->setArg(11, (unsigned int) numBlocks);
        }
        forceRebuildNeighborList = true;
    }
}

void MetalNonbondedUtilities::createKernelsForGroups(int groups) {
    KernelSet kernels;
    string source;
    for (int i = 0; i < 32; i++) {
        if ((groups&(1<<i)) != 0) {
            source += groupKernelSource[i];
        }
    }
    kernels.hasForces = (source.size() > 0);
    kernels.source = source;
    if (useCutoff) {
        double paddedCutoff = padCutoff(maxCutoff);
        map<string, string> defines;
        defines["TILE_SIZE"] = context.intToString(MetalContext::TileSize);
        defines["NUM_ATOMS"] = context.intToString(context.getNumAtoms());
        defines["PADDING"] = context.doubleToString(paddedCutoff-maxCutoff);
        defines["PADDED_CUTOFF"] = context.doubleToString(paddedCutoff);
        defines["PADDED_CUTOFF_SQUARED"] = context.doubleToString(paddedCutoff*paddedCutoff);
        defines["NUM_TILES_WITH_EXCLUSIONS"] = context.intToString(exclusionTiles.getSize());
        defines["NUM_BLOCKS"] = context.intToString(context.getNumAtomBlocks());
        defines["SIMD_WIDTH"] = context.intToString(context.getSIMDWidth());
        if (usePeriodic)
            defines["USE_PERIODIC"] = "1";
        if (context.getBoxIsTriclinic())
            defines["TRICLINIC"] = "1";
        if (useLargeBlocks)
            defines["USE_LARGE_BLOCKS"] = "1";
        defines["MAX_EXCLUSIONS"] = context.intToString(maxExclusions);
        int binShift = 1;
        while (1<<binShift <= context.getNumAtomBlocks())
            binShift++;
        defines["BIN_SHIFT"] = context.intToString(binShift);
        defines["BLOCK_INDEX_MASK"] = context.intToString((1<<binShift)-1);
        int groupSize = 256;
        while (true) {
            defines["GROUP_SIZE"] = context.intToString(groupSize);
            ComputeProgram program = context.compileProgram(MetalKernelSources::findInteractingBlocks, defines);

            // Metal binds arguments by position, so every argument is added in order.  Empty
            // arguments are set before each execution.

            kernels.findBlockBoundsKernel = program->createKernel("findBlockBounds");
            kernels.findBlockBoundsKernel->addArg(context.getNumAtoms());
            for (int i = 0; i < 5; i++)
                kernels.findBlockBoundsKernel->addArg();
            kernels.findBlockBoundsKernel->addArg(context.getPosq());
            kernels.findBlockBoundsKernel->addArg(blockCenter);
            kernels.findBlockBoundsKernel->addArg(blockBoundingBox);
            kernels.findBlockBoundsKernel->addArg(rebuildNeighborList);
            kernels.findBlockBoundsKernel->addArg(blockSizeRange);
            kernels.computeSortKeysKernel = program->createKernel("computeSortKeys");
            kernels.computeSortKeysKernel->addArg(blockBoundingBox);
            kernels.computeSortKeysKernel->addArg(sortedBlocks);
            kernels.computeSortKeysKernel->addArg(blockSizeRange);
            kernels.computeSortKeysKernel->addArg(numBlockSizes);
            kernels.sortBoxDataKernel = program->createKernel("sortBoxData");
            kernels.sortBoxDataKernel->addArg(sortedBlocks);
            kernels.sortBoxDataKernel->addArg(blockCenter);
            kernels.sortBoxDataKernel->addArg(blockBoundingBox);
            kernels.sortBoxDataKernel->addArg(sortedBlockCenter);
            kernels.sortBoxDataKernel->addArg(sortedBlockBoundingBox);
            kernels.sortBoxDataKernel->addArg(context.getPosq());
            kernels.sortBoxDataKernel->addArg(oldPositions);
            kernels.sortBoxDataKernel->addArg(interactionCount);
            kernels.sortBoxDataKernel->addArg(rebuildNeighborList);
            kernels.sortBoxDataKernel->addArg((int) true);
            if (useLargeBlocks) {
                kernels.sortBoxDataKernel->addArg(largeBlockCenter);
                kernels.sortBoxDataKernel->addArg(largeBlockBoundingBox);
                for (int i = 0; i < 5; i++)
                    kernels.sortBoxDataKernel->addArg();
            }
            kernels.findInteractingBlocksKernel = program->createKernel("findBlocksWithInteractions");
            for (int i = 0; i < 5; i++)
                kernels.findInteractingBlocksKernel->addArg();
            kernels.findInteractingBlocksKernel->addArg(interactionCount);
            kernels.findInteractingBlocksKernel->addArg(interactingTiles);
            kernels.findInteractingBlocksKernel->addArg(interactingAtoms);
            kernels.findInteractingBlocksKernel->addArg(context.getPosq());
            kernels.findInteractingBlocksKernel->addArg((unsigned int) interactingTiles.getSize());
            kernels.findInteractingBlocksKernel->addArg((unsigned int) startBlockIndex);
            kernels.findInteractingBlocksKernel->addArg((unsigned int) numBlocks);
            kernels.findInteractingBlocksKernel->addArg(sortedBlocks);
            kernels.findInteractingBlocksKernel->addArg(sortedBlockCenter);
            kernels.findInteractingBlocksKernel->addArg(sortedBlockBoundingBox);
            kernels.findInteractingBlocksKernel->addArg(exclusionIndices);
            kernels.findInteractingBlocksKernel->addArg(exclusionRowIndices);
            kernels.findInteractingBlocksKernel->addArg(oldPositions);
            kernels.findInteractingBlocksKernel->addArg(rebuildNeighborList);
            if (useLargeBlocks) {
                kernels.findInteractingBlocksKernel->addArg(largeBlockCenter);
                kernels.findInteractingBlocksKernel->addArg(largeBlockBoundingBox);
            }
            if (kernels.findInteractingBlocksKernel->getMaxBlockSize() < groupSize) {
                // The device can't handle this block size, so reduce it.

                groupSize -= 32;
                if (groupSize < 32)
                    throw OpenMMException("Failed to create findInteractingBlocks kernel");
                continue;
            }
            break;
        }
        interactingBlocksThreadBlockSize = groupSize;
    }
    groupKernels[groups] = kernels;
}

ComputeKernel MetalNonbondedUtilities::createInteractionKernel(const string& source, vector<ComputeParameterInfo>& params, vector<ComputeParameterInfo>& arguments, bool useExclusions, bool isSymmetric, int groups, bool includeForces, bool includeEnergy) {
    map<string, string> replacements;
    replacements["COMPUTE_INTERACTION"] = source;
    const string suffixes[] = {"x", "y", "z", "w"};
    stringstream localData;
    int localDataSize = 0;
    for (const ComputeParameterInfo& param : params) {
        if (param.getNumComponents() == 1)
            localData<<param.getType()<<" "<<param.getName()<<";\n";
        else {
            for (int j = 0; j < param.getNumComponents(); ++j)
                localData<<param.getComponentType()<<" "<<param.getName()<<"_"<<suffixes[j]<<";\n";
        }
        localDataSize += param.getSize();
    }
    replacements["ATOM_PARAMETER_DATA"] = localData.str();
    stringstream args;
    for (const ComputeParameterInfo& param : params) {
        args << ", GLOBAL ";
        if (param.isConstant())
            args << "const ";
        if (param.getNumComponents() == 3)
            args << param.getComponentType();
        else
            args << param.getType();
        args << "* RESTRICT global_";
        args << param.getName();
    }
    for (const ComputeParameterInfo& arg : arguments) {
        args << ", GLOBAL ";
        if (arg.isConstant())
            args << "const ";
        args << arg.getType();
        args << "* RESTRICT ";
        args << arg.getName();
    }
    if (energyParameterDerivatives.size() > 0)
        args << ", GLOBAL mixed* RESTRICT energyParamDerivs";
    replacements["PARAMETER_ARGUMENTS"] = args.str();
    stringstream loadLocal1;
    for (const ComputeParameterInfo& param : params) {
        if (param.getNumComponents() == 1) {
            loadLocal1<<"localData[localAtomIndex]."<<param.getName()<<" = "<<param.getName()<<"1;\n";
        }
        else {
            for (int j = 0; j < param.getNumComponents(); ++j)
                loadLocal1<<"localData[localAtomIndex]."<<param.getName()<<"_"<<suffixes[j]<<" = "<<param.getName()<<"1."<<suffixes[j]<<";\n";
        }
    }
    replacements["LOAD_LOCAL_PARAMETERS_FROM_1"] = loadLocal1.str();
    replacements["DECLARE_LOCAL_PARAMETERS"] = "";
    stringstream loadLocal2;
    for (const ComputeParameterInfo& param : params) {
        if (param.getNumComponents() == 1) {
            loadLocal2<<"localData[localAtomIndex]."<<param.getName()<<" = global_"<<param.getName()<<"[j];\n";
        }
        else {
            if (param.getNumComponents() == 3)
                loadLocal2<<param.getType()<<" temp_"<<param.getName()<<" = make_"<<param.getType()<<"(global_"<<param.getName()<<"[3*j], global_"<<param.getName()<<"[3*j+1], global_"<<param.getName()<<"[3*j+2]);\n";
            else
                loadLocal2<<param.getType()<<" temp_"<<param.getName()<<" = global_"<<param.getName()<<"[j];\n";
            for (int j = 0; j < param.getNumComponents(); ++j)
                loadLocal2<<"localData[localAtomIndex]."<<param.getName()<<"_"<<suffixes[j]<<" = temp_"<<param.getName()<<"."<<suffixes[j]<<";\n";
        }
    }
    replacements["LOAD_LOCAL_PARAMETERS_FROM_GLOBAL"] = loadLocal2.str();
    stringstream load1;
    for (const ComputeParameterInfo& param : params) {
        load1<<param.getType()<<" "<<param.getName()<<"1 = ";
        if (param.getNumComponents() == 3)
            load1<<"make_"<<param.getType()<<"(global_"<<param.getName()<<"[3*atom1], global_"<<param.getName()<<"[3*atom1+1], global_"<<param.getName()<<"[3*atom1+2]);\n";
        else
            load1<<"global_"<<param.getName()<<"[atom1];\n";
    }
    replacements["LOAD_ATOM1_PARAMETERS"] = load1.str();
    stringstream load2j;
    for (const ComputeParameterInfo& param : params) {
        if (param.getNumComponents() == 1) {
            load2j<<param.getType()<<" "<<param.getName()<<"2 = localData[atom2]."<<param.getName()<<";\n";
        }
        else {
            load2j<<param.getType()<<" "<<param.getName()<<"2 = "<<param.getType()<<"(";
            for (int j = 0; j < param.getNumComponents(); ++j) {
                if (j > 0)
                    load2j<<", ";
                load2j<<"localData[atom2]."<<param.getName()<<"_"<<suffixes[j];
            }
            load2j<<");\n";
        }
    }
    replacements["LOAD_ATOM2_PARAMETERS"] = load2j.str();
    stringstream clearLocal;
    for (const ComputeParameterInfo& param : params) {
        if (param.getNumComponents() == 1)
            clearLocal<<"localData[localAtomIndex]."<<param.getName()<<" = 0;\n";
        else
            for (int j = 0; j < param.getNumComponents(); ++j)
                clearLocal<<"localData[localAtomIndex]."<<param.getName()<<"_"<<suffixes[j]<<" = 0;\n";
    }
    replacements["CLEAR_LOCAL_PARAMETERS"] = clearLocal.str();
    stringstream initDerivs;
    for (int i = 0; i < energyParameterDerivatives.size(); i++)
        initDerivs<<"mixed energyParamDeriv"<<i<<" = 0;\n";
    replacements["INIT_DERIVATIVES"] = initDerivs.str();
    stringstream saveDerivs;
    const vector<string>& allParamDerivNames = context.getEnergyParamDerivNames();
    int numDerivs = allParamDerivNames.size();
    for (int i = 0; i < energyParameterDerivatives.size(); i++)
        for (int index = 0; index < numDerivs; index++)
            if (allParamDerivNames[index] == energyParameterDerivatives[i])
                saveDerivs<<"energyParamDerivs[GLOBAL_ID*"<<numDerivs<<"+"<<index<<"] += energyParamDeriv"<<i<<";\n";
    replacements["SAVE_DERIVATIVES"] = saveDerivs.str();
    map<string, string> defines;
    if (useCutoff)
        defines["USE_CUTOFF"] = "1";
    if (usePeriodic)
        defines["USE_PERIODIC"] = "1";
    if (useExclusions)
        defines["USE_EXCLUSIONS"] = "1";
    if (isSymmetric)
        defines["USE_SYMMETRIC"] = "1";
    if (useNeighborList)
        defines["USE_NEIGHBOR_LIST"] = "1";
    if (includeForces)
        defines["INCLUDE_FORCES"] = "1";
    if (includeEnergy)
        defines["INCLUDE_ENERGY"] = "1";
    defines["THREAD_BLOCK_SIZE"] = context.intToString(forceThreadBlockSize);
    defines["FORCE_WORK_GROUP_SIZE"] = context.intToString(forceThreadBlockSize);
    double maxCutoff = 0.0;
    for (int i = 0; i < 32; i++) {
        if ((groups&(1<<i)) != 0) {
            double cutoff = groupCutoff[i];
            maxCutoff = max(maxCutoff, cutoff);
            defines["CUTOFF_"+context.intToString(i)+"_SQUARED"] = context.doubleToString(cutoff*cutoff);
            defines["CUTOFF_"+context.intToString(i)] = context.doubleToString(cutoff);
        }
    }
    defines["MAX_CUTOFF"] = context.doubleToString(maxCutoff);
    defines["NUM_ATOMS"] = context.intToString(context.getNumAtoms());
    defines["PADDED_NUM_ATOMS"] = context.intToString(context.getPaddedNumAtoms());
    defines["NUM_BLOCKS"] = context.intToString(context.getNumAtomBlocks());
    defines["TILE_SIZE"] = context.intToString(MetalContext::TileSize);
    int numExclusionTiles = exclusionTiles.getSize();
    defines["NUM_TILES_WITH_EXCLUSIONS"] = context.intToString(numExclusionTiles);
    int numContexts = context.getPlatformData().contexts.size();
    int startExclusionIndex = context.getContextIndex()*numExclusionTiles/numContexts;
    int endExclusionIndex = (context.getContextIndex()+1)*numExclusionTiles/numContexts;
    defines["FIRST_EXCLUSION_TILE"] = context.intToString(startExclusionIndex);
    defines["LAST_EXCLUSION_TILE"] = context.intToString(endExclusionIndex);
    if ((localDataSize/4)%2 == 0)
        defines["PARAMETER_SIZE_IS_EVEN"] = "1";
    ComputeProgram program = context.compileProgram(context.replaceStrings(kernelSource, replacements), defines);
    ComputeKernel kernel = program->createKernel("computeNonbonded");

    // Set arguments to the Kernel.

    kernel->addArg(context.getLongForceBuffer());
    kernel->addArg(context.getEnergyBuffer());
    kernel->addArg(context.getPosq());
    kernel->addArg(exclusions);
    kernel->addArg(exclusionTiles);
    kernel->addArg((unsigned int) startTileIndex);
    kernel->addArg((unsigned long long) numTiles);
    if (useCutoff) {
        kernel->addArg(interactingTiles);
        kernel->addArg(interactionCount);
        for (int i = 0; i < 5; i++)
            kernel->addArg(); // The periodic box size arguments are set when the kernel is executed.
        kernel->addArg((unsigned int) interactingTiles.getSize());
        kernel->addArg(blockCenter);
        kernel->addArg(blockBoundingBox);
        kernel->addArg(interactingAtoms);
    }
    for (ComputeParameterInfo& param : params)
        kernel->addArg(param.getArray());
    for (ComputeParameterInfo& arg : arguments)
        kernel->addArg(arg.getArray());
    if (energyParameterDerivatives.size() > 0)
        kernel->addArg(context.getEnergyParamDerivBuffer());
    return kernel;
}

void MetalNonbondedUtilities::setKernelSource(const string& source) {
    kernelSource = source;
}
