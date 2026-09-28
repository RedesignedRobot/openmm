/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2009-2026 Stanford University and the Authors.      *
 * Portions copyright (c) 2020-2023 Advanced Micro Devices, Inc.              *
 * Authors: Peter Eastman, Nicholas Curtis                                    *
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
#include "MetalSort.h"
#include "MetalKernelSources.h"
#include "MetalExpressionUtilities.h"
#include <algorithm>
#include <atomic>
#include <unistd.h>
#include <map>
#include <set>
#include <utility>

using namespace OpenMM;
using namespace std;

// Steps without a host wait share a command buffer, up to this many.  Fewer buffers mean fewer boundaries
// where the GPU can idle, and committing every few steps keeps committed work queued ahead of the GPU, so
// it doesn't wait for MetalQueue's idle commit.
static const int StepsPerCommandBuffer = 8;

// findBlockBounds handles one atom block per SIMD group.  Eight blocks per threadgroup take it from 18.6 to 2.5 us per
// step on apoa1 on an M3 Ultra (with the parallel periodic wrap), mostly by making one pair of atomics on the block
// size range per threadgroup instead of per block.
static const int FindBlockBoundsThreadBlockSize = 256;

// prepareRebuildDispatch sorts up to this many blocks in its one threadgroup.  Larger systems sort in
// three dispatches of BinSortThreadBlockSize threads (at most 256, since a block's rank inside its
// threadgroup takes 8 bits).
static const int MaxBlocksSortedInOneGroup = 1024;
static const int BinSortThreadBlockSize = 256;
// computeSortKeys' number of block size bins.
static const int NumSizeBins = 20;

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

MetalNonbondedUtilities::MetalNonbondedUtilities(MetalContext& context) : context(context), useCutoff(false), usePeriodic(false), useNeighborList(false), anyExclusions(false), usePadding(true),
        pinnedCountBuffer(NULL), countCopies(0), forceRebuildNeighborList(true), groupFlags(0), canUsePairList(true), waitForCount(true), tilesAfterReorder(0),
        uncommittedSteps(0), commitCountAfterLastStep(0) {
    // On a step that doesn't rebuild the list, each dispatch of the rebuild chain still costs the Apple9 GPUs
    // a few microseconds, so the GPU sizes that chain itself and skips it.  The Apple8 GPUs gained nothing from
    // this.  On apoa1 (2882 blocks) the list rebuilds on half the steps, and each skipped step still spent
    // 49 us on an M3 Ultra in the sort, sortBoxData and an empty findBlocksWithInteractions.
    useIndirectRebuild = context.getDevice()->supportsFamily(MTL::GPUFamilyApple9);
    sortBlocksByBins = useIndirectRebuild && context.getNumAtomBlocks() > MaxBlocksSortedInOneGroup;
    // Decide how many thread blocks to use.

    // The tile count, the single pair count, and how many times copyInteractionCounts has run.
    pinnedCountDeviceMemory = context.getDevice()->newBuffer(4*sizeof(unsigned int), MTL::ResourceStorageModeShared);
    pinnedCountBuffer = (unsigned int*) pinnedCountDeviceMemory->contents();
    memset(pinnedCountBuffer, 0, 4*sizeof(unsigned int));
    // Large systems have enough neighbor list tiles to split finely, and 160 force blocks per core instead of 40 make
    // apoa1, cellulose and stmv 4-11% faster on an M3 Ultra and 1-2% faster on an M2.  Small systems lose at 160 (rf
    // 5% on the M3 Ultra, gbsa 1.4% on the M2): their tiles are already spread thin, and every block adds 64 entries
    // to the energy buffer that is cleared each step.
    numForceThreadBlocks = (context.getNumAtoms() > 90000 ? 160 : 40)*context.getMultiprocessors();
    forceThreadBlockSize = 64;
    findInteractingBlocksThreadBlockSize = context.getSIMDWidth();

    // When building the neighbor list, we can optionally use large blocks (32 * warpSize atoms) to
    // accelerate the process.  This makes building the neighbor list faster, but it prevents
    // us from sorting atom blocks by size, which leads to a slightly less efficient neighbor
    // list.  We guess based on system size which will be faster.

    useLargeBlocks = (context.getNumAtoms() > 90000);
    setKernelSource(MetalKernelSources::nonbonded);
}

MetalNonbondedUtilities::~MetalNonbondedUtilities() {
    pinnedCountDeviceMemory->release();
}

void MetalNonbondedUtilities::addInteraction(bool usesCutoff, bool usesPeriodic, bool usesExclusions, double cutoffDistance,
        const vector<vector<int> >& exclusionList, const string& kernel, int forceGroup, bool usesNeighborList, bool supportsPairList) {
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
    useNeighborList |= (usesNeighborList && useCutoff);
    groupCutoff[forceGroup] = cutoffDistance;
    groupFlags |= 1<<forceGroup;
    canUsePairList &= supportsPairList;
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

static bool compareInt2(int2 a, int2 b) {
    return ((a.y < b.y) || (a.y == b.y && a.x < b.x));
}

static bool compareInt2LargeSIMD(int2 a, int2 b) {
    // This version is used on devices with SIMD width greater than tile size.  It puts diagonal tiles before off-diagonal
    // ones to reduce thread divergence.

    if (a.x == a.y) {
        if (b.x == b.y)
            return (a.x < b.x);
        return true;
    }
    if (b.x == b.y)
        return false;
    return ((a.y < b.y) || (a.y == b.y && a.x < b.x));
}

void MetalNonbondedUtilities::initialize(const System& system) {
    string errorMessage = "Error initializing nonbonded utilities";
    if (atomExclusions.size() == 0) {
        // No exclusions were specifically requested, so just mark every atom as not interacting with itself.

        atomExclusions.resize(context.getNumAtoms());
        for (int i = 0; i < (int) atomExclusions.size(); i++)
            atomExclusions[i].push_back(i);
    }

    // Create the list of tiles.

    numAtoms = context.getNumAtoms();
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
    vector<int2> exclusionTilesVec;
    for (set<pair<int, int> >::const_iterator iter = tilesWithExclusions.begin(); iter != tilesWithExclusions.end(); ++iter)
        exclusionTilesVec.push_back(make_int2(iter->first, iter->second));
    sort(exclusionTilesVec.begin(), exclusionTilesVec.end(), context.getSIMDWidth() <= 32 || !useNeighborList ? compareInt2 : compareInt2LargeSIMD);
    exclusionTiles.initialize<int2>(context, exclusionTilesVec.size(), "exclusionTiles");
    exclusionTiles.upload(exclusionTilesVec);
    map<pair<int, int>, int> exclusionTileMap;
    for (int i = 0; i < (int) exclusionTilesVec.size(); i++) {
        int2 tile = exclusionTilesVec[i];
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

    exclusions.initialize<tileflags>(context, tilesWithExclusions.size()*MetalContext::TileSize, "exclusions");
    tileflags allFlags = (tileflags) -1;
    vector<tileflags> exclusionVec(exclusions.getSize(), allFlags);
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
        // Select a size for the arrays that hold the neighbor list.  If arrays for every tile and single pair
        // the list could ever hold fit in 1/256 of the device's unused working set, it can't overflow, and
        // the host only waits for the interaction count on steps that set the reordering baseline.  With
        // several contexts, one context's blocks can have more than its share of tiles.  Otherwise we have to
        // make a fairly arbitrary guess, but if this turns out to be too small we'll increase it later.

        long long maxPossiblePairs = getMaxBitsForPairs()*(long long) MetalContext::TileSize*numTiles;
        long long maxListBytes = numTiles*(MetalContext::TileSize+1)*(long long) sizeof(int)+maxPossiblePairs*(long long) sizeof(int2);
        long long unusedMemory = (long long) context.getDevice()->recommendedMaxWorkingSetSize()-(long long) context.getDevice()->currentAllocatedSize();
        waitForCount = (numContexts > 1 || maxListBytes > unusedMemory/256);
        if (!waitForCount)
            context.getCurrentStream().commitWhenIdle();
        maxTiles = 20*numAtomBlocks;
        if (maxTiles > numTiles || !waitForCount)
            maxTiles = numTiles;
        if (maxTiles < 1)
            maxTiles = 1;
        maxSinglePairs = 5*numAtoms;
        if (!waitForCount && maxSinglePairs < maxPossiblePairs)
            maxSinglePairs = maxPossiblePairs;
        // Small systems leave most GPU cores idle during the neighbor list build, so give each atom block several warps.
        numTilesInBatch = (numAtomBlocks < 64*context.getMultiprocessors() ? 4 : 1);
        interactingTiles.initialize<int>(context, maxTiles, "interactingTiles");
        interactingAtoms.initialize<int>(context, MetalContext::TileSize*maxTiles, "interactingAtoms");
        interactionCount.initialize<unsigned int>(context, 2, "interactionCount");
        singlePairs.initialize<int2>(context, maxSinglePairs, "singlePairs");
        int elementSize = (context.getUseDoublePrecision() ? sizeof(double) : sizeof(float));
        blockCenter.initialize(context, numAtomBlocks, 4*elementSize, "blockCenter");
        blockBoundingBox.initialize(context, numAtomBlocks, 4*elementSize, "blockBoundingBox");
        sortedBlocks.initialize<unsigned int>(context, numAtomBlocks, "sortedBlocks");
        sortedBlockCenter.initialize(context, numAtomBlocks+1, 4*elementSize, "sortedBlockCenter");
        sortedBlockBoundingBox.initialize(context, numAtomBlocks+1, 4*elementSize, "sortedBlockBoundingBox");
        blockSizeRange.initialize(context, 2, elementSize, "blockSizeRange");
        largeBlockCenter.initialize(context, numAtomBlocks, 4*elementSize, "largeBlockCenter");
        largeBlockBoundingBox.initialize(context, numAtomBlocks*4, elementSize, "largeBlockBoundingBox");
        oldPositions.initialize(context, numAtoms, 4*elementSize, "oldPositions");
        rebuildNeighborList.initialize<int>(context, 1, "rebuildNeighborList");
        if (useIndirectRebuild) {
            displacementFlags.initialize<unsigned int>(context, numAtomBlocks, "displacementFlags");
            rebuildDispatch.initialize<unsigned int>(context, 12, "rebuildDispatch");
        }
        if (sortBlocksByBins) {
            int binSortGroups = (numAtomBlocks+BinSortThreadBlockSize-1)/BinSortThreadBlockSize;
            binRanks.initialize<unsigned int>(context, numAtomBlocks, "binRanks");
            binCounts.initialize<unsigned int>(context, NumSizeBins*binSortGroups, "binCounts");
            binOffsets.initialize<unsigned int>(context, NumSizeBins*binSortGroups, "binOffsets");
        }
        if (!useIndirectRebuild)
            blockSorter = context.createSort(new BlockSortTrait(), numAtomBlocks, false);
        vector<unsigned int> count(2, 0);
        interactionCount.upload(count);
        rebuildNeighborList.upload(&count[0]);
        if (context.getUseDoublePrecision()) {
            blockSizeRange.upload(vector<double>{1e38, 0});
        } else {
            blockSizeRange.upload(vector<float>{1e38, 0});
        }
    }

    // Record arguments for kernels.

    forceArgs.push_back(&context.unwrap(context.getLongForceBuffer()).getDevicePointer());
    forceArgs.push_back(&context.unwrap(context.getEnergyBuffer()).getDevicePointer());
    forceArgs.push_back(&context.unwrap(context.getPosq()).getDevicePointer());
    forceArgs.push_back(&exclusions.getDevicePointer());
    forceArgs.push_back(&exclusionTiles.getDevicePointer());
    forceArgs.push_back(&startTileIndex);
    forceArgs.push_back(&numTiles);
    if (useCutoff) {
        forceArgs.push_back(&interactingTiles.getDevicePointer());
        forceArgs.push_back(&interactionCount.getDevicePointer());
        forceArgs.push_back(context.getPeriodicBoxSizePointer());
        forceArgs.push_back(context.getInvPeriodicBoxSizePointer());
        forceArgs.push_back(context.getPeriodicBoxVecXPointer());
        forceArgs.push_back(context.getPeriodicBoxVecYPointer());
        forceArgs.push_back(context.getPeriodicBoxVecZPointer());
        forceArgs.push_back(&maxTiles);
        forceArgs.push_back(&blockCenter.getDevicePointer());
        forceArgs.push_back(&blockBoundingBox.getDevicePointer());
        forceArgs.push_back(&interactingAtoms.getDevicePointer());
        forceArgs.push_back(&maxSinglePairs);
        forceArgs.push_back(&singlePairs.getDevicePointer());
    }
    hasInitializedParams = false;
    paramStartIndex = forceArgs.size();
    for (int i = 0; i < parameters.size()+arguments.size(); i++)
        forceArgs.push_back(NULL);
    if (energyParameterDerivatives.size() > 0)
        forceArgs.push_back(&context.unwrap(context.getEnergyParamDerivBuffer()).getDevicePointer());
    if (useCutoff) {
        findBlockBoundsArgs.push_back(&numAtoms);
        findBlockBoundsArgs.push_back(context.getPeriodicBoxSizePointer());
        findBlockBoundsArgs.push_back(context.getInvPeriodicBoxSizePointer());
        findBlockBoundsArgs.push_back(context.getPeriodicBoxVecXPointer());
        findBlockBoundsArgs.push_back(context.getPeriodicBoxVecYPointer());
        findBlockBoundsArgs.push_back(context.getPeriodicBoxVecZPointer());
        findBlockBoundsArgs.push_back(&context.unwrap(context.getPosq()).getDevicePointer());
        findBlockBoundsArgs.push_back(&blockCenter.getDevicePointer());
        findBlockBoundsArgs.push_back(&blockBoundingBox.getDevicePointer());
        findBlockBoundsArgs.push_back(&rebuildNeighborList.getDevicePointer());
        findBlockBoundsArgs.push_back(&blockSizeRange.getDevicePointer());
        if (useIndirectRebuild) {
            findBlockBoundsArgs.push_back(&oldPositions.getDevicePointer());
            findBlockBoundsArgs.push_back(&displacementFlags.getDevicePointer());
            findBlockBoundsArgs.push_back(&forceRebuildNeighborList);
        }
        computeSortKeysArgs.push_back(&blockBoundingBox.getDevicePointer());
        computeSortKeysArgs.push_back(&sortedBlocks.getDevicePointer());
        computeSortKeysArgs.push_back(&blockSizeRange.getDevicePointer());
        sortBoxDataArgs.push_back(&sortedBlocks.getDevicePointer());
        sortBoxDataArgs.push_back(&blockCenter.getDevicePointer());
        sortBoxDataArgs.push_back(&blockBoundingBox.getDevicePointer());
        sortBoxDataArgs.push_back(&sortedBlockCenter.getDevicePointer());
        sortBoxDataArgs.push_back(&sortedBlockBoundingBox.getDevicePointer());
        if (useLargeBlocks) {
            sortBoxDataArgs.push_back(&largeBlockCenter.getDevicePointer());
            sortBoxDataArgs.push_back(&largeBlockBoundingBox.getDevicePointer());
            sortBoxDataArgs.push_back(context.getPeriodicBoxSizePointer());
            sortBoxDataArgs.push_back(context.getInvPeriodicBoxSizePointer());
            sortBoxDataArgs.push_back(context.getPeriodicBoxVecXPointer());
            sortBoxDataArgs.push_back(context.getPeriodicBoxVecYPointer());
            sortBoxDataArgs.push_back(context.getPeriodicBoxVecZPointer());
        }
        sortBoxDataArgs.push_back(&context.unwrap(context.getPosq()).getDevicePointer());
        sortBoxDataArgs.push_back(&oldPositions.getDevicePointer());
        sortBoxDataArgs.push_back(&interactionCount.getDevicePointer());
        sortBoxDataArgs.push_back(&rebuildNeighborList.getDevicePointer());
        sortBoxDataArgs.push_back(&forceRebuildNeighborList);
        sortBoxDataArgs.push_back(&blockSizeRange.getDevicePointer());
        findInteractingBlocksArgs.push_back(context.getPeriodicBoxSizePointer());
        findInteractingBlocksArgs.push_back(context.getInvPeriodicBoxSizePointer());
        findInteractingBlocksArgs.push_back(context.getPeriodicBoxVecXPointer());
        findInteractingBlocksArgs.push_back(context.getPeriodicBoxVecYPointer());
        findInteractingBlocksArgs.push_back(context.getPeriodicBoxVecZPointer());
        findInteractingBlocksArgs.push_back(&interactionCount.getDevicePointer());
        findInteractingBlocksArgs.push_back(&interactingTiles.getDevicePointer());
        findInteractingBlocksArgs.push_back(&interactingAtoms.getDevicePointer());
        findInteractingBlocksArgs.push_back(&singlePairs.getDevicePointer());
        findInteractingBlocksArgs.push_back(&context.unwrap(context.getPosq()).getDevicePointer());
        findInteractingBlocksArgs.push_back(&maxTiles);
        findInteractingBlocksArgs.push_back(&maxSinglePairs);
        findInteractingBlocksArgs.push_back(&startBlockIndex);
        findInteractingBlocksArgs.push_back(&numBlocks);
        findInteractingBlocksArgs.push_back(&sortedBlocks.getDevicePointer());
        findInteractingBlocksArgs.push_back(&sortedBlockCenter.getDevicePointer());
        findInteractingBlocksArgs.push_back(&sortedBlockBoundingBox.getDevicePointer());
        if (useLargeBlocks) {
            findInteractingBlocksArgs.push_back(&largeBlockCenter.getDevicePointer());
            findInteractingBlocksArgs.push_back(&largeBlockBoundingBox.getDevicePointer());
        }
        findInteractingBlocksArgs.push_back(&exclusionIndices.getDevicePointer());
        findInteractingBlocksArgs.push_back(&exclusionRowIndices.getDevicePointer());
        findInteractingBlocksArgs.push_back(&oldPositions.getDevicePointer());
        findInteractingBlocksArgs.push_back(&rebuildNeighborList.getDevicePointer());
        copyInteractionCountsArgs.push_back(&interactionCount.getDevicePointer());
        copyInteractionCountsArgs.push_back(&pinnedCountDeviceMemory);
    }
}

double MetalNonbondedUtilities::getMaxCutoffDistance() {
    double cutoff = 0.0;
    for (map<int, double>::const_iterator iter = groupCutoff.begin(); iter != groupCutoff.end(); ++iter)
        cutoff = max(cutoff, iter->second);
    return cutoff;
}

double MetalNonbondedUtilities::padCutoff(double cutoff) {
    double padding = (usePadding ? 0.08*cutoff : 0.0);
    return cutoff+padding;
}

void MetalNonbondedUtilities::prepareInteractions(int forceGroups) {
    if ((forceGroups&groupFlags) == 0)
        return;
    if (groupKernels.find(forceGroups) == groupKernels.end())
        createKernelsForGroups(forceGroups);
    KernelSet& kernels = groupKernels[forceGroups];
    if (useCutoff && usePeriodic) {
        double4 box = context.getPeriodicBoxSize();
        double minAllowedSize = 1.999999*maxCutoff;
        if (box.x < minAllowedSize || box.y < minAllowedSize || box.z < minAllowedSize)
            throw OpenMMException("The periodic box size has decreased to less than twice the nonbonded cutoff.");
    }
    if (!useNeighborList)
        return;
    if (numTiles == 0)
        return;

    // Compute the neighbor list.

    context.executeKernelFlat(kernels.findBlockBoundsKernel, &findBlockBoundsArgs[0], context.getPaddedNumAtoms(), FindBlockBoundsThreadBlockSize);
    if (useIndirectRebuild) {
        void* args[] = {&displacementFlags.getDevicePointer(), &rebuildNeighborList.getDevicePointer(),
                &interactionCount.getDevicePointer(), &pinnedCountDeviceMemory, &blockSizeRange.getDevicePointer(),
                &blockBoundingBox.getDevicePointer(), &sortedBlocks.getDevicePointer(), &rebuildDispatch.getDevicePointer()};
        int prepareThreads = min((context.getNumAtomBlocks()+31)/32*32, (int) kernels.prepareRebuildDispatchKernel->maxTotalThreadsPerThreadgroup());
        context.executeKernelFlat(kernels.prepareRebuildDispatchKernel, args, prepareThreads, prepareThreads);
        MTL::Buffer* dispatch = rebuildDispatch.getDevicePointer();
        if (sortBlocksByBins) {
            void* countArgs[] = {&blockBoundingBox.getDevicePointer(), &blockSizeRange.getDevicePointer(),
                    &binRanks.getDevicePointer(), &binCounts.getDevicePointer()};
            void* sumArgs[] = {&binCounts.getDevicePointer(), &binOffsets.getDevicePointer()};
            void* scatterArgs[] = {&binRanks.getDevicePointer(), &binOffsets.getDevicePointer(), &sortedBlocks.getDevicePointer()};
            context.executeKernelFlat(kernels.countBlockBinsKernel, countArgs, 0, BinSortThreadBlockSize, 0, dispatch, 6*sizeof(unsigned int));
            context.executeKernelFlat(kernels.sumBlockBinsKernel, sumArgs, 0, 32, 0, dispatch, 9*sizeof(unsigned int));
            context.executeKernelFlat(kernels.scatterBlocksKernel, scatterArgs, 0, BinSortThreadBlockSize, 0, dispatch, 6*sizeof(unsigned int));
        }
        context.executeKernelFlat(kernels.sortBoxDataKernel, &sortBoxDataArgs[0], context.getNumAtoms(), 64, 0, dispatch, 0);
        context.executeKernelFlat(kernels.findInteractingBlocksKernel, &findInteractingBlocksArgs[0], context.getNumAtomBlocks()*context.getSIMDWidth()*numTilesInBatch,
                findInteractingBlocksThreadBlockSize, 0, dispatch, 3*sizeof(unsigned int));
    }
    else {
        context.executeKernelFlat(kernels.computeSortKeysKernel, &computeSortKeysArgs[0], context.getNumAtomBlocks());
        blockSorter->sort(sortedBlocks);
        context.executeKernelFlat(kernels.sortBoxDataKernel, &sortBoxDataArgs[0], context.getNumAtoms(), 64);
        context.executeKernelFlat(kernels.findInteractingBlocksKernel, &findInteractingBlocksArgs[0], context.getNumAtomBlocks() * context.getSIMDWidth() * numTilesInBatch, findInteractingBlocksThreadBlockSize);
    }
    forceRebuildNeighborList = false;
    // With the indirect rebuild, prepareRebuildDispatch already copied a count, at most one list old,
    // which is all a step without the wait needs.
    bool wait = (waitForCount || context.getStepsSinceReorder() == 0);
    // Each command buffer boundary leaves the GPU idle, so the force kernels go into this buffer
    // too.  computeInteractions() commits it before waiting for the count.
    if (wait || !useIndirectRebuild) {
        context.executeKernelFlat(kernels.copyInteractionCountsKernel, &copyInteractionCountsArgs[0], 1, 1);
        countCopies++;
    }
}

void MetalNonbondedUtilities::initParamArgs() {
    int index = paramStartIndex;
    for (ComputeParameterInfo& param : parameters)
        forceArgs[index++] = &context.unwrap(param.getArray()).getDevicePointer();
    for (ComputeParameterInfo& arg : arguments)
        forceArgs[index++] = &context.unwrap(arg.getArray()).getDevicePointer();
    hasInitializedParams = true;
}

void MetalNonbondedUtilities::computeInteractions(int forceGroups, bool includeForces, bool includeEnergy) {
    if ((forceGroups&groupFlags) == 0)
        return;
    KernelSet& kernels = groupKernels[forceGroups];
    if (kernels.hasForces && (includeForces || includeEnergy)) {
        metalFunction_t& kernel = (includeForces ? (includeEnergy ? kernels.forceEnergyKernel : kernels.forceKernel) : kernels.energyKernel);
        if (kernel == NULL)
            kernel = createInteractionKernel(kernels.source, parameters, arguments, true, true, forceGroups, includeForces, includeEnergy);
        if (!hasInitializedParams)
            initParamArgs();
        context.executeKernelFlat(kernel, &forceArgs[0], numForceThreadBlocks*forceThreadBlockSize, forceThreadBlockSize);
    }
    if (useNeighborList && numTiles > 0) {
        // Without the wait, the count comes from an earlier step.  It can't exceed the arrays, so it only
        // guides reordering, which needs the exact count right after a reorder.  Steps without the wait
        // share a command buffer, up to StepsPerCommandBuffer of them, unless the GPU has run out of work.
        MetalQueue& queue = context.getCurrentStream();
        bool wait = (waitForCount || context.getStepsSinceReorder() == 0);
        if (queue.getCommitCount() != commitCountAfterLastStep)
            uncommittedSteps = 0;
        if (wait || ++uncommittedSteps >= StepsPerCommandBuffer || queue.isIdle()) {
            queue.commit();
            uncommittedSteps = 0;
        }
        commitCountAfterLastStep = queue.getCommitCount();
        if (wait)
            waitForCountCopy();
        updateNeighborListSize();
    }
}

void MetalNonbondedUtilities::waitForCountCopy() {
    // The host watches the copy counter in pinned memory instead of waiting on a shared event.  A signal
    // between the neighbor list and the force kernels stalled the force kernels on the M3 Ultra until the
    // host's next commit, 47 us per step on apoa1.  The counter usually shows up while the force kernels
    // run; the idle check covers a buffer that ends without it, which only an error does.
    volatile unsigned int* copies = pinnedCountBuffer+2;
    MetalQueue& queue = context.getCurrentStream();
    for (int i = 1; *copies != countCopies; i++) {
        usleep(10);
        if (i%32 == 0 && queue.isIdle() && *copies != countCopies) {
            queue.finish();
            throw OpenMMException("The neighbor list count never arrived");
        }
    }
    std::atomic_thread_fence(std::memory_order_acquire);
}

bool MetalNonbondedUtilities::updateNeighborListSize() {
    if (!useCutoff)
        return false;
    if (context.getStepsSinceReorder() == 0 || tilesAfterReorder == 0)
        tilesAfterReorder = pinnedCountBuffer[0];
    else if (context.getStepsSinceReorder() > 25 && pinnedCountBuffer[0] > 1.1*tilesAfterReorder)
        context.forceReorder();
    if (pinnedCountBuffer[0] <= maxTiles && pinnedCountBuffer[1] <= maxSinglePairs)
        return false;

    // The most recent timestep had too many interactions to fit in the arrays.  Make the arrays bigger to prevent
    // this from happening in the future.

    if (pinnedCountBuffer[0] > maxTiles) {
        maxTiles = (unsigned int) (1.2*pinnedCountBuffer[0]);
        unsigned int numBlocks = context.getNumAtomBlocks();
        int totalTiles = numBlocks*(numBlocks+1)/2;
        if (maxTiles > totalTiles)
            maxTiles = totalTiles;
        interactingTiles.resize(maxTiles);
        interactingAtoms.resize(MetalContext::TileSize*(size_t) maxTiles);
        if (forceArgs.size() > 0)
            forceArgs[7] = &interactingTiles.getDevicePointer();
        findInteractingBlocksArgs[6] = &interactingTiles.getDevicePointer();
        if (forceArgs.size() > 0)
            forceArgs[17] = &interactingAtoms.getDevicePointer();
        findInteractingBlocksArgs[7] = &interactingAtoms.getDevicePointer();
    }
    if (pinnedCountBuffer[1] > maxSinglePairs) {
        maxSinglePairs = (unsigned int) (1.2*pinnedCountBuffer[1]);
        singlePairs.resize(maxSinglePairs);
        if (forceArgs.size() > 0)
            forceArgs[19] = &singlePairs.getDevicePointer();
        findInteractingBlocksArgs[8] = &singlePairs.getDevicePointer();
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
    startTileIndex = (int) (startFraction*totalTiles);
    numTiles = (long long) (endFraction*totalTiles)-startTileIndex;
    forceRebuildNeighborList = true;
}

int MetalNonbondedUtilities::getMaxBitsForPairs() const {
    // Small non-LJPME systems avoid sparse-pair atomics on Apple9 and later.
    // Preserve the large-system policy: disabling its two-bit pair list regresses
    // cellulose.  Older devices and double precision retain their existing policy.
    // Zero bits shrink the worst-case list, so more systems fit the wait-free budget in
    // initialize() and allocate every tile: growth stays under 1/256 of the unused working set.
    if (!useLJPME && !context.getUseDoublePrecision() && context.getNumAtoms() < 100000 &&
            context.getDevice()->supportsFamily(MTL::GPUFamilyApple9))
        return 0;
    int maxBits = 0;
    if (canUsePairList) {
        if (context.getUseDoublePrecision()) {
            maxBits = 4;
        }
        else {
            if (context.getSIMDWidth() > 32) {
                // CDNA
                if (context.getNumAtoms() < 100000)
                    maxBits = 4;
                else // Large systems
                    maxBits = 0;
            }
            else {
                // RDNA
                if (context.getNumAtoms() < 100000)
                    maxBits = 4;
                else if (context.getNumAtoms() < 500000)
                    maxBits = 2;
                else // Very large systems
                    maxBits = 0;
            }
        }
    }
    return maxBits;
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
    kernels.forceKernel = kernels.energyKernel = kernels.forceEnergyKernel = NULL;
    if (useCutoff) {
        double paddedCutoff = padCutoff(maxCutoff);
        map<string, string> defines;
        defines["TILE_SIZE"] = context.intToString(MetalContext::TileSize);
        defines["NUM_BLOCKS"] = context.intToString(context.getNumAtomBlocks());
        defines["NUM_ATOMS"] = context.intToString(context.getNumAtoms());
        defines["PADDED_NUM_ATOMS"] = context.intToString(context.getPaddedNumAtoms());
        defines["PADDING"] = context.doubleToString(paddedCutoff-maxCutoff);
        defines["PADDED_CUTOFF"] = context.doubleToString(paddedCutoff);
        defines["PADDED_CUTOFF_SQUARED"] = context.doubleToString(paddedCutoff*paddedCutoff);
        defines["NUM_TILES_WITH_EXCLUSIONS"] = context.intToString(exclusionTiles.getSize());
        if (usePeriodic)
            defines["USE_PERIODIC"] = "1";
        if (context.getBoxIsTriclinic())
            defines["TRICLINIC"] = "1";
        if (useLargeBlocks)
            defines["USE_LARGE_BLOCKS"] = "1";
        defines["MAX_EXCLUSIONS"] = context.intToString(maxExclusions);
        defines["MAX_BITS_FOR_PAIRS"] = context.intToString(getMaxBitsForPairs());
        defines["NUM_TILES_IN_BATCH"] = context.intToString(numTilesInBatch);
        defines["GROUP_SIZE"] = context.intToString(findInteractingBlocksThreadBlockSize);
        if (useIndirectRebuild)
            defines["INDIRECT_REBUILD"] = "1";
        if (sortBlocksByBins) {
            defines["SORT_BLOCKS_BY_BINS"] = "1";
            defines["BIN_SORT_GROUP"] = context.intToString(BinSortThreadBlockSize);
            defines["NUM_SIZE_BINS"] = context.intToString(NumSizeBins);
        }
        defines["FIND_BLOCK_BOUNDS_GROUP"] = context.intToString(FindBlockBoundsThreadBlockSize);
        int binShift = 1;
        while (1<<binShift <= context.getNumAtomBlocks())
            binShift++;
        defines["BIN_SHIFT"] = context.intToString(binShift);
        defines["BLOCK_INDEX_MASK"] = context.intToString((1<<binShift)-1);
        metalModule_t interactingBlocksProgram = context.createModule(MetalKernelSources::vectorOps+MetalKernelSources::findInteractingBlocks, defines);
        kernels.findBlockBoundsKernel = context.getKernel(interactingBlocksProgram, "findBlockBounds");
        if (useIndirectRebuild)
            kernels.prepareRebuildDispatchKernel = context.getKernel(interactingBlocksProgram, "prepareRebuildDispatch");
        if (sortBlocksByBins) {
            kernels.countBlockBinsKernel = context.getKernel(interactingBlocksProgram, "countBlockBins");
            kernels.sumBlockBinsKernel = context.getKernel(interactingBlocksProgram, "sumBlockBins");
            kernels.scatterBlocksKernel = context.getKernel(interactingBlocksProgram, "scatterBlocks");
        }
        kernels.computeSortKeysKernel = context.getKernel(interactingBlocksProgram, "computeSortKeys");
        kernels.sortBoxDataKernel = context.getKernel(interactingBlocksProgram, "sortBoxData");
        kernels.findInteractingBlocksKernel = context.getKernel(interactingBlocksProgram, "findBlocksWithInteractions");
        kernels.copyInteractionCountsKernel = context.getKernel(interactingBlocksProgram, "copyInteractionCounts");
    }
    groupKernels[groups] = kernels;
}

/**
 * A three component parameter is stored as packed scalars, because a Metal vector of three
 * takes the space of four.
 */
static string loadParameter(const ComputeParameterInfo& param, const string& index) {
    string array = "global_"+param.getName();
    if (param.getNumComponents() != 3)
        return array+"["+index+"]";
    return "make_"+param.getType()+"("+array+"[3*"+index+"], "+array+"[3*"+index+"+1], "+array+"[3*"+index+"+2])";
}

metalFunction_t MetalNonbondedUtilities::createInteractionKernel(const string& source, vector<ComputeParameterInfo>& params, vector<ComputeParameterInfo>& arguments, bool useExclusions, bool isSymmetric, int groups, bool includeForces, bool includeEnergy) {
    map<string, string> replacements;
    replacements["COMPUTE_INTERACTION"] = source;
    const string suffixes[] = {"x", "y", "z", "w"};
    stringstream args;
    for (const ComputeParameterInfo& param : params) {
        args << ", ";
        if (param.isConstant())
            args << "const ";
        args << (param.getNumComponents() == 3 ? param.getComponentType() : param.getType());
        args << "* __restrict__ global_";
        args << param.getName();
    }
    for (const ComputeParameterInfo& arg : arguments) {
        args << ", ";
        if (arg.isConstant())
            args << "const ";
        args << arg.getType();
        args << "* __restrict__ ";
        args << arg.getName();
    }
    if (energyParameterDerivatives.size() > 0)
        args << ", mixed* __restrict__ energyParamDerivs";
    replacements["PARAMETER_ARGUMENTS"] = args.str();

    stringstream load1;
    for (const ComputeParameterInfo& param : params) {
        load1 << param.getType();
        load1 << " ";
        load1 << param.getName();
        load1 << "1 = ";
        load1 << loadParameter(param, "atom1");
        load1 << ";\n";
    }
    replacements["LOAD_ATOM1_PARAMETERS"] = load1.str();

    // Part 1. Defines for on diagonal exclusion tiles

    stringstream broadcastWarpData;
    broadcastWarpData << "posq2.x = SHFL(shflPosq.x, j);\n";
    broadcastWarpData << "posq2.y = SHFL(shflPosq.y, j);\n";
    broadcastWarpData << "posq2.z = SHFL(shflPosq.z, j);\n";
    broadcastWarpData << "posq2.w = SHFL(shflPosq.w, j);\n";
    for (const ComputeParameterInfo& param : params) {
        broadcastWarpData << param.getType() << " shfl" << param.getName() << ";\n";
        for (int j = 0; j < param.getNumComponents(); j++) {
            if (param.getNumComponents() == 1)
                broadcastWarpData << "shfl" << param.getName() << "=SHFL(" << param.getName() <<"1,j);\n";
            else
                broadcastWarpData << "shfl" << param.getName()+"."+suffixes[j] << "=SHFL(" << param.getName()+"1."+suffixes[j] <<",j);\n";
        }
    }
    replacements["BROADCAST_WARP_DATA"] = broadcastWarpData.str();

    // Part 2. Defines for off-diagonal exclusions, and neighborlist tiles.
    stringstream declareLocal2;
    for (const ComputeParameterInfo& param : params)
        declareLocal2<<param.getType()<<" shfl"<<param.getName()<<";\n";
    replacements["DECLARE_LOCAL_PARAMETERS"] = declareLocal2.str();

    stringstream loadLocal2;
    for (const ComputeParameterInfo& param : params)
        loadLocal2<<"shfl"<<param.getName()<<" = "<<loadParameter(param, "j")<<";\n";
    replacements["LOAD_LOCAL_PARAMETERS_FROM_GLOBAL"] = loadLocal2.str();

    stringstream load2j;
    for (const ComputeParameterInfo& param : params)
        load2j<<param.getType()<<" "<<param.getName()<<"2 = shfl"<<param.getName()<<";\n";
    replacements["LOAD_ATOM2_PARAMETERS"] = load2j.str();

    stringstream clearLocal;
    for (const ComputeParameterInfo& param : params) {
        clearLocal<<"shfl";
        clearLocal<<param.getName()<<" = ";
        if (param.getNumComponents() == 1)
            clearLocal<<"0;\n";
        else
            clearLocal<<"make_"<<param.getType()<<"(0);\n";
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

    stringstream shuffleWarpData;
    shuffleWarpData << "shflPosq = warpRotateLeft(shflPosq);\n";
    shuffleWarpData << "shflForce = warpRotateLeft(shflForce);\n";
    for (const ComputeParameterInfo& param : params) {
        shuffleWarpData<<"shfl"<<param.getName()<<"=warpRotateLeft(shfl"<<param.getName()<<");\n";
    }
    replacements["SHUFFLE_WARP_DATA"] = shuffleWarpData.str();

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
    defines["ENABLE_SHUFFLE"] = "1"; // Used only in metalpoNonbonded.cc
    if (includeForces)
        defines["INCLUDE_FORCES"] = "1";
    if (includeEnergy)
        defines["INCLUDE_ENERGY"] = "1";
    defines["THREAD_BLOCK_SIZE"] = context.intToString(forceThreadBlockSize);
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
    defines["MAX_CUTOFF_SQUARED"] = context.doubleToString(maxCutoff*maxCutoff);
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
    metalModule_t program = context.createModule(MetalKernelSources::vectorOps+context.replaceStrings(kernelSource, replacements), defines);
    metalFunction_t kernel = context.getKernel(program, "computeNonbonded");
    return kernel;
}

void MetalNonbondedUtilities::setKernelSource(const string& source) {
    kernelSource = source;
}
