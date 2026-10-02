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
#include "MetalAtomReorder.h"
#include "MetalArray.h"
#include "MetalContext.h"
#include "MetalSort.h"
#include "MetalKernelSources.h"
#include "MetalExpressionUtilities.h"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <map>
#include <set>
#include <utility>

using namespace OpenMM;
using namespace std;

#ifdef OPENMM_METAL_TEST_HOOKS
void (*MetalNonbondedUtilities::exactCountHook)(MetalContext&) = nullptr;
#endif

// Steps without a host wait share a command buffer, up to this many.  Fewer buffers mean fewer boundaries
// where the GPU can idle, and committing every few steps keeps committed work queued ahead of the GPU, so
// it doesn't wait for MetalQueue's idle commit.
static const int StepsPerCommandBuffer = 8;

// findBlockBounds handles one atom block per SIMD group.  Eight blocks per threadgroup take it from 18.6 to 2.5 us per
// step on apoa1 on an M3 Ultra (with the parallel periodic wrap), mostly by making one pair of atomics on the block
// size range per threadgroup instead of per block.
static const int FindBlockBoundsThreadBlockSize = 256;

// sortBlocksInOneGroup sorts up to this many blocks in its one threadgroup, one chunk of blocks per pass
// over its threads.  Larger systems sort in three dispatches of BinSortThreadBlockSize threads (at most 256,
// since a block's rank inside its threadgroup takes 8 bits), which beat a long serial chunk loop.
static const int MaxBlocksSortedInOneGroup = 4096;
static const int BinSortThreadBlockSize = 256;
// computeSortKeys' number of block size bins.
static const int NumSizeBins = 20;
// interactingAtoms uses a 32-bit unsigned tile offset.
static const uint64_t MaxNeighborTiles = numeric_limits<unsigned int>::max()/MetalContext::TileSize;

struct MetalNonbondedUtilities::CountSnapshots {
    // A slot cannot be reused until its completion handler has copied both counts.
    // If callbacks lag behind, skip sampling instead of waiting or allocating more slots.
    static const int NumSlots = 4;
    metalDeviceptr_t buffers[NumSlots] = {};
    bool busy[NumSlots] = {};
    mutex lock;
    unsigned long long nextSequence = 0, sequence = 0, epoch = 0;
    unsigned int counts[2] = {};
    ~CountSnapshots() {
        for (auto buffer : buffers)
            if (buffer != NULL)
                buffer->release();
    }
};

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
        pinnedCountBuffer(NULL), forceRebuildNeighborList(true), groupFlags(0), canUsePairList(true), waitForCount(true), tilesAfterReorder(0),
        uncommittedSteps(0), commitCountAfterLastStep(0) {
    // On a step that doesn't rebuild the list, each dispatch of the rebuild chain still costs the Apple9 GPUs
    // a few microseconds, so the GPU sizes that chain itself and skips it.  The Apple8 GPUs gained nothing from
    // this.  On apoa1 (2882 blocks) the list rebuilds on half the steps, and each skipped step still spent
    // 49 us on an M3 Ultra in the sort, sortBoxData and an empty findBlocksWithInteractions.
    useIndirectRebuild = context.getDevice()->supportsFamily(MTL::GPUFamilyApple9);
    sortBlocksByBins = useIndirectRebuild && context.getNumAtomBlocks() > MaxBlocksSortedInOneGroup;
    // Decide how many thread blocks to use.

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
    // Allocate last so a failure above cannot leak the counts buffer.
    // The host reads these counts only after all queued writes have completed.
    pinnedCountDeviceMemory = context.createBuffer(2*sizeof(unsigned int), "pinnedCountDeviceMemory");
    pinnedCountBuffer = (unsigned int*) pinnedCountDeviceMemory->contents();
    memset(pinnedCountBuffer, 0, 2*sizeof(unsigned int));
}

MetalNonbondedUtilities::~MetalNonbondedUtilities() {
    if (atomReorder != nullptr)
        context.setDeviceReorder(NULL);
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
    numInteractions++;
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
    useRoleOrder = chooseRoleOrder(system);

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

    if (useCutoff && usePeriodic && !context.getUseDoublePrecision() && context.getNumContexts() == 1) {
        atomReorder.reset(new MetalAtomReorder(context));
        context.setDeviceReorder(atomReorder.get());
    }
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
        if (!waitForCount) {
            countSnapshots = make_shared<CountSnapshots>();
            for (auto& buffer : countSnapshots->buffers)
                buffer = context.createBuffer(2*sizeof(unsigned int), "interactionCountSnapshot");
            context.getCurrentStream().commitWhenIdle();
        }
        uint64_t initialTiles = max<uint64_t>(1, waitForCount ? min<uint64_t>(20ULL*numAtomBlocks, numTiles) : numTiles);
        uint64_t initialPairs = waitForCount ? 5ULL*numAtoms : max<uint64_t>(5ULL*numAtoms, maxPossiblePairs);
        // Leave room for the signed single-pair loop index's final stride.
        uint64_t pairLimit = numeric_limits<int>::max()-(uint64_t) numForceThreadBlocks*forceThreadBlockSize;
        if (initialTiles > MaxNeighborTiles)
            throw OpenMMException("Neighbor list tile capacity "+to_string(initialTiles)+" exceeds the 32-bit atom index limit of "+to_string(MaxNeighborTiles));
        if (initialPairs > pairLimit)
            throw OpenMMException("Neighbor list pair capacity "+to_string(initialPairs)+" exceeds the 32-bit pair index limit of "+to_string(pairLimit));
        maxTiles = (unsigned int) initialTiles;
        maxSinglePairs = (unsigned int) initialPairs;
        // Small systems leave most GPU cores idle during the neighbor list build, so give each atom block several warps.
        numTilesInBatch = (numAtomBlocks < 64*context.getMultiprocessors() ? 4 : 1);
        interactingTiles.initialize<int>(context, maxTiles, "interactingTiles");
        interactingAtoms.initialize<int>(context, (size_t) MetalContext::TileSize*maxTiles, "interactingAtoms");
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
            // Five indirect dispatch triples, then findBlockBounds' arrival counter and moved flag, which start at 0.
            rebuildDispatch.initialize<unsigned int>(context, 20, "rebuildDispatch");
            rebuildDispatch.upload(vector<unsigned int>(20, 0));
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

    MetalArray& listPosq = (useRoleOrder ? orderedPosq : context.unwrap(context.getPosq()));
    forceArgs.push_back(useRoleOrder ? &orderedForces.getDevicePointer() : &context.unwrap(context.getLongForceBuffer()).getDevicePointer());
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
        if (useRoleOrder)
            forceArgs.push_back(&atomOrder.getDevicePointer());
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
        findBlockBoundsArgs.push_back(&listPosq.getDevicePointer());
        findBlockBoundsArgs.push_back(&blockCenter.getDevicePointer());
        findBlockBoundsArgs.push_back(&blockBoundingBox.getDevicePointer());
        findBlockBoundsArgs.push_back(&rebuildNeighborList.getDevicePointer());
        findBlockBoundsArgs.push_back(&blockSizeRange.getDevicePointer());
        if (useIndirectRebuild) {
            findBlockBoundsArgs.push_back(&oldPositions.getDevicePointer());
            findBlockBoundsArgs.push_back(&forceRebuildNeighborList);
            findBlockBoundsArgs.push_back(&interactionCount.getDevicePointer());
            findBlockBoundsArgs.push_back(&pinnedCountDeviceMemory);
            findBlockBoundsArgs.push_back(&rebuildDispatch.getDevicePointer());
            if (useRoleOrder) {
                findBlockBoundsArgs.push_back(&context.unwrap(context.getPosq()).getDevicePointer());
                findBlockBoundsArgs.push_back(&atomOrder.getDevicePointer());
            }
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
        sortBoxDataArgs.push_back(&listPosq.getDevicePointer());
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
        findInteractingBlocksArgs.push_back(&listPosq.getDevicePointer());
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
        if (useRoleOrder)
            findInteractingBlocksArgs.push_back(&atomRoles.getDevicePointer());
        copyInteractionCountsArgs.push_back(&interactionCount.getDevicePointer());
        copyInteractionCountsArgs.push_back(&pinnedCountDeviceMemory);
    }
}

// Role of each particle in a NonbondedForce: 1 if it has LJ, 2 if it has charge, counting parameter offsets.
static vector<int> nonbondedRoles(const NonbondedForce& force) {
    vector<int> roles(force.getNumParticles());
    for (int i = 0; i < force.getNumParticles(); i++) {
        double charge, sigma, epsilon;
        force.getParticleParameters(i, charge, sigma, epsilon);
        roles[i] = (epsilon != 0 ? 1 : 0) | (charge != 0 ? 2 : 0);
    }
    for (int i = 0; i < force.getNumParticleParameterOffsets(); i++) {
        string parameter;
        int particle;
        double chargeScale, sigmaScale, epsilonScale;
        force.getParticleParameterOffset(i, parameter, particle, chargeScale, sigmaScale, epsilonScale);
        roles[particle] |= (epsilonScale != 0 ? 1 : 0) | (chargeScale != 0 ? 2 : 0);
    }
    return roles;
}

void MetalNonbondedUtilities::checkRoleOrder(const NonbondedForce& force) const {
    if (!useRoleOrder)
        return;
    vector<int> roles = nonbondedRoles(force);
    for (int i = 0; i < (int) roles.size(); i++)
        if ((roles[i] & ~contextRoles[i]) != 0)
            throw OpenMMException("updateParametersInContext: A particle gained charge or Lennard-Jones parameters it did not have when the Context was created");
}

// A water model whose oxygen has LJ but no charge and whose other sites have charge but no LJ (TIP4P-Ew, TIP5P, OPC)
// has site pairs that never interact: in a tile of such waters 6 of 16 site pairs do nothing.  Role order puts the
// atoms of small molecules that have only one role after all other atoms, LJ only first and charge only second,
// and the neighbor list then lists an atom only for x atoms that share a role with it.  On the FAH tip4pew unit
// this was priced at 0.68-0.77 of today's nonbonded tile cost (lab lanes/r3-explicit.md).  Pairs of atoms with no
// shared role contribute nothing to any NonbondedForce term, so their exclusions are dropped too.  The roles come
// from NonbondedForce's parameters and offsets, so it must be the only interaction.  TIP3P has no LJ-only sites
// and gets more tiles in this order, so both roles must each hold a tenth of the atoms.
bool MetalNonbondedUtilities::chooseRoleOrder(const System& system) {
    if (!useNeighborList || !useIndirectRebuild || numInteractions != 1 || context.getPlatformData().contexts.size() != 1)
        return false;
    const NonbondedForce* nonbonded = NULL;
    for (int i = 0; i < system.getNumForces(); i++) {
        const NonbondedForce* force = dynamic_cast<const NonbondedForce*>(&system.getForce(i));
        if (force == NULL)
            continue;
        if (nonbonded != NULL)
            return false;
        nonbonded = force;
    }
    if (nonbonded == NULL || nonbonded->getNonbondedMethod() == NonbondedForce::LJPME || nonbonded->getNumParticles() != numAtoms)
        return false;
    vector<int> roles = nonbondedRoles(*nonbonded);

    // Molecules are the groups the exclusions connect.  Only atoms of molecules up to 8 atoms move, so the atoms of
    // a protein stay together.
    vector<int> parent(numAtoms);
    for (int i = 0; i < numAtoms; i++)
        parent[i] = i;
    auto root = [&parent](int i) {
        while (parent[i] != i)
            i = parent[i] = parent[parent[i]];
        return i;
    };
    for (int i = 0; i < numAtoms; i++)
        for (int j : atomExclusions[i])
            parent[root(i)] = root(j);
    vector<int> moleculeSize(numAtoms, 0);
    for (int i = 0; i < numAtoms; i++)
        moleculeSize[root(i)]++;
    vector<vector<int> > regions(4);
    for (int i = 0; i < numAtoms; i++) {
        bool moves = (moleculeSize[root(i)] <= 8 && roles[i] != 3);
        regions[moves ? (roles[i] == 0 ? 3 : roles[i]) : 0].push_back(i);
    }
    if (regions[1].size() < numAtoms/10 || regions[2].size() < numAtoms/10)
        return false;

    int paddedNumAtoms = context.getPaddedNumAtoms();
    vector<int> order, orderedRoles(paddedNumAtoms, 0), position(paddedNumAtoms);
    for (vector<int>& region : regions)
        order.insert(order.end(), region.begin(), region.end());
    for (int i = numAtoms; i < paddedNumAtoms; i++)
        order.push_back(i);
    for (int i = 0; i < paddedNumAtoms; i++) {
        position[order[i]] = i;
        if (i < numAtoms)
            orderedRoles[i] = roles[order[i]];
    }
    vector<vector<int> > orderedExclusions(numAtoms);
    for (int i = 0; i < numAtoms; i++) {
        int atom = order[i];
        for (int j : atomExclusions[atom])
            if (j == atom || (roles[atom] & roles[j]) != 0)
                orderedExclusions[i].push_back(position[j]);
    }
    atomExclusions = orderedExclusions;
    contextRoles = roles;
    atomOrder.initialize<int>(context, paddedNumAtoms, "atomOrder");
    atomRoles.initialize<int>(context, paddedNumAtoms, "atomRoles");
    orderedPosq.initialize(context, paddedNumAtoms, context.getPosq().getElementSize(), "orderedPosq");
    orderedForces.initialize<unsigned long long>(context, 3*paddedNumAtoms, "orderedForces");
    orderedForces.upload(vector<unsigned long long>(3*paddedNumAtoms, 0));
    atomOrder.upload(order);
    inverseAtomOrder.initialize<int>(context, paddedNumAtoms, "inverseRoleAtomOrder");
    inverseAtomOrder.upload(position);
    atomRoles.upload(orderedRoles);
    return true;
}

void MetalNonbondedUtilities::scatterRoleForces() {
    if (!roleForcesPending)
        return;
    roleForcesPending = false;
    void* args[] = {&orderedForces.getDevicePointer(), &atomOrder.getDevicePointer(), &context.unwrap(context.getLongForceBuffer()).getDevicePointer()};
    context.executeKernelFlat(scatterOrderedForcesKernel, args, context.getPaddedNumAtoms(), 64);
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
    if (useCutoff && usePeriodic && !context.getUseDeviceBox()) {
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

    // With the indirect rebuild findBlockBounds gathers the ordered positions itself and its last threadgroup
    // decides the rebuild and sizes the indirect dispatches below (empty on a step without a rebuild).
    if (useRoleOrder && !useIndirectRebuild) {
        void* gatherArgs[] = {&context.unwrap(context.getPosq()).getDevicePointer(), &atomOrder.getDevicePointer(), &orderedPosq.getDevicePointer()};
        context.executeKernelFlat(kernels.gatherOrderedPositionsKernel, gatherArgs, context.getPaddedNumAtoms(), 64);
    }
    context.executeKernelFlat(kernels.findBlockBoundsKernel, &findBlockBoundsArgs[0], context.getPaddedNumAtoms(), FindBlockBoundsThreadBlockSize);
    if (useIndirectRebuild) {
        MTL::Buffer* dispatch = rebuildDispatch.getDevicePointer();
        if (!sortBlocksByBins) {
            void* sortArgs[] = {&blockSizeRange.getDevicePointer(), &blockBoundingBox.getDevicePointer(), &sortedBlocks.getDevicePointer()};
            int sortThreads = min((context.getNumAtomBlocks()+31)/32*32, (int) kernels.sortBlocksKernel->maxTotalThreadsPerThreadgroup());
            context.executeKernelFlat(kernels.sortBlocksKernel, sortArgs, 0, sortThreads, 0, dispatch, 12*sizeof(unsigned int));
        }
        else {
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
    // With the indirect rebuild, findBlockBounds already copied a count, at most one list old,
    // which is all a step without the wait needs.
    // Registered device trials need no fresh count for a fully allocated list.
    // Lists requiring host overflow checks must still wait for exact counts.
    bool wait = (waitForCount || (context.deviceBarostatState == nullptr && context.getStepsSinceReorder() == 0));
    if (context.getStepsSinceReorder() == 0)
        ++reorderEpoch;
    // Each command buffer boundary leaves the GPU idle, so the force kernels go into this buffer
    // too.  computeInteractions() commits it before waiting for the count.
    if (wait || !useIndirectRebuild) {
        context.executeKernelFlat(kernels.copyInteractionCountsKernel, &copyInteractionCountsArgs[0], 1, 1);
    }
    if (!wait && context.getStepsSinceReorder()%StepsPerCommandBuffer == 0)
        sampleInteractionCounts(kernels);
}

void MetalNonbondedUtilities::sampleInteractionCounts(KernelSet& kernels) {
    auto snapshots = countSnapshots;
    int slot = 0;
    unsigned long long sequence;
    {
        lock_guard<mutex> guard(snapshots->lock);
        while (slot < CountSnapshots::NumSlots && snapshots->busy[slot])
            ++slot;
        if (slot == CountSnapshots::NumSlots)
            return;
        snapshots->busy[slot] = true;
        sequence = ++snapshots->nextSequence;
    }
    MetalQueue& queue = context.getCurrentStream();
    // Keep idle commits from separating the GPU copy from its completion handler.
    lock_guard<recursive_mutex> guard(queue.getLock());
    void* args[] = {&interactionCount.getDevicePointer(), &snapshots->buffers[slot]};
    context.executeKernelFlat(kernels.copyInteractionCountsKernel, args, 1, 1);
    unsigned long long epoch = reorderEpoch;
    queue.addCompletedHandler([snapshots, slot, sequence, epoch](MTL::CommandBuffer* buffer) {
        lock_guard<mutex> guard(snapshots->lock);
        if (buffer->status() == MTL::CommandBufferStatusCompleted && sequence > snapshots->sequence) {
            const unsigned int* counts = (const unsigned int*) snapshots->buffers[slot]->contents();
            snapshots->counts[0] = counts[0];
            snapshots->counts[1] = counts[1];
            snapshots->sequence = sequence;
            snapshots->epoch = epoch;
        }
        snapshots->busy[slot] = false;
    });
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
        if (useRoleOrder && includeForces)
            roleForcesPending = true;
    }
    if (useNeighborList && numTiles > 0) {
        // Without the wait, the count comes from an earlier step.  It can't exceed the arrays, so it only
        // guides reordering through completed snapshots.  Exact counts require queue completion.
        // Steps without the wait share a command buffer, up to StepsPerCommandBuffer of them,
        // unless the GPU has run out of work.
        MetalQueue& queue = context.getCurrentStream();
        bool wait = (waitForCount || (context.deviceBarostatState == nullptr && context.getStepsSinceReorder() == 0));
        if (queue.getCommitCount() != commitCountAfterLastStep)
            uncommittedSteps = 0;
        if (wait || ++uncommittedSteps >= StepsPerCommandBuffer || queue.isIdle()) {
            queue.commit();
            uncommittedSteps = 0;
        }
        commitCountAfterLastStep = queue.getCommitCount();
        if (wait) {
            ++countWaitCalls;
            queue.finish();
        }
#ifdef OPENMM_METAL_TEST_HOOKS
        if (wait && exactCountHook)
            exactCountHook(context);
#endif
        updateNeighborListSize();
    }
}

bool MetalNonbondedUtilities::updateNeighborListSize() {
    if (!useCutoff)
        return false;
    bool drained = context.getCurrentStream().isDrained();
    if (drained && (context.getStepsSinceReorder() == 0 || tilesAfterReorder == 0))
        tilesAfterReorder = pinnedCountBuffer[0];
    // Completed count snapshots depend on queue progress and are not checkpoint state.
    // Deterministic contexts must use the fixed reorder interval saved by Common.
    else if (context.getStepsSinceReorder() > 25) {
        unsigned int tiles = 0;
        if (waitForCount && drained)
            tiles = pinnedCountBuffer[0];
        else if (countSnapshots) {
            lock_guard<mutex> guard(countSnapshots->lock);
            if (countSnapshots->epoch == reorderEpoch)
                tiles = countSnapshots->counts[0];
        }
        if (tiles > 1.1*tilesAfterReorder) {
            context.forceReorder();
            ++adaptiveReorderCount;
        }
    }
    // Overflow decisions still require the exact counts after finish().  Snapshots are
    // only for the heuristic; an open buffer can still write the shared exact counts.
    if (!drained)
        return false;
    if (pinnedCountBuffer[0] <= maxTiles && pinnedCountBuffer[1] <= maxSinglePairs)
        return false;

    // The most recent timestep had too many interactions to fit in the arrays.  Make the arrays bigger to prevent
    // this from happening in the future.

    // Prepare every replacement before changing any buffer or capacity.
    unique_ptr<MetalArray> newTiles, newAtoms, newPairs;
    unsigned int newMaxTiles = maxTiles, newMaxSinglePairs = maxSinglePairs;
    if (pinnedCountBuffer[0] > maxTiles) {
        uint64_t numBlocks = context.getNumAtomBlocks();
        uint64_t totalTiles = numBlocks*(numBlocks+1)/2;
        uint64_t limit = min(totalTiles, MaxNeighborTiles);
        if (pinnedCountBuffer[0] > limit)
            throw OpenMMException("Neighbor list requires "+to_string(pinnedCountBuffer[0])+" tiles, exceeding the tile capacity limit of "+to_string(limit));
        newMaxTiles = (unsigned int) min<uint64_t>(6ULL*pinnedCountBuffer[0]/5, limit);
        newTiles.reset(new MetalArray(context, newMaxTiles, sizeof(int), "interactingTiles"));
        newAtoms.reset(new MetalArray(context, (size_t) MetalContext::TileSize*newMaxTiles, sizeof(int), "interactingAtoms"));
    }
    if (pinnedCountBuffer[1] > maxSinglePairs) {
        uint64_t pairLimit = numeric_limits<int>::max()-(uint64_t) numForceThreadBlocks*forceThreadBlockSize;
        if (pinnedCountBuffer[1] > pairLimit)
            throw OpenMMException("Neighbor list requires "+to_string(pinnedCountBuffer[1])+" pairs, exceeding the 32-bit pair index limit of "+to_string(pairLimit));
        newMaxSinglePairs = (unsigned int) min<uint64_t>(6ULL*pinnedCountBuffer[1]/5, pairLimit);
        newPairs.reset(new MetalArray(context, newMaxSinglePairs, sizeof(int2), "singlePairs"));
    }
    if (newTiles) {
        interactingTiles.swap(*newTiles);
        interactingAtoms.swap(*newAtoms);
        maxTiles = newMaxTiles;
        if (forceArgs.size() > 0)
            forceArgs[7] = &interactingTiles.getDevicePointer();
        findInteractingBlocksArgs[6] = &interactingTiles.getDevicePointer();
        if (forceArgs.size() > 0)
            forceArgs[17] = &interactingAtoms.getDevicePointer();
        findInteractingBlocksArgs[7] = &interactingAtoms.getDevicePointer();
    }
    if (newPairs) {
        singlePairs.swap(*newPairs);
        maxSinglePairs = newMaxSinglePairs;
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
        defines["NUM_SIZE_BINS"] = context.intToString(NumSizeBins);
        if (useRoleOrder)
            defines["ROLE_ORDER"] = "1";
        if (sortBlocksByBins) {
            defines["SORT_BLOCKS_BY_BINS"] = "1";
            defines["BIN_SORT_GROUP"] = context.intToString(BinSortThreadBlockSize);
        }
        defines["FIND_BLOCK_BOUNDS_GROUP"] = context.intToString(FindBlockBoundsThreadBlockSize);
        int binShift = 1;
        while (1<<binShift <= context.getNumAtomBlocks())
            binShift++;
        defines["BIN_SHIFT"] = context.intToString(binShift);
        defines["BLOCK_INDEX_MASK"] = context.intToString((1<<binShift)-1);
        metalModule_t interactingBlocksProgram = context.createModule(MetalKernelSources::vectorOps+MetalKernelSources::findInteractingBlocks, defines);
        kernels.findBlockBoundsKernel = context.getKernel(interactingBlocksProgram, "findBlockBounds");
        if (useIndirectRebuild && !sortBlocksByBins)
            kernels.sortBlocksKernel = context.getKernel(interactingBlocksProgram, "sortBlocksInOneGroup");
        if (sortBlocksByBins) {
            kernels.countBlockBinsKernel = context.getKernel(interactingBlocksProgram, "countBlockBins");
            kernels.sumBlockBinsKernel = context.getKernel(interactingBlocksProgram, "sumBlockBins");
            kernels.scatterBlocksKernel = context.getKernel(interactingBlocksProgram, "scatterBlocks");
        }
        kernels.computeSortKeysKernel = context.getKernel(interactingBlocksProgram, "computeSortKeys");
        kernels.sortBoxDataKernel = context.getKernel(interactingBlocksProgram, "sortBoxData");
        kernels.findInteractingBlocksKernel = context.getKernel(interactingBlocksProgram, "findBlocksWithInteractions");
        kernels.copyInteractionCountsKernel = context.getKernel(interactingBlocksProgram, "copyInteractionCounts");
        if (useRoleOrder) {
            kernels.gatherOrderedPositionsKernel = context.getKernel(interactingBlocksProgram, "gatherOrderedPositions");
            scatterOrderedForcesKernel = context.getKernel(interactingBlocksProgram, "scatterOrderedForces");
        }
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
        load1 << loadParameter(param, useRoleOrder ? "NB_ATOM(atom1)" : "atom1");
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
        loadLocal2<<"shfl"<<param.getName()<<" = "<<loadParameter(param, useRoleOrder ? "NB_ATOM(j)" : "j")<<";\n";
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
    if (useRoleOrder)
        defines["ROLE_ORDER"] = "1";
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
