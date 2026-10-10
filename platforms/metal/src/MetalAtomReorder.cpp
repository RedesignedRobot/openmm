/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 * -------------------------------------------------------------------------- */

#include "MetalAtomReorder.h"
#include "MetalContext.h"
#include "MetalKernelSources.h"
#include "MetalQueue.h"
#include "openmm/OpenMMException.h"
#include <algorithm>
#include <cstring>
#include <map>

using namespace OpenMM;
using namespace std;

// The key packs group, bin and molecule index into 12, 28 and 24 bits (GROUP_SHIFT and BIN_SHIFT in
// reorderAtoms.metal).  A system that doesn't fit is reordered on the host.
static const int MaxGroups = 1<<12;
static const long long MaxBins = 1<<28;
static const int MaxMolecules = 1<<24;

// Molecules with more atoms than this get a threadgroup of LargeBlockSize threads to sum their center.
static const int LargeMolecule = 32;
static const int LargeBlockSize = 256;

class MoleculeKeyTrait : public ComputeSortImpl::SortTrait {
public:
    int getDataSize() const {return sizeof(unsigned long long);}
    int getKeySize() const {return sizeof(unsigned long long);}
    const char* getDataType() const {return "ulong";}
    const char* getKeyType() const {return "ulong";}
    const char* getMinKey() const {return "0";}
    const char* getMaxKey() const {return "0xFFFFFFFFFFFFFFFFUL";}
    const char* getMaxValue() const {return "0xFFFFFFFFFFFFFFFFUL";}
    const char* getSortKey() const {return "value";}
};

MetalAtomReorder::MetalAtomReorder(MetalContext& context) : context(context), tablesVersion(-1), tablesUsable(false),
        hostStale(false), deviceOffsetsStale(true), numLarge(0) {
    map<string, string> defines;
    defines["LARGE_MOLECULE"] = context.intToString(LargeMolecule);
    defines["LARGE_BLOCK_SIZE"] = context.intToString(LargeBlockSize);
    if (context.getUseDeviceBox())
        defines["DEVICE_BOX_ENERGY"] = "1";
    ComputeProgram program = context.compileProgram(MetalKernelSources::reorderAtoms, defines);
    if (context.getUseDeviceBox()) {
        scaleKernel = program->createKernel("scaleMoleculeCenters");
        for (int i = 0; i < 5; i++)
            scaleKernel->addArg();
    }
    bool mixed = context.getUseMixedPrecision();
    keysKernel = program->createKernel("computeMoleculeKeys");
    for (int i = 0; i < 15; i++)
        keysKernel->addArg();
    largeKeysKernel = program->createKernel("computeLargeMoleculeKeys");
    for (int i = 0; i < 16; i++)
        largeKeysKernel->addArg();
    gatherKernel = program->createKernel("gatherAtoms");
    for (int i = 0; i < (mixed ? 20 : 19); i++)
        gatherKernel->addArg();
    storeKernel = program->createKernel("storeAtoms");
    for (int i = 0; i < (mixed ? 10 : 9); i++)
        storeKernel->addArg();
    int paddedNumAtoms = context.getPaddedNumAtoms();
    scratch.reset(new MetalArray(context, (size_t) paddedNumAtoms*(mixed ? 4 : 2), sizeof(mm_float4), "reorderScratch"));
    scratchIndex.reset(new MetalArray(context, paddedNumAtoms, sizeof(int), "reorderScratchIndex"));
    scratchOffsets.reset(new MetalArray(context, paddedNumAtoms, sizeof(mm_int4), "reorderScratchOffsets"));
    nanFlag.reset(new MetalArray(context, 1, sizeof(int), "reorderNaNFlag"));
    *(int*) nanFlag->getDevicePointer()->contents() = 0;
    if (context.getUseDeviceBox()) {
        cellOffsets.reset(new MetalArray(context, context.getNumAtoms(), sizeof(mm_int4), "reorderCellOffsets"));
        memset(cellOffsets->getDevicePointer()->contents(), 0, context.getNumAtoms()*sizeof(mm_int4));
        deviceOffsetsStale = false;
    }
}

bool MetalAtomReorder::buildTables() {
    // Fresh buffers are not in use by the GPU, so the host fills them directly.

    tablesVersion = context.getMoleculeGroupsVersion();
    tablesUsable = false;
    const vector<ComputeContext::MoleculeGroup>& groups = context.getMoleculeGroups();
    int numAtoms = context.getNumAtoms();
    if (groups.size() == 0 || groups.size() > MaxGroups)
        return false;
    vector<mm_int4> moleculeVec;
    vector<int> templateVec;
    vector<int> slotVec(numAtoms, -1);
    vector<int> largeVec;
    for (int g = 0; g < (int) groups.size(); g++) {
        const vector<int>& atoms = groups[g].atoms;
        const vector<int>& offsets = groups[g].offsets;
        bool useHilbert = (offsets.size() > 5000 || atoms.size() > 8); // As in ComputeContext::findAtomOrder().
        int templateStart = templateVec.size();
        templateVec.insert(templateVec.end(), atoms.begin(), atoms.end());
        for (int offset : offsets) {
            int m = moleculeVec.size();
            if (m >= MaxMolecules)
                return false;
            moleculeVec.push_back(mm_int4(offset, 2*g+(useHilbert ? 1 : 0), templateStart, atoms.size()));
            if (atoms.size() > LargeMolecule)
                largeVec.push_back(m);
            for (int atom : atoms) {
                int slot = offset+atom;
                if (slot < 0 || slot >= numAtoms || slotVec[slot] != -1)
                    return false;
                slotVec[slot] = m;
            }
        }
    }
    for (int m : slotVec)
        if (m == -1)
            return false;
    int numMolecules = moleculeVec.size();
    molecules.reset(new MetalArray(context, numMolecules, sizeof(mm_int4), "reorderMolecules"));
    memcpy(molecules->getDevicePointer()->contents(), moleculeVec.data(), numMolecules*sizeof(mm_int4));
    templateAtoms.reset(new MetalArray(context, templateVec.size(), sizeof(int), "reorderTemplateAtoms"));
    memcpy(templateAtoms->getDevicePointer()->contents(), templateVec.data(), templateVec.size()*sizeof(int));
    slotMolecule.reset(new MetalArray(context, numAtoms, sizeof(int), "reorderSlotMolecule"));
    memcpy(slotMolecule->getDevicePointer()->contents(), slotVec.data(), numAtoms*sizeof(int));
    numLarge = largeVec.size();
    largeMolecules.reset();
    if (numLarge > 0) {
        largeMolecules.reset(new MetalArray(context, numLarge, sizeof(int), "reorderLargeMolecules"));
        memcpy(largeMolecules->getDevicePointer()->contents(), largeVec.data(), numLarge*sizeof(int));
    }
    keys.reset(new MetalArray(context, numMolecules, sizeof(unsigned long long), "reorderKeys"));
    moleculeCells.reset(new MetalArray(context, numMolecules, sizeof(mm_int4), "reorderMoleculeCells"));
    sort = context.createSort(new MoleculeKeyTrait(), numMolecules, false);
    tablesUsable = true;
    return true;
}

void MetalAtomReorder::uploadCellOffsets() {
    const vector<mm_int4>& offsets = hostCellOffsets(context);
    cellOffsets.reset(new MetalArray(context, offsets.size(), sizeof(mm_int4), "reorderCellOffsets"));
    memcpy(cellOffsets->getDevicePointer()->contents(), offsets.data(), offsets.size()*sizeof(mm_int4));
    deviceOffsetsStale = false;
}

void MetalAtomReorder::checkForNaN() {
    if (context.getCurrentStream().isDrained() && *(int*) nanFlag->getDevicePointer()->contents() != 0) {
        // Report this operation once, so explicit position replacement can recover.
        context.clearBuffer(*nanFlag);
        throw OpenMMException("Particle coordinate is NaN.  For more information, see https://github.com/openmm/openmm/wiki/Frequently-Asked-Questions#nan");
    }
}

bool MetalAtomReorder::discardNaN() {
    context.getCurrentStream().finish();
    int& flag = *(int*) nanFlag->getDevicePointer()->contents();
    bool pending = (flag != 0);
    flag = 0;
    return pending;
}

void MetalAtomReorder::restoreNaN() {
    context.getCurrentStream().finish();
    *(int*) nanFlag->getDevicePointer()->contents() = 1;
}

bool MetalAtomReorder::hasUsableTables() {
    if (tablesVersion != context.getMoleculeGroupsVersion())
        buildTables();
    return tablesUsable;
}

bool MetalAtomReorder::apply() {
    if (!hasUsableTables())
        return false;
    checkForNaN();

    // Bins as in ComputeContext::findAtomOrder() for a periodic system, whose range is the box.

    float zigzagWidth = (float) (0.2*context.getNonbondedUtilities().getMaxCutoffDistance());
    float invZigzagWidth = (float) (1.0/zigzagWidth);
    float invHilbertWidth = 0;
    int zigzagBins[3] = {};
    mm_float4 hi[3] = {}, lo[3] = {}, invBoxSize = {};
    if (!context.getUseDeviceBox()) {
        Vec3 box[3];
        context.getPeriodicBoxVectors(box[0], box[1], box[2]);
        float boxSize[3] = {(float) box[0][0], (float) box[1][1], (float) box[2][2]};
        float hilbertWidth = (float) (max(max(boxSize[0], boxSize[1]), boxSize[2])/255.0);
        invHilbertWidth = (float) (1.0/hilbertWidth);
        for (int i = 0; i < 3; i++)
            zigzagBins[i] = 1+(int) (boxSize[i]*invZigzagWidth);
        if ((long long) zigzagBins[0]*zigzagBins[1]*(zigzagBins[2]+1) >= MaxBins)
            return false;
        for (int i = 0; i < 3; i++) {
            float h[3], l[3];
            for (int j = 0; j < 3; j++) {
                h[j] = (float) box[i][j];
                l[j] = (float) (box[i][j]-h[j]);
            }
            hi[i] = mm_float4(h[0], h[1], h[2], 0);
            lo[i] = mm_float4(l[0], l[1], l[2], 0);
        }
        invBoxSize = mm_float4((float) (1.0/box[0][0]), (float) (1.0/box[1][1]), (float) (1.0/box[2][2]), 0);
    }
    if (deviceOffsetsStale)
        uploadCellOffsets();
    int numMolecules = molecules->getSize();
    int numAtoms = context.getNumAtoms();
    int paddedNumAtoms = context.getPaddedNumAtoms();
    bool mixed = context.getUseMixedPrecision();
    int arg = 0;
    keysKernel->setArg(arg++, numMolecules);
    keysKernel->setArg(arg++, *molecules);
    keysKernel->setArg(arg++, *templateAtoms);
    keysKernel->setArg(arg++, context.getPosq());
    for (int i = 0; i < 3; i++)
        keysKernel->setArg(arg++, hi[i]);
    keysKernel->setArg(arg++, invBoxSize);
    keysKernel->setArg(arg++, invHilbertWidth);
    keysKernel->setArg(arg++, invZigzagWidth);
    keysKernel->setArg(arg++, zigzagBins[0]);
    keysKernel->setArg(arg++, zigzagBins[1]);
    keysKernel->setArg(arg++, *keys);
    keysKernel->setArg(arg++, *moleculeCells);
    keysKernel->setArg(arg++, *nanFlag);
    // Clear in stream order before either key kernel can report this operation.
    context.clearBuffer(*nanFlag);
    keysKernel->execute(numMolecules);
    if (numLarge > 0) {
        arg = 0;
        largeKeysKernel->setArg(arg++, numLarge);
        largeKeysKernel->setArg(arg++, *largeMolecules);
        largeKeysKernel->setArg(arg++, *molecules);
        largeKeysKernel->setArg(arg++, *templateAtoms);
        largeKeysKernel->setArg(arg++, context.getPosq());
        for (int i = 0; i < 3; i++)
            largeKeysKernel->setArg(arg++, hi[i]);
        largeKeysKernel->setArg(arg++, invBoxSize);
        largeKeysKernel->setArg(arg++, invHilbertWidth);
        largeKeysKernel->setArg(arg++, invZigzagWidth);
        largeKeysKernel->setArg(arg++, zigzagBins[0]);
        largeKeysKernel->setArg(arg++, zigzagBins[1]);
        largeKeysKernel->setArg(arg++, *keys);
        largeKeysKernel->setArg(arg++, *moleculeCells);
        largeKeysKernel->setArg(arg++, *nanFlag);
        largeKeysKernel->execute(numLarge*LargeBlockSize, LargeBlockSize);
    }
    sort->sort(*keys);
    arg = 0;
    gatherKernel->setArg(arg++, numAtoms);
    gatherKernel->setArg(arg++, paddedNumAtoms);
    gatherKernel->setArg(arg++, *slotMolecule);
    gatherKernel->setArg(arg++, *molecules);
    gatherKernel->setArg(arg++, *keys);
    gatherKernel->setArg(arg++, *moleculeCells);
    gatherKernel->setArg(arg++, context.getPosq());
    if (mixed)
        gatherKernel->setArg(arg++, context.getPosqCorrection());
    gatherKernel->setArg(arg++, context.getVelm());
    gatherKernel->setArg(arg++, context.getAtomIndexArray());
    gatherKernel->setArg(arg++, *cellOffsets);
    gatherKernel->setArg(arg++, *scratch);
    gatherKernel->setArg(arg++, *scratchIndex);
    gatherKernel->setArg(arg++, *scratchOffsets);
    for (int i = 0; i < 3; i++)
        gatherKernel->setArg(arg++, hi[i]);
    for (int i = 0; i < 3; i++)
        gatherKernel->setArg(arg++, lo[i]);
    gatherKernel->execute(paddedNumAtoms);
    arg = 0;
    storeKernel->setArg(arg++, numAtoms);
    storeKernel->setArg(arg++, paddedNumAtoms);
    storeKernel->setArg(arg++, *scratch);
    storeKernel->setArg(arg++, *scratchIndex);
    storeKernel->setArg(arg++, *scratchOffsets);
    storeKernel->setArg(arg++, context.getPosq());
    if (mixed)
        storeKernel->setArg(arg++, context.getPosqCorrection());
    storeKernel->setArg(arg++, context.getVelm());
    storeKernel->setArg(arg++, context.getAtomIndexArray());
    storeKernel->setArg(arg++, *cellOffsets);
    storeKernel->execute(paddedNumAtoms);
    hostStale = true;
    return true;
}

void MetalAtomReorder::syncHost() {
    // Readers on other threads must not see the flag cleared before the copies are done.
    if (!hostStale.load(memory_order_acquire))
        return;
    lock_guard<mutex> lock(syncLock);
    if (!hostStale.load(memory_order_relaxed))
        return;
    context.getCurrentStream().finish();
    int numAtoms = context.getNumAtoms();
    const int* index = (const int*) context.unwrap(context.getAtomIndexArray()).getDevicePointer()->contents();
    const mm_int4* offsets = (const mm_int4*) cellOffsets->getDevicePointer()->contents();
    copy(index, index+numAtoms, hostAtomIndex(context).begin());
    copy(offsets, offsets+numAtoms, hostCellOffsets(context).begin());
    hostStale.store(false, memory_order_release);
    checkForNaN();
}

void MetalAtomReorder::hostChanged() {
    // getPositions() reaches this once per atom from every thread pool worker.  Storing only when the flag
    // is clear keeps the workers from fighting over one cache line.
    if (!deviceOffsetsStale.load(memory_order_relaxed))
        deviceOffsetsStale.store(true, memory_order_relaxed);
}

MetalArray& MetalAtomReorder::getCellOffsets() {
    if (deviceOffsetsStale)
        uploadCellOffsets();
    return *cellOffsets;
}

void MetalAtomReorder::deviceOffsetsChanged() {
    deviceOffsetsStale = false;
    hostStale = true;
}

void MetalAtomReorder::prepareScale() {
    if (!hasUsableTables())
        throw OpenMMException("Device box scaling does not support this molecule layout");
}

void MetalAtomReorder::scaleCoordinates(MetalArray& scaleFactor) {
    scaleKernel->setArg(0, molecules->getSize());
    scaleKernel->setArg(1, *molecules);
    scaleKernel->setArg(2, *templateAtoms);
    scaleKernel->setArg(3, context.getPosq());
    scaleKernel->setArg(4, scaleFactor);
    scaleKernel->execute(molecules->getSize());
}
