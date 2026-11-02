#ifndef CURRENT_CBC_UNIT_STATE_SHARED_HLSL
#define CURRENT_CBC_UNIT_STATE_SHARED_HLSL

// CBC selection has adopted the physical graph for this unit. This says
// nothing about black-colour confidence or a learned rim width. Inside's
// independent graph has a different readiness contract and does not use this.
// Scalar-only code is also included by the CPU contract truth-table test.
bool CurrentCBCPhysicalRefinementAdopted(uint evaluated, uint fullyObserved,
                                       uint enclosedBodyPixels, uint selectionStatus,
                                       uint graphReady) {
    return evaluated!=0u&&fullyObserved==1u&&enclosedBodyPixels>0u&&
        selectionStatus==1u&&graphReady==1u;
}

#endif
