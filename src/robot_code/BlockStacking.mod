MODULE BlockStacking
    !=====================================================================
    ! BLOCK STACKING TASK
    !
    !=====================================================================

    !--------------------- DATA TYPES -----------------------------------
    RECORD Blocks
        robtarget pos;      ! block pose, as reported by the localizer
        num colorId;        ! detected color ID (1, 2, or 3)
    ENDRECORD

    !--------------------- TOOL / INPUT DATA -----------------------------
    LOCAL PERS tooldata gripper:=[TRUE,[[0,0,117.85],[0.966,0,0,0.259]],[0.59,[0.15,0.01,57.98],[1,0,0,0],0,0,0]];
    LOCAL PERS wobjdata wWorkArea:=[FALSE,TRUE,"",[[0,0,0],[1,0,0,0]],[[0,0,0],[1,0,0,0]]];

    PERS Blocks localizedBlocks{6};

    CONST robtarget pyramidBase := [[-400,0, 20],[1,0,0,0],[0,0,0,0],[9E9,9E9,9E9,9E9,9E9,9E9]];

    ! Safe home position between operations - update as needed.
    LOCAL CONST robtarget pHome:=[[-310,85,370],[0.00853,0.98276,0.18332,-0.02265],[0,0,0,0],[9E9,9E9,9E9,9E9,9E9,9E9]];

    !--------------------- CONFIGURATION -----
    CONST num BLOCK_PITCH        := 100;  ! [mm] center-to-center spacing, Layer 1 (X axis)
    CONST num BLOCK_HEIGHT       := 50;   ! [mm] block height = vertical step between layers
    CONST num APPROACH_CLEARANCE := 50;   ! [mm] vertical clearance for safe approach/depart
    CONST num tGripDelay         := 0.3;  ! [s] settle time after gripper actuation

    CONST speeddata vApproach := [300, 500, 5000, 1000];  ! precise pick/place speed
    CONST speeddata vTransit  := [1000, 1000, 5000, 1000]; ! fast point-to-point speed

    ! Per-layer X offsets from pyramidBase (uniform spacing within a layer).
    CONST num Layer1XOffset{3} := [-BLOCK_PITCH, 0, BLOCK_PITCH];
    CONST num Layer2XOffset{2} := [-BLOCK_PITCH/2, BLOCK_PITCH/2];
    CONST num Layer3XOffset{1} := [0];

    ! Per-layer Z offset from pyramidBase (stacking height).
    CONST num LayerZOffset{3} := [0, BLOCK_HEIGHT, 2*BLOCK_HEIGHT];

    !=====================================================================
    FUNC robtarget GetLayerTarget(num layer, num posIdx)
        VAR num xOff;

        TEST layer
        CASE 1:
            xOff := Layer1XOffset{posIdx};
        CASE 2:
            xOff := Layer2XOffset{posIdx};
        CASE 3:
            xOff := Layer3XOffset{posIdx};
        DEFAULT:
            TPWrite "GetLayerTarget: invalid layer index " + NumToStr(layer,0);
            Stop;
        ENDTEST

        RETURN Offs(pyramidBase, xOff, 0, LayerZOffset{layer});
    ENDFUNC

    !=====================================================================
    PROC PickAndPlaceBlock(robtarget pickTarget, robtarget placeTarget)
        ! ----- Pick -----
        MoveJ Offs(pickTarget, 0, 0, APPROACH_CLEARANCE), vTransit, z10, tGripper;
        MoveL pickTarget, vApproach, fine, tGripper;
        EGP_CLOSE;
        WaitTime tGripDelay;
        MoveL Offs(pickTarget, 0, 0, APPROACH_CLEARANCE), vApproach, z10, tGripper;

        ! ----- Place -----
        MoveJ Offs(placeTarget, 0, 0, APPROACH_CLEARANCE), vTransit, z10, tGripper;
        MoveL placeTarget, vApproach, fine, tGripper;
        EGP_OPEN;
        WaitTime tGripDelay;
        MoveL Offs(placeTarget, 0, 0, APPROACH_CLEARANCE), vApproach, z10, tGripper;
    ENDPROC

    !=====================================================================
    PROC Main()
        VAR num layer;
        VAR num posIdx;
        VAR num i;
        VAR num nBlocks;

        nBlocks := Dim(localizedBlocks, 1);

        MoveJ pHome, vTransit, fine, tGripper;
        EGP_OPEN;

        FOR layer FROM 1 TO 3 DO
            posIdx := 1;
            FOR i FROM 1 TO nBlocks DO
                IF localizedBlocks{i}.colorId = layer THEN
                    PickAndPlaceBlock(localizedBlocks{i}.pos, GetLayerTarget(layer, posIdx));
                    posIdx := posIdx + 1;
                ENDIF
            ENDFOR
        ENDFOR

        MoveJ pHome, vTransit, fine, tGripper;
    ENDPROC

ENDMODULE
