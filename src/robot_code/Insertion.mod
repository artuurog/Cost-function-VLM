MODULE BeadInsertion
    !=======================================================================
    ! Bead-on-stick insertion task
    ! Picks NUM_BEADS beads from the table and inserts each onto its
    !=======================================================================

    !--- Task configuration ------------------------------------------------
    CONST num NUM_BEADS := 3;

    ! Clearance / offset distances [mm]
    CONST num zApproachTable := 40;  ! clearance above bead before/after grasp
    CONST num zApproachStick := 50;  ! clearance above stick before insertion
    CONST num zDepartStick   := 50;  ! retreat height after releasing bead

    ! Settle times [s]
    CONST num tGripSettle    := 0.3; ! after closing gripper on bead
    CONST num tReleaseSettle := 0.3; ! after opening gripper on stick

    ! Speed data
    CONST speeddata vTransit  := v300; ! free-space transit moves (MoveJ)
    CONST speeddata vApproach := v100; ! approach/depart moves (MoveL)
    CONST speeddata vInsert   := v20;  ! slow, precise insertion (position controlled, no compliance)

    ! Zone data
    CONST zonedata zTransit := z10;

    !--- Tool / work object (calibrate for the actual cell) ----------------
    PERS tooldata tGripper := [TRUE,[[0,0,117.85],[0.966,0,0,0.259]],[0.59,[0.15,0.01,57.98],[1,0,0,0],0,0,0]];
    PERS wobjdata wobjStation := [FALSE,TRUE,"",[[0,0,0],[1,0,0,0]],[[0,0,0],[1,0,0,0]]];

    LOCAL CONST robtarget pHome:=[[-310,85,370],[0.00853,0.98276,0.18332,-0.02265],[0,0,0,0],[9E9,9E9,9E9,9E9,9E9,9E9]];

    VAR robtarget beadPos{NUM_BEADS};
    VAR robtarget stickPos{NUM_BEADS};

!---------------------------------------------------------------------

    PROC Main()
        Initialize;

        FOR i FROM 1 TO NUM_BEADS DO
            PickAndInsertBead i;
        ENDFOR

        MoveAbsJ jHome, vTransit, fine, tGripper\WObj:=wobjStation;
    ENDPROC

    PROC Initialize()
        EGP_OPEN;
        beads_positions(beadPos);
        target_positions(stickPos);
    ENDPROC

    PROC PickAndInsertBead(num idx)
        PickBead beadPos{idx};
        InsertBead stickPos{idx};
    ENDPROC

    ! Grasp a bead resting on the table
    PROC PickBead(robtarget pBead)
        MoveJ Offs(pBead, 0, 0, zApproachTable), vTransit, zTransit, tGripper\WObj:=wobjStation;
        EGP_OPEN;
        MoveL pBead, vApproach, fine, tGripper\WObj:=wobjStation;
        EGP_CLOSE;
        WaitTime tGripSettle;
        MoveL Offs(pBead, 0, 0, zApproachTable), vApproach, zTransit, tGripper\WObj:=wobjStation;
    ENDPROC

    ! Carry the held bead to its stick and insert it
    PROC InsertBead(robtarget pStick)
        MoveJ Offs(pStick, 0, 0, zApproachStick), vTransit, zTransit, tGripper\WObj:=wobjStation;
        MoveL pStick, vInsert, fine, tGripper\WObj:=wobjStation;
        EGP_OPEN;
        WaitTime tReleaseSettle;
        MoveL Offs(pStick, 0, 0, zDepartStick), vApproach, zTransit, tGripper\WObj:=wobjStation;
    ENDPROC

ENDMODULE
