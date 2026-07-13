MODULE Sorting
    !======================================================================
    ! SortingModule
    !
    !======================================================================

    !----------------------------------------------------------------------
    ! Configuration constants
    !----------------------------------------------------------------------
    CONST num nNumObjects    := 5;     ! total items to sort
    CONST num nNumContainers := 2;     ! total containers in use
    CONST num nClearance     := 50;    ! approach/depart clearance above target [mm]
    CONST num nStackStep     := 15;    ! vertical rise per item
    CONST num nGripDwell     := 0.3; 

    !---------------------------------------------------------
    ! TOOL & WORK OBJECT
    !---------------------------------------------------------
    LOCAL PERS tooldata gripper:=[TRUE,[[0,0,117.85],[0.966,0,0,0.259]],[0.59,[0.15,0.01,57.98],[1,0,0,0],0,0,0]];
    LOCAL PERS wobjdata wWorkArea:=[FALSE,TRUE,"",[[0,0,0],[1,0,0,0]],[[0,0,0],[1,0,0,0]]];

    ! Symbolic container IDs (match container_A / container_B in the PDDL problem)
    CONST num CONTAINER_A := 1;
    CONST num CONTAINER_B := 2;

    ! Sorting rule: nContainerOfObject{objNo} = target container for that object
    ! (directly mirrors the assigned-to facts in sorting_problem.pddl)
    CONST num nContainerOfObject{5} := [CONTAINER_A, CONTAINER_A, CONTAINER_A,
                                         CONTAINER_B, CONTAINER_B];

    ! Speed / zone data
    VAR speeddata vApproach := v1000;  ! transit moves
    VAR speeddata vPrecise  := v200;   ! final approach / depart near objects
    VAR zonedata  zTransit  := z10;    ! blended zone for transit moves

    VAR num nItemsInContainer{2};

    ! TODO: replace with actual calibrated home position
    LOCAL CONST robtarget pHome:=[[-310,85,370],[0.00853,0.98276,0.18332,-0.02265],[0,0,0,0],[9E9,9E9,9E9,9E9,9E9,9E9]];

    !----------------------------------------------------------------------
    ! PROC main
    ! Entry point: sorts all configured objects, then returns home.
    !----------------------------------------------------------------------
    PROC main()
        Initialize;

        FOR nObj FROM 1 TO nNumObjects DO
            SortObject nObj, nContainerOfObject{nObj};
        ENDFOR

        MoveAbsJ pHome, vApproach, zTransit, tGripper\WObj:=wobjTable;
    ENDPROC

    !----------------------------------------------------------------------
    ! PROC Initialize
    ! Resets gripper and per-container item counters before a run.
    !----------------------------------------------------------------------
    PROC Initialize()
        EGP_OPEN;

        FOR nCont FROM 1 TO nNumContainers DO
            nItemsInContainer{nCont} := 0;
        ENDFOR
    ENDPROC

    PROC SortObject(num nObj, num nCont)
        VAR robtarget pPick;
        VAR robtarget pPlace;
        VAR robtarget pPickApproach;
        VAR robtarget pPlaceApproach;

        ! --- Resolve positions from external functions ---
        pPick  := GetObjectPosition(nObj);
        pPlace := GetContainerTarget(nCont);

        ! Raise drop point so each new item clears items already in the container
        pPlace := Offs(pPlace, 0, 0, nItemsInContainer{nCont} * nStackStep);

        pPickApproach  := Offs(pPick,  0, 0, nClearance);
        pPlaceApproach := Offs(pPlace, 0, 0, nClearance);

        ! --- Pick ---
        MoveJ pPickApproach, vApproach, zTransit, tGripper\WObj:=wobjTable;
        MoveL pPick, vPrecise, fine, tGripper\WObj:=wobjTable;
        EGP_CLOSE;
        WaitTime nGripDwell;
        MoveL pPickApproach, vPrecise, zTransit, tGripper\WObj:=wobjTable;

        ! --- Place ---
        MoveJ pPlaceApproach, vApproach, zTransit, tGripper\WObj:=wobjTable;
        MoveL pPlace, vPrecise, fine, tGripper\WObj:=wobjTable;
        EGP_OPEN;
        WaitTime nGripDwell;
        MoveL pPlaceApproach, vPrecise, zTransit, tGripper\WObj:=wobjTable;

        nItemsInContainer{nCont} := nItemsInContainer{nCont} + 1;
    ENDPROC

ENDMODULE
