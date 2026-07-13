MODULE BowlStacking
    !=====================================================================
    ! BOWL STACKING MODULE
    !
    ! Task: Detect a set of bowls of different sizes scattered across the
    ! workspace, then nest them into a single stack (largest at the
    ! bottom, smallest on top) without moving the largest bowl.
    !
    !=====================================================================

    !--------------------------------------------------------------
    ! Configuration constants
    !--------------------------------------------------------------
    CONST num MAX_BOWLS      := 10;    ! max array size RAPID can hold
    CONST num APPROACH_HEIGHT:= 80;    ! mm, safe clearance above a bowl
    CONST num NEST_PITCH     := 20;    ! mm, vertical rise per nested bowl
    CONST num GRIP_DWELL     := 0.3;   ! s, pause after each gripper action

    VAR speeddata vApproach := v300;   ! transit speed
    VAR speeddata vPrecise  := v100;   ! near-surface speed
    VAR zonedata  zApproach := z10;    ! blended zone, non-critical points
    VAR zonedata  zPrecise  := fine;   ! exact stop, critical points (grip/release)

    ! Tool and workobject
    PERS tooldata  tGripper  := [TRUE,[[0,0,117.85],[0.966,0,0,0.259]],[0.59,[0.15,0.01,57.98],[1,0,0,0],0,0,0]];
    PERS wobjdata  wobjTable := [FALSE,TRUE,"",[[0,0,0],[1,0,0,0]],[[0,0,0],[1,0,0,0]]];

    !--------------------------------------------------------------
    ! Bowl data type
    !--------------------------------------------------------------
    RECORD bowldata
        robtarget pos;
        num radius;
    ENDRECORD

    VAR bowldata bowls{MAX_BOWLS};
    VAR num nBowls := 0;

    !=====================================================================
    ! MAIN
    !=====================================================================
    PROC main()
        bowl_localizer bowls, nBowls;

        IF nBowls < 2 THEN
            TPWrite "Fewer than 2 bowls detected - nothing to stack.";
            RETURN;
        ENDIF

        SortBowlsBySize;   ! largest -> bowls{1}, smallest -> bowls{nBowls}
        StackBowls;        ! nest bowls{2..nBowls} onto bowls{1} in place

        TPWrite "Bowl stacking complete.";
    ENDPROC

    !=====================================================================
    ! External localization interface (implementation provided elsewhere)
    !=====================================================================
    PROC bowl_localizer(INOUT bowldata bowlList{*}, INOUT num count)
        ! Populates bowlList{1..count} with each detected bowl's
        ! position and radius. Not implemented in this module.
    ENDPROC

    !=====================================================================
    ! Sort bowls largest -> smallest by radius (bubble sort)
    !=====================================================================
    PROC SortBowlsBySize()
        VAR bowldata temp;
        VAR num i;
        VAR num j;

        FOR i FROM 1 TO nBowls - 1 DO
            FOR j FROM 1 TO nBowls - i DO
                IF bowls{j}.radius < bowls{j + 1}.radius THEN
                    temp        := bowls{j};
                    bowls{j}    := bowls{j + 1};
                    bowls{j + 1}:= temp;
                ENDIF
            ENDFOR
        ENDFOR
    ENDPROC

    !=====================================================================
    ! Nest every bowl except the largest onto the largest bowl's position
    !=====================================================================
    PROC StackBowls()
        VAR robtarget stackBase;
        VAR num stackHeight;
        VAR num i;

        stackBase  := bowls{1}.pos;   ! largest bowl stays in place
        stackHeight:= 0;

        FOR i FROM 2 TO nBowls DO
            stackHeight := stackHeight + NEST_PITCH;
            PickAndPlaceBowl bowls{i}.pos, Offs(stackBase, 0, 0, stackHeight);
        ENDFOR
    ENDPROC

    !=====================================================================
    ! Pick-and-place routine for a single bowl
    !=====================================================================
    PROC PickAndPlaceBowl(robtarget fromPos, robtarget toPos)
        ! --- Pick ---
        MoveJ Offs(fromPos, 0, 0, APPROACH_HEIGHT), vApproach, zApproach, tGripper \WObj:=wobjTable;
        EGP_OPEN;
        MoveL fromPos, vPrecise, zPrecise, tGripper \WObj:=wobjTable;
        EGP_CLOSE;
        WaitTime GRIP_DWELL;
        MoveL Offs(fromPos, 0, 0, APPROACH_HEIGHT), vPrecise, zApproach, tGripper \WObj:=wobjTable;

        ! --- Place ---
        MoveJ Offs(toPos, 0, 0, APPROACH_HEIGHT), vApproach, zApproach, tGripper \WObj:=wobjTable;
        MoveL toPos, vPrecise, zPrecise, tGripper \WObj:=wobjTable;
        EGP_OPEN;
        WaitTime GRIP_DWELL;
        MoveL Offs(toPos, 0, 0, APPROACH_HEIGHT), vPrecise, zApproach, tGripper \WObj:=wobjTable;
    ENDPROC

ENDMODULE
