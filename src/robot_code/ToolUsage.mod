MODULE ToolUsage

    !-----------------------------------------------------------
    ! Configuration - centralized for easy tuning
    !-----------------------------------------------------------
    CONST num CLEARANCE_Z := 30;        ! mm, safe height above target/rack before final descent
    CONST num GRIP_SETTLE_TIME := 0.3;  ! s, dwell after gripper open/close

    VAR speeddata vApproach := v200;    ! transit speed
    VAR speeddata vContact  := v50;     ! slow, controlled speed for tip/grasp contact
    VAR zonedata  zTransit  := z10;     ! blended zone for transit moves
    VAR zonedata  zFine     := fine;    ! exact stop at grasp/contact points

    PERS tooldata tGripper := [TRUE,[[0,0,117.85],[0.966,0,0,0.259]],[0.59,[0.15,0.01,57.98],[1,0,0,0],0,0,0]];
        ! TCP of the empty gripper 

    PERS tooldata tToolTip := [TRUE,[[0,0,0],[1,0,0,0]],[0.5,[0,0,1],[1,0,0,0],0,0,0]];
        ! TCP for the grasped tool's tip

    PERS wobjdata wobjStation := [FALSE,TRUE,"",[[0,0,0],[1,0,0,0]],[[0,0,0],[1,0,0,0]]];
        ! Station work object

    CONST robtarget rToolRack := [[400,0,50],[0,0,1,0],[0,0,0,0],[9E9,9E9,9E9,9E9,9E9,9E9]];
        ! Pick pose for tool


    PROC Main(robtarget targetPoints{*})
        Initialize;
        GraspTool;
        LocalizeTip;
        PlaceToolAtTargets targetPoints;
        ReleaseTool;
    ENDPROC

    !-----------------------------------------------------------
    ! One-time setup
    !-----------------------------------------------------------
    PROC Initialize()
        EGP_OPEN;
        WaitTime GRIP_SETTLE_TIME;
    ENDPROC

    !-----------------------------------------------------------
    ! Move to the rack, grasp the tool, retract
    !-----------------------------------------------------------
    PROC GraspTool()
        MoveJ Offs(rToolRack, 0, 0, CLEARANCE_Z), vApproach, zTransit, tGripper\WObj:=wobjStation;
        MoveL rToolRack, vContact, zFine, tGripper\WObj:=wobjStation;

        EGP_CLOSE;
        WaitTime GRIP_SETTLE_TIME;

        MoveL Offs(rToolRack, 0, 0, CLEARANCE_Z), vContact, zTransit, tGripper\WObj:=wobjStation;
    ENDPROC

    !-----------------------------------------------------------
    ! Call the external tip localization once, right after grasp.
    ! Result is reused for every placement in this cycle.
    !-----------------------------------------------------------
    PROC LocalizeTip()
        tToolTip := LocalizeToolTip();
    ENDPROC

    !-----------------------------------------------------------
    ! Loop over all targets and place the tip on each
    !-----------------------------------------------------------
    PROC PlaceToolAtTargets(robtarget targetPoints{*})
        VAR num nTargets;
        VAR num i;

        nTargets := Dim(targetPoints, 1);

        FOR i FROM 1 TO nTargets DO
            PlaceTipAtTarget targetPoints{i};
        ENDFOR
    ENDPROC

    !-----------------------------------------------------------
    ! Approach, contact, and retract for a single target
    !-----------------------------------------------------------
    PROC PlaceTipAtTarget(robtarget target)
        MoveJ Offs(target, 0, 0, CLEARANCE_Z), vApproach, zTransit, tToolTip\WObj:=wobjStation;
        MoveL target, vContact, zFine, tToolTip\WObj:=wobjStation;
        MoveL Offs(target, 0, 0, CLEARANCE_Z), vContact, zTransit, tToolTip\WObj:=wobjStation;
    ENDPROC

    !-----------------------------------------------------------
    ! Return the tool to the rack and release it
    !-----------------------------------------------------------
    PROC ReleaseTool()
        MoveJ Offs(rToolRack, 0, 0, CLEARANCE_Z), vApproach, zTransit, tGripper\WObj:=wobjStation;
        MoveL rToolRack, vContact, zFine, tGripper\WObj:=wobjStation;

        EGP_OPEN;
        WaitTime GRIP_SETTLE_TIME;

        MoveL Offs(rToolRack, 0, 0, CLEARANCE_Z), vContact, zTransit, tGripper\WObj:=wobjStation;
    ENDPROC

    FUNC tooldata LocalizeToolTip()
        RETURN tToolTip;
    ENDFUNC

ENDMODULE
