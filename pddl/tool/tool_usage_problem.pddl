; ============================================================
; PROBLEM: place_hammer_on_target
; Robot: gofa_robot places the tool on target_loc
; ============================================================

(define (problem place_hammer_on_target)
  (:domain tool_usage)

  ; ----------------------------------------------------------
  ; OBJECTS
  ; ----------------------------------------------------------
  (:objects
    gofa_robot             - robot

    ; tools available in the scene
    hammer_tool            - hammer
    screwdriver_tool       - screwdriver

    ; physical parts of the hammer
    hammer_handle          - grasp_point
    hammer_tip             - tip          ; the flat striking face

    ; physical parts of the screwdriver
    screwdriver_handle     - grasp_point
    screwdriver_tip        - tip          ; the blade/bit end

    ; locations in the workspace
    home_loc               - location     ; robot home position
    tool_rack_loc          - location     ; where tools rest initially
    above_target_loc       - location     ; waypoint above the target
    target_loc             - target       ; final contact point

    ; orientation descriptors
    vertical_orient        - orientation  ; tool axis pointing straight down
    hammer_strike_orient   - orientation  ; hammer flat-face pointing down
  )

  ; ----------------------------------------------------------
  ; INITIAL STATE
  ; ----------------------------------------------------------
  (:init
    ; robot starts at home, gripper empty
    (robot_at          gofa_robot    home_loc)
    (gripper_empty     gofa_robot)

    ; tools are on the rack
    (tool_at           hammer_tool       tool_rack_loc)
    (tool_at           screwdriver_tool  tool_rack_loc)

    ; structural tool properties
    (tool_has_handle   hammer_tool       hammer_handle)
    (tool_has_tip      hammer_tool       hammer_tip)
    (tool_is_hammer    hammer_tool)

    (tool_has_handle   screwdriver_tool  screwdriver_handle)
    (tool_has_tip      screwdriver_tool  screwdriver_tip)
    (tool_is_screwdriver screwdriver_tool)

    ; spatial layout
    (above             above_target_loc  target_loc)

    ; orientation type flags
    (is_vertical_orient       vertical_orient)
    (is_hammer_strike_orient  hammer_strike_orient)
  )

  ; ----------------------------------------------------------
  ; GOAL
  ; ----------------------------------------------------------
  ; The flat face of the hammer tip must be in contact with the target,
  ; achieved by executing the full action sequence with the hammer.
  (:goal
    (and
      (task_complete hammer_tool target_loc)
      (at_target     hammer_tip  target_loc)
    )
  )
)
