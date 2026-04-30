; ============================================================
; PLAN: place_hammer_on_target
; Ground action sequence for gofa_robot to pick the hammer
; by its handle and place the flat face on target_loc.
; ============================================================

; Step 1 – Move from home to the tool rack
(move gofa_robot home_loc tool_rack_loc)

; Step 2 – Grasp the hammer at its handle (NOT the head)
(grasp gofa_robot hammer_tool hammer_handle tool_rack_loc)

; Step 3 – Verify the grip is secure on the handle
(verify_grasp gofa_robot hammer_tool hammer_handle)

; Step 4 – Orient the end-effector so the hammer flat-face points down
(orient_hammer_strike gofa_robot hammer_tool hammer_strike_orient tool_rack_loc)

; Step 5 – Move to the waypoint directly above the target
(move gofa_robot tool_rack_loc above_target_loc)

; Step 6 – Execute the approach descent to the pre-contact pose
(approach_target gofa_robot hammer_tool target_loc above_target_loc)

; Step 7 – Lower the hammer tip (flat face) onto the target
(place gofa_robot hammer_tool hammer_tip target_loc above_target_loc)

; ---- TASK COMPLETE ----
; Postcondition:
;   (at_target    hammer_tip  target_loc)  -> TRUE
;   (task_complete hammer_tool target_loc) -> TRUE
;   (gripper_empty gofa_robot)             -> TRUE
