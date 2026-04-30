; ============================================================
; DOMAIN: tool_usage
; Robot: gofa_robot
; Tools: hammer, screwdriver
; Task:  pick the tool by its handle and place its tip on a target
; ============================================================

(define (domain tool_usage)

  (:requirements
    :typing
    :negative-preconditions
    :conditional-effects
  )

  ; ----------------------------------------------------------
  ; TYPES
  ; ----------------------------------------------------------
  (:types
    robot          ; the manipulator (gofa_robot)
    tool           ; any manipulable tool
    hammer         - tool   ; subtype: hammer
    screwdriver    - tool   ; subtype: screwdriver
    location       ; a pose / position in the workspace
    target         - location  ; the goal contact location
    grasp_point    ; the designated handle region of a tool
    tip            ; the functional end of a tool
    orientation    ; an end-effector orientation descriptor
  )

  ; ----------------------------------------------------------
  ; PREDICATES
  ; ----------------------------------------------------------
  (:predicates

    ; --- robot state ---
    (robot_at          ?r - robot    ?l - location)
    (gripper_empty     ?r - robot)
    (gripper_holding   ?r - robot    ?t - tool)
    (gripper_secure    ?r - robot    ?t - tool)   ; confirmed grip at handle

    ; --- tool properties ---
    (tool_at           ?t - tool     ?l - location)
    (tool_has_handle   ?t - tool     ?g - grasp_point)
    (tool_has_tip      ?t - tool     ?tp - tip)
    (tool_is_hammer    ?t - tool)                 ; flags tool subtype
    (tool_is_screwdriver ?t - tool)

    ; --- spatial relations ---
    (above             ?l1 - location ?l2 - location)  ; l1 is directly above l2
    (at_target         ?tp - tip      ?tgt - target)   ; tip in contact with target

    ; --- orientation constraints ---
    (end_effector_orientation ?r - robot ?o - orientation)
    (is_vertical_orient       ?o - orientation)  ; for screwdriver task
    (is_hammer_strike_orient  ?o - orientation)  ; flat-face-down for hammer

    ; --- orientation satisfied flags ---
    (screwdriver_ready ?t - tool ?r - robot)  ; vertical orient + secure grip
    (hammer_ready      ?t - tool ?r - robot)  ; strike orient + secure grip

    ; --- task progress ---
    (pre_contact_reached ?r - robot ?tgt - target)  ; robot above target
    (task_complete       ?t - tool  ?tgt - target)
  )

  ; ----------------------------------------------------------
  ; ACTIONS
  ; ----------------------------------------------------------

  ; --- 1. GRASP ---------------------------------------------------
  ; Move the gripper to the tool handle and close the gripper.
  ; The robot must be at the tool location and the gripper must be free.
  (:action grasp
    :parameters (?r - robot ?t - tool ?g - grasp_point ?l - location)
    :precondition (and
      (robot_at        ?r ?l)
      (tool_at         ?t ?l)
      (tool_has_handle ?t ?g)
      (gripper_empty   ?r)
    )
    :effect (and
      (gripper_holding ?r ?t)
      (not (gripper_empty ?r))
      (not (tool_at ?t ?l))
    )
  )

  ; --- 2. VERIFY_GRASP --------------------------------------------
  ; Confirm that the gripper has a secure grip on the tool handle.
  ; This check is mandatory before any motion with the tool.
  (:action verify_grasp
    :parameters (?r - robot ?t - tool ?g - grasp_point)
    :precondition (and
      (gripper_holding ?r ?t)
      (tool_has_handle ?t ?g)
    )
    :effect (and
      (gripper_secure ?r ?t)
    )
  )

  ; --- 3. ORIENT_VERTICAL ----------------------------------------
  ; Rotate the end-effector so the tool axis is vertical (pointing down).
  ; Required for the screwdriver before contacting the target.
  (:action orient_vertical
    :parameters (?r - robot ?t - tool ?o - orientation ?l - location)
    :precondition (and
      (tool_is_screwdriver ?t)
      (gripper_secure      ?r ?t)
      (robot_at            ?r ?l)
      (is_vertical_orient  ?o)
    )
    :effect (and
      (end_effector_orientation ?r ?o)
      (screwdriver_ready        ?t ?r)
    )
  )

  ; --- 4. ORIENT_HAMMER_STRIKE ------------------------------------
  ; Rotate the end-effector so the hammer flat-face is aligned downward,
  ; ready to contact the target surface correctly.
  (:action orient_hammer_strike
    :parameters (?r - robot ?t - tool ?o - orientation ?l - location)
    :precondition (and
      (tool_is_hammer          ?t)
      (gripper_secure          ?r ?t)
      (robot_at                ?r ?l)
      (is_hammer_strike_orient ?o)
    )
    :effect (and
      (end_effector_orientation ?r ?o)
      (hammer_ready             ?t ?r)
    )
  )

  ; --- 5. MOVE ----------------------------------------------------
  ; Move the robot (with or without a tool) from one location to another.
  ; If carrying a tool the orientation state is preserved.
  (:action move
    :parameters (?r - robot ?from - location ?to - location)
    :precondition (and
      (robot_at ?r ?from)
    )
    :effect (and
      (robot_at     ?r ?to)
      (not (robot_at ?r ?from))
    )
  )

  ; --- 6. APPROACH_TARGET -----------------------------------------
  ; Move to a safe pre-contact waypoint directly above the target.
  ; Ensures the robot approaches along the correct axis.
  (:action approach_target
    :parameters (?r - robot ?t - tool ?tgt - target ?above_tgt - location)
    :precondition (and
      (gripper_holding ?r ?t)
      (gripper_secure  ?r ?t)
      (robot_at        ?r ?above_tgt)
      (above           ?above_tgt ?tgt)
      ; at least one orientation constraint must have been satisfied
      (or (screwdriver_ready ?t ?r)
          (hammer_ready      ?t ?r))
    )
    :effect (and
      (pre_contact_reached ?r ?tgt)
    )
  )

  ; --- 7. PLACE ---------------------------------------------------
  ; Lower the tool tip onto the target.
  ; Enforces tool-specific orientation constraints:
  ;   * Screwdriver -> must be in vertical orientation
  ;   * Hammer      -> must be in hammer-strike (flat-face-down) orientation
  (:action place
    :parameters (?r - robot ?t - tool ?tp - tip ?tgt - target ?above_tgt - location)
    :precondition (and
      (gripper_holding     ?r ?t)
      (gripper_secure      ?r ?t)
      (tool_has_tip        ?t ?tp)
      (robot_at            ?r ?above_tgt)
      (above               ?above_tgt ?tgt)
      (pre_contact_reached ?r ?tgt)
      ; tool-specific orientation guard
      (or
        (and (tool_is_screwdriver ?t) (screwdriver_ready ?t ?r))
        (and (tool_is_hammer      ?t) (hammer_ready      ?t ?r))
      )
    )
    :effect (and
      (at_target    ?tp  ?tgt)
      (task_complete ?t  ?tgt)
      ; robot releases tool at target
      (gripper_empty   ?r)
      (not (gripper_holding ?r ?t))
      (not (gripper_secure  ?r ?t))
    )
  )

  ; --- 8. RELEASE -------------------------------------------------
  ; Open the gripper to release the tool (e.g. abort or regrasp).
  (:action release
    :parameters (?r - robot ?t - tool ?l - location)
    :precondition (and
      (gripper_holding ?r ?t)
      (robot_at        ?r ?l)
    )
    :effect (and
      (gripper_empty      ?r)
      (tool_at            ?t ?l)
      (not (gripper_holding ?r ?t))
      (not (gripper_secure  ?r ?t))
    )
  )

)
