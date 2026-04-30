(define (domain bowl-stacking)

  ;; -----------------------------------------------------------------------
  ;; Requirements
  ;; -----------------------------------------------------------------------
  (:requirements
    :typing                   ; typed objects
    :negative-preconditions   ; (not ...) allowed in preconditions
    :universal-preconditions  ; (forall ...) allowed in preconditions
    :conditional-effects      ; (when ...) allowed in effects
  )

  ;; -----------------------------------------------------------------------
  ;; Types
  ;; -----------------------------------------------------------------------
  ;; - robot       : the manipulator (gofa_robot)
  ;; - bowl        : any bowl entity, including the fixed target bowl
  ;; - location    : abstract spatial position
  ;; -----------------------------------------------------------------------
  (:types
    robot
    bowl
    location
  )

  ;; -----------------------------------------------------------------------
  ;; Predicates
  ;; -----------------------------------------------------------------------
  (:predicates

    ;; --- Robot state ---
    ;; Robot end-effector is at a given location
    (robot-at ?r - robot ?loc - location)

    ;; Gripper is empty (not holding anything)
    (gripper-empty ?r - robot)

    ;; Robot is currently holding a bowl
    (holding ?r - robot ?b - bowl)

    ;; --- Bowl state ---
    ;; Bowl is resting on the table (not yet stacked)
    (on-table ?b - bowl)

    ;; Bowl has been successfully placed inside the target bowl
    (stacked-in-target ?b - bowl)

    ;; Bowl is currently at a given location
    (bowl-at ?b - bowl ?loc - location)

    ;; --- Size / ordering relation ---
    ;; (larger-than ?b1 ?b2) is TRUE if bowl b1 is larger than bowl b2.
    ;; Declared as static facts in (:init); never modified by actions.
    ;; Used to enforce bottom-up stacking order:
    ;;   before placing b2, every b1 with (larger-than b1 b2) must already
    ;;   satisfy (stacked-in-target b1).
    (larger-than ?b1 - bowl ?b2 - bowl)

    ;; --- Target bowl ---
    ;; Marks the fixed bowl that acts as the stacking receptacle.
    ;; The target bowl itself is never grasped or moved.
    (is-target ?b - bowl)

    ;; Location of the target bowl (centre, used as release point)
    (target-at ?loc - location)
  )

  ;; -----------------------------------------------------------------------
  ;; Action: grasp
  ;; -----------------------------------------------------------------------
  ;; Robot closes gripper around a bowl that is still on the table.
  ;;
  ;; Parameters : robot r, bowl b, location loc
  ;; Pre        : robot at loc, bowl at loc, gripper empty,
  ;;              bowl is on the table (not yet stacked),
  ;;              bowl is NOT the fixed target receptacle
  ;; Effect     : robot holds b, gripper occupied,
  ;;              bowl leaves the table and its location
  ;; -----------------------------------------------------------------------
  (:action grasp
    :parameters (?r - robot ?b - bowl ?loc - location)
    :precondition
      (and
        (robot-at     ?r  ?loc)   ; robot co-located with the bowl
        (bowl-at      ?b  ?loc)   ; bowl is reachable at this location
        (gripper-empty ?r)        ; gripper must be free
        (on-table     ?b)         ; bowl must still be unsorted on the table
        (not (is-target ?b))      ; cannot grasp the fixed target bowl
      )
    :effect
      (and
        (holding          ?r  ?b)   ; robot now holds the bowl
        (not (gripper-empty ?r))    ; gripper is occupied
        (not (on-table    ?b))      ; bowl leaves the table
        (not (bowl-at     ?b ?loc)) ; bowl is no longer at its table position
      )
  )

  ;; -----------------------------------------------------------------------
  ;; Action: move
  ;; -----------------------------------------------------------------------
  ;; Robot carries the grasped bowl from its current location to the
  ;; centre of the target bowl.
  ;;
  ;; Parameters : robot r, bowl b (held), source loc from, target loc to
  ;; Pre        : robot at 'from', holding b, target location is 'to'
  ;; Effect     : robot is now at 'to'
  ;; -----------------------------------------------------------------------
  (:action move
    :parameters (?r - robot ?b - bowl ?from - location ?to - location)
    :precondition
      (and
        (robot-at  ?r  ?from)   ; robot starts at source location
        (holding   ?r  ?b)      ; robot must be carrying the bowl
        (target-at ?to)         ; destination must be the target centre
      )
    :effect
      (and
        (robot-at      ?r  ?to)    ; robot arrives at target location
        (not (robot-at ?r  ?from)) ; robot leaves source location
      )
  )

  ;; -----------------------------------------------------------------------
  ;; Action: release
  ;; -----------------------------------------------------------------------
  ;; Robot opens its gripper and deposits the bowl into the target bowl.
  ;; The stacking ORDER constraint is enforced here:
  ;;   every bowl that is larger than the current one must already be
  ;;   stacked inside the target bowl before this bowl can be released.
  ;;
  ;; Parameters : robot r, bowl b, target bowl tb, location loc
  ;; Pre        : robot at loc (= target centre), holding b,
  ;;              tb is the target bowl and is at loc,
  ;;              for ALL bowls b2: if (larger-than b2 b) then
  ;;                (stacked-in-target b2)  [bottom-up order]
  ;; Effect     : b is stacked in target, gripper free
  ;; -----------------------------------------------------------------------
  (:action release
    :parameters (?r - robot ?b - bowl ?tb - bowl ?loc - location)
    :precondition
      (and
        (robot-at    ?r   ?loc)   ; robot is above the target bowl
        (holding     ?r   ?b)     ; robot is holding bowl b
        (is-target   ?tb)         ; tb is the fixed receptacle
        (bowl-at     ?tb  ?loc)   ; target bowl is at this location
        (target-at   ?loc)        ; confirm this is the designated target loc

        ;; Ordering constraint: every bowl larger than b must already be
        ;; inside the target before b can be placed (large → medium → small)
        (forall (?b2 - bowl)
          (imply (larger-than ?b2 ?b)
                 (stacked-in-target ?b2)))
      )
    :effect
      (and
        (stacked-in-target ?b)      ; bowl b is now inside the target bowl
        (gripper-empty     ?r)      ; gripper is free again
        (not (holding      ?r  ?b)) ; robot no longer holds b
      )
  )

)
