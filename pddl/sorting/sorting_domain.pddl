(define (domain sorting-task)

  ;; -----------------------------------------------------------------------
  ;; Requirements
  ;; -----------------------------------------------------------------------
  (:requirements
    :typing          ; allow typed objects
    :negative-preconditions  ; allow (not ...) in preconditions
  )

  ;; -----------------------------------------------------------------------
  ;; Types
  ;; -----------------------------------------------------------------------
  ;; - robot     : the manipulator (gofa_robot)
  ;; - item      : any graspable object to be sorted
  ;; - container : a target bin / box that receives items
  ;; - location  : abstract spatial position (table, inside a container, etc.)
  ;; -----------------------------------------------------------------------
  (:types
    robot
    item
    container
    location
  )

  ;; -----------------------------------------------------------------------
  ;; Predicates
  ;; -----------------------------------------------------------------------
  (:predicates

    ;; --- Robot state ---
    ;; The robot arm is currently at a given location
    (robot-at ?r - robot ?loc - location)

    ;; The robot gripper is empty (not holding any item)
    (gripper-empty ?r - robot)

    ;; The robot is currently holding a specific item
    (holding ?r - robot ?obj - item)

    ;; --- Item state ---
    ;; An item is resting on the table (not yet sorted)
    (on-table ?obj - item)

    ;; An item has been placed inside a specific container
    (in-container ?obj - item ?c - container)

    ;; The item is currently at a given location (used for grasp reachability)
    (item-at ?obj - item ?loc - location)

    ;; --- Container state ---
    ;; A container is located at a specific location
    (container-at ?c - container ?loc - location)

    ;; --- Assignment ---
    ;; Encodes which container is the correct target for each item
    ;; (set in the problem file to represent the sorting rule)
    (assigned-to ?obj - item ?c - container)
  )

  ;; -----------------------------------------------------------------------
  ;; Action: grasp
  ;; -----------------------------------------------------------------------
  ;; The robot moves its end-effector to the item location and closes
  ;; the gripper around the item.
  ;;
  ;; Parameters : robot r, item obj, location loc
  ;; Pre        : robot is at loc, item is at loc, gripper is empty,
  ;;              item is still on the table
  ;; Effect     : robot is holding obj, gripper is no longer empty,
  ;;              item is no longer on the table
  ;; -----------------------------------------------------------------------
  (:action grasp
    :parameters (?r - robot ?obj - item ?loc - location)
    :precondition
      (and
        (robot-at    ?r  ?loc)   ; robot must be co-located with the item
        (item-at     ?obj ?loc)  ; item must be reachable at this location
        (gripper-empty ?r)       ; gripper must be free before grasping
        (on-table    ?obj)       ; item must still be unsorted on the table
      )
    :effect
      (and
        (holding       ?r  ?obj)   ; robot now holds the item
        (not (gripper-empty ?r))   ; gripper is occupied
        (not (on-table ?obj))      ; item leaves the table
        (not (item-at  ?obj ?loc)) ; item is no longer at its original location
      )
  )

  ;; -----------------------------------------------------------------------
  ;; Action: move
  ;; -----------------------------------------------------------------------
  ;; The robot transports the grasped item from the current location to
  ;; the location of the target container.
  ;;
  ;; Parameters : robot r, item obj, source location from,
  ;;              target container c, target location to
  ;; Pre        : robot is at 'from', robot is holding obj,
  ;;              container c is at 'to', obj is assigned to c
  ;; Effect     : robot is now at 'to', no longer at 'from'
  ;; -----------------------------------------------------------------------
  (:action move
    :parameters (?r - robot ?obj - item ?from - location
                 ?c - container ?to - location)
    :precondition
      (and
        (robot-at      ?r  ?from)  ; robot starts at source location
        (holding       ?r  ?obj)   ; robot must be carrying the item
        (container-at  ?c  ?to)    ; destination container is at target location
        (assigned-to   ?obj ?c)    ; item must be destined for this container
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
  ;; The robot opens its gripper and deposits the item into the container
  ;; it is currently positioned above.
  ;;
  ;; Parameters : robot r, item obj, container c, location loc
  ;; Pre        : robot is at loc, container c is at loc,
  ;;              robot is holding obj, obj is assigned to c
  ;; Effect     : item is inside the container, gripper is empty,
  ;;              robot is no longer holding the item
  ;; -----------------------------------------------------------------------
  (:action release
    :parameters (?r - robot ?obj - item ?c - container ?loc - location)
    :precondition
      (and
        (robot-at     ?r   ?loc)   ; robot must be above / at the container
        (container-at ?c   ?loc)   ; container must be at the same location
        (holding      ?r   ?obj)   ; robot must be holding the item to release
        (assigned-to  ?obj ?c)     ; item must belong to this container
      )
    :effect
      (and
        (in-container  ?obj ?c)    ; item is now inside the container
        (gripper-empty ?r)         ; gripper is free again
        (not (holding  ?r  ?obj))  ; robot no longer holds the item
      )
  )

)
