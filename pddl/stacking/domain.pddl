; PDDL Domain – Block Stacking Pyramid
; Robot: gofa_robot
; Actions: grasp, move, release

(define (domain block-stacking)

  (:requirements :strips :typing :equality)

  ; ----- Types -----
  (:types
    robot    ; the manipulator arm
    block    ; rectangular wooden block to be stacked
    location ; abstract 3-D position (table slot or on-block position)
  )

  ; ----- Predicates -----
  (:predicates
    ; robot arm is free (gripper empty)
    (arm-free ?r - robot)

    ; robot arm is holding a specific block
    (holding ?r - robot ?b - block)

    ; block ?b is resting on location ?l (table slot or top of another block)
    (on ?b - block ?l - location)

    ; location ?l is currently unoccupied (no block placed there yet)
    (location-free ?l - location)

    ; block ?b is clear on top (nothing placed on it)
    (clear ?b - block)

    ; block ?b has been placed at its goal location (task-level flag)
    (placed ?b - block)

    ; robot end-effector is positioned above location ?l
    (above ?r - robot ?l - location)
  )

  ; ----- Actions -----

  ; GRASP: the robot grasps a block that is clear and resting on some location.
  ; Preconditions: arm must be free, block must be clear and placed somewhere.
  ; Effects: arm holds the block, block is no longer on its previous location,
  ;          that location becomes free, block loses its 'clear' status from
  ;          the perspective of the location below.
  (:action grasp
    :parameters (?r - robot ?b - block ?l - location)
    :precondition (and
      (arm-free ?r)           ; gripper is empty
      (clear ?b)              ; nothing is stacked on top of ?b
      (on ?b ?l)              ; ?b is currently at location ?l
    )
    :effect (and
      (not (arm-free ?r))     ; gripper is now occupied
      (holding ?r ?b)         ; robot holds the block
      (not (on ?b ?l))        ; block is no longer at ?l
      (location-free ?l)      ; ?l becomes available
    )
  )

  ; MOVE: the robot moves a held block to be above a target location.
  ; This action represents the Cartesian trajectory of the end-effector.
  ; Preconditions: robot must be holding the block, target location must be free.
  ; Effects: end-effector is positioned above the target location.
  (:action move
    :parameters (?r - robot ?b - block ?l_target - location)
    :precondition (and
      (holding ?r ?b)           ; robot is carrying the block
      (location-free ?l_target) ; target slot is unoccupied
    )
    :effect (and
      (above ?r ?l_target)      ; end-effector is now above the target
    )
  )

  ; RELEASE: the robot places the held block at the target location.
  ; Preconditions: robot must be holding the block and positioned above the target.
  ; Effects: block is placed at the target, arm becomes free, location is occupied,
  ;          the block is now clear (nothing on top of it yet), and is marked as placed.
  (:action release
    :parameters (?r - robot ?b - block ?l_target - location)
    :precondition (and
      (holding ?r ?b)           ; robot still holds the block
      (above ?r ?l_target)      ; end-effector is above the target location
      (location-free ?l_target) ; location is still free (no race condition)
    )
    :effect (and
      (arm-free ?r)             ; gripper is released
      (not (holding ?r ?b))     ; robot no longer holds the block
      (on ?b ?l_target)         ; block is now at the target location
      (not (location-free ?l_target)) ; target is now occupied
      (not (above ?r ?l_target))      ; end-effector is no longer locked above
      (clear ?b)                ; the placed block has a clear top surface
      (placed ?b)               ; task-level goal flag
    )
  )

)
