; ==========================================================================
; PDDL Domain — Insertion Task
;
; ==========================================================================

(define (domain insertion_task)

  (:requirements
    :strips                    ; basic add/delete effects
    :typing                    ; typed objects
    :negative-preconditions    ; (not ...) allowed in preconditions
  )

  ; ------------------------------------------------------------------------
  ; Types
  ; ------------------------------------------------------------------------
  (:types
    robot    ; the GoFa manipulator arm
    bead     ; wooden perforated bead
    stick    ; vertical stick on the abacus
    location ; workspace region (table or abacus)
  )

  ; ------------------------------------------------------------------------
  ; Predicates
  ; ------------------------------------------------------------------------
  (:predicates

    ; robot state
    (robot-free     ?r - robot)
    ; True when the gripper is open (not holding any bead).

    (robot-holding  ?r - robot  ?b - bead)
    ; True when the gripper is closed around bead ?b.

    (robot-at       ?r - robot  ?loc - location)
    ; True when the end-effector is positioned above location ?loc.

    ; bead state
    (bead-on-table  ?b - bead)
    ; True when bead ?b is resting on the table, not yet picked up.

    (bead-inserted  ?b - bead   ?s - stick)
    ; True after bead ?b has been successfully inserted onto stick ?s.

    ; stick state
    (stick-available ?s - stick)
    ; True when stick ?s can still receive a bead.

    ; location descriptors
    (is-table  ?loc - location)
    (is-abacus ?loc - location)
  )

  ; ------------------------------------------------------------------------
  ; Actions
  ; ------------------------------------------------------------------------

  ; --- pick_bead -----------------------------------------------------------
  ; The robot grasps a bead lying on the table.
  ; Precondition: gripper is free, bead is on the table, robot is at table.
  (:action pick_bead
    :parameters (?r - robot  ?b - bead  ?table - location)
    :precondition (and
      (robot-free    ?r)
      (bead-on-table ?b)
      (robot-at      ?r ?table)
      (is-table      ?table)
    )
    :effect (and
      (not (robot-free    ?r))
      (robot-holding      ?r ?b)
      (not (bead-on-table ?b))
    )
  )

  ; --- move_to_stick -------------------------------------------------------
  ; The robot carries the held bead from the table region to the abacus,
  ; aligning the end-effector above the chosen stick.
  (:action move_to_stick
    :parameters (?r - robot  ?b - bead
                 ?from - location  ?abacus - location  ?s - stick)
    :precondition (and
      (robot-holding   ?r ?b)
      (robot-at        ?r ?from)
      (is-abacus       ?abacus)
      (stick-available ?s)
      (not (robot-at   ?r ?abacus))
    )
    :effect (and
      (not (robot-at ?r ?from))
      (robot-at      ?r ?abacus)
    )
  )

  ; --- insert_bead ---------------------------------------------------------
  ; The robot inserts the held bead onto the target stick.
  ; This step involves contact forces at the end-effector (noted as a
  ; challenge in the paper — see Insertion Task description and Table III).
  (:action insert_bead
    :parameters (?r - robot  ?b - bead  ?s - stick  ?abacus - location)
    :precondition (and
      (robot-holding   ?r ?b)
      (robot-at        ?r ?abacus)
      (is-abacus       ?abacus)
      (stick-available ?s)
    )
    :effect (and
      (not (robot-holding ?r ?b))
      (robot-free         ?r)
      (bead-inserted      ?b ?s)
      ; stick-available stays true: multiple beads per stick are allowed.
      ; To model single-bead sticks, uncomment the line below:
      ; (not (stick-available ?s))
    )
  )

  ; --- move_to_table -------------------------------------------------------
  ; The robot returns its empty gripper to the table to pick the next bead.
  (:action move_to_table
    :parameters (?r - robot  ?abacus - location  ?table - location)
    :precondition (and
      (robot-free  ?r)
      (robot-at    ?r ?abacus)
      (is-abacus   ?abacus)
      (is-table    ?table)
    )
    :effect (and
      (not (robot-at ?r ?abacus))
      (robot-at      ?r ?table)
    )
  )

)
