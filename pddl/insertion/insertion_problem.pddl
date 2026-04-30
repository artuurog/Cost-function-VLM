; ==========================================================================
; PDDL Problem — Insertion Task
;   N beads  = 3
;   T sticks = 3
;   Robot    = gofa_robot
; ==========================================================================
(define (problem insertion_problem)
  (:domain insertion_task)

  (:objects
    gofa_robot          - robot      ; ABB GoFa manipulator
  
    bead1 bead2 bead3  - bead       ; 3 wooden bead(s) on the table
    stick1 stick2 stick3 - stick      ; 3 vertical stick(s) on the abacus
  
    table_loc             - location   ; table workspace region
    abacus_loc            - location   ; abacus workspace region
  )

  (:init
    ; robot initial state
    (robot-free  gofa_robot)
    (robot-at    gofa_robot table_loc)  ; gripper starts above the table
    
    ; all beads start on the table
    (bead-on-table bead1)
    (bead-on-table bead2)
    (bead-on-table bead3)
    
    ; all sticks are available
    (stick-available stick1)
    (stick-available stick2)
    (stick-available stick3)
    
    ; location type tags
    (is-table  table_loc)
    (is-abacus abacus_loc)
  )

  (:goal
    (and
      ; every bead must be inserted onto a stick
      (bead-inserted bead1 stick1)
      (bead-inserted bead2 stick2)
      (bead-inserted bead3 stick3)
    )
  )

)
