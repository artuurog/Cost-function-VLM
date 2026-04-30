(define (problem sorting-5-objects)

  ;; Reference the domain defined above
  (:domain sorting-task)

  ;; -----------------------------------------------------------------------
  ;; Objects
  ;; -----------------------------------------------------------------------
  (:objects
    gofa_robot                          - robot

    obj1 obj2 obj3 obj4 obj5            - item

    container_A container_B             - container

    table_loc container_A_loc container_B_loc - location
  )

  ;; -----------------------------------------------------------------------
  ;; Initial State
  ;; -----------------------------------------------------------------------
  (:init

    ;; --- Robot initial configuration ---
    (robot-at    gofa_robot table_loc)  ; arm starts at the table
    (gripper-empty gofa_robot)          ; gripper starts open / empty

    ;; --- Items on the table ---
    (on-table    obj1)  (item-at obj1 table_loc)
    (on-table    obj2)  (item-at obj2 table_loc)
    (on-table    obj3)  (item-at obj3 table_loc)
    (on-table    obj4)  (item-at obj4 table_loc)
    (on-table    obj5)  (item-at obj5 table_loc)

    ;; --- Container positions ---
    (container-at container_A container_A_loc)
    (container-at container_B container_B_loc)

    ;; --- Sorting assignments (encodes the sorting rule) ---
    ;; obj1, obj2, obj3 belong to container_A (e.g. red category)
    (assigned-to obj1 container_A)
    (assigned-to obj2 container_A)
    (assigned-to obj3 container_A)
    ;; obj4, obj5 belong to container_B (e.g. blue category)
    (assigned-to obj4 container_B)
    (assigned-to obj5 container_B)
  )

  ;; -----------------------------------------------------------------------
  ;; Goal
  ;; -----------------------------------------------------------------------
  ;; All items must end up inside their assigned containers.
  ;; The order of operations is unconstrained at the PDDL level;
  ;; the planner will find an optimal sequence of grasp/move/release actions.
  ;; -----------------------------------------------------------------------
  (:goal
    (and
      (in-container obj1 container_A)
      (in-container obj2 container_A)
      (in-container obj3 container_A)
      (in-container obj4 container_B)
      (in-container obj5 container_B)
    )
  )

)
