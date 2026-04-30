(define (problem bowl-stacking-3)

  (:domain bowl-stacking)

  ;; -----------------------------------------------------------------------
  ;; Objects
  ;; -----------------------------------------------------------------------
  (:objects
    gofa_robot                                    - robot

    large_bowl medium_bowl small_bowl target_bowl - bowl

    large_bowl_loc medium_bowl_loc small_bowl_loc
    target_loc                                    - location
  )

  ;; -----------------------------------------------------------------------
  ;; Initial State
  ;; -----------------------------------------------------------------------
  (:init

    ;; --- Robot initial configuration ---
    ;; Robot starts at the table; we pick any bowl location as the
    ;; starting position (here: large_bowl_loc).
    (robot-at     gofa_robot  large_bowl_loc)
    (gripper-empty gofa_robot)

    ;; --- Bowls on the table ---
    (on-table    large_bowl)   (bowl-at large_bowl   large_bowl_loc)
    (on-table    medium_bowl)  (bowl-at medium_bowl  medium_bowl_loc)
    (on-table    small_bowl)   (bowl-at small_bowl   small_bowl_loc)

    ;; --- Target bowl is fixed at target_loc (never moved) ---
    (is-target   target_bowl)
    (bowl-at     target_bowl  target_loc)
    (target-at   target_loc)

    ;; --- Size ordering (static facts, never modified) ---
    ;; large_bowl is larger than both medium_bowl and small_bowl
    (larger-than large_bowl  medium_bowl)
    (larger-than large_bowl  small_bowl)
    ;; medium_bowl is larger than small_bowl
    (larger-than medium_bowl small_bowl)
    ;; Note: (larger-than X X) is never declared (irreflexive by construction)
  )

  ;; -----------------------------------------------------------------------
  ;; Goal
  ;; -----------------------------------------------------------------------
  ;; All three bowls must be stacked inside target_bowl.
  ;; The ordering constraint in the 'release' precondition guarantees
  ;; the planner will produce the sequence:
  ;;   grasp(large) → move → release(large)
  ;;   grasp(medium) → move → release(medium)
  ;;   grasp(small) → move → release(small)
  ;; -----------------------------------------------------------------------
  (:goal
    (and
      (stacked-in-target large_bowl)
      (stacked-in-target medium_bowl)
      (stacked-in-target small_bowl)
    )
  )

)
