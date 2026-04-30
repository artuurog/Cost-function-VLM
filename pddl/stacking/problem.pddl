; PDDL Problem – 6-Block Pyramid Stacking
;
; Pyramid structure (Z increases with layer number):
;
;           [ block_6 ]              <- Layer 3 (highest Z)
;        [ block_4 ][ block_5 ]     <- Layer 2
;   [ block_1 ][ block_2 ][ block_3 ] <- Layer 1 (lowest Z, on table)
;
; Initial state: all blocks are on the table in arbitrary table slots.
; Goal state:    each block is placed at its designated pyramid location.

(define (problem pyramid-6-blocks)
  (:domain block-stacking)

  ; ----- Objects -----
  (:objects
    gofa_robot - robot

    ; Six rectangular blocks
    block_1 block_2 block_3   - block   ; layer 1 (bottom row)
    block_4 block_5           - block   ; layer 2 (middle row)
    block_6                   - block   ; layer 3 (apex)

    ; Table slots where blocks rest initially (arbitrary positions)
    table_slot_1 table_slot_2 table_slot_3
    table_slot_4 table_slot_5 table_slot_6 - location

    ; Target pyramid locations – naming convention: loc_LX_PY
    ;   L = layer number (1=lowest, 3=highest Z)
    ;   P = position index within the layer
    loc_L1_P1 loc_L1_P2 loc_L1_P3 - location  ; layer 1 targets
    loc_L2_P1 loc_L2_P2           - location  ; layer 2 targets
    loc_L3_P1                     - location  ; layer 3 target (apex)
  )

  ; ----- Initial State -----
  (:init
    ; Robot arm starts free (gripper empty)
    (arm-free gofa_robot)

    ; All blocks are on the table, each in a different slot
    (on block_1 table_slot_1)
    (on block_2 table_slot_2)
    (on block_3 table_slot_3)
    (on block_4 table_slot_4)
    (on block_5 table_slot_5)
    (on block_6 table_slot_6)

    ; All blocks have a clear top surface initially
    (clear block_1)
    (clear block_2)
    (clear block_3)
    (clear block_4)
    (clear block_5)
    (clear block_6)

    ; All pyramid target locations are free
    (location-free loc_L1_P1)
    (location-free loc_L1_P2)
    (location-free loc_L1_P3)
    (location-free loc_L2_P1)
    (location-free loc_L2_P2)
    (location-free loc_L3_P1)

    ; Table slots are occupied (not free) since blocks are already there
    ; NOTE: table slots are NOT listed as location-free because blocks occupy them.
    ; They become free after grasp actions lift the blocks.
  )

  ; ----- Goal -----
  ; All six blocks must be placed at their respective pyramid target locations.
  ; Layer 1 must be completed before layer 2, and layer 2 before layer 3,
  ; because upper-layer blocks rest on lower-layer blocks (support constraint).
  (:goal
    (and
      ; Layer 1 – three blocks forming the base (lowest Z)
      (on block_1 loc_L1_P1)
      (on block_2 loc_L1_P2)
      (on block_3 loc_L1_P3)

      ; Layer 2 – two blocks resting on layer 1 (mid Z)
      (on block_4 loc_L2_P1)
      (on block_5 loc_L2_P2)

      ; Layer 3 – apex block (highest Z)
      (on block_6 loc_L3_P1)
    )
  )

)
