; PDDL Plan – Sequential action sequence for gofa_robot
;
; Execution order respects the bottom-up stacking constraint:
;   Step 1-9  : place layer-1 blocks (block_1, block_2, block_3)
;   Step 10-18: place layer-2 blocks (block_4, block_5)
;   Step 19-21: place layer-3 apex  (block_6)
;
; Each block manipulation = grasp + move + release  (3 actions per block)

; ===== LAYER 1 – base row (lowest Z) =====

; --- block_1 -> loc_L1_P1 ---
(grasp   gofa_robot block_1 table_slot_1)  ; pick block_1 from table
(move    gofa_robot block_1 loc_L1_P1)    ; carry to layer-1 position 1
(release gofa_robot block_1 loc_L1_P1)    ; place block_1 at L1-P1

; --- block_2 -> loc_L1_P2 ---
(grasp   gofa_robot block_2 table_slot_2)  ; pick block_2 from table
(move    gofa_robot block_2 loc_L1_P2)    ; carry to layer-1 position 2
(release gofa_robot block_2 loc_L1_P2)    ; place block_2 at L1-P2

; --- block_3 -> loc_L1_P3 ---
(grasp   gofa_robot block_3 table_slot_3)  ; pick block_3 from table
(move    gofa_robot block_3 loc_L1_P3)    ; carry to layer-1 position 3
(release gofa_robot block_3 loc_L1_P3)    ; place block_3 at L1-P3

; ===== LAYER 2 – middle row (mid Z) =====
; Precondition: layer-1 blocks must already be placed to serve as supports.

; --- block_4 -> loc_L2_P1 ---
(grasp   gofa_robot block_4 table_slot_4)  ; pick block_4 from table
(move    gofa_robot block_4 loc_L2_P1)    ; carry to layer-2 position 1
(release gofa_robot block_4 loc_L2_P1)    ; place block_4 at L2-P1

; --- block_5 -> loc_L2_P2 ---
(grasp   gofa_robot block_5 table_slot_5)  ; pick block_5 from table
(move    gofa_robot block_5 loc_L2_P2)    ; carry to layer-2 position 2
(release gofa_robot block_5 loc_L2_P2)    ; place block_5 at L2-P2

; ===== LAYER 3 – apex (highest Z) =====
; Precondition: layer-2 blocks must already be placed to serve as support.

; --- block_6 -> loc_L3_P1 ---
(grasp   gofa_robot block_6 table_slot_6)  ; pick block_6 from table
(move    gofa_robot block_6 loc_L3_P1)    ; carry to apex position
(release gofa_robot block_6 loc_L3_P1)    ; place block_6 at L3-P1 (apex)

; ===== END OF PLAN =====
; Final state: pyramid fully assembled
;           [ block_6 ]
;        [ block_4 ][ block_5 ]
;   [ block_1 ][ block_2 ][ block_3 ]
