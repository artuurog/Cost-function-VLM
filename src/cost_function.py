"""
Compute cost function terms and total cost
"""
import numpy as np

VIDEO_PATH = "C:/Users/user/Desktop/PoliMi/DOTTORATO/hand object interaction/video/my_demos/red_block1.mp4"


if __name__=="__main__":
    
    track_combined(VIDEO_PATH)
    
    distance_cost()
    
    hand_velocity_cost()
    
    hand_direction_cost()
    
    obj_velocity_cost()
    
    gesture_cost()    # hand compactness + hand enclosure costs
    
    coupling_term()
    
    # compute total cost
    J = 