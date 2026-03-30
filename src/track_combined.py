"""
Combine hand tracking and object tracking.


"""

VIDEO_PATH = "C:/Users/user/Desktop/PoliMi/DOTTORATO/hand object interaction/video/my_demos/red_block1.mp4"


def get_hand_results(VIDEO_PATH: str):
    """
    Process video with hand tracking analyzer
    """
    
def get_objects_results(VIDEO_PATH: str):
    """
    Process video with object tracker
    """
    
def calc_rel_dist(hand_traj, obj_traj):
    """
    Compute relative hand-object distance for every time step

    """
def calc_v_rel(rel_dist, timestep: float):
    """
    Compute relative hand-object velocity for every time step
    """

if __name__=="__main__":
    
    get_hand_results()
    
    get_objects_results()
    
    # computations
    calc_v_rel()
    
    calc_rel_dist()