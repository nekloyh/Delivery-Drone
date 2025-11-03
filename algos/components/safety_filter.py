
import numpy as np

def project_to_safe(state_xy, action_xy, dist_to_obstacle, margin=0.3, hard_stop=True):
    """Heuristic shield:
      - if too close to obstacle border (< margin), scale down or zero the action
      - pushes the action away from obstacles (simple radial strategy)
    """
    if dist_to_obstacle is None:
        return action_xy

    if dist_to_obstacle >= margin:
        return action_xy

    scale = max(0.0, (dist_to_obstacle / margin))
    safe = action_xy * scale
    if hard_stop and dist_to_obstacle < 0.2 * margin:
        safe = np.zeros_like(action_xy)
    return safe
