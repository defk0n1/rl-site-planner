import gym
from gym import spaces
import numpy as np
import torch
from lte_utils import compute_rsrp_map, compute_sinr_map, build_candidate_graph

class LTEPlannerEnv(gym.Env):
    def __init__(self, clutter_map, candidate_positions, clutter_lookup, k=5, resolution=1.0):
        super().__init__()
        self.clutter_map = clutter_map
        self.candidate_positions = candidate_positions
        self.clutter_lookup = clutter_lookup
        self.k = k
        self.resolution = resolution
        self.num_nodes = len(candidate_positions)
        self.coverage_target = 0.95  # Target coverage percentage


    # Action space
        self.action_space = spaces.Dict({
            "placement": spaces.MultiBinary(self.num_nodes),
            "height": spaces.Box(low=0, high=1, shape=(self.num_nodes,)),  # Scaled to [0,1]
            "tilt": spaces.Box(low=0, high=1, shape=(self.num_nodes,)),    # Scaled to [0,1]
            "azimuth": spaces.Box(low=0, high=1, shape=(self.num_nodes,))  # Scaled to [0,1]
        })
        
        # Observation space remains graph-based
        self.observation_space = spaces.Dict({
            "x": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_nodes, 5)),
            "edge_index": spaces.Tuple((
                spaces.Box(low=0, high=self.num_nodes-1, dtype=np.int64),
                spaces.Box(low=0, high=self.num_nodes-1, dtype=np.int64)
            ))
        })

    def reset(self):
        self.active_nodes = np.zeros((self.num_nodes, 4))
        return self._get_state()

    def step(self, action):
        # Process action
        # print(action)
        placements = action["placement"]
        heights = np.where(action["height"] > 0.5, 45.0, 28.0)  # Discrete heights
        tilts = action["tilt"] * 12.0                           # [0,12] degrees
        azimuths = action["azimuth"] * 360.0                     # [0,360] degrees
        processed_action = np.column_stack([placements, heights, tilts, azimuths])
        
        # Compute metrics
        rsrp_map = compute_rsrp_map(self.clutter_map,self.candidate_positions, processed_action,clutter_lookup=self.clutter_lookup)
        sinr_map = compute_sinr_map(rsrp_map, self.candidate_positions , processed_action)

        # Calculate coverage
        coverage = (rsrp_map > -95).mean()    

        # Calculate reward
        reward = self._compute_reward(rsrp_map, sinr_map, processed_action)
        
        # Termination
        done = (
            coverage >= self.coverage_target or
            self.steps >= 20  # Max steps per episode
        )

        info = {
            "rsrp_map": rsrp_map,
            "sinr_map": sinr_map,
            "active_tx": processed_action[placements > 0.5]
        }
        
        return self._get_state(), reward, done, info

    def _get_state(self):
        return build_candidate_graph(self.clutter_map, self.candidate_positions, k=self.k)

    def _compute_reward(self, rsrp_map, sinr_map, action):
        coverage = (rsrp_map > -95).mean()
        # print(coverage)
        good_sinr = (sinr_map > 5).mean()
        total_power = action[:, 1].sum()
        num_tx = (action[:, 0] > 0.5).sum()
        return 2.0 * coverage + 1.0 * good_sinr - 0.1 * total_power - 0.2 * num_tx
