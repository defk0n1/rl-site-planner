import gym
from gym import spaces
import numpy as np
import torch
from lte_utils import compute_rsrp_map, compute_sinr_map,compute_sinr_map_auto, build_candidate_graph

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LTEPlannerEnv(gym.Env):
    def __init__(self, clutter_map, candidate_positions, clutter_lookup, k=5, resolution=1.0,freq=1800,tx_power_dbm=15.2):
        super().__init__()
        self.clutter_map = torch.tensor(clutter_map, dtype=torch.int, device=device)
        self.candidate_positions = candidate_positions  
        self.clutter_lookup = clutter_lookup
        self.k = k
        self.resolution = resolution
        self.num_nodes = len(candidate_positions)
        self.coverage_target = 0.95
        self.max_steps_per_episode = 20
        self.current_step = 0
        self.freq = freq
        self.tx_power_dbm = tx_power_dbm 

        self.action_space = spaces.Dict({
            "placement": spaces.MultiBinary(self.num_nodes),
            "height": spaces.Box(low=0, high=1, shape=(self.num_nodes,)),
            "tilt": spaces.Box(low=0, high=1, shape=(self.num_nodes,)),
            # "azimuth": spaces.Box(low=0, high=1, shape=(self.num_nodes,))
            "azimuth":spaces.MultiDiscrete([71] * self.num_nodes)
        })

        self.observation_space = spaces.Dict({
            "x": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_nodes, 5)),
            "edge_index": spaces.Tuple((
                spaces.Box(low=0, high=self.num_nodes - 1, dtype=np.int64),
                spaces.Box(low=0, high=self.num_nodes - 1, dtype=np.int64)
            ))
        })

    def reset(self):
        self.current_step = 0
        self.active_nodes = torch.zeros((self.num_nodes, 4), device=device)
        return self._get_state()

    def step(self, action):
        self.current_step += 1

        # Convert action to processed format
        placement = torch.tensor(action["placement"], dtype=torch.float32, device=device)
        height = torch.tensor(action["height"], device=device)
        tilt = torch.tensor(action["tilt"], device=device)
        azimuth = torch.tensor(action["azimuth"], device=device)


        processed_action = torch.stack([placement, height, tilt, azimuth], dim=1)

        # Compute metrics on GPU
        rsrp_map , rsrp_maps_per_bs , valid_mask = compute_rsrp_map(
            self.clutter_map, self.candidate_positions, processed_action,
            clutter_lookup=self.clutter_lookup, resolution=self.resolution , freq_mhz=self.freq ,tx_power_dbm=self.tx_power_dbm
        )
        sinr_map , _ = compute_sinr_map_auto(rsrp_maps_per_bs, processed_action,valid_mask,device="cuda")

        # Compute coverage
        coverage = ((rsrp_map > -95) & valid_mask).float().mean().item()
        good_sinr = ((sinr_map > 5) & valid_mask).float().mean()

        # Compute reward
        reward = self._compute_reward(coverage, good_sinr, processed_action)

        done = (
            coverage >= self.coverage_target or
            self.current_step >= self.max_steps_per_episode
        )

        info = {
            "rsrp_map": rsrp_map.detach().cpu().numpy(),
            "sinr_map": sinr_map.detach().cpu().numpy(),
            "active_tx": processed_action[placement > 0.5].detach().cpu().numpy()
        }

        return self._get_state(), reward, done, info

    def _get_state(self):
        return build_candidate_graph(self.clutter_map, self.candidate_positions, k=self.k)

    def _compute_reward(self, coverage, good_sinr, action):
        
        num_tx = (action[:, 0] > 0.5).sum()

        # print(f'Coverage :{coverage} , SINR : {good_sinr} , Number of transmitters : {num_tx} ' , )

        return (
            2.0 * coverage +
            1.0 * good_sinr -
            0.01 * num_tx.item()
        )
