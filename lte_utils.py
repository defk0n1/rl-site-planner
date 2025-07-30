import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

def build_candidate_graph(clutter_map, candidate_positions, k=5):
    h, w = clutter_map.shape
    node_features = []
    for (x, y) in candidate_positions:
        clutter = clutter_map[y, x]
        node_feat = [x / w, y / h, clutter / clutter_map.max()]
        node_features.append(node_feat)
    x = torch.tensor(node_features, dtype=torch.float , requires_grad=True)
    nbrs = NearestNeighbors(n_neighbors=k).fit(candidate_positions)
    _, indices = nbrs.kneighbors(candidate_positions)
    edge_list = []
    for i in range(len(candidate_positions)):
        for j in indices[i][1:]:
            edge_list.append([i, j])
            edge_list.append([j, i])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)
    

def cost231_hata(d, f, h_b, h_m, clutter_class):
    d = np.maximum(d, 0.01)  # Clamp to 10 meters to avoid log10(0)
    a_h_m = (1.1 * np.log10(f) - 0.7) * h_m - (1.56 * np.log10(f) - 0.8)
    c = 3 if clutter_class == "urban" else 0
    L = (46.3 + 33.9 * np.log10(f) - 13.82 * np.log10(h_b)
         - a_h_m + (44.9 - 6.55 * np.log10(h_b)) * np.log10(d) + c)
    return L

def sectorized_path_loss(d, f, h_b, h_m, clutter_class, azimuth, user_azimuth, tilt):
    """Calculate path loss with sector antenna pattern"""
    # Base path loss (COST-231 Hata)
    L_hata = cost231_hata(d, f, h_b, h_m, clutter_class)
    
    # Horizontal pattern (3GPP TR 36.814)
    azimuth_diff = np.abs((user_azimuth - azimuth + 180) % 360 - 180)
    A_h = -np.minimum(12 * (azimuth_diff / 65)**2, 20)  # 65° 3dB beamwidth
    
    # Vertical pattern (including downtilt)
    theta_3db = 10  # Typical vertical beamwidth
    elevation = np.degrees(np.arctan2(h_b - h_m, d*1000))  # Convert km to m
    A_v = -np.minimum(12 * ((elevation - tilt) / theta_3db)**2, 20)
    
    # Total antenna gain
    G_max = 18  # Max antenna gain (dBi)
    total_gain = G_max + A_h + A_v
    
    return L_hata - total_gain  # Note: L_hata is loss, so subtract gain


def compute_rsrp_map(clutter_map, candidate_positions, configs, 
                    tx_power_dbm=15.2, freq_mhz=1800, h_m=1.5, 
                    clutter_lookup=None, resolution=1.0):
    h, w = clutter_map.shape
    rsrp_map = np.full((h, w), -150.0)
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    user_positions = np.stack([xx.ravel(), yy.ravel()], axis=1)
    
    for idx, ((x, y), (place, height, tilt, azimuth)) in enumerate(zip(candidate_positions, configs)):
        if place < 0.5:
            continue
            
        tx_pos = np.array([x, y])
        distances = np.linalg.norm(user_positions - tx_pos, axis=1) * resolution
        distances_km = distances / 1000.0
        
        # Calculate user azimuths relative to transmitter
        dx = user_positions[:, 0] - x
        dy = user_positions[:, 1] - y
        user_azimuths = np.degrees(np.arctan2(dy, dx)) % 360
        
        # Get clutter type at transmitter location
        clutter_type = clutter_lookup[clutter_map[y, x]][0]
        
        # Calculate path loss for all users
        path_loss = np.array([
            sectorized_path_loss(
                d=distances_km[i],
                f=freq_mhz,
                h_b=height,
                h_m=h_m,
                clutter_class=clutter_type,
                azimuth=azimuth,
                user_azimuth=user_azimuths[i],
                tilt=tilt
            )
            for i in range(len(distances_km))
        ])
        
        # Additional clutter loss
        user_clutter = clutter_map[user_positions[:, 1], user_positions[:, 0]]
        clutter_loss = np.array([clutter_lookup[cid][1] for cid in user_clutter])
        
        # Total received power
        received_power = tx_power_dbm - path_loss - clutter_loss
        rsrp_map_flat = rsrp_map.ravel()
        rsrp_map_flat = np.maximum(rsrp_map_flat, received_power)
        rsrp_map = rsrp_map_flat.reshape((h, w))
    
    return rsrp_map

def compute_sinr_map(rsrp_map, candidate_positions, configs, noise_floor_dbm=-104):
    signal = rsrp_map
    interference = np.full_like(signal, 0.0)
    for idx, (place, *_rest) in enumerate(configs):
        if place < 0.5:
            continue
        interference += np.random.uniform(-120, -100, size=signal.shape)
    sinr = signal - 10 * np.log10(10 ** (interference / 10) + 10 ** (noise_floor_dbm / 10))
    return sinr