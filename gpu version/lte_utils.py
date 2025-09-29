import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def build_candidate_graph(clutter_map, candidate_positions, k=5):
    h, w = clutter_map.shape
    node_features = []
    num_nodes = len(candidate_positions)

    for (x, y) in candidate_positions:
        clutter = clutter_map[y, x]
        node_feat = [x / w, y / h, clutter / clutter_map.max()]
        node_features.append(node_feat)
    x = torch.tensor(node_features, dtype=torch.float, requires_grad=True).to(device)


    k_desired = k
    k_fraction = 0.5
    k_safe = min(k_desired, max(1, int(num_nodes * k_fraction)), num_nodes - 1) 
    
    nbrs = NearestNeighbors(n_neighbors=k_safe).fit(candidate_positions)  # CPU only
    _, indices = nbrs.kneighbors(candidate_positions)
    
    edge_list = []
    for i in range(len(candidate_positions)):
        for j in indices[i][1:]:
            edge_list.append([i, j])
            edge_list.append([j, i])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(device)
    
    return Data(x=x, edge_index=edge_index)

def cost231_hata(d, f, h_b, h_m, clutter_class):
    d = torch.clamp(d, min=0.01)
    a_h_m = (1.1 * torch.log10(f) - 0.7) * h_m - (1.56 * torch.log10(f) - 0.8)
    c = 3.0 if clutter_class == "urban" else 0.0
    L = (46.3 + 33.9 * torch.log10(f) - 13.82 * torch.log10(h_b)
         - a_h_m + (44.9 - 6.55 * torch.log10(h_b)) * torch.log10(d) + c)
    return L

def sectorized_path_loss(d, f, h_b, h_m, clutter_class, azimuth, user_azimuth, tilt):
    L_hata = cost231_hata(d, f, h_b, h_m, clutter_class)

    azimuth_diff = torch.remainder(user_azimuth - azimuth + 180, 360) - 180
    A_h = -torch.minimum(12 * (azimuth_diff / 65) ** 2, torch.tensor(20.0, device=d.device))
    
    theta_3db = 10
    elevation = torch.rad2deg(torch.atan2(h_b - h_m, d * 1000))
    A_v = -torch.minimum(12 * ((elevation - tilt) / theta_3db) ** 2, torch.tensor(20.0, device=d.device))

    G_max = 18
    total_gain = G_max + A_h + A_v

    return L_hata - total_gain

def compute_rsrp_map(clutter_map, candidate_positions, configs,
                     tx_power_dbm=15.2, freq_mhz=1800, h_m=1.5, 
                     clutter_lookup=None, resolution=1.0,
                     testing=False):

    # print(clutter_lookup)
    # print(configs)
    h, w = clutter_map.shape
    num_bs = len(candidate_positions)

    # print("clutter_map shape:" , clutter_map.shape)
    rsrp_map = torch.full((h, w), -150.0, dtype=torch.float, device=device)
    rsrp_maps_per_bs = torch.full((num_bs, h, w), -150.0, dtype=torch.float, device=device)

    
    yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    user_positions = torch.stack([xx.flatten(), yy.flatten()], dim=1).to(device)

    # mask out pixels marked as -1 (out of polygon)
    clutter_flat = clutter_map.flatten()
    unflattend_mask = clutter_map !=-1
    valid_mask = clutter_flat != -1
    valid_positions = user_positions[valid_mask]

    for idx, ((x, y), (place, height, tilt, azimuth)) in enumerate(zip(candidate_positions, configs)):
        if place < 0.5:
            continue

        # print("(x,y): ", x , y)
            
        tx_pos = torch.tensor([x, y], dtype=torch.float, device=device)
        distances = torch.norm(valid_positions - tx_pos, dim=1) * resolution
        distances_km = distances / 1000.0

        dx = valid_positions[:, 0] - x  
        dy = valid_positions[:, 1] - y
        user_azimuths = torch.remainder(torch.rad2deg(torch.atan2(dy, dx)), 360)
        
        # print("ERROR HERE : " , int(clutter_map[int(y), int(x)]))

        clutter_type = clutter_lookup[int(clutter_map[int(y), int(x)])]

        d = distances_km
        f = torch.tensor(freq_mhz, dtype=torch.float, device=device)
        # h_b = torch.tensor(height, dtype=torch.float, device=device)
        h_b = height.detach().clone().to(device).requires_grad_(True)

        h_m_tensor = torch.tensor(h_m, dtype=torch.float, device=device)
        # tilt_tensor = torch.tensor(tilt, dtype=torch.float, device=device)
        tilt_tensor = tilt.detach().clone().to(device).requires_grad_(True)

        # az_tensor = torch.tensor(azimuth, dtype=torch.float, device=device)
        az_tensor = azimuth.detach().clone().to(device).requires_grad_(True)


        path_loss = sectorized_path_loss(d, f, h_b, h_m_tensor, clutter_type, az_tensor, user_azimuths, tilt_tensor)



        user_clutter = clutter_map[valid_positions[:, 1].long().cpu(), valid_positions[:, 0].long().cpu()]

        # print(user_clutter)
        # print(clutter_lookup)


        clutter_loss = torch.tensor([clutter_lookup[cid][1] for cid in user_clutter.cpu().numpy()], dtype=torch.float, device=device) if not testing else torch.tensor([clutter_lookup[cid][1] for cid in user_clutter], dtype=torch.float, device=device)
        received_power = tx_power_dbm - path_loss - clutter_loss

         # write only valid pixels back into the RSRP map
        rsrp_bs = torch.full((h*w,), -150.0, dtype=torch.float, device=device)
        rsrp_bs[valid_mask] = received_power
        rsrp_maps_per_bs[idx] = rsrp_bs.reshape((h, w))


        # rsrp_maps_per_bs[idx] = received_power.reshape((h, w))
        
    # rsrp_map = torch.maximum(rsrp_map.flatten(), received_power).reshape((h, w))    
    rsrp_max_map, _ = torch.max(rsrp_maps_per_bs, dim=0)
    # print(torch.unique(rsrp_max_map))
    return rsrp_max_map , rsrp_maps_per_bs , unflattend_mask

def compute_sinr_map(rsrp_map, configs, noise_floor_dbm=-104):
    signal = rsrp_map
    interference = torch.zeros_like(signal, device=device)
    for (place, *_rest) in configs:
        if place < 0.5:
            continue
        # Fake interference – could be enhanced
        interference += torch.empty_like(signal).uniform_(-120, -100)
    
    sinr = signal - 10 * torch.log10(10 ** (interference / 10) + 10 ** (noise_floor_dbm / 10))
    return sinr


def compute_sinr_map_auto(rsrp_maps, configs,valid_mask, noise_floor_dbm=-104, device="cpu"):
    """
    Compute SINR map by auto-selecting serving BS per pixel.
    """
    num_bs, H, W = rsrp_maps.shape

    print(valid_mask.dtype)
# Mask invalid pixels
    masked_rsrp = rsrp_maps.clone()
    masked_rsrp = masked_rsrp.masked_fill(~valid_mask.unsqueeze(0), -150.0)


    # Serving BS per pixel
    serving_rsrp, serving_idx = torch.max(masked_rsrp, dim=0)  # [H,W]

    # Signal in mW
    signal_mw = 10 ** (serving_rsrp / 10.0)

    # Interference (sum of all other BSs)
    interference_mw = torch.zeros_like(signal_mw, device=device)
    for bs_idx, (place, *_rest) in enumerate(configs):
        if place < 0.5:
            continue
        mask = (serving_idx != bs_idx) & valid_mask
        interf_dbm = masked_rsrp[bs_idx]
        interf_mw = 10 ** (interf_dbm / 10.0)
        interference_mw += interf_mw * mask

    # Noise in mW
    noise_mw = 10 ** (noise_floor_dbm / 10.0)

    # SINR
    sinr_linear = signal_mw / (interference_mw + noise_mw)
    sinr_db = 10 * torch.log10(sinr_linear + 1e-12)

    # Ensure invalid pixels remain at very low value
    sinr_db[~valid_mask] = -150.0

    return sinr_db, serving_idx