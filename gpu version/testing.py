import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import os
import json
from affine import Affine
import rasterio
import pyproj
import pandas as pd
from datetime import datetime
from matplotlib.patches import Patch


from lte_env import LTEPlannerEnv
from gnn_models import GNNPolicy
from lte_utils import build_candidate_graph, compute_rsrp_map, compute_sinr_map , compute_sinr_map_auto


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str , required=True, help="Path to the input file or directory")

args = parser.parse_args()

print(f"The provided path is: {args.path}")


def select_action(policy_net, state):
    state = state.to(device)
    with torch.no_grad():
        logits = policy_net(state)
        print(logits)

    # --------------------------
    # Placement (Bernoulli)
    # --------------------------
    place_logits = logits[:, 0]
    place_probs = torch.sigmoid(place_logits)
    place_dist = torch.distributions.Bernoulli(probs=place_probs)
    place_sample = place_dist.sample()

    # --------------------------
    # Height (Continuous Normal)
    # --------------------------
    height_mean = logits[:, 1]      # raw output
    height_std = 0.5                # or learnable
    height_dist = torch.distributions.Normal(height_mean, height_std)
    height_sample = height_dist.rsample()  # reparameterized sample
    height_env = torch.clamp(height_sample * 12 + 28, 28.0, 45.0)  # map to env range

    # --------------------------
    # Tilt (Continuous Normal)
    # --------------------------
    tilt_mean = logits[:, 2]
    tilt_std = 0.5
    tilt_dist = torch.distributions.Normal(tilt_mean, tilt_std)
    tilt_sample = tilt_dist.rsample()
    tilt_env = torch.clamp(tilt_sample * 12, 0.0, 12.0)

    # --------------------------
    # Azimuth (Categorical)
    # --------------------------
    azimuth_logits = logits[:, 3:]
    azimuth_dist = torch.distributions.Categorical(logits=azimuth_logits)
    azimuth_sample = azimuth_dist.sample()
    azimuth_env = azimuth_sample * 5.0  # map to env range

    # --------------------------
    # Log probabilities
    # --------------------------
    log_prob = (
        place_dist.log_prob(place_sample).sum()
        + height_dist.log_prob(height_sample).sum()
        + tilt_dist.log_prob(tilt_sample).sum()
        + azimuth_dist.log_prob(azimuth_sample).sum()
    )

    entropy = (
        place_dist.entropy().sum()
        + height_dist.entropy().sum()
        + tilt_dist.entropy().sum()
        + azimuth_dist.entropy().sum()
    )

    action_dict = {
        "placement": place_sample.cpu().numpy(),
        "height": height_env.cpu().numpy(),
        "tilt": tilt_env.cpu().numpy(),
        "azimuth": azimuth_env.cpu().numpy()
    }

    action_tensor = torch.cat([
        place_sample.unsqueeze(1),
        height_sample.unsqueeze(1),
        tilt_sample.unsqueeze(1),
        azimuth_sample.float().unsqueeze(1)
    ], dim=1)

    return {
        "action_dict": action_dict,
        "action_tensor": action_tensor,
        "log_prob": log_prob,
        "logits": logits,
        "entropy": entropy,
        "dists": {
        "place": place_probs.detach(),         # store probs as tensor
        "height": (height_mean.detach(), torch.tensor(height_std)), 
        "tilt": (tilt_mean.detach(), torch.tensor(tilt_std)),
        "azimuth": azimuth_logits.detach()     # store logits as tensor
    }



        
    }

# === Load saved model ===
def load_model(path="gnn_policy_weights.pth"):
    model = GNNPolicy().to(device)
    model.load_state_dict(torch.load(path))
    model.eval()

    print(f"✅ Model loaded from {path}")
    print(model)
    return model

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# clutter_map = load_clutter_map("index_clutter.txt")

base_dir = "polygon_data"
project_name = "Untitled project (2)"
npy_path = os.path.join(base_dir, project_name, "mosaic.npy")

meta_path = os.path.join(base_dir, project_name, "metadata.json")

with open(meta_path, "r") as f:
    metadata = json.load(f)



transform = Affine.from_gdal(*metadata["transform"])

print(transform)


shape = metadata["shape"]


clutter_map_unmapped = np.load(npy_path,mmap_mode='r')
h, w = clutter_map_unmapped.shape

original_values = np.array([
    256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560,
    2816, 3072, 3328, 3584, 3840, 4096, 4352, 4608, 5120, 61912
])
new_classes = np.where(original_values == 61912, 0, original_values // 256)
mapping = dict(zip(original_values, new_classes))

# --- Extract unique values in the raster ---
unique_vals = np.unique(clutter_map_unmapped)

# --- Build a dict that includes defaults (-1) ---
remap_dict = {val: mapping.get(val, -1) for val in unique_vals}

# --- Vectorized remapping using np.take with indexer ---
# Step 1: make an array of remapped values in the order of unique_vals
remap_array = np.array([remap_dict[v] for v in unique_vals], dtype=np.int32)

# Step 2: use np.searchsorted to map all pixels via unique indices
indices = np.searchsorted(unique_vals, clutter_map_unmapped)
clutter_map = remap_array[indices]


print(clutter_map)
print(clutter_map.shape)
print(np.unique(clutter_map))




# Environment Setup
# h, w = 500, 500
# clutter_map = np.ones((h, w), dtype=int) * 2
# clutter_map[150:360, 150:360] = 5
# clutter_map[180:330, 180:330] = 6
# clutter_map[:100, :100] = 3
# clutter_map[420:, 420:] = 1
# clutter_map[100:200, 400:500] = 4
# clutter_map[300:380, 100:160] = 7
candidate_positions = [(x, y) for x in range(0, h, 25) for y in range(0, w, 25) if x < h and y < w and clutter_map[x , y] != -1]

# candidate_positions = [(25, 25), (25, 50), (25, 75), (50, 25), (50, 50), (50, 75), (75, 25), (75, 50), (75, 75), (75, 90)]
print(candidate_positions)


clutter_lookup = {
    0: ("urban", 0), 1: ("rural", 1), 2: ("rural", 0), 3: ("rural", 0.5),
    4: ("urban", 6), 5: ("rural", 9), 6: ("rural", 11), 7: ("rural", 15),
    8: ("rural", 5), 9: ("rural", 8), 10: ("suburban", 3), 11: ("urban", 10),
    12: ("suburban", 3), 13: ("urban", 8), 14: ("urban", 18), 15: ("suburban", 5),
    16: ("urban", 7), 17: ("rural", 2), 18: ("suburban", 4), 20: ("urban", 1.5),
    -1:("outofbounds",-1)
}


gpu_clutter_map = torch.Tensor(clutter_map).to(device)

# # === Candidate transmitter positions on a grid ===
# gpu_candidate_positions =torch.Tensor( [(x, y) for x in range(50, 450, 100) for y in range(50, 450, 100)]).to(device)
# candidate_positions = [(x, y) for x in range(50, 500, 100) for y in range(50, 500, 100)]

# # === Clutter mapping (merged) ===
# clutter_to_env = {
#     1: 'urban', 2: 'urban', 3: 'suburban', 4: 'urban',
#     5: 'rural', 6: 'rural', 7: 'rural', 8: 'rural',
#     9: 'rural', 10: 'suburban', 11: 'urban', 12: 'suburban',
#     13: 'urban', 14: 'urban', 15: 'suburban', 16: 'urban',
#     17: 'rural', 18: 'suburban', 20: 'urban'
# }
# clutter_loss_table = {
#     0: 0, 1: 1, 2: 0, 3: 0.5, 4: 6, 5: 9, 6: 11, 7: 15,
#     8: 5, 9: 8, 10: 3, 11: 10, 12: 3, 13: 8, 14: 18,
#     15: 5, 16: 7, 17: 2, 18: 4, 20: 1.5
# }
# clutter_lookup = {
#     i: (clutter_to_env.get(i, "urban"), clutter_loss_table.get(i, 5))
#     for i in range(0, 21)
# }


print(clutter_lookup)
# === Create environment and policy model ===
env = LTEPlannerEnv(clutter_map, candidate_positions, clutter_lookup, resolution=50)

if os.path.exists(args.path):
        policy_net = load_model(args.path)
else :
    raise Exception("path not found, exiting!")



# === Run the policy on the new state ===
state = env.reset()

print(state)

action_data = select_action(policy_net, state)



action = action_data['action_dict']
placement = torch.tensor(action["placement"], dtype=torch.float32, device=device)
height = torch.where(
            torch.tensor(action["height"], device=device) > 0.5,
            torch.tensor(45.0, device=device),
            torch.tensor(28.0, device=device)
        )
tilt = torch.tensor(action["tilt"], device=device) * 12.0
azimuth = torch.tensor(action["azimuth"], device=device) * 360.0

processed_action = torch.stack([placement, height, tilt, azimuth], dim=1)



_, reward, _, info = env.step(action)

print(info)

print(f"\n📈 Reward on new map: {reward:.2f}")
print(f"📍 Active Transmitters: {int(action["placement"].sum()) } / {len(candidate_positions)}")
 
# === Plotting ===
# def plot_transmitters(clutter_map, candidate_positions, action, title="Transmitter Placement"):
#     clutter_colors = {
#     -1: "#d3d3d3",  # UNKNOWN - light gray
#     1: "#a8e6ff",  # OPEN - light sky blue
#     2: "#91d5ff",  # SEA - ocean blue
#     3: "#87c3ff",  # INLAND WATER - lake blue
#     4: "#bdbdbd",  # MEAN INDIVIDUAL - gray
#     5: "#999999",  # MEAN COLLECTIVE - medium gray
#     6: "#5c5c5c",  # DENSE COLLECTIVE - dark gray
#     7: "#b5651d",  # SKYSCRAPERS - brown
#     8: "#f4a460",  # VILLAGE - sand brown
#     9: "#8b8b8b",  # INDUSTRIAL - steel gray
#     10: "#c0f9ff", # OPEN IN URBAN - very light blue
#     11: "#228B22", # FOREST - forest green
#     12: "#90ee90", # PARK - light green
#     13: "#006400", # DENSE INDIVIDUAL - dark green
#     14: "#8b4513", # GROUP OF SKYSCRAPERS - dark brown
#     15: "#6fbf73", # SPARSE FOREST - medium green
#     16: "#ffc107", # SCATTERED URBAN - amber
#     17: "#a2d149", # GRASS AGRICULTURE - greenish yellow
#     18: "#9acd32", # SWAMP - yellow green
#     20: "#ffe4b5", # AIRPORT - moccasin (light tan)
# }

#     clutter_rgb = np.zeros((h, w, 3))
#     for code, color in clutter_colors.items():
#         clutter_rgb[clutter_map == code] = plt.matplotlib.colors.to_rgb(color)
#     plt.figure(figsize=(6, 6))
#     plt.imshow(clutter_rgb, origin='lower')
#     # cmap='tab20', origin='lower') 
#     for i, (x, y) in enumerate(candidate_positions):
#         if action["placement"][i] > 0.5:
#             plt.plot(x, y, 'ro', label='Active' if i == 0 else "")
#         else:
#             plt.plot(x, y, 'bo', label='Inactive' if i == 0 else "")
#     plt.title(title)
#     plt.legend(loc="upper right")
#     plt.grid(True)
#     plt.show()



import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

def plot_transmitters(clutter_map, candidate_positions, action, title="Transmitter Placement"):
    clutter_colors = {
        -1: ("#d3d3d3", "UNKNOWN"),
        1: ("#a8e6ff", "OPEN"),
        2: ("#91d5ff", "SEA"),
        3: ("#87c3ff", "INLAND WATER"),
        4: ("#bdbdbd", "MEAN INDIVIDUAL"),
        5: ("#999999", "MEAN COLLECTIVE"),
        6: ("#5c5c5c", "DENSE COLLECTIVE"),
        7: ("#b5651d", "SKYSCRAPERS"),
        8: ("#f4a460", "VILLAGE"),
        9: ("#8b8b8b", "INDUSTRIAL"),
        10: ("#c0f9ff", "OPEN IN URBAN"),
        11: ("#228B22", "FOREST"),
        12: ("#90ee90", "PARK"),
        13: ("#006400", "DENSE INDIVIDUAL"),
        14: ("#8b4513", "GROUP OF SKYSCRAPERS"),
        15: ("#6fbf73", "SPARSE FOREST"),
        16: ("#ffc107", "SCATTERED URBAN"),
        17: ("#a2d149", "GRASS AGRICULTURE"),
        18: ("#9acd32", "SWAMP"),
        20: ("#ffe4b5", "AIRPORT"),
    }

    h, w = clutter_map.shape
    clutter_rgb = np.zeros((h, w, 3))
    for code, (color, _) in clutter_colors.items():
        clutter_rgb[clutter_map == code] = plt.matplotlib.colors.to_rgb(color)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(clutter_rgb, origin="lower")

    # Plot transmitters
    for i, (x, y) in enumerate(candidate_positions):
        if action["placement"][i] > 0.5:
            ax.plot(x, y, "ro", label="Active" if i == 0 else "")
        else:
            ax.plot(x, y, "bo", label="Inactive" if i == 0 else "")

    # Build dynamic legend for clutter classes present in the map
    present_classes = np.unique(clutter_map)
    legend_elements = []
    for cls in present_classes:
        if cls in clutter_colors:
            color, name = clutter_colors[cls]
            legend_elements.append(Patch(facecolor=color, edgecolor="black", label=name))

    # Add transmitter legend separately
    legend_elements.append(Patch(facecolor="red", edgecolor="black", label="Active Transmitter"))
    legend_elements.append(Patch(facecolor="blue", edgecolor="black", label="Inactive Transmitter"))

    ax.set_title(title)
    ax.grid(True)

    # Place legend outside plot (to the right)
    ax.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0,
        frameon=True
    )

    plt.tight_layout()
    plt.show()



def pixel_to_lonlat(row, col, transform):
    x, y = transform * (row, col)  # note: col=x, row=y
    transformer = pyproj.Transformer.from_crs('EPSG:32633', 'EPSG:4326', always_xy=True)
    lon, lat = transformer.transform(x, y)

    return lon, lat

# Lon/lat to pixel
def lonlat_to_pixel(lon, lat, transform):
    """Convert longitude/latitude to pixel coordinates (inverse function)"""
    # Step 1: Convert lon/lat to projected coordinates
    transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:32633', always_xy=True)
    x, y = transformer.transform(lon, lat)
    
    # Step 2: Convert projected coordinates to pixel coordinates
    inv_transform = ~transform  # Inverse of the affine transform
    col, row = inv_transform * (x, y)
    
    # Step 3: Round to integers and return in same order as input function
    return int(round(row)), int(round(col))


def save_result(candidate_positions, action , clutter_lookup , clutter_map , env , transform, filename=None):
    deployed_transmitters = []
    for i, (x,y) in enumerate(candidate_positions):
        if action["placement"][i] > 0.5:
            lon, lat = pixel_to_lonlat(y, x, transform)
            transmitter_config = {
                "Transceiver Name": "PLACEHOLDER",
                "Cell Name": f"PLACEHOLDER_{i}",
                "Longitude": lon,
                "Latitude": lat,
                "Frequency Band": env.freq,
                "Scenario":clutter_lookup[clutter_map[y,x]][0],
                "Antenna Type":"1800MHz 65deg 17dBi 0Tilt",
                "Azimuth":action["azimuth"][i].item(),
                "Height":action["height"][i].item(),
                "Mechanical Downtilt":0,
                "Electrical Downtilt":action["tilt"][i].item(),
                "Max Power(dBm)": env.tx_power_dbm}

            deployed_transmitters.append(transmitter_config)
    

        # Save to CSV if there are deployed transmitters
    if deployed_transmitters:
        # Create DataFrame
        df = pd.DataFrame(deployed_transmitters)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.join(base_dir, project_name), exist_ok=True)
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"deployed_transmitters_{timestamp}.csv"
        
        # Ensure .csv extension
        if not filename.endswith('.csv'):
            filename += '.csv'
            
        filepath = os.path.join(base_dir, project_name, filename)
        
        # Save to CSV
        df.to_csv(filepath, index=False)
        
        # Print summary
        print(f"✅ Saved {len(deployed_transmitters)} transmitters to: {filepath}")
        print(f"📊 Columns: {list(df.columns)}")
        print(f"📍 Coordinate range: Lon [{df['Longitude'].min():.6f}, {df['Longitude'].max():.6f}], "
              f"Lat [{df['Latitude'].min():.6f}, {df['Latitude'].max():.6f}]")
        
        return deployed_transmitters, filepath
    else:
        print("⚠️  No transmitters were deployed (all placement values <= 0.5)")
        return deployed_transmitters, None

        


# === Visualize placement ===
plot_transmitters(clutter_map, candidate_positions, action, title="Test Episode - Transmitters")

print("clutter_map :", clutter_map)
print("processed_action :", processed_action)


# === Visualize RSRP and SINR maps ===
rsrp_max_map , rsrp_maps_per_bs , unflattend_mask = compute_rsrp_map(clutter_map, candidate_positions, processed_action, clutter_lookup = clutter_lookup , testing = True)
# sinr_map , _ = compute_sinr_map_auto(rsrp_maps_per_bs, processed_action,unflattend_mask,device="cuda")

print(torch.unique(rsrp_max_map))

print("\nCoverage Statistics:")
print(f"- Area > -95dBm: {(rsrp_max_map > -95).float().mean()*100:.1f}%")
print(f"- Area > -100dBm: {(rsrp_max_map > -100).float().mean()*100:.1f}%")



plt.figure(figsize=(6, 5))
plt.imshow(rsrp_max_map.detach().cpu(), cmap='viridis', origin='lower')
plt.title("RSRP Map (dBm)") 
plt.colorbar()
plt.show()




    


deployed, filepath = save_result(candidate_positions, action, clutter_lookup, 
                                 clutter_map, env, transform,  
                                 filename="results.csv")





# plt.figure(figsize=(6, 5))
# plt.imshow(sinr_map.detach().cpu(), cmap='plasma', origin='lower')
# plt.title("SINR Map (dB)")
# plt.colorbar()
# plt.show()
