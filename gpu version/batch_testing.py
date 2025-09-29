import numpy as np
import torch
import argparse
import os
import json
from affine import Affine
import pyproj
import pandas as pd
from datetime import datetime
import traceback

from lte_env import LTEPlannerEnv
from gnn_models import GNNPolicy
from lte_utils import compute_rsrp_map

# -------------------
# CLI arguments
# -------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Path to trained GNN model weights")
parser.add_argument("--base_dir", type=str, default="polygon_data", help="Base directory of projects")
args = parser.parse_args()

# Use the base_dir from args for global access
base_dir = args.base_dir

# -------------------
# Device setup
# -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------
# Action selection
# -------------------
def select_action(policy_net, state):
    state = state.to(device)
    with torch.no_grad():
        logits = policy_net(state)

    place_logits = logits[:, 0]
    place_probs = torch.sigmoid(place_logits)
    place_dist = torch.distributions.Bernoulli(probs=place_probs)
    place_sample = place_dist.sample()

    height_mean = logits[:, 1]
    height_dist = torch.distributions.Normal(height_mean, 0.5)
    height_sample = height_dist.rsample()
    height_env = torch.clamp(height_sample * 12 + 28, 28.0, 45.0)

    tilt_mean = logits[:, 2]
    tilt_dist = torch.distributions.Normal(tilt_mean, 0.5)
    tilt_sample = tilt_dist.rsample()
    tilt_env = torch.clamp(tilt_sample * 12, 0.0, 12.0)

    azimuth_logits = logits[:, 3:]
    azimuth_dist = torch.distributions.Categorical(logits=azimuth_logits)
    azimuth_sample = azimuth_dist.sample()
    azimuth_env = azimuth_sample * 5.0

    action_dict = {
        "placement": place_sample.cpu().numpy(),
        "height": height_env.cpu().numpy(),
        "tilt": tilt_env.cpu().numpy(),
        "azimuth": azimuth_env.cpu().numpy()
    }

    return action_dict

# -------------------
# Model loader
# -------------------
def load_model(path):
    model = GNNPolicy().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"✅ Model loaded from {path}")
    return model

# -------------------
# Helpers for geo
# -------------------
def pixel_to_lonlat(row, col, transform):
    x, y = transform * (row, col)
    transformer = pyproj.Transformer.from_crs('EPSG:32633', 'EPSG:4326', always_xy=True)
    lon, lat = transformer.transform(x, y)
    return lon, lat

# -------------------
# Your save_result function
# -------------------
def save_result(candidate_positions, action, clutter_lookup, clutter_map, env, transform, project_name, filename=None):
    deployed_transmitters = []
    for i, (x, y) in enumerate(candidate_positions):
        if action["placement"][i] > 0.5:
            lon, lat = pixel_to_lonlat(y, x, transform)
            transmitter_config = {
                "Transceiver Name": "PLACEHOLDER",
                "Cell Name": f"PLACEHOLDER_{i}",
                "Longitude": lon,
                "Latitude": lat,
                "Frequency Band": env.freq,
                "Scenario": clutter_lookup[clutter_map[y, x]][0],
                "Antenna Type": "1800MHz 65deg 17dBi 0Tilt",
                "Azimuth": action["azimuth"][i].item(),
                "Height": action["height"][i].item(),
                "Mechanical Downtilt": 0,
                "Electrical Downtilt": action["tilt"][i].item(),
                "Max Power(dBm)": env.tx_power_dbm
            }
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

# -------------------
# Save aggregated results (modified to work with your format)
# -------------------
def save_aggregate(all_transmitters, summary_list, base_dir):
    if not all_transmitters and not summary_list:
        print("⚠️ No results to save")
        return
        
    out_dir = os.path.join(base_dir, "aggregated_results")
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    if all_transmitters:
        # Flatten all transmitters from all projects
        all_tx_flat = []
        for project_tx_list in all_transmitters:
            all_tx_flat.extend(project_tx_list)
        
        if all_tx_flat:
            df_all = pd.DataFrame(all_tx_flat)
            df_all.to_csv(os.path.join(out_dir, f"all_transmitters_{timestamp}.csv"), index=False)
            print(f"✅ All transmitters saved: {len(all_tx_flat)} records")

    if summary_list:
        df_summary = pd.DataFrame(summary_list)
        df_summary.to_csv(os.path.join(out_dir, f"summary_{timestamp}.csv"), index=False)
        print(f"✅ Summary saved: {len(summary_list)} projects")
        print(df_summary.head())

# -------------------
# Main pipeline
# -------------------
policy_net = load_model(args.model)

clutter_lookup = {
    0: ("urban", 0), 1: ("rural", 1), 2: ("rural", 0), 3: ("rural", 0.5),
    4: ("urban", 6), 5: ("rural", 9), 6: ("rural", 11), 7: ("rural", 15),
    8: ("rural", 5), 9: ("rural", 8), 10: ("suburban", 3), 11: ("urban", 10),
    12: ("suburban", 3), 13: ("urban", 8), 14: ("urban", 18), 15: ("suburban", 5),
    16: ("urban", 7), 17: ("rural", 2), 18: ("suburban", 4), 20: ("urban", 1.5),
    -1: ("outofbounds", -1)
}

all_transmitters = []  # Will store lists of transmitters for each project
summary_list = []
failed_projects = []

# Get list of all projects
project_list = [name for name in os.listdir(base_dir) 
                if os.path.isdir(os.path.join(base_dir, name))]

print(f"🔍 Found {len(project_list)} projects to process")

for project_name in project_list:
    try:
        print(f"\n🚀 Processing project: {project_name}")
        
        project_path = os.path.join(base_dir, project_name)
        npy_path = os.path.join(project_path, "mosaic.npy")
        meta_path = os.path.join(project_path, "metadata.json")

        if not (os.path.exists(npy_path) and os.path.exists(meta_path)):
            print(f"  ⚠️ Skipping {project_name}, missing files")
            failed_projects.append({
                "project": project_name,
                "error": "Missing required files (mosaic.npy or metadata.json)"
            })
            continue

        with open(meta_path, "r") as f:
            metadata = json.load(f)

        transform = Affine.from_gdal(*metadata["transform"])
        clutter_map_unmapped = np.load(npy_path, mmap_mode="r")
        h, w = clutter_map_unmapped.shape

        # --- Remap clutter ---
        original_values = np.array([
            256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560,
            2816, 3072, 3328, 3584, 3840, 4096, 4352, 4608, 5120, 61912
        ])
        new_classes = np.where(original_values == 61912, 0, original_values // 256)
        mapping = dict(zip(original_values, new_classes))
        unique_vals = np.unique(clutter_map_unmapped)
        remap_dict = {val: mapping.get(val, -1) for val in unique_vals}
        remap_array = np.array([remap_dict[v] for v in unique_vals], dtype=np.int32)
        indices = np.searchsorted(unique_vals, clutter_map_unmapped)
        clutter_map = remap_array[indices]

        if len(np.unique(clutter_map)) < 3 :
            print(f"  ⚠️ Clutter map too simple for {project_name}")
            failed_projects.append({
                "project": project_name,
                "error": "Clutter map too simple for"
            })
            continue


        candidate_positions = [
            (x, y) for x in range(0, h, 10) for y in range(0, w, 10)
            if clutter_map[x, y] != -1
        ]

        if not candidate_positions:
            print(f"  ⚠️ No valid candidate positions for {project_name}")
            failed_projects.append({
                "project": project_name,
                "error": "No valid candidate positions found"
            })
            continue

        env = LTEPlannerEnv(clutter_map, candidate_positions, clutter_lookup, resolution=50)
        state = env.reset()
        
        # Try to select action
        try:
            action = select_action(policy_net, state)
        except Exception as e:
            print(f"  ❌ Action selection failed for {project_name}: {str(e)}")
            failed_projects.append({
                "project": project_name,
                "error": f"Action selection failed: {str(e)}"
            })
            continue

        placement = torch.tensor(action["placement"], dtype=torch.float32, device=device)
        height = torch.tensor(action["height"], device=device)
        tilt = torch.tensor(action["tilt"], device=device)
        azimuth = torch.tensor(action["azimuth"], device=device)
        processed_action = torch.stack([placement, height, tilt, azimuth], dim=1)

        _, reward, _, info = env.step(action)

        rsrp_max_map, _, _ = compute_rsrp_map(
            clutter_map, candidate_positions, processed_action,
            clutter_lookup=clutter_lookup, testing=True
        )

        cov95 = float((rsrp_max_map > -95).float().mean() * 100)
        cov100 = float((rsrp_max_map > -100).float().mean() * 100)
        num_tx = int(action["placement"].sum())

        # Use your save_result function to save transmitters for this project
        deployed_transmitters, filepath = save_result(
            candidate_positions, action, clutter_lookup, clutter_map, 
            env, transform, project_name
        )
        
        # Add to aggregated results
        if deployed_transmitters:
            all_transmitters.append(deployed_transmitters)

        # Create summary for this project
        project_summary = {
            "Project": project_name,
            "Coverage > -95dBm (%)": cov95,
            "Coverage > -100dBm (%)": cov100,
            "Num_Transmitters": num_tx,
            "Reward": reward,
            "Output_File": filepath if filepath else "No transmitters deployed"
        }
        summary_list.append(project_summary)
        
        print(f"  ✅ {project_name} completed successfully")
        print(f"    - Coverage > -95dBm: {cov95:.2f}%")
        print(f"    - Coverage > -100dBm: {cov100:.2f}%")
        print(f"    - Transmitters: {num_tx}")
        print(f"    - Reward: {reward:.4f}")

    except Exception as e:
        print(f"  ❌ Error processing {project_name}: {str(e)}")
        print(f"  📋 Traceback: {traceback.format_exc()}")
        failed_projects.append({
            "project": project_name,
            "error": str(e),
            "traceback": traceback.format_exc()
        })
        continue

# Save aggregated results
print(f"\n📊 Processing complete!")
print(f"✅ Successful projects: {len(summary_list)}")
print(f"❌ Failed projects: {len(failed_projects)}")

if failed_projects:
    print(f"\n⚠️ Failed projects:")
    for failed in failed_projects:
        print(f"  - {failed['project']}: {failed['error']}")
    
    # Save failed projects log
    failed_df = pd.DataFrame(failed_projects)
    failed_path = os.path.join(base_dir, "failed_projects.csv")
    failed_df.to_csv(failed_path, index=False)
    print(f"📁 Failed projects log saved: {failed_path}")

save_aggregate(all_transmitters, summary_list, base_dir)