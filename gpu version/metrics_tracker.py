# metrics_tracker.py
import csv
import os
from torch.utils.tensorboard import SummaryWriter

class MetricsTracker:
    def __init__(self, log_dir="logs", csv_name="ppo_metrics.csv", use_tensorboard=True):
        self.csv_path = os.path.join(log_dir, csv_name)
        self.use_tensorboard = use_tensorboard
        os.makedirs(log_dir, exist_ok=True)
        self.metrics = []  

        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "reward", "entropy", "kl_div", "variance"])
        
                

        self.writer = SummaryWriter(log_dir=log_dir) if use_tensorboard else None

    def log(self, episode, reward, entropy=None, kl_div=None, variance=None):
        with open(self.csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                episode,
                reward,
                entropy if entropy is not None else '',
                kl_div if kl_div is not None else '',
                variance if variance is not None else ''
            ])
        if self.writer:
            self.writer.add_scalar("Reward", reward, episode)
            if entropy is not None:
                self.writer.add_scalar("Entropy", entropy, episode)
            if kl_div is not None:
                self.writer.add_scalar("KL_Divergence", kl_div, episode)
            if variance is not None:
                self.writer.add_scalar("Action_Variance", variance, episode)
        
        self.metrics.append({  # <-- ADD THIS
            "episode": episode,
            "reward": reward,
            "entropy": entropy,
            "kl_div": kl_div,
            "variance": variance
        })

    def close(self):    
        if self.writer:
            self.writer.close()
