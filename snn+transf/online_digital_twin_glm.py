#!/usr/bin/env python3
"""
Online Hawkes-GLM "digital twin" for extracellular spike recordings
-------------------------------------------------------------------
- Loads a Braingeneers-style ACQM zip with a nested qm.npz
- Builds binned spike matrix X [N, T] from per-unit spike sample indices
- Splits into contiguous 60/20/20 train/val/test segments
- Trains an online Hawkes-GLM with exponential basis history (both self & cross)
  * Signed couplings; sparsity via weighted L1
  * Spatially weighted penalty encouraging local connectivity using unit positions
  * Optional k-NN mask for incoming connections (reduces parameters)
  * Per-neuron reconstruction metrics (MAE, correlation, spike timing accuracy)
  * Per-neuron loss tracking for optimization monitoring
- Evaluates NLL, per-unit firing rates; generates a simulated test segment
- Saves quick-look plots and a torch model checkpoint

Author: you + ChatGPT (Sebas digital twin edition)
Date: 2025-11-02 (Updated with per-neuron reconstruction metrics)
"""
from __future__ import annotations
import os, io, zipfile, math, json
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------------------------------
# Per-Neuron Reconstruction Metrics
# -------------------------------

def compute_per_neuron_metrics(X_real: np.ndarray, X_gen: np.ndarray, dt: float) -> Dict:
    """
    Compute detailed per-neuron reconstruction metrics.
    
    Args:
        X_real: Real spike matrix [N, T]
        X_gen: Generated spike matrix [N, T]
        dt: Bin width in seconds
        
    Returns:
        Dictionary with per-neuron metrics and aggregated statistics
    """
    N, T = X_real.shape
    
    metrics = {
        'per_neuron': {},
        'summary': {}
    }
    
    mae_list = []
    mse_list = []
    corr_list = []
    rate_error_list = []
    spike_count_real_list = []
    spike_count_gen_list = []
    
    for i in range(N):
        real = X_real[i, :]
        gen = X_gen[i, :]
        
        # Mean Absolute Error (per bin)
        mae = mean_absolute_error(real, gen)
        
        # Mean Squared Error (per bin)
        mse = mean_squared_error(real, gen)
        
        # Correlation coefficient
        if real.std() > 1e-9 and gen.std() > 1e-9:
            corr = np.corrcoef(real, gen)[0, 1]
        else:
            corr = 0.0
        
        # Firing rate comparison
        rate_real = real.mean() / dt  # Hz
        rate_gen = gen.mean() / dt    # Hz
        rate_error = abs(rate_real - rate_gen)
        rate_error_pct = (rate_error / max(rate_real, 1e-6)) * 100
        
        # Spike counts
        spike_count_real = int(real.sum())
        spike_count_gen = int(gen.sum())
        
        # Spike timing precision (Van Rossum-like metric)
        # Convolve with Gaussian kernel and compute MSE
        sigma_bins = max(1, int(0.010 / dt))  # 10ms smoothing window
        from scipy.ndimage import gaussian_filter1d
        real_smooth = gaussian_filter1d(real.astype(float), sigma=sigma_bins)
        gen_smooth = gaussian_filter1d(gen.astype(float), sigma=sigma_bins)
        timing_mse = mean_squared_error(real_smooth, gen_smooth)
        
        # Store per-neuron metrics
        metrics['per_neuron'][i] = {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(np.sqrt(mse)),
            'correlation': float(corr),
            'rate_real_hz': float(rate_real),
            'rate_gen_hz': float(rate_gen),
            'rate_error_hz': float(rate_error),
            'rate_error_pct': float(rate_error_pct),
            'spike_count_real': spike_count_real,
            'spike_count_gen': spike_count_gen,
            'timing_mse': float(timing_mse)
        }
        
        mae_list.append(mae)
        mse_list.append(mse)
        corr_list.append(corr)
        rate_error_list.append(rate_error)
        spike_count_real_list.append(spike_count_real)
        spike_count_gen_list.append(spike_count_gen)
    
    # Summary statistics
    metrics['summary'] = {
        'mae_mean': float(np.mean(mae_list)),
        'mae_std': float(np.std(mae_list)),
        'mae_median': float(np.median(mae_list)),
        'mse_mean': float(np.mean(mse_list)),
        'rmse_mean': float(np.sqrt(np.mean(mse_list))),
        'correlation_mean': float(np.mean(corr_list)),
        'correlation_std': float(np.std(corr_list)),
        'correlation_median': float(np.median(corr_list)),
        'rate_error_mean_hz': float(np.mean(rate_error_list)),
        'rate_error_std_hz': float(np.std(rate_error_list)),
        'total_spikes_real': int(np.sum(spike_count_real_list)),
        'total_spikes_gen': int(np.sum(spike_count_gen_list)),
        'spike_count_error_pct': float(abs(np.sum(spike_count_gen_list) - np.sum(spike_count_real_list)) / max(np.sum(spike_count_real_list), 1) * 100)
    }
    
    return metrics


# -------------------------------
# I/O and preprocessing
# -------------------------------

def load_acqm_zip(acqm_zip_path: str):
    """Load nested qm.npz from the provided ACQM zip.
    Returns:
      spikes_dict: dict[int -> np.ndarray of spike sample indices]
      neuron_data: dict[int -> dict(...)] (must contain 'position')
      fs: float sampling frequency (Hz)
    """
    with zipfile.ZipFile(acqm_zip_path, 'r') as zf:
        # heuristic: inner file is named 'qm.npz'
        inner_name = 'qm.npz'
        if inner_name not in zf.namelist():
            # try to find .npz inside
            candidates = [n for n in zf.namelist() if n.lower().endswith('.npz')]
            if not candidates:
                raise FileNotFoundError("No .npz found inside the zip")
            inner_name = candidates[0]
        with zf.open(inner_name, 'r') as f:
            content = f.read()
    inner = np.load(io.BytesIO(content), allow_pickle=True)

    # train: 0-d object np.ndarray containing dict[int->np.ndarray]
    train_obj = inner['train'].item()
    neuron_data = inner['neuron_data'].item()
    fs = float(inner['fs'].item() if hasattr(inner['fs'], 'item') else float(inner['fs']))

    # Some datasets may have keys as np.int64; convert to int
    spikes_dict = {int(k): np.asarray(v, dtype=np.int64) for k, v in train_obj.items()}
    return spikes_dict, neuron_data, fs


def build_positions(neuron_data: Dict[int, dict], unit_ids: List[int]) -> np.ndarray:
    """Extract XY positions for the ordered list of unit_ids. Returns array [N, 2]."""
    pos = []
    for uid in unit_ids:
        entry = neuron_data.get(uid, None)
        if entry is None:
            raise KeyError(f"Unit id {uid} not found in neuron_data")
        p = np.asarray(entry['position'], dtype=float).ravel()
        if p.size < 2:
            raise ValueError(f"Position for unit {uid} has wrong shape {p.shape}")
        pos.append(p[:2])
    return np.vstack(pos)


def spikes_to_binned_matrix(spikes_dict: Dict[int, np.ndarray], fs: float, dt: float,
                             unit_ids: List[int], t_max_samples: int | None = None) -> Tuple[np.ndarray, int]:
    """Convert per-unit spike sample indices into a binned binary matrix X [N, T].
    Args:
      spikes_dict: unit_id -> int64 sample indices
      fs: sampling rate (Hz)
      dt: bin width (seconds)
      unit_ids: ordered units to keep
      t_max_samples: optional cap on total samples (e.g. to trim tail)
    Returns: (X, T) where X is uint8 with shape [N, T]
    """
    # Determine total duration
    max_sample = 0
    for uid in unit_ids:
        s = spikes_dict[uid]
        if s.size:
            max_sample = max(max_sample, int(s.max()))
    if t_max_samples is not None:
        max_sample = min(max_sample, int(t_max_samples))

    total_seconds = max_sample / fs
    T = int(math.floor(total_seconds / dt)) + 1

    N = len(unit_ids)
    X = np.zeros((N, T), dtype=np.uint8)

    for i, uid in enumerate(unit_ids):
        s = spikes_dict[uid]
        if s.size == 0:
            continue
        # convert to bin indices
        tb = (s / fs / dt).astype(np.int64)
        tb = tb[(tb >= 0) & (tb < T)]
        if tb.size:
            # set bins to 1 (clip duplicates)
            X[i, np.unique(tb)] = 1
    return X, T


# -------------------------------
# Online Hawkes-GLM
# -------------------------------
@dataclass
class GLMConfig:
    dt: float = 0.002  # 2 ms bins for better temporal resolution
    taus: Tuple[float, ...] = (0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128)  # include short scales right after delay
    delay_ms: float = 2.0  # physiologic min synaptic delay
    lr: float = 5e-3
    l1: float = 1e-3
    l2: float = 0.0
    l1_group: float = 5e-4  # group lasso across bases (edge-wise sparsity)
    device: str = 'auto'  # 'cpu' or 'cuda' or 'auto'
    max_rate_hz: float = 1000.0  # numerical clamp
    loglam_min: float = -20.0    # clamp for log-lambda to avoid -inf
    loglam_max: float = 7.0      # clamp for log-lambda to avoid +inf
    grad_clip: float = 1.0       # gradient norm clip
    pos_mass_penalty: float = 0.0  # penalty weight for positive kernel mass (stability)
    self_pos_penalty: float = 5e-3  # penalty on positive self-mass for refractoriness
    knn_k: int = 20               # incoming connections per neuron (kNN mask); set 0 to disable
    spatial_ell: float = 400.0    # microns; larger -> weaker spatial penalty
    batch_bins: int = 250         # log every ~0.5 s @ 2 ms
    train_bins: int | None = None # if set, limit training bins


class OnlineHawkesGLM(nn.Module):
    def __init__(self, N: int, M: int, mask_in: torch.Tensor | None, dist_mat: torch.Tensor | None,
                 cfg: GLMConfig):
        super().__init__()
        self.N, self.M = N, M
        self.cfg = cfg
        # parameters
        self.b = nn.Parameter(torch.zeros(N))              # baseline log-rate
        self.H = nn.Parameter(torch.zeros(N, N, M))        # coupling kernels (signed)

        # delay buffer implementation
        self.delay_bins = max(1, int(round((cfg.delay_ms/1000.0)/cfg.dt)))
        self.register_buffer('inj_buf', torch.zeros(self.delay_bins, N))
        self.inj_head = 0

        # masks & penalty weights
        if mask_in is None:
            self.register_buffer('mask', torch.ones(N, N, dtype=torch.bool))
        else:
            self.register_buffer('mask', mask_in.to(torch.bool))

        if dist_mat is None:
            self.register_buffer('penalty_w', torch.ones(N, N, M))
        else:
            # distance-weighted L1: w = 1 + d/ell  (broadcast over M)
            ell = max(1e-6, float(cfg.spatial_ell))
            w2d = 1.0 + (dist_mat / ell)
            self.register_buffer('penalty_w', w2d[:, :, None].repeat(1, 1, M))

        # basis decays
        taus = torch.tensor(cfg.taus, dtype=torch.float32)
        self.register_buffer('alpha', torch.exp(-torch.tensor(cfg.dt) / taus))  # [M]

        # runtime state (not parameters)
        self.register_buffer('z', torch.zeros(N, M))   # basis traces

    def reset_state(self):
        self.z.zero_()
        self.inj_buf.zero_()
        self.inj_head = 0

    def forward_intensity(self):
        """Compute log-lambda and lambda from current traces z and params.
        Returns: loglam [N], lam [N]
        """
        H_eff = self.H * self.mask.unsqueeze(-1)
        loglam = self.b + torch.einsum('ijn,jn->i', H_eff, self.z)
        loglam = torch.clamp(loglam, min=self.cfg.loglam_min, max=self.cfg.loglam_max)
        lam = torch.exp(loglam)
        return loglam, lam

    def step_online(self, x_t: torch.Tensor, return_per_neuron: bool = False) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """One online update for a single bin x_t in {0,1}^N with exact Bernoulli likelihood.
        
        Args:
            x_t: Current spike vector [N]
            return_per_neuron: If True, return (total_loss, per_neuron_nll)
            
        Returns:
            total_loss: Scalar loss
            per_neuron_nll: [N] per-neuron negative log-likelihood (if return_per_neuron=True)
        """
        # update traces with delay: use delayed injection from buffer
        with torch.no_grad():
            delayed = self.inj_buf[self.inj_head]                 # x_{t - delay}
            self.z.mul_(self.alpha)
            self.z.add_((1.0 - self.alpha) * delayed[:, None])
            self.inj_buf[self.inj_head] = x_t                     # push current spikes
            self.inj_head = (self.inj_head + 1) % self.delay_bins

        loglam, lam = self.forward_intensity()
        dt = self.cfg.dt
        # Exact Bernoulli likelihood (better than small-Δt Poisson at moderate rates)
        p = 1.0 - torch.exp(-lam * dt)
        eps = 1e-12
        per_neuron_nll = -(x_t * torch.log(p + eps) + (1 - x_t) * torch.log(1 - p + eps))
        nll = per_neuron_nll.mean()

        # penalties
        l1 = self.cfg.l1 * (torch.abs(self.H) * self.penalty_w).mean()
        l2 = self.cfg.l2 * (self.H ** 2).mean() if self.cfg.l2 > 0 else 0.0
        
        # Stability: penalize positive kernel mass (approx discrete integral)
        one_minus_alpha = (1.0 - self.alpha)  # [M]
        mass = (self.H * self.mask.unsqueeze(-1)) * one_minus_alpha  # [N,N,M]
        mass = mass.sum(dim=2)  # [N,N]
        pos_pen = self.cfg.pos_mass_penalty * F.relu(mass).mean()
        
        # Refractoriness: penalize positive self-mass
        self_mass = mass.diag()  # [N]
        pos_self = self.cfg.self_pos_penalty * F.relu(self_mass).mean()
        
        # Group lasso across bases (edge-wise sparsity)
        group_lasso = 0.0
        if self.cfg.l1_group > 0:
            group = torch.sqrt(1e-8 + (self.H**2).sum(dim=2))  # [N,N]
            group_lasso = self.cfg.l1_group * (group * self.penalty_w[:,:,0]).mean()
        
        total_loss = nll + l1 + l2 + pos_pen + pos_self + group_lasso
        
        # Debug guard
        if torch.isnan(total_loss) or total_loss < -0.1:
            print(f"Debug: nll={nll:.6f}, l1={l1:.6f}, l2={l2:.6f}, pos_pen={pos_pen:.6f}, pos_self={pos_self:.6f}, group={group_lasso:.6f}")
            print(f"Debug: lam_mean={lam.mean():.6f}, lam_max={lam.max():.6f}")
            print(f"Debug: x_t_sum={x_t.sum():.0f}, p_mean={p.mean():.6f}")

        if return_per_neuron:
            return total_loss, per_neuron_nll
        return total_loss

    @torch.no_grad()
    def eval_segment(self, X: np.ndarray, start: int, end: int, reset: bool = True) -> Tuple[float, float]:
        """Compute average NLL and average Hz over [start, end) without updating weights."""
        if reset:
            self.reset_state()
        N = self.N; dt = self.cfg.dt
        total_loss = 0.0; total_spikes = 0.0; T = end - start
        
        for t in range(start, end):
            # get current spikes
            x_t = torch.from_numpy(X[:, t].astype(np.float32)).to(self.z.device)
            
            # update traces with delay
            delayed = self.inj_buf[self.inj_head]
            self.z.mul_(self.alpha)
            self.z.add_((1.0 - self.alpha) * delayed[:, None])
            self.inj_buf[self.inj_head] = x_t
            self.inj_head = (self.inj_head + 1) % self.delay_bins

            # intensity
            loglam, lam = self.forward_intensity()
            # Exact Bernoulli NLL (consistent with training)
            p = 1.0 - torch.exp(-lam * dt)
            eps = 1e-12
            nll = -(x_t * torch.log(p + eps) + (1 - x_t) * torch.log(1 - p + eps)).mean()
            total_loss += float(nll.item())
            total_spikes += float(x_t.sum().item())

        avg_nll = total_loss / max(1, T)
        avg_hz = (total_spikes / N) / (T * dt)
        return avg_nll, avg_hz

    @torch.no_grad()
    def eval_per_neuron_loss(self, X: np.ndarray, start: int, end: int, reset: bool = True) -> Dict:
        """Compute per-neuron loss statistics over [start, end).
        
        Returns:
            Dictionary with per-neuron average NLL and firing rates
        """
        if reset:
            self.reset_state()
        N = self.N; dt = self.cfg.dt
        T = end - start
        
        # Accumulators for per-neuron metrics
        per_neuron_nll = np.zeros(N)
        per_neuron_spikes = np.zeros(N)
        
        for t in range(start, end):
            # get current spikes
            x_t = torch.from_numpy(X[:, t].astype(np.float32)).to(self.z.device)
            
            # update traces with delay
            delayed = self.inj_buf[self.inj_head]
            self.z.mul_(self.alpha)
            self.z.add_((1.0 - self.alpha) * delayed[:, None])
            self.inj_buf[self.inj_head] = x_t
            self.inj_head = (self.inj_head + 1) % self.delay_bins

            # intensity
            loglam, lam = self.forward_intensity()
            # Exact Bernoulli NLL per neuron
            p = 1.0 - torch.exp(-lam * dt)
            eps = 1e-12
            nll_per_neuron = -(x_t * torch.log(p + eps) + (1 - x_t) * torch.log(1 - p + eps))
            
            per_neuron_nll += nll_per_neuron.cpu().numpy()
            per_neuron_spikes += x_t.cpu().numpy()
        
        # Average over time
        results = {
            'per_neuron_avg_nll': (per_neuron_nll / T).tolist(),
            'per_neuron_avg_hz': (per_neuron_spikes / (T * dt)).tolist(),
            'per_neuron_total_spikes': per_neuron_spikes.astype(int).tolist()
        }
        
        return results

    @torch.no_grad()
    def generate_segment(self, T: int, priming: np.ndarray | None = None, reset: bool = True) -> np.ndarray:
        """Simulate a spike matrix [N, T] given current parameters. If priming provided, warm up traces."""
        if reset:
            self.reset_state()
        N = self.N; dt = self.cfg.dt
        if priming is not None and priming.size > 0:
            for t in range(priming.shape[1]):
                x_t = torch.from_numpy(priming[:, t].astype(np.float32)).to(self.z.device)
                delayed = self.inj_buf[self.inj_head]
                self.z.mul_(self.alpha)
                self.z.add_((1.0 - self.alpha) * delayed[:, None])
                self.inj_buf[self.inj_head] = x_t
                self.inj_head = (self.inj_head + 1) % self.delay_bins
                
        Y = np.zeros((N, T), dtype=np.uint8)
        for t in range(T):
            loglam, lam = self.forward_intensity()
            # small-bin Bernoulli approx: p = 1 - exp(-lam*dt)
            p = 1.0 - torch.exp(-lam * dt)
            x_t = torch.bernoulli(p.clamp(0.0, 1.0)).float()
            
            # update traces for next step with delay
            delayed = self.inj_buf[self.inj_head]
            self.z.mul_(self.alpha)
            self.z.add_((1.0 - self.alpha) * delayed[:, None])
            self.inj_buf[self.inj_head] = x_t
            self.inj_head = (self.inj_head + 1) % self.delay_bins
            
            Y[:, t] = x_t.cpu().numpy().astype(np.uint8)
        return Y


# -------------------------------
# Helpers: masks, distances, kNN
# -------------------------------

def pairwise_dist(xy: np.ndarray) -> np.ndarray:
    # xy: [N,2]
    diffs = xy[:, None, :] - xy[None, :, :]
    D = np.sqrt(np.sum(diffs * diffs, axis=2))
    return D


def knn_incoming_mask(D: np.ndarray, k: int) -> np.ndarray:
    """Return boolean mask [N_post, N_pre] allowing only k nearest presyn for each postsyn.
    Includes self by default (you can remove diag if desired)."""
    N = D.shape[0]
    mask = np.zeros((N, N), dtype=bool)
    # smaller distance => closer neighbor
    idx_sorted = np.argsort(D, axis=1)  # for each post (row), pres sorted by distance
    k_eff = min(k, N)
    for i in range(N):
        pres = idx_sorted[i, :k_eff]
        mask[i, pres] = True
    return mask


# -------------------------------
# Plotting
# -------------------------------

def quick_plots(outdir: str, X: np.ndarray, Xgen: np.ndarray, split_idx: Tuple[int,int,int,int],
                metrics_dict: Dict, dt: float = 0.00005):
    os.makedirs(outdir, exist_ok=True)
    t0, t1, v0, v1 = split_idx  # train [0:t1], val [t1:v1], test [v1:T]
    
    # Plot ALL units over ALL test time (not just subset)
    T_test = X.shape[1] - v1  # test duration in bins
    print(f'  Creating raster plots for ALL {X.shape[0]} units over full test segment ({T_test} bins = {T_test*dt:.1f}s)')
    
    # Real raster - ALL units, ALL test time
    plt.figure(figsize=(16,8))
    sub_real = X[:, v1:]  # ALL units, ALL test time
    i, j = np.nonzero(sub_real)
    plt.scatter(j * dt, i, s=0.1, alpha=0.7)
    plt.title(f'Real raster - ALL {X.shape[0]} units, full test segment ({T_test*dt:.1f}s)')
    plt.xlabel('Time (s)')
    plt.ylabel('Unit index')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'raster_real.png'), dpi=150)
    plt.close()
    print(f'    Real: {len(i):,} spikes plotted')

    # Generated raster - ALL units, ALL generated time
    plt.figure(figsize=(16,8))
    sub_gen = Xgen  # Already the full generated matrix
    i, j = np.nonzero(sub_gen)
    plt.scatter(j * dt, i, s=0.1, alpha=0.7)
    plt.title(f'Generated raster - ALL {Xgen.shape[0]} units, full generated segment ({Xgen.shape[1]*dt:.1f}s)')
    plt.xlabel('Time (s)')
    plt.ylabel('Unit index')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'raster_gen.png'), dpi=150)
    plt.close()
    print(f'    Generated: {len(i):,} spikes plotted')

    # Firing rates comparison over test - ALL units
    fr_real = X[:, v1:].mean(axis=1) / dt  # spikes/bin to Hz for ALL units
    fr_gen  = Xgen.mean(axis=1) / dt       # ALL units
    lim = max(1e-6, float(max(fr_real.max(), fr_gen.max())))
    plt.figure(figsize=(8,6))
    plt.plot([0, lim], [0, lim], 'k--', linewidth=1, label='Perfect match')
    plt.scatter(fr_real, fr_gen, s=12, alpha=0.7)
    plt.xlabel('Real firing rate (Hz)')
    plt.ylabel('Generated firing rate (Hz)')
    plt.title(f'Per-unit firing rate comparison (ALL {X.shape[0]} units)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'fr_scatter.png'), dpi=160)
    plt.close()
    
    # Print statistics
    print(f'    Real firing rates: {fr_real.mean():.2f} ± {fr_real.std():.2f} Hz')
    print(f'    Generated firing rates: {fr_gen.mean():.2f} ± {fr_gen.std():.2f} Hz')
    print(f'    Correlation: {np.corrcoef(fr_real, fr_gen)[0,1]:.3f}')
    
    # --- NEW: Per-neuron reconstruction quality plots ---
    
    # 1. MAE distribution across neurons
    mae_values = [metrics_dict['per_neuron'][i]['mae'] for i in range(len(metrics_dict['per_neuron']))]
    plt.figure(figsize=(10, 5))
    plt.hist(mae_values, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(mae_values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(mae_values):.4f}')
    plt.axvline(np.median(mae_values), color='blue', linestyle='--', linewidth=2, label=f'Median: {np.median(mae_values):.4f}')
    plt.xlabel('Mean Absolute Error (spikes/bin)')
    plt.ylabel('Number of neurons')
    plt.title('Per-Neuron Reconstruction MAE Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'mae_distribution.png'), dpi=150)
    plt.close()
    
    # 2. Correlation distribution
    corr_values = [metrics_dict['per_neuron'][i]['correlation'] for i in range(len(metrics_dict['per_neuron']))]
    plt.figure(figsize=(10, 5))
    plt.hist(corr_values, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(corr_values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(corr_values):.3f}')
    plt.axvline(np.median(corr_values), color='blue', linestyle='--', linewidth=2, label=f'Median: {np.median(corr_values):.3f}')
    plt.xlabel('Correlation coefficient')
    plt.ylabel('Number of neurons')
    plt.title('Per-Neuron Spike Train Correlation Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'correlation_distribution.png'), dpi=150)
    plt.close()
    
    # 3. Per-neuron firing rate error
    rate_errors = [metrics_dict['per_neuron'][i]['rate_error_pct'] for i in range(len(metrics_dict['per_neuron']))]
    plt.figure(figsize=(10, 5))
    plt.hist(rate_errors, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(rate_errors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rate_errors):.1f}%')
    plt.axvline(np.median(rate_errors), color='blue', linestyle='--', linewidth=2, label=f'Median: {np.median(rate_errors):.1f}%')
    plt.xlabel('Firing rate error (%)')
    plt.ylabel('Number of neurons')
    plt.title('Per-Neuron Firing Rate Error Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'rate_error_distribution.png'), dpi=150)
    plt.close()
    
    # 4. Best and worst reconstructed neurons (by correlation)
    N = len(metrics_dict['per_neuron'])
    corr_with_idx = [(i, metrics_dict['per_neuron'][i]['correlation']) for i in range(N)]
    corr_with_idx.sort(key=lambda x: x[1], reverse=True)
    
    # Plot top 5 best and worst
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    for idx in range(5):
        # Best neurons
        neuron_id, corr = corr_with_idx[idx]
        ax = axes[0, idx]
        t_axis = np.arange(Xgen.shape[1]) * dt
        # Smooth for visualization
        from scipy.ndimage import gaussian_filter1d
        sigma = max(1, int(0.050 / dt))  # 50ms smoothing
        real_smooth = gaussian_filter1d(X[neuron_id, v1:].astype(float), sigma=sigma)
        gen_smooth = gaussian_filter1d(Xgen[neuron_id, :].astype(float), sigma=sigma)
        ax.plot(t_axis, real_smooth, 'b-', alpha=0.6, linewidth=1, label='Real')
        ax.plot(t_axis, gen_smooth, 'r-', alpha=0.6, linewidth=1, label='Generated')
        ax.set_title(f'Best #{idx+1}: Neuron {neuron_id}\nCorr={corr:.3f}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Smoothed rate')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Worst neurons
        neuron_id, corr = corr_with_idx[-(idx+1)]
        ax = axes[1, idx]
        real_smooth = gaussian_filter1d(X[neuron_id, v1:].astype(float), sigma=sigma)
        gen_smooth = gaussian_filter1d(Xgen[neuron_id, :].astype(float), sigma=sigma)
        ax.plot(t_axis, real_smooth, 'b-', alpha=0.6, linewidth=1, label='Real')
        ax.plot(t_axis, gen_smooth, 'r-', alpha=0.6, linewidth=1, label='Generated')
        ax.set_title(f'Worst #{idx+1}: Neuron {neuron_id}\nCorr={corr:.3f}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Smoothed rate')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'best_worst_neurons.png'), dpi=150)
    plt.close()
    
    # 5. Scatter plot: MAE vs Firing Rate
    rates_real = [metrics_dict['per_neuron'][i]['rate_real_hz'] for i in range(N)]
    plt.figure(figsize=(8, 6))
    plt.scatter(rates_real, mae_values, alpha=0.6, s=20)
    plt.xlabel('Real firing rate (Hz)')
    plt.ylabel('MAE (spikes/bin)')
    plt.title('Reconstruction Error vs Firing Rate')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'mae_vs_rate.png'), dpi=150)
    plt.close()
    
    print(f'  Per-neuron reconstruction metrics saved to {outdir}/')


# -------------------------------
# Main
# -------------------------------

def main(acqm_zip_path: str,
         outdir: str = '/Users/sebas/Desktop/Ephys/trial-1/glm',
         cfg: GLMConfig = GLMConfig()):
    os.makedirs(outdir, exist_ok=True)
    print('[*] Loading ACQM zip ...')
    spikes_dict, neuron_data, fs = load_acqm_zip(acqm_zip_path)

    # stable unit ordering: intersect keys present in both dicts
    unit_ids = sorted(list(set(spikes_dict.keys()) & set(neuron_data.keys())))
    N = len(unit_ids)
    print(f'[*] Units: {N}, fs={fs:.1f} Hz')

    # positions & distances
    XY = build_positions(neuron_data, unit_ids)   # [N,2] in microns
    D = pairwise_dist(XY)

    # kNN incoming mask (for each postsyn neuron pick k nearest presyn)
    if cfg.knn_k and cfg.knn_k > 0:
        mask_np = knn_incoming_mask(D, cfg.knn_k)
    else:
        mask_np = np.ones((N, N), dtype=bool)

    # binning
    dt = cfg.dt
    print('[*] Binning spikes ...')
    X, T = spikes_to_binned_matrix(spikes_dict, fs, dt, unit_ids)
    print(f'[*] Binned matrix shape: {X.shape} (N,T)')

    # splits (contiguous): 60/20/20
    t_train = int(0.6 * T)
    t_val   = t_train + int(0.2 * T)
    split_idx = (0, t_train, t_train, t_val)
    print(f'[*] Splits: train [0:{t_train}), val [{t_train}:{t_val}), test [{t_val}:{T})')

    # device
    dev = torch.device('cuda' if (cfg.device == 'cuda' or (cfg.device=='auto' and torch.cuda.is_available())) else 'cpu')
    print(f'[*] Device: {dev}')

    # Plot raster before training to verify data
    print('[*] Creating pre-training raster plot...')
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(16, 10))
    
    # Plot ALL units and ALL time
    print(f'  Plotting raster for all {N} units over {T*dt:.1f} seconds...')
    sub = X
    i, j = np.nonzero(sub)
    plt.scatter(j * dt, i, s=0.1, alpha=0.6)
    plt.title(f'Complete spike raster ({N} units, {T*dt:.1f}s total)')
    plt.xlabel('Time (s)')
    plt.ylabel('Unit index')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'raster_raw_data.png'), dpi=150)
    plt.close()
    print(f'  Saved complete raster plot with {len(i):,} spikes')
    
    # Also plot firing rate distribution
    fr_per_unit = X.mean(axis=1) / dt  # spikes per second
    plt.figure(figsize=(8, 4))
    plt.hist(fr_per_unit, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Firing rate (Hz)')
    plt.ylabel('Number of units')
    plt.title(f'Firing rate distribution (N={N} units)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'firing_rate_distribution.png'), dpi=150)
    plt.close()
    
    print(f'  Total spikes: {X.sum():,}')
    print(f'  Mean firing rate: {fr_per_unit.mean():.2f} ± {fr_per_unit.std():.2f} Hz')
    print(f'  Sparsity: {1 - X.mean():.6f} (fraction of zeros)')
    print(f'  Recording duration: {T*dt:.1f} seconds ({T*dt/60:.1f} minutes)')
    print(f'  Data density: {X.sum()/(N*T)*100:.4f}% (spikes per bin)')

    # model
    M = len(cfg.taus)
    model = OnlineHawkesGLM(N=N, M=M,
                            mask_in=torch.from_numpy(mask_np),
                            dist_mat=torch.from_numpy(D.astype(np.float32)),
                            cfg=cfg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # ---------------- train ----------------
    print('[*] Training (online) ...')
    model.train(); model.reset_state()
    batch_bins = cfg.batch_bins
    max_bins = cfg.train_bins if cfg.train_bins is not None else t_train
    running = 0.0; count = 0
    
    # Track per-neuron losses during training
    per_neuron_loss_history = {i: [] for i in range(N)}
    batch_per_neuron_loss = np.zeros(N)

    for t in range(1, max_bins):
        x_t = torch.from_numpy(X[:, t].astype(np.float32)).to(dev)
        opt.zero_grad(set_to_none=True)
        loss, per_neuron_nll = model.step_online(x_t, return_per_neuron=True)
        if torch.isnan(loss):
            print(f'  t={t:6d}/{t_train}  Loss is NaN, stopping training.')
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        # Hard constraint: project self-coupling to non-positive
        with torch.no_grad():
            Hself = model.H.diagonal(dim1=0, dim2=1)  # [N,M]
            Hself.clamp_(max=0.0)
        running += float(loss.item())
        batch_per_neuron_loss += per_neuron_nll.detach().cpu().numpy()
        count += 1
        
        if (t % batch_bins) == 0:
            with torch.no_grad():
                loglam_temp, _ = model.forward_intensity()
                lo = (loglam_temp <= model.cfg.loglam_min).float().mean().item()
                hi = (loglam_temp >= model.cfg.loglam_max).float().mean().item()
            avg_batch_loss = running / max(1, count)
            avg_per_neuron = batch_per_neuron_loss / max(1, count)
            
            # Store per-neuron losses
            for i in range(N):
                per_neuron_loss_history[i].append(float(avg_per_neuron[i]))
            
            # Show worst 3 neurons (highest loss)
            worst_idx = np.argsort(avg_per_neuron)[-3:][::-1]
            worst_str = ', '.join([f'N{i}:{avg_per_neuron[i]:.3f}' for i in worst_idx])
            
            print(f'  t={t:6d}/{t_train}  avg_loss={avg_batch_loss:.4f}  clamp_lo={lo:.3f}, clamp_hi={hi:.3f}  worst_losses=[{worst_str}]')
            running = 0.0
            count = 0
            batch_per_neuron_loss = np.zeros(N)

    # ---------------- post-hoc rate calibration ----------------
    print('[*] Post-hoc rate calibration ...')
    with torch.no_grad():
        model.reset_state()
        lam_sum = torch.zeros(model.N, device=dev)
        for t in range(t_train):
            # get current spikes
            x_t = torch.from_numpy(X[:, t].astype(np.float32)).to(dev)
            # update traces with delay (same logic as eval)
            delayed = model.inj_buf[model.inj_head]
            model.z.mul_(model.alpha)
            model.z.add_((1.0 - model.alpha) * delayed[:, None])
            model.inj_buf[model.inj_head] = x_t
            model.inj_head = (model.inj_head + 1) % model.delay_bins
            loglam, lam = model.forward_intensity()
            lam_sum += lam
        lam_mean = lam_sum / t_train
        real_hz = torch.from_numpy(X[:, :t_train].mean(axis=1)/cfg.dt).to(dev)
        adj = (real_hz.clamp(1e-6).log() - lam_mean.clamp(1e-6).log())
        model.b.add_(adj)  # one-shot per-neuron bias correction
        print(f'  Applied bias corrections: mean={adj.mean():.4f}, std={adj.std():.4f}')

    # ---------------- eval val ----------------
    print('[*] Validation ...')
    model.eval()
    val_nll, val_hz = model.eval_segment(X, start=t_train, end=t_val, reset=True)
    print(f'  Val avg NLL/bin={val_nll:.6f}, Val avg Hz/unit={val_hz:.3f}')
    
    # Per-neuron validation metrics
    val_per_neuron = model.eval_per_neuron_loss(X, start=t_train, end=t_val, reset=True)
    print(f'  Val per-neuron NLL: mean={np.mean(val_per_neuron["per_neuron_avg_nll"]):.4f}, std={np.std(val_per_neuron["per_neuron_avg_nll"]):.4f}')

    # ---------------- eval test + generate ----------------
    print('[*] Test evaluation ...')
    test_nll, test_hz = model.eval_segment(X, start=t_val, end=T, reset=True)
    print(f'  Test avg NLL/bin={test_nll:.6f}, Test avg Hz/unit={test_hz:.3f}')
    
    # Per-neuron test metrics
    test_per_neuron = model.eval_per_neuron_loss(X, start=t_val, end=T, reset=True)
    print(f'  Test per-neuron NLL: mean={np.mean(test_per_neuron["per_neuron_avg_nll"]):.4f}, std={np.std(test_per_neuron["per_neuron_avg_nll"]):.4f}')

    print('[*] Generating simulated test segment ...')
    # prime with 1 s of real test to initialize traces
    prime_bins = int(1.0 / dt)
    prime = X[:, t_val : min(T, t_val + prime_bins)]
    Y = model.generate_segment(T=(T - t_val), priming=prime, reset=True)
    
    # ---------------- Per-neuron reconstruction metrics ----------------
    print('[*] Computing per-neuron reconstruction metrics ...')
    reconstruction_metrics = compute_per_neuron_metrics(X[:, t_val:], Y, dt)
    
    print(f'  Overall MAE: {reconstruction_metrics["summary"]["mae_mean"]:.6f} ± {reconstruction_metrics["summary"]["mae_std"]:.6f}')
    print(f'  Overall correlation: {reconstruction_metrics["summary"]["correlation_mean"]:.3f} ± {reconstruction_metrics["summary"]["correlation_std"]:.3f}')
    print(f'  Total spike count error: {reconstruction_metrics["summary"]["spike_count_error_pct"]:.2f}%')
    
    # Find best and worst reconstructed neurons
    neuron_corrs = [(i, reconstruction_metrics['per_neuron'][i]['correlation']) 
                    for i in range(N)]
    neuron_corrs.sort(key=lambda x: x[1], reverse=True)
    
    print(f'  Best 5 neurons (by correlation):')
    for i, (neuron_id, corr) in enumerate(neuron_corrs[:5]):
        mae = reconstruction_metrics['per_neuron'][neuron_id]['mae']
        rate_real = reconstruction_metrics['per_neuron'][neuron_id]['rate_real_hz']
        rate_gen = reconstruction_metrics['per_neuron'][neuron_id]['rate_gen_hz']
        print(f'    {i+1}. Neuron {neuron_id}: corr={corr:.3f}, MAE={mae:.6f}, rate={rate_real:.1f}→{rate_gen:.1f} Hz')
    
    print(f'  Worst 5 neurons (by correlation):')
    for i, (neuron_id, corr) in enumerate(neuron_corrs[-5:][::-1]):
        mae = reconstruction_metrics['per_neuron'][neuron_id]['mae']
        rate_real = reconstruction_metrics['per_neuron'][neuron_id]['rate_real_hz']
        rate_gen = reconstruction_metrics['per_neuron'][neuron_id]['rate_gen_hz']
        print(f'    {i+1}. Neuron {neuron_id}: corr={corr:.3f}, MAE={mae:.6f}, rate={rate_real:.1f}→{rate_gen:.1f} Hz')

    # ---------------- save & plots ----------------
    torch.save({'b': model.b.detach().cpu(),
                'H': (model.H * model.mask.unsqueeze(-1)).detach().cpu(),
                'cfg': cfg.__dict__,
                'unit_ids': unit_ids,
                'taus': cfg.taus,
                'dt': cfg.dt,
                'XY': XY,
                'mask': mask_np,
                'per_neuron_loss_history': per_neuron_loss_history},
               os.path.join(outdir, 'glm_model.pt'))

    # quick plots with reconstruction metrics
    quick_plots(outdir, X, Y, split_idx, reconstruction_metrics, dt=cfg.dt)

    # dump metrics
    metrics = {
        'N': N,
        'T': T,
        'dt': dt,
        'train_bins': t_train,
        'val_bins': (t_val - t_train),
        'test_bins': (T - t_val),
        'val_nll_per_bin': val_nll,
        'test_nll_per_bin': test_nll,
        'val_avg_hz_per_unit': val_hz,
        'test_avg_hz_per_unit': test_hz,
        'val_per_neuron': val_per_neuron,
        'test_per_neuron': test_per_neuron,
        'reconstruction_metrics': reconstruction_metrics,
    }
    with open(os.path.join(outdir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save per-neuron loss history plot
    if per_neuron_loss_history and len(per_neuron_loss_history[0]) > 0:
        plt.figure(figsize=(12, 6))
        # Plot a sample of neurons (max 20 to avoid clutter)
        neurons_to_plot = min(20, N)
        sample_idx = np.linspace(0, N-1, neurons_to_plot, dtype=int)
        for i in sample_idx:
            plt.plot(per_neuron_loss_history[i], alpha=0.6, linewidth=1, label=f'N{i}' if neurons_to_plot <= 10 else None)
        plt.xlabel('Training batch')
        plt.ylabel('Per-neuron NLL')
        plt.title(f'Per-Neuron Loss Evolution During Training ({neurons_to_plot} neurons shown)')
        if neurons_to_plot <= 10:
            plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'per_neuron_loss_evolution.png'), dpi=150)
        plt.close()
    
    print('[*] Done. Outputs in', outdir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Online Hawkes-GLM Digital Twin with Per-Neuron Reconstruction Metrics')
    parser.add_argument('--acqm', type=str, 
                        default='/Users/sebas/Desktop/Ephys/trial-1/Trace_20251018_20_09_51-LC-kolf2.2-day51-shank3_acqm.zip',
                        help='Path to ACQM zip with nested qm.npz')
    parser.add_argument('--outdir', type=str, default='/Users/sebas/Desktop/Ephys/trial-1/glm')
    parser.add_argument('--dt', type=float, default=0.002)
    parser.add_argument('--taus', type=float, nargs='+', default=[0.002,0.004,0.008,0.016,0.032,0.064,0.128])
    parser.add_argument('--delay_ms', type=float, default=2.0)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--l1', type=float, default=1e-3)
    parser.add_argument('--l2', type=float, default=0.0)
    parser.add_argument('--l1_group', type=float, default=5e-4)
    parser.add_argument('--knn_k', type=int, default=20)
    parser.add_argument('--ell', type=float, default=400.0)
    parser.add_argument('--batch_bins', type=int, default=250)
    parser.add_argument('--train_bins', type=int, default=-1, help='Limit train bins (for quick tests)')
    parser.add_argument('--device', type=str, default='auto', choices=['auto','cpu','cuda'])
    parser.add_argument('--loglam_min', type=float, default=-10.0)
    parser.add_argument('--loglam_max', type=float, default=7.0)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--pos_mass_penalty', type=float, default=0.0)
    parser.add_argument('--self_pos_penalty', type=float, default=5e-3)
    args = parser.parse_args()

    cfg = GLMConfig(dt=args.dt,
                    taus=tuple(args.taus),
                    delay_ms=args.delay_ms,
                    lr=args.lr,
                    l1=args.l1,
                    l2=args.l2,
                    l1_group=args.l1_group,
                    device=args.device,
                    knn_k=args.knn_k,
                    spatial_ell=args.ell,
                    batch_bins=args.batch_bins,
                    train_bins=None if (args.train_bins is None or args.train_bins < 0) else args.train_bins,
                    loglam_min=args.loglam_min,
                    loglam_max=args.loglam_max,
                    grad_clip=args.grad_clip,
                    pos_mass_penalty=args.pos_mass_penalty,
                    self_pos_penalty=args.self_pos_penalty)

    main(args.acqm, outdir=args.outdir, cfg=cfg)
