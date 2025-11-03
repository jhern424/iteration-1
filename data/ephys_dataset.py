"""
Electrophysiology Spike Dataset for PyTorch

This dataset loads spike train data from .npz files and creates sliding windows
for spike forecasting tasks. Each sample represents a single neuron's activity
over time.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import zipfile
import os
import tempfile
from typing import List, Tuple, Optional, Dict
import warnings


class EphysDataset(Dataset):
    """
    PyTorch Dataset for electrophysiology spike forecasting.

    Creates per-neuron sliding windows for binary spike prediction.

    Args:
        zip_path: Path to the .zip file containing .npz spike data
        history_ms: Length of input history window in milliseconds (default: 200ms)
        forecast_ms: Length of future window to predict in milliseconds (default: 100ms)
        bin_size_ms: Size of time bins in milliseconds (default: 5ms)
        stride_ms: Stride for sliding window in milliseconds (default: 50ms)
        min_spikes: Minimum number of spikes a neuron must have to be included (default: 10)
        max_samples_per_neuron: Maximum number of windows per neuron (None = all)
    """

    def __init__(
        self,
        zip_path: str,
        history_ms: float = 200.0,
        forecast_ms: float = 100.0,
        bin_size_ms: float = 5.0,
        stride_ms: float = 50.0,
        min_spikes: int = 10,
        max_samples_per_neuron: Optional[int] = None,
        fs: int = 20000,  # Default sampling rate
    ):
        self.zip_path = zip_path
        self.history_ms = history_ms
        self.forecast_ms = forecast_ms
        self.bin_size_ms = bin_size_ms
        self.stride_ms = stride_ms
        self.min_spikes = min_spikes
        self.max_samples_per_neuron = max_samples_per_neuron
        self.fs = fs

        # Calculate number of bins
        self.n_history_bins = int(history_ms / bin_size_ms)
        self.n_forecast_bins = int(forecast_ms / bin_size_ms)
        self.n_total_bins = self.n_history_bins + self.n_forecast_bins

        # Load spike data
        self.spike_trains_ms, self.recording_length_ms = self._load_spike_data()

        # Generate sliding windows
        self.windows = self._generate_windows()

        print(f"Dataset created with {len(self.windows)} windows from {len(self.spike_trains_ms)} neurons")
        print(f"History: {history_ms}ms ({self.n_history_bins} bins), Forecast: {forecast_ms}ms ({self.n_forecast_bins} bins)")
        print(f"Recording length: {self.recording_length_ms:.2f}ms ({self.recording_length_ms/1000:.2f}s)")

    def _load_spike_data(self) -> Tuple[List[np.ndarray], float]:
        """Load spike trains from zip file."""
        print(f"Loading spike data from {self.zip_path}...")

        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            # Find the .npz file
            npz_files = [f for f in zip_ref.namelist() if f.endswith('.npz')]
            if not npz_files:
                raise FileNotFoundError(f"No .npz file found in {self.zip_path}")

            npz_file = npz_files[0]

            # Extract to temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                zip_ref.extract(npz_file, temp_dir)
                extracted_path = os.path.join(temp_dir, npz_file)

                # Load the .npz file
                npz_data = np.load(extracted_path, allow_pickle=True)

                # Extract spike times
                train = npz_data['train'].item()  # Dictionary of neuron_id -> spike times
                if 'fs' in npz_data.keys():
                    self.fs = npz_data['fs'].item()

                # Convert to list and sort by neuron ID
                neuron_ids = sorted(train.keys())
                spike_trains = [train[nid] for nid in neuron_ids]

                # Convert from samples to milliseconds
                spike_trains_ms = []
                for st in spike_trains:
                    if len(st) > 0:
                        spike_times_ms = (np.array(st) / self.fs) * 1000
                        spike_trains_ms.append(spike_times_ms)
                    else:
                        spike_trains_ms.append(np.array([]))

                # Calculate recording length
                if len(spike_trains_ms) > 0:
                    max_time = max(max(st) if len(st) > 0 else 0 for st in spike_trains_ms)
                    recording_length_ms = max_time
                else:
                    recording_length_ms = 0

        # Filter neurons with sufficient spikes
        filtered_trains = [st for st in spike_trains_ms if len(st) >= self.min_spikes]
        print(f"Loaded {len(spike_trains_ms)} neurons, kept {len(filtered_trains)} with >={self.min_spikes} spikes")

        return filtered_trains, recording_length_ms

    def _spike_times_to_binary_vector(
        self,
        spike_times: np.ndarray,
        start_ms: float,
        end_ms: float
    ) -> np.ndarray:
        """
        Convert spike times to binary vector (binned).

        Args:
            spike_times: Array of spike times in milliseconds
            start_ms: Start of time window
            end_ms: End of time window

        Returns:
            Binary array of shape (n_bins,) where 1 indicates spike presence
        """
        n_bins = int((end_ms - start_ms) / self.bin_size_ms)
        binary_vector = np.zeros(n_bins, dtype=np.float32)

        # Find spikes in this time window
        window_spikes = spike_times[(spike_times >= start_ms) & (spike_times < end_ms)]

        # Bin the spikes
        if len(window_spikes) > 0:
            bin_indices = ((window_spikes - start_ms) / self.bin_size_ms).astype(int)
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)
            binary_vector[bin_indices] = 1.0

        return binary_vector

    def _generate_windows(self) -> List[Dict]:
        """
        Generate sliding windows for all neurons.

        Returns:
            List of window dictionaries containing neuron_idx, start_ms, and end_ms
        """
        windows = []

        for neuron_idx, spike_train in enumerate(self.spike_trains_ms):
            if len(spike_train) == 0:
                continue

            # Determine valid time range for windows
            # We need enough time for history + forecast
            window_duration_ms = self.history_ms + self.forecast_ms
            max_start_time = self.recording_length_ms - window_duration_ms

            if max_start_time <= 0:
                warnings.warn(f"Neuron {neuron_idx}: Recording too short for window generation")
                continue

            # Generate sliding windows
            start_times = np.arange(0, max_start_time, self.stride_ms)

            neuron_windows = []
            for start_ms in start_times:
                end_ms = start_ms + window_duration_ms

                window = {
                    'neuron_idx': neuron_idx,
                    'start_ms': start_ms,
                    'end_ms': end_ms,
                }
                neuron_windows.append(window)

            # Limit samples per neuron if specified
            if self.max_samples_per_neuron is not None:
                if len(neuron_windows) > self.max_samples_per_neuron:
                    # Randomly sample
                    indices = np.random.choice(
                        len(neuron_windows),
                        self.max_samples_per_neuron,
                        replace=False
                    )
                    neuron_windows = [neuron_windows[i] for i in sorted(indices)]

            windows.extend(neuron_windows)

        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single training sample.

        Returns:
            history: Tensor of shape (n_history_bins,) - binary input
            target: Tensor of shape (n_forecast_bins,) - binary target
        """
        window = self.windows[idx]
        neuron_idx = window['neuron_idx']
        start_ms = window['start_ms']

        spike_train = self.spike_trains_ms[neuron_idx]

        # Extract history window
        history_start = start_ms
        history_end = start_ms + self.history_ms
        history = self._spike_times_to_binary_vector(
            spike_train, history_start, history_end
        )

        # Extract forecast window
        forecast_start = history_end
        forecast_end = forecast_start + self.forecast_ms
        target = self._spike_times_to_binary_vector(
            spike_train, forecast_start, forecast_end
        )

        return torch.from_numpy(history), torch.from_numpy(target)

    def get_spike_statistics(self) -> Dict:
        """Get statistics about spike rates and patterns."""
        total_spikes = sum(len(st) for st in self.spike_trains_ms)

        firing_rates = []
        for st in self.spike_trains_ms:
            if len(st) > 0:
                rate = len(st) / (self.recording_length_ms / 1000)  # spikes per second
                firing_rates.append(rate)

        return {
            'n_neurons': len(self.spike_trains_ms),
            'total_spikes': total_spikes,
            'mean_firing_rate': np.mean(firing_rates) if firing_rates else 0,
            'std_firing_rate': np.std(firing_rates) if firing_rates else 0,
            'min_firing_rate': np.min(firing_rates) if firing_rates else 0,
            'max_firing_rate': np.max(firing_rates) if firing_rates else 0,
        }


def get_class_weights(dataset: EphysDataset, device='cpu') -> torch.Tensor:
    """
    Calculate class weights for handling spike sparsity.

    Args:
        dataset: EphysDataset instance
        device: Device to place weights on

    Returns:
        Tensor of shape (2,) with weights for [no_spike, spike]
    """
    total_bins = 0
    total_spikes = 0

    print("Calculating class weights...")
    for i in range(len(dataset)):
        _, target = dataset[i]
        total_bins += len(target)
        total_spikes += target.sum().item()

    n_no_spikes = total_bins - total_spikes

    # Weight inversely proportional to frequency
    weight_no_spike = 1.0
    weight_spike = n_no_spikes / total_spikes if total_spikes > 0 else 1.0

    weights = torch.tensor([weight_no_spike, weight_spike], dtype=torch.float32).to(device)

    print(f"Class weights - No spike: {weight_no_spike:.4f}, Spike: {weight_spike:.4f}")
    print(f"Spike rate: {total_spikes/total_bins:.4f} ({total_spikes}/{total_bins} bins)")

    return weights


if __name__ == "__main__":
    # Test the dataset
    print("Testing EphysDataset...")

    # Test with shank3 data
    shank3_path = "/Users/sebas/Desktop/Ephys/spikeformer/dta_sebas/Trace_20251018_20_09_51-LC-kolf2.2-day51-shank3_acqm.zip"

    dataset = EphysDataset(
        zip_path=shank3_path,
        history_ms=200.0,
        forecast_ms=100.0,
        bin_size_ms=5.0,
        stride_ms=50.0,
        min_spikes=10,
        max_samples_per_neuron=100,  # Limit for testing
    )

    print(f"\nDataset statistics:")
    stats = dataset.get_spike_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print(f"\nSample data:")
    history, target = dataset[0]
    print(f"  History shape: {history.shape}, Target shape: {target.shape}")
    print(f"  History spikes: {history.sum()}, Target spikes: {target.sum()}")

    # Test DataLoader
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    batch_history, batch_target = next(iter(loader))
    print(f"\nBatch shapes:")
    print(f"  History: {batch_history.shape}, Target: {batch_target.shape}")

    print("\n✓ Dataset test successful!")
