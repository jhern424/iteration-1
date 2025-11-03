# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from braingeneers.analysis import SpikeData
import zipfile
import os
import tempfile

# Set up matplotlib for better plots
plt.style.use('default')
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12

# Load LC recording dataset
lc_file_path = "/Users/sebas/Desktop/Ephys/lc/Trace_20251018_20_09_51-LC-kolf2.2-day51-shank3_acqm.zip"

print("Loading LC recording dataset...")
try:
    # Extract and load the .npz file directly
    with zipfile.ZipFile(lc_file_path, 'r') as zip_ref:
        # Find the .npz file
        npz_files = [f for f in zip_ref.namelist() if f.endswith('.npz')]
        if not npz_files:
            raise FileNotFoundError("No .npz file found in the archive")
        
        npz_file = npz_files[0]
        print(f"Found .npz file: {npz_file}")
        
        # Extract to a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_ref.extract(npz_file, temp_dir)
            extracted_path = os.path.join(temp_dir, npz_file)
            
            # Load the .npz file
            npz_data = np.load(extracted_path, allow_pickle=True)
            
            # Extract the relevant data
            train = npz_data['train'].item()  # Dictionary of spike times per neuron
            neuron_data = npz_data['neuron_data']
            fs = npz_data['fs'].item() if 'fs' in npz_data.keys() else 20000  # Sampling rate
            
            print(f"Number of neurons in train data: {len(train)}")
            print(f"Sampling rate: {fs} Hz")
            
            # Convert train dict to list format expected by SpikeData
            neuron_ids = sorted(train.keys())
            spike_trains = [train[nid] for nid in neuron_ids]
            
            # Check sample data and convert spike times from samples to milliseconds
            if len(spike_trains) > 0 and len(spike_trains[0]) > 0:
                sample_times = spike_trains[0][:5] if len(spike_trains[0]) > 5 else spike_trains[0]
                print(f"Sample spike times from first neuron: {sample_times}")
                
                max_time_samples = max(max(st) if len(st) > 0 else 0 for st in spike_trains)
                recording_length_ms = (max_time_samples / fs) * 1000
                print(f"Estimated recording length: {recording_length_ms:.2f} ms ({recording_length_ms/1000:.2f} seconds)")
                
                # Convert spike times from samples to milliseconds
                spike_trains_ms = []
                for st in spike_trains:
                    if len(st) > 0:
                        spike_times_ms = (np.array(st) / fs) * 1000  # Convert to ms
                        spike_trains_ms.append(spike_times_ms)
                    else:
                        spike_trains_ms.append(np.array([]))
            else:
                spike_trains_ms = spike_trains
            
            # Create SpikeData object
            lc_spike_data = SpikeData(train=spike_trains_ms)
            print("Created SpikeData successfully")
    
    print(f"\nShank3 dataset loaded successfully!")
    print(f"Number of neurons: {lc_spike_data.N}")
    print(f"Recording length: {lc_spike_data.length:.2f} ms ({lc_spike_data.length/1000:.2f} seconds)")
    print(f"Total spikes: {sum(len(train) for train in lc_spike_data.train)}")
    
except Exception as e:
    print(f"Error loading Shank3 dataset: {e}")
    import traceback
    traceback.print_exc()
    lc_spike_data = None

# Create raster plot for the entire recording - SHANK3
if lc_spike_data is not None:
    print("\n" + "="*60)
    print("Creating SHANK3 raster plot for entire recording...")
    print("="*60)
    
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Plot raster
    for neuron_idx, spike_times in enumerate(lc_spike_data.train):
        if len(spike_times) > 0:
            # Convert to seconds for better readability
            spike_times_sec = spike_times / 1000
            ax.scatter(spike_times_sec, [neuron_idx] * len(spike_times), 
                      s=1, c='black', marker='|', linewidths=0.5)
    
    ax.set_xlabel('Time (seconds)', fontsize=14)
    ax.set_ylabel('Neuron Index', fontsize=14)
    ax.set_title('Raster Plot: SHANK3 (kolf2.2, day 51)', fontsize=16)
    ax.set_xlim(0, lc_spike_data.length / 1000)
    ax.set_ylim(-1, lc_spike_data.N)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('/Users/sebas/Desktop/Ephys/lc/raster_plot_full_shank3.png', dpi=300, bbox_inches='tight')
    print(f"Shank3 full raster plot saved to: /Users/sebas/Desktop/Ephys/lc/raster_plot_full_shank3.png")
    plt.show()

# Create raster plot for a specific time window (first 60 seconds) - SHANK3
if lc_spike_data is not None:
    print("\n" + "="*60)
    print("Creating SHANK3 raster plot for time window (0-60s)...")
    print("="*60)
    
    # Define time window (in seconds)
    start_time = 0
    end_time = 60  # First 60 seconds
    
    # Make sure end_time doesn't exceed recording length
    end_time = min(end_time, lc_spike_data.length / 1000)
    
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Plot raster for the specified time window
    for neuron_idx, spike_times in enumerate(lc_spike_data.train):
        if len(spike_times) > 0:
            # Convert to seconds
            spike_times_sec = spike_times / 1000
            # Filter spikes within the time window
            mask = (spike_times_sec >= start_time) & (spike_times_sec <= end_time)
            filtered_spikes = spike_times_sec[mask]
            
            if len(filtered_spikes) > 0:
                ax.scatter(filtered_spikes, [neuron_idx] * len(filtered_spikes), 
                          s=2, c='black', marker='|', linewidths=0.8)
    
    ax.set_xlabel('Time (seconds)', fontsize=14)
    ax.set_ylabel('Neuron Index', fontsize=14)
    ax.set_title(f'Raster Plot: SHANK3 (kolf2.2, day 51) - {start_time}s to {end_time}s', fontsize=16)
    ax.set_xlim(start_time, end_time)
    ax.set_ylim(-1, lc_spike_data.N)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('/Users/sebas/Desktop/Ephys/lc/raster_plot_60s_shank3.png', dpi=300, bbox_inches='tight')
    print(f"Shank3 60s raster plot saved to: /Users/sebas/Desktop/Ephys/lc/raster_plot_60s_shank3.png")
    plt.show()

# Calculate and display firing rate statistics per neuron
if lc_spike_data is not None:
    print("\n" + "="*60)
    print("FIRING RATE STATISTICS PER NEURON")
    print("="*60)
    
    # Calculate firing rate for each neuron (spikes per second)
    recording_duration_sec = lc_spike_data.length / 1000
    firing_rates = []
    
    for neuron_idx, spike_times in enumerate(lc_spike_data.train):
        firing_rate = len(spike_times) / recording_duration_sec
        firing_rates.append(firing_rate)
    
    firing_rates = np.array(firing_rates)
    
    print(f"\nNumber of neurons: {len(firing_rates)}")
    print(f"Recording duration: {recording_duration_sec:.2f} seconds")
    print(f"\nFiring rate statistics:")
    print(f"  Mean: {np.mean(firing_rates):.2f} Hz")
    print(f"  Median: {np.median(firing_rates):.2f} Hz")
    print(f"  Std: {np.std(firing_rates):.2f} Hz")
    print(f"  Min: {np.min(firing_rates):.2f} Hz")
    print(f"  Max: {np.max(firing_rates):.2f} Hz")
    
    # Find most and least active neurons
    most_active_idx = np.argmax(firing_rates)
    least_active_idx = np.argmin(firing_rates)
    
    print(f"\nMost active neuron: Index {most_active_idx} with {firing_rates[most_active_idx]:.2f} Hz ({len(lc_spike_data.train[most_active_idx])} spikes)")
    print(f"Least active neuron: Index {least_active_idx} with {firing_rates[least_active_idx]:.2f} Hz ({len(lc_spike_data.train[least_active_idx])} spikes)")
    
    # Plot histogram of firing rates
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(firing_rates, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(firing_rates), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(firing_rates):.2f} Hz')
    ax.axvline(np.median(firing_rates), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(firing_rates):.2f} Hz')
    ax.set_xlabel('Firing Rate (Hz)', fontsize=14)
    ax.set_ylabel('Number of Neurons', fontsize=14)
    ax.set_title('Distribution of Firing Rates Across Neurons', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/Users/sebas/Desktop/Ephys/lc/firing_rate_distribution_shank3.png', dpi=300, bbox_inches='tight')
    print(f"\nShank3 firing rate distribution saved to: /Users/sebas/Desktop/Ephys/lc/firing_rate_distribution_shank3.png")
    plt.show()

# Population firing rate over time - SHANK3
if lc_spike_data is not None:
    print("\n" + "="*60)
    print("POPULATION FIRING RATE ANALYSIS")
    print("="*60)
    
    bin_size = 1000  # 1 second bins (1000 ms)
    smoothing_window = 10  # 10 bins for smoothing
    
    print("Calculating population firing rate...")
    
    # Calculate population firing rate
    bins, pop_rate = lc_spike_data.population_firing_rate(
        bin_size=bin_size, 
        w=smoothing_window, 
        average=False  # Total population rate, not average
    )
    
    # Convert to time in seconds for better visualization
    time_s = bins[:-1] / 1000
    
    print(f"Mean population firing rate: {np.mean(pop_rate):.2f} spikes/s")
    print(f"Max population firing rate: {np.max(pop_rate):.2f} spikes/s")
    print(f"Min population firing rate: {np.min(pop_rate):.2f} spikes/s")
    
    # Plot population firing rate
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.plot(time_s, pop_rate, 'b-', linewidth=1.5)
    ax.set_xlabel('Time (seconds)', fontsize=14)
    ax.set_ylabel('Population Firing Rate (spikes/s)', fontsize=14)
    ax.set_title('Population Firing Rate Over Time: SHANK3', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, lc_spike_data.length / 1000)
    plt.tight_layout()
    plt.savefig('/Users/sebas/Desktop/Ephys/lc/population_firing_rate_shank3.png', dpi=300, bbox_inches='tight')
    print(f"Shank3 population firing rate plot saved to: /Users/sebas/Desktop/Ephys/lc/population_firing_rate_shank3.png")
    plt.show()

# Burst detection analysis
if lc_spike_data is not None:
    print("\n" + "="*60)
    print("BURST DETECTION ANALYSIS")
    print("="*60)
    
    from scipy.signal import find_peaks
    from scipy.stats import variation
    
    # Calculate recording duration from spike trains
    duration = lc_spike_data.length  # in ms
    
    # Calculate population activity
    bins = np.arange(0, duration + 1, 1)  # bin_size=1 ms
    population_activity = np.zeros(len(bins) - 1)
    
    for spikes in lc_spike_data.train:
        hist, _ = np.histogram(spikes, bins=bins)
        population_activity += hist
    
    # Smooth population activity
    kernel = np.ones(50) / 50  # w=50
    smoothed_activity = np.convolve(population_activity, kernel, mode='same')
    
    # Find burst peaks with stringent criteria
    baseline = np.percentile(smoothed_activity, 25)  # Use 25th percentile as baseline
    peak_thr = baseline + 2.5 * np.std(smoothed_activity)  # Dynamic threshold
    peak_indices, peak_info = find_peaks(smoothed_activity, 
                                       height=peak_thr,
                                       distance=200,  # Minimum distance between peaks
                                       prominence=0.5)  # Prominence requirement
    
    print(f"\nBurst Detection:")
    print(f"  Baseline activity: {baseline:.2f} spikes/ms")
    print(f"  Peak threshold: {peak_thr:.2f} spikes/ms")
    print(f"  Number of peaks detected: {len(peak_indices)}")
    
    # Find burst edges
    burst_edges = []
    min_burst_width = 20  # Minimum burst width in ms
    max_burst_width = 500  # Maximum burst width in ms
    
    if len(peak_indices) > 0:
        edge_thrs = baseline + 0.2 * (peak_info['peak_heights'] - baseline)
        
        for peak_i, edge_thr in zip(peak_indices, edge_thrs):
            # Look for burst start
            burst_start = None
            for i in range(peak_i, max(peak_i - max_burst_width, 0), -1):
                if all(smoothed_activity[max(0, i-5):i+1] < edge_thr):
                    burst_start = i
                    break
            
            # Look for burst end
            burst_end = None
            for i in range(peak_i, min(peak_i + max_burst_width, len(smoothed_activity))):
                if all(smoothed_activity[i:min(i+5, len(smoothed_activity))] < edge_thr):
                    burst_end = i
                    break
                    
            if (burst_start is not None and 
                burst_end is not None and 
                min_burst_width <= (burst_end - burst_start) <= max_burst_width):
                burst_edges.append([burst_start, burst_end])
        
        burst_edges = np.array(burst_edges)
    else:
        burst_edges = np.array([]).reshape(0, 2)
    
    print(f"\nBurst Statistics:")
    print(f"  Total bursts detected: {len(burst_edges)}")
    
    if len(burst_edges) > 0:
        # Calculate burst metrics
        burst_widths = burst_edges[:, 1] - burst_edges[:, 0]
        burst_amplitudes = peak_info['peak_heights']
        recording_duration_sec = duration / 1000
        burst_frequency = len(burst_edges) / recording_duration_sec
        
        print(f"  Mean burst width: {np.mean(burst_widths):.2f} ± {np.std(burst_widths):.2f} ms")
        print(f"  Median burst width: {np.median(burst_widths):.2f} ms")
        print(f"  Mean burst amplitude: {np.mean(burst_amplitudes):.2f} ± {np.std(burst_amplitudes):.2f} spikes/ms")
        print(f"  Burst frequency: {burst_frequency:.4f} bursts/s ({burst_frequency*60:.2f} bursts/min)")
        
        # Calculate interburst intervals
        if len(burst_edges) > 1:
            interburst_intervals = burst_edges[1:, 0] - burst_edges[:-1, 1]
            max_ibi = 1 * 60 * 1000  # 1 minute in ms
            filtered_ibis = interburst_intervals[interburst_intervals < max_ibi]
            
            if len(filtered_ibis) > 0:
                mean_ibi = np.mean(filtered_ibis)
                std_ibi = np.std(filtered_ibis)
                cv_ibi = variation(filtered_ibis) if len(filtered_ibis) > 1 else np.nan
                
                print(f"  Mean interburst interval: {mean_ibi:.2f} ± {std_ibi:.2f} ms ({mean_ibi/1000:.2f}s)")
                print(f"  CV of interburst intervals: {cv_ibi:.3f}")
        
        # Calculate burst involvement (which neurons participate in bursts)
        total_bursts = len(burst_edges)
        total_neurons = lc_spike_data.N
        burst_involvement_matrix = np.zeros((total_neurons, total_bursts), dtype=int)
        
        for i, (burst_start, burst_end) in enumerate(burst_edges):
            for unit in range(total_neurons):
                spikes = lc_spike_data.train[unit]
                spikes_in_burst = np.sum((spikes >= burst_start) & (spikes <= burst_end + 1))
                burst_involvement_matrix[unit, i] = 1 if spikes_in_burst >= 2 else 0
        
        # Burst involvement coefficient (BIC) - fraction of bursts each neuron participates in
        bic_matrix = np.sum(burst_involvement_matrix, axis=1) / total_bursts
        
        # Classify neurons as rigid (participate in ≥90% of bursts) or nonrigid
        rigid_neurons = np.where(bic_matrix >= 0.9)[0]
        nonrigid_neurons = np.where(bic_matrix < 0.9)[0]
        
        print(f"\nNeuron Classification:")
        print(f"  Rigid backbone neurons (BIC ≥ 0.9): {len(rigid_neurons)} ({100*len(rigid_neurons)/total_neurons:.1f}%)")
        print(f"  Non-rigid neurons (BIC < 0.9): {len(nonrigid_neurons)} ({100*len(nonrigid_neurons)/total_neurons:.1f}%)")
        print(f"  Mean BIC: {np.mean(bic_matrix):.3f}")
        print(f"  Median BIC: {np.median(bic_matrix):.3f}")
        
        # Plot burst detection results
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
        
        # Main plot: Population activity with detected bursts
        ax1 = fig.add_subplot(gs[0:2, :])
        time_sec = np.arange(len(smoothed_activity)) / 1000
        ax1.plot(time_sec, smoothed_activity, 'k', alpha=0.7, linewidth=1)
        ax1.axhline(baseline, color='gray', linestyle='--', alpha=0.5, label='Baseline')
        ax1.axhline(peak_thr, color='orange', linestyle='--', alpha=0.5, label='Peak threshold')
        
        for start, end in burst_edges:
            ax1.axvspan(start/1000, end/1000, color='red', alpha=0.2)
        
        ax1.scatter(peak_indices/1000, peak_info['peak_heights'], 
                   color='red', s=50, zorder=5, label='Burst peaks')
        ax1.set_xlabel('Time (seconds)', fontsize=12)
        ax1.set_ylabel('Population Activity (spikes/ms)', fontsize=12)
        ax1.set_title(f'Burst Detection - {len(burst_edges)} bursts detected', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, duration/1000)
        
        # Burst width distribution
        ax2 = fig.add_subplot(gs[2, 0])
        ax2.hist(burst_widths, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax2.axvline(np.mean(burst_widths), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(burst_widths):.1f}ms')
        ax2.set_xlabel('Burst Width (ms)', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Burst Width Distribution', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Burst amplitude distribution
        ax3 = fig.add_subplot(gs[2, 1])
        ax3.hist(burst_amplitudes, bins=30, color='coral', edgecolor='black', alpha=0.7)
        ax3.axvline(np.mean(burst_amplitudes), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(burst_amplitudes):.2f}')
        ax3.set_xlabel('Burst Amplitude (spikes/ms)', fontsize=12)
        ax3.set_ylabel('Count', fontsize=12)
        ax3.set_title('Burst Amplitude Distribution', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Burst involvement coefficient (BIC) distribution
        ax4 = fig.add_subplot(gs[3, 0])
        ax4.hist(bic_matrix, bins=30, color='purple', edgecolor='black', alpha=0.7)
        ax4.axvline(0.9, color='red', linestyle='--', linewidth=2, label='Rigid threshold (0.9)')
        ax4.axvline(np.median(bic_matrix), color='green', linestyle='--', 
                   linewidth=2, label=f'Median: {np.median(bic_matrix):.3f}')
        ax4.set_xlabel('Burst Involvement Coefficient', fontsize=12)
        ax4.set_ylabel('Number of Neurons', fontsize=12)
        ax4.set_title('Neuron Burst Involvement', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Interburst interval distribution (if available)
        ax5 = fig.add_subplot(gs[3, 1])
        if len(burst_edges) > 1:
            ax5.hist(filtered_ibis/1000, bins=30, color='teal', edgecolor='black', alpha=0.7)
            ax5.axvline(mean_ibi/1000, color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {mean_ibi/1000:.2f}s')
            ax5.set_xlabel('Interburst Interval (seconds)', fontsize=12)
            ax5.set_ylabel('Count', fontsize=12)
            ax5.set_title('Interburst Interval Distribution', fontsize=12)
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'Not enough bursts\nfor IBI analysis', 
                    ha='center', va='center', transform=ax5.transAxes, fontsize=12)
            ax5.set_title('Interburst Interval Distribution', fontsize=12)
        
        plt.savefig('/Users/sebas/Desktop/Ephys/lc/burst_detection_analysis_shank3.png', dpi=300, bbox_inches='tight')
        print(f"\nShank3 burst detection plot saved to: /Users/sebas/Desktop/Ephys/lc/burst_detection_analysis_shank3.png")
        plt.show()
        
        # Create raster plot with bursts highlighted
        print("\nGenerating SHANK3 raster plot with bursts highlighted...")
        
        fig, ax = plt.subplots(figsize=(20, 10))
        
        # First, shade burst regions
        for start, end in burst_edges:
            ax.axvspan(start/1000, end/1000, color='red', alpha=0.15, zorder=0)
        
        # Plot neurons, highlighting rigid backbone neurons
        for neuron_idx, spike_times in enumerate(lc_spike_data.train):
            if len(spike_times) > 0:
                spike_times_sec = spike_times / 1000
                
                # Check if this is a rigid backbone neuron
                if neuron_idx in rigid_neurons:
                    color = 'blue'
                    alpha = 0.8
                    s = 2
                else:
                    color = 'black'
                    alpha = 0.5
                    s = 1
                
                ax.scatter(spike_times_sec, [neuron_idx] * len(spike_times), 
                          s=s, c=color, marker='|', linewidths=0.5, alpha=alpha)
        
        ax.set_xlabel('Time (seconds)', fontsize=14)
        ax.set_ylabel('Neuron Index', fontsize=14)
        ax.set_title(f'Raster Plot with Bursts: SHANK3 - Rigid Neurons in Blue', fontsize=16)
        ax.set_xlim(0, lc_spike_data.length / 1000)
        ax.set_ylim(-1, lc_spike_data.N)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add legend
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = [
            Patch(facecolor='red', alpha=0.3, label=f'Burst periods (n={len(burst_edges)})'),
            Line2D([0], [0], marker='|', color='w', markerfacecolor='blue', 
                   markersize=10, label=f'Rigid backbone neurons (n={len(rigid_neurons)})'),
            Line2D([0], [0], marker='|', color='w', markerfacecolor='black', 
                   markersize=10, label='Non-rigid neurons')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('/Users/sebas/Desktop/Ephys/lc/raster_plot_bursts_highlighted_shank3.png', dpi=300, bbox_inches='tight')
        print(f"Shank3 burst raster plot saved to: /Users/sebas/Desktop/Ephys/lc/raster_plot_bursts_highlighted_shank3.png")
        plt.show()
        
    else:
        print("  No bursts detected with current parameters.")
        print("  Try adjusting detection threshold or smoothing parameters.")

# Load wildtype dataset for comparison
wt_file_path = "/Users/sebas/Desktop/Ephys/lc/Trace_20251018_20_09_51-LC-kolf2.2-day51-wt_acqm.zip"

print("\n" + "="*60)
print("LOADING WILDTYPE DATASET FOR COMPARISON")
print("="*60)

try:
    with zipfile.ZipFile(wt_file_path, 'r') as zip_ref:
        npz_files = [f for f in zip_ref.namelist() if f.endswith('.npz')]
        if not npz_files:
            raise FileNotFoundError("No .npz file found in the archive")
        
        npz_file = npz_files[0]
        print(f"Found .npz file: {npz_file}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_ref.extract(npz_file, temp_dir)
            extracted_path = os.path.join(temp_dir, npz_file)
            
            npz_data = np.load(extracted_path, allow_pickle=True)
            train = npz_data['train'].item()
            fs = npz_data['fs'].item() if 'fs' in npz_data.keys() else 20000
            
            neuron_ids = sorted(train.keys())
            spike_trains = [train[nid] for nid in neuron_ids]
            
            if len(spike_trains) > 0 and len(spike_trains[0]) > 0:
                max_time_samples = max(max(st) if len(st) > 0 else 0 for st in spike_trains)
                recording_length_ms = (max_time_samples / fs) * 1000
                
                spike_trains_ms = []
                for st in spike_trains:
                    if len(st) > 0:
                        spike_times_ms = (np.array(st) / fs) * 1000
                        spike_trains_ms.append(spike_times_ms)
                    else:
                        spike_trains_ms.append(np.array([]))
            else:
                spike_trains_ms = spike_trains
            
            wt_spike_data = SpikeData(train=spike_trains_ms)
    
    print(f"Wildtype dataset loaded successfully!")
    print(f"Number of neurons: {wt_spike_data.N}")
    print(f"Recording length: {wt_spike_data.length:.2f} ms ({wt_spike_data.length/1000:.2f} seconds)")
    print(f"Total spikes: {sum(len(train) for train in wt_spike_data.train)}")
    
except Exception as e:
    print(f"Error loading wildtype dataset: {e}")
    import traceback
    traceback.print_exc()
    wt_spike_data = None

# Analyze wildtype bursts
if wt_spike_data is not None:
    print("\n" + "="*60)
    print("WILDTYPE BURST DETECTION")
    print("="*60)
    
    from scipy.signal import find_peaks
    from scipy.stats import variation
    
    duration_wt = wt_spike_data.length
    bins_wt = np.arange(0, duration_wt + 1, 1)
    population_activity_wt = np.zeros(len(bins_wt) - 1)
    
    for spikes in wt_spike_data.train:
        hist, _ = np.histogram(spikes, bins=bins_wt)
        population_activity_wt += hist
    
    kernel = np.ones(50) / 50
    smoothed_activity_wt = np.convolve(population_activity_wt, kernel, mode='same')
    
    baseline_wt = np.percentile(smoothed_activity_wt, 25)
    peak_thr_wt = baseline_wt + 2.5 * np.std(smoothed_activity_wt)
    peak_indices_wt, peak_info_wt = find_peaks(smoothed_activity_wt, 
                                                height=peak_thr_wt,
                                                distance=200,
                                                prominence=0.5)
    
    burst_edges_wt = []
    min_burst_width = 20
    max_burst_width = 500
    
    if len(peak_indices_wt) > 0:
        edge_thrs_wt = baseline_wt + 0.2 * (peak_info_wt['peak_heights'] - baseline_wt)
        
        for peak_i, edge_thr in zip(peak_indices_wt, edge_thrs_wt):
            burst_start = None
            for i in range(peak_i, max(peak_i - max_burst_width, 0), -1):
                if all(smoothed_activity_wt[max(0, i-5):i+1] < edge_thr):
                    burst_start = i
                    break
            
            burst_end = None
            for i in range(peak_i, min(peak_i + max_burst_width, len(smoothed_activity_wt))):
                if all(smoothed_activity_wt[i:min(i+5, len(smoothed_activity_wt))] < edge_thr):
                    burst_end = i
                    break
                    
            if (burst_start is not None and 
                burst_end is not None and 
                min_burst_width <= (burst_end - burst_start) <= max_burst_width):
                burst_edges_wt.append([burst_start, burst_end])
        
        burst_edges_wt = np.array(burst_edges_wt)
    else:
        burst_edges_wt = np.array([]).reshape(0, 2)
    
    print(f"Total bursts detected: {len(burst_edges_wt)}")
    
    # Calculate wildtype burst metrics
    if len(burst_edges_wt) > 0:
        burst_widths_wt = burst_edges_wt[:, 1] - burst_edges_wt[:, 0]
        burst_amplitudes_wt = peak_info_wt['peak_heights']
        recording_duration_sec_wt = duration_wt / 1000
        burst_frequency_wt = len(burst_edges_wt) / recording_duration_sec_wt
        
        print(f"Mean burst width: {np.mean(burst_widths_wt):.2f} ± {np.std(burst_widths_wt):.2f} ms")
        print(f"Burst frequency: {burst_frequency_wt:.4f} bursts/s ({burst_frequency_wt*60:.2f} bursts/min)")
        
        if len(burst_edges_wt) > 1:
            interburst_intervals_wt = burst_edges_wt[1:, 0] - burst_edges_wt[:-1, 1]
            max_ibi = 1 * 60 * 1000
            filtered_ibis_wt = interburst_intervals_wt[interburst_intervals_wt < max_ibi]
            
            if len(filtered_ibis_wt) > 0:
                mean_ibi_wt = np.mean(filtered_ibis_wt)
                cv_ibi_wt = variation(filtered_ibis_wt) if len(filtered_ibis_wt) > 1 else np.nan
                print(f"Mean interburst interval: {mean_ibi_wt:.2f} ms ({mean_ibi_wt/1000:.2f}s)")
                print(f"CV of interburst intervals: {cv_ibi_wt:.3f}")
        
        # Calculate burst involvement for wildtype
        total_bursts_wt = len(burst_edges_wt)
        total_neurons_wt = wt_spike_data.N
        burst_involvement_matrix_wt = np.zeros((total_neurons_wt, total_bursts_wt), dtype=int)
        
        for i, (burst_start, burst_end) in enumerate(burst_edges_wt):
            for unit in range(total_neurons_wt):
                spikes = wt_spike_data.train[unit]
                spikes_in_burst = np.sum((spikes >= burst_start) & (spikes <= burst_end + 1))
                burst_involvement_matrix_wt[unit, i] = 1 if spikes_in_burst >= 2 else 0
        
        bic_matrix_wt = np.sum(burst_involvement_matrix_wt, axis=1) / total_bursts_wt
        rigid_neurons_wt = np.where(bic_matrix_wt >= 0.9)[0]
        
        print(f"Rigid backbone neurons: {len(rigid_neurons_wt)} ({100*len(rigid_neurons_wt)/total_neurons_wt:.1f}%)")
        print(f"Mean BIC: {np.mean(bic_matrix_wt):.3f}")
        
        # Plot wildtype burst detection results
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
        
        # Main plot: Population activity with detected bursts
        ax1 = fig.add_subplot(gs[0:2, :])
        time_sec_wt = np.arange(len(smoothed_activity_wt)) / 1000
        ax1.plot(time_sec_wt, smoothed_activity_wt, 'k', alpha=0.7, linewidth=1)
        ax1.axhline(baseline_wt, color='gray', linestyle='--', alpha=0.5, label='Baseline')
        ax1.axhline(peak_thr_wt, color='orange', linestyle='--', alpha=0.5, label='Peak threshold')
        
        for start, end in burst_edges_wt:
            ax1.axvspan(start/1000, end/1000, color='red', alpha=0.2)
        
        ax1.scatter(peak_indices_wt/1000, peak_info_wt['peak_heights'], 
                   color='red', s=50, zorder=5, label='Burst peaks')
        ax1.set_xlabel('Time (seconds)', fontsize=12)
        ax1.set_ylabel('Population Activity (spikes/ms)', fontsize=12)
        ax1.set_title(f'Burst Detection: WILDTYPE - {len(burst_edges_wt)} bursts detected', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, duration_wt/1000)
        
        # Burst width distribution
        ax2 = fig.add_subplot(gs[2, 0])
        ax2.hist(burst_widths_wt, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax2.axvline(np.mean(burst_widths_wt), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(burst_widths_wt):.1f}ms')
        ax2.set_xlabel('Burst Width (ms)', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Burst Width Distribution', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Burst amplitude distribution
        ax3 = fig.add_subplot(gs[2, 1])
        ax3.hist(burst_amplitudes_wt, bins=30, color='coral', edgecolor='black', alpha=0.7)
        ax3.axvline(np.mean(burst_amplitudes_wt), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(burst_amplitudes_wt):.2f}')
        ax3.set_xlabel('Burst Amplitude (spikes/ms)', fontsize=12)
        ax3.set_ylabel('Count', fontsize=12)
        ax3.set_title('Burst Amplitude Distribution', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Burst involvement coefficient (BIC) distribution
        ax4 = fig.add_subplot(gs[3, 0])
        ax4.hist(bic_matrix_wt, bins=30, color='purple', edgecolor='black', alpha=0.7)
        ax4.axvline(0.9, color='red', linestyle='--', linewidth=2, label='Rigid threshold (0.9)')
        ax4.axvline(np.median(bic_matrix_wt), color='green', linestyle='--', 
                   linewidth=2, label=f'Median: {np.median(bic_matrix_wt):.3f}')
        ax4.set_xlabel('Burst Involvement Coefficient', fontsize=12)
        ax4.set_ylabel('Number of Neurons', fontsize=12)
        ax4.set_title('Neuron Burst Involvement', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Interburst interval distribution (if available)
        ax5 = fig.add_subplot(gs[3, 1])
        if len(burst_edges_wt) > 1:
            ax5.hist(filtered_ibis_wt/1000, bins=30, color='teal', edgecolor='black', alpha=0.7)
            ax5.axvline(mean_ibi_wt/1000, color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {mean_ibi_wt/1000:.2f}s')
            ax5.set_xlabel('Interburst Interval (seconds)', fontsize=12)
            ax5.set_ylabel('Count', fontsize=12)
            ax5.set_title('Interburst Interval Distribution', fontsize=12)
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'Not enough bursts\nfor IBI analysis', 
                    ha='center', va='center', transform=ax5.transAxes, fontsize=12)
            ax5.set_title('Interburst Interval Distribution', fontsize=12)
        
        plt.savefig('/Users/sebas/Desktop/Ephys/lc/burst_detection_analysis_wildtype.png', dpi=300, bbox_inches='tight')
        print(f"\nWildtype burst detection plot saved to: /Users/sebas/Desktop/Ephys/lc/burst_detection_analysis_wildtype.png")
        plt.show()
        
        # Create raster plot with bursts highlighted for wildtype
        print("\nGenerating WILDTYPE raster plot with bursts highlighted...")
        
        fig, ax = plt.subplots(figsize=(20, 10))
        
        # First, shade burst regions
        for start, end in burst_edges_wt:
            ax.axvspan(start/1000, end/1000, color='red', alpha=0.15, zorder=0)
        
        # Plot neurons, highlighting rigid backbone neurons
        for neuron_idx, spike_times in enumerate(wt_spike_data.train):
            if len(spike_times) > 0:
                spike_times_sec = spike_times / 1000
                
                # Check if this is a rigid backbone neuron
                if neuron_idx in rigid_neurons_wt:
                    color = 'blue'
                    alpha = 0.8
                    s = 2
                else:
                    color = 'black'
                    alpha = 0.5
                    s = 1
                
                ax.scatter(spike_times_sec, [neuron_idx] * len(spike_times), 
                          s=s, c=color, marker='|', linewidths=0.5, alpha=alpha)
        
        ax.set_xlabel('Time (seconds)', fontsize=14)
        ax.set_ylabel('Neuron Index', fontsize=14)
        ax.set_title(f'Raster Plot with Bursts: WILDTYPE - Rigid Neurons in Blue', fontsize=16)
        ax.set_xlim(0, wt_spike_data.length / 1000)
        ax.set_ylim(-1, wt_spike_data.N)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add legend
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = [
            Patch(facecolor='red', alpha=0.3, label=f'Burst periods (n={len(burst_edges_wt)})'),
            Line2D([0], [0], marker='|', color='w', markerfacecolor='blue', 
                   markersize=10, label=f'Rigid backbone neurons (n={len(rigid_neurons_wt)})'),
            Line2D([0], [0], marker='|', color='w', markerfacecolor='black', 
                   markersize=10, label='Non-rigid neurons')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('/Users/sebas/Desktop/Ephys/lc/raster_plot_bursts_highlighted_wildtype.png', dpi=300, bbox_inches='tight')
        print(f"Wildtype burst raster plot saved to: /Users/sebas/Desktop/Ephys/lc/raster_plot_bursts_highlighted_wildtype.png")
        plt.show()
    
    # Create wildtype raster plots (full and 60s window)
    print("\nGenerating WILDTYPE basic raster plots...")
    
    # Full raster plot
    fig, ax = plt.subplots(figsize=(20, 10))
    for neuron_idx, spike_times in enumerate(wt_spike_data.train):
        if len(spike_times) > 0:
            spike_times_sec = spike_times / 1000
            ax.scatter(spike_times_sec, [neuron_idx] * len(spike_times), 
                      s=1, c='black', marker='|', linewidths=0.5)
    ax.set_xlabel('Time (seconds)', fontsize=14)
    ax.set_ylabel('Neuron Index', fontsize=14)
    ax.set_title('Raster Plot: WILDTYPE (kolf2.2, day 51)', fontsize=16)
    ax.set_xlim(0, wt_spike_data.length / 1000)
    ax.set_ylim(-1, wt_spike_data.N)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('/Users/sebas/Desktop/Ephys/lc/raster_plot_full_wildtype.png', dpi=300, bbox_inches='tight')
    print(f"Wildtype full raster plot saved to: /Users/sebas/Desktop/Ephys/lc/raster_plot_full_wildtype.png")
    plt.show()
    
    # 60s window raster plot
    start_time = 0
    end_time = min(60, wt_spike_data.length / 1000)
    fig, ax = plt.subplots(figsize=(20, 10))
    for neuron_idx, spike_times in enumerate(wt_spike_data.train):
        if len(spike_times) > 0:
            spike_times_sec = spike_times / 1000
            mask = (spike_times_sec >= start_time) & (spike_times_sec <= end_time)
            filtered_spikes = spike_times_sec[mask]
            if len(filtered_spikes) > 0:
                ax.scatter(filtered_spikes, [neuron_idx] * len(filtered_spikes), 
                          s=2, c='black', marker='|', linewidths=0.8)
    ax.set_xlabel('Time (seconds)', fontsize=14)
    ax.set_ylabel('Neuron Index', fontsize=14)
    ax.set_title(f'Raster Plot: WILDTYPE (kolf2.2, day 51) - {start_time}s to {end_time}s', fontsize=16)
    ax.set_xlim(start_time, end_time)
    ax.set_ylim(-1, wt_spike_data.N)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('/Users/sebas/Desktop/Ephys/lc/raster_plot_60s_wildtype.png', dpi=300, bbox_inches='tight')
    print(f"Wildtype 60s raster plot saved to: /Users/sebas/Desktop/Ephys/lc/raster_plot_60s_wildtype.png")
    plt.show()
    
    # Wildtype firing rate distribution
    print("\nGenerating WILDTYPE firing rate statistics...")
    recording_duration_sec_wt = wt_spike_data.length / 1000
    firing_rates_wt = np.array([len(train) / recording_duration_sec_wt 
                                for train in wt_spike_data.train])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(firing_rates_wt, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(firing_rates_wt), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(firing_rates_wt):.2f} Hz')
    ax.axvline(np.median(firing_rates_wt), color='green', linestyle='--', linewidth=2, 
               label=f'Median: {np.median(firing_rates_wt):.2f} Hz')
    ax.set_xlabel('Firing Rate (Hz)', fontsize=14)
    ax.set_ylabel('Number of Neurons', fontsize=14)
    ax.set_title('Distribution of Firing Rates Across Neurons: WILDTYPE', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/Users/sebas/Desktop/Ephys/lc/firing_rate_distribution_wildtype.png', dpi=300, bbox_inches='tight')
    print(f"Wildtype firing rate distribution saved to: /Users/sebas/Desktop/Ephys/lc/firing_rate_distribution_wildtype.png")
    plt.show()
    
    # Wildtype population firing rate
    print("\nGenerating WILDTYPE population firing rate...")
    bin_size = 1000
    smoothing_window = 10
    bins_wt, pop_rate_wt = wt_spike_data.population_firing_rate(
        bin_size=bin_size, 
        w=smoothing_window, 
        average=False
    )
    time_s_wt = bins_wt[:-1] / 1000
    
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.plot(time_s_wt, pop_rate_wt, 'b-', linewidth=1.5)
    ax.set_xlabel('Time (seconds)', fontsize=14)
    ax.set_ylabel('Population Firing Rate (spikes/s)', fontsize=14)
    ax.set_title('Population Firing Rate Over Time: WILDTYPE', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, wt_spike_data.length / 1000)
    plt.tight_layout()
    plt.savefig('/Users/sebas/Desktop/Ephys/lc/population_firing_rate_wildtype.png', dpi=300, bbox_inches='tight')
    print(f"Wildtype population firing rate plot saved to: /Users/sebas/Desktop/Ephys/lc/population_firing_rate_wildtype.png")
    plt.show()

# Comparative Analysis: Shank3 vs Wildtype
if lc_spike_data is not None and wt_spike_data is not None and len(burst_edges) > 0 and len(burst_edges_wt) > 0:
    print("\n" + "="*60)
    print("COMPARATIVE ANALYSIS: SHANK3 vs WILDTYPE")
    print("="*60)
    
    # Calculate firing rates for both datasets
    recording_duration_sec_shank3 = lc_spike_data.length / 1000
    recording_duration_sec_wt = wt_spike_data.length / 1000
    
    firing_rates_shank3 = np.array([len(train) / recording_duration_sec_shank3 
                                     for train in lc_spike_data.train])
    firing_rates_wt = np.array([len(train) / recording_duration_sec_wt 
                                for train in wt_spike_data.train])
    
    # Prepare data for violin plots
    import pandas as pd
    
    # Create dataframe for violin plots
    data_list = []
    
    # Firing rates
    for fr in firing_rates_shank3:
        data_list.append({'Genotype': 'Shank3', 'Metric': 'Firing Rate (Hz)', 'Value': fr})
    for fr in firing_rates_wt:
        data_list.append({'Genotype': 'Wildtype', 'Metric': 'Firing Rate (Hz)', 'Value': fr})
    
    # Burst widths
    for bw in burst_widths:
        data_list.append({'Genotype': 'Shank3', 'Metric': 'Burst Width (ms)', 'Value': bw})
    for bw in burst_widths_wt:
        data_list.append({'Genotype': 'Wildtype', 'Metric': 'Burst Width (ms)', 'Value': bw})
    
    # Burst amplitudes
    for ba in burst_amplitudes:
        data_list.append({'Genotype': 'Shank3', 'Metric': 'Burst Amplitude (spikes/ms)', 'Value': ba})
    for ba in burst_amplitudes_wt:
        data_list.append({'Genotype': 'Wildtype', 'Metric': 'Burst Amplitude (spikes/ms)', 'Value': ba})
    
    # Burst Involvement Coefficient (BIC)
    for bic in bic_matrix:
        data_list.append({'Genotype': 'Shank3', 'Metric': 'Burst Involvement Coefficient', 'Value': bic})
    for bic in bic_matrix_wt:
        data_list.append({'Genotype': 'Wildtype', 'Metric': 'Burst Involvement Coefficient', 'Value': bic})
    
    # Interburst intervals
    if len(filtered_ibis) > 0:
        for ibi in filtered_ibis / 1000:  # Convert to seconds
            data_list.append({'Genotype': 'Shank3', 'Metric': 'Interburst Interval (s)', 'Value': ibi})
    if len(filtered_ibis_wt) > 0:
        for ibi in filtered_ibis_wt / 1000:  # Convert to seconds
            data_list.append({'Genotype': 'Wildtype', 'Metric': 'Interburst Interval (s)', 'Value': ibi})
    
    df = pd.DataFrame(data_list)
    
    # Create violin plots
    metrics = ['Firing Rate (Hz)', 'Burst Width (ms)', 'Burst Amplitude (spikes/ms)', 
               'Burst Involvement Coefficient', 'Interburst Interval (s)']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    colors = {'Shank3': '#FF6B6B', 'Wildtype': '#4ECDC4'}
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        metric_data = df[df['Metric'] == metric]
        
        if len(metric_data) > 0:
            # Create violin plot
            positions = [1, 2]
            genotypes = ['Shank3', 'Wildtype']
            data_to_plot = [metric_data[metric_data['Genotype'] == g]['Value'].values 
                           for g in genotypes]
            
            # Remove empty arrays
            data_to_plot = [d for d in data_to_plot if len(d) > 0]
            valid_genotypes = [g for g, d in zip(genotypes, data_to_plot) if len(d) > 0]
            
            if len(data_to_plot) > 0:
                parts = ax.violinplot(data_to_plot, positions=range(len(data_to_plot)), 
                                     showmeans=True, showmedians=True)
                
                # Color the violins
                for i, pc in enumerate(parts['bodies']):
                    pc.set_facecolor(colors[valid_genotypes[i]])
                    pc.set_alpha(0.7)
                
                # Add scatter points
                for i, (genotype, data) in enumerate(zip(valid_genotypes, data_to_plot)):
                    x_pos = np.random.normal(i, 0.04, size=len(data))
                    ax.scatter(x_pos, data, alpha=0.3, s=20, color=colors[genotype])
                
                ax.set_xticks(range(len(valid_genotypes)))
                ax.set_xticklabels(valid_genotypes)
                ax.set_ylabel(metric, fontsize=11)
                ax.set_title(metric, fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add statistics
                from scipy import stats
                if len(data_to_plot) == 2:
                    # Perform t-test
                    t_stat, p_value = stats.ttest_ind(data_to_plot[0], data_to_plot[1])
                    
                    # Add p-value annotation
                    if p_value < 0.001:
                        sig_text = '***'
                    elif p_value < 0.01:
                        sig_text = '**'
                    elif p_value < 0.05:
                        sig_text = '*'
                    else:
                        sig_text = 'ns'
                    
                    y_max = max([d.max() for d in data_to_plot])
                    y_min = min([d.min() for d in data_to_plot])
                    y_range = y_max - y_min
                    
                    ax.text(0.5, y_max + 0.1 * y_range, 
                           f'p={p_value:.4f} {sig_text}',
                           ha='center', fontsize=10, fontweight='bold')
    
    # Remove extra subplot
    axes[-1].axis('off')
    
    # Add summary statistics table
    ax_table = axes[-1]
    ax_table.axis('tight')
    ax_table.axis('off')
    
    summary_stats = []
    summary_stats.append(['Metric', 'Shank3', 'Wildtype'])
    summary_stats.append(['N neurons', f'{lc_spike_data.N}', f'{wt_spike_data.N}'])
    summary_stats.append(['N bursts', f'{len(burst_edges)}', f'{len(burst_edges_wt)}'])
    summary_stats.append(['Burst freq (bursts/min)', 
                          f'{burst_frequency*60:.2f}', f'{burst_frequency_wt*60:.2f}'])
    summary_stats.append(['Mean firing rate (Hz)', 
                          f'{np.mean(firing_rates_shank3):.2f}', f'{np.mean(firing_rates_wt):.2f}'])
    summary_stats.append(['N rigid neurons', 
                          f'{len(rigid_neurons)} ({100*len(rigid_neurons)/lc_spike_data.N:.1f}%)', 
                          f'{len(rigid_neurons_wt)} ({100*len(rigid_neurons_wt)/wt_spike_data.N:.1f}%)'])
    
    table = ax_table.table(cellText=summary_stats, cellLoc='left', loc='center',
                          colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#E8E8E8')
        table[(0, i)].set_text_props(weight='bold')
    
    plt.suptitle('Shank3 vs Wildtype Comparison (LC, Day 51)', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('/Users/sebas/Desktop/Ephys/lc/shank3_vs_wildtype_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: /Users/sebas/Desktop/Ephys/lc/shank3_vs_wildtype_comparison.png")
    plt.show()
    
    # Print statistical summary
    print("\n" + "="*60)
    print("STATISTICAL COMPARISON SUMMARY")
    print("="*60)
    
    from scipy import stats
    
    # Firing rate comparison
    t_stat, p_val = stats.ttest_ind(firing_rates_shank3, firing_rates_wt)
    print(f"\nFiring Rate:")
    print(f"  Shank3: {np.mean(firing_rates_shank3):.2f} ± {np.std(firing_rates_shank3):.2f} Hz")
    print(f"  Wildtype: {np.mean(firing_rates_wt):.2f} ± {np.std(firing_rates_wt):.2f} Hz")
    print(f"  t-test: t={t_stat:.3f}, p={p_val:.4f}")
    
    # Burst width comparison
    t_stat, p_val = stats.ttest_ind(burst_widths, burst_widths_wt)
    print(f"\nBurst Width:")
    print(f"  Shank3: {np.mean(burst_widths):.2f} ± {np.std(burst_widths):.2f} ms")
    print(f"  Wildtype: {np.mean(burst_widths_wt):.2f} ± {np.std(burst_widths_wt):.2f} ms")
    print(f"  t-test: t={t_stat:.3f}, p={p_val:.4f}")
    
    # Burst amplitude comparison
    t_stat, p_val = stats.ttest_ind(burst_amplitudes, burst_amplitudes_wt)
    print(f"\nBurst Amplitude:")
    print(f"  Shank3: {np.mean(burst_amplitudes):.2f} ± {np.std(burst_amplitudes):.2f} spikes/ms")
    print(f"  Wildtype: {np.mean(burst_amplitudes_wt):.2f} ± {np.std(burst_amplitudes_wt):.2f} spikes/ms")
    print(f"  t-test: t={t_stat:.3f}, p={p_val:.4f}")
    
    # BIC comparison
    t_stat, p_val = stats.ttest_ind(bic_matrix, bic_matrix_wt)
    print(f"\nBurst Involvement Coefficient:")
    print(f"  Shank3: {np.mean(bic_matrix):.3f} ± {np.std(bic_matrix):.3f}")
    print(f"  Wildtype: {np.mean(bic_matrix_wt):.3f} ± {np.std(bic_matrix_wt):.3f}")
    print(f"  t-test: t={t_stat:.3f}, p={p_val:.4f}")
    
    # Burst frequency comparison (single values)
    print(f"\nBurst Frequency:")
    print(f"  Shank3: {burst_frequency*60:.2f} bursts/min")
    print(f"  Wildtype: {burst_frequency_wt*60:.2f} bursts/min")
    print(f"  Difference: {(burst_frequency - burst_frequency_wt)*60:.2f} bursts/min")

print("\n" + "="*60)
print("Analysis complete!")
print("="*60)
