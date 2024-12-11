import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def set_custom_style():
    """Set up custom plotting style"""
    # Create custom color palette
    custom_colors = ["#2ecc71", "#e74c3c", "#3498db", "#f1c40f", "#9b59b6"]
    
    # Basic style settings
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 16,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'lines.linewidth': 2.0,
        'lines.markersize': 8.0,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.facecolor': '#f8f9fa',
        'figure.facecolor': 'white',
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.spines.right': True,
        'axes.spines.top': True,
    })
    
    # Set seaborn style and palette
    sns.set_palette(custom_colors)
    sns.set_style("whitegrid", {
        'axes.linewidth': 2,
        'axes.edgecolor': 'black',
        'grid.color': '.8',
        'grid.linestyle': '-'
    })
    
    # Custom color map for heatmaps
    colors = ['#000033', '#0000FF', '#00FFFF', '#FFFF00', '#FF0000', '#FF00FF']
    n_bins = 100
    custom_cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)
    
    return custom_cmap

def read_binary_file(filename):
    """Read binary data from Armadillo file format"""
    with open(filename, 'rb') as f:
        header1 = f.readline()
        header2 = f.readline()
        data = np.fromfile(f, dtype=np.float64)
        dims = [int(x) for x in header2.split()]
        return data.reshape(dims) if len(dims) > 1 else data

def plot_probability_conservation():
    """Plot probability conservation over time"""
    prob_no_barrier = read_binary_file("prob_history_no_barrier.bin")
    prob_with_barrier = read_binary_file("prob_history_with_barrier.bin")
    
    dt = 2.5e-5
    T = 0.008
    t_points = np.arange(0, T + dt, dt)
    
    plt.figure(figsize=(10, 6))
    
    # Enhanced plotting with markers and different line styles
    plt.plot(t_points[:len(prob_no_barrier)], np.abs(prob_no_barrier - 1), 
             label='No slits', marker='o', markersize=4, linestyle='-', alpha=0.7)
    plt.plot(t_points[:len(prob_with_barrier)], np.abs(prob_with_barrier - 1), 
             label='2 slits', marker='s', markersize=4, linestyle='--', alpha=0.7)
    
    plt.xlabel('Time')
    plt.ylabel('|p(x,y;t)|² - 1|')
    plt.legend(frameon=True, facecolor='white', edgecolor='black')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.gca().set_facecolor('#f8f9fa')
    plt.tight_layout()
    plt.savefig('probability.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_state_evolution():
    """Plot state evolution at different times"""
    times = {
        'initial': (0, '0'),
        'middle': (0.001, '40'),
        'final': (0.002, '80')
    }
    
    components = ['prob', 'real', 'imag']
    titles = ['Probability', 'Real Part', 'Imaginary Part']
    
    h = 0.005
    x_points = np.arange(0, 1+h, h)
    y_points = np.arange(0, 1+h, h)
    x_min, x_max = x_points[0], x_points[-1]
    y_min, y_max = y_points[0], y_points[-1]
    
    custom_cmap = set_custom_style()
    
    for time_label, (t, idx) in times.items():
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Time Evolution: t = {t}', y=1.05, fontsize=18)
        
        for i, (comp, title) in enumerate(zip(components, titles)):
            filename = f'{time_label}_{comp}_{idx}.bin'
            data = read_binary_file(filename)
            
            axes[i].grid(False)
            
            vmax = np.max(data)
            vmin = np.min(data)
            if comp == 'prob':
                norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
            else:
                abs_max = max(abs(vmin), abs(vmax))
                norm = mpl.colors.Normalize(vmin=-abs_max, vmax=abs_max)
            
            img = axes[i].imshow(data.T, extent=[x_min,x_max,y_min,y_max], 
                               cmap=custom_cmap if comp == 'prob' else 'RdBu_r',
                               norm=norm, origin='lower')
            
            cbar = fig.colorbar(img, ax=axes[i], orientation="horizontal", 
                              pad=0.2, fraction=0.05)
            cbar.set_label(f'{title} Magnitude')
            
            axes[i].set_xlabel('x')
            axes[i].set_ylabel('y')
            axes[i].set_title(title)
            
            # Add box around plot
            for spine in axes[i].spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1)
                spine.set_edgecolor('black')
        
        plt.tight_layout()
        plt.savefig(f'state_evolution_{time_label}.pdf', dpi=300, bbox_inches='tight')
        plt.close()

def plot_detector_analysis():
    """Plot detector screen analysis"""
    h = 0.005
    x_coord = 0.8
    t = 0.002
    
    plt.figure(figsize=(12, 8))
    y_points = np.linspace(0, 1, 199)
    
    line_styles = ['-', '--', '-.']
    markers = ['o', 's', '^']
    
    for i, n_slits in enumerate([1, 2, 3]):
        data = read_binary_file(f'detector_prob_{n_slits}slits.bin')
        plt.plot(y_points, data / np.sum(data), 
                label=f'{n_slits} slit{"s" if n_slits > 1 else ""}',
                linestyle=line_styles[i], marker=markers[i], 
                markersize=6, markevery=10, alpha=0.8)
    
    plt.xlabel('y')
    plt.ylabel(f'p(y|x={x_coord};t={t})')
    plt.legend(frameon=True, facecolor='white', edgecolor='black')
    plt.grid(True, alpha=0.3)
    plt.gca().set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    plt.savefig('detector_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    custom_cmap = set_custom_style()
    
    print("Creating probability conservation plot...")
    plot_probability_conservation()
    
    print("Creating state evolution plots...")
    plot_state_evolution()
    
    print("Creating detector analysis plot...")
    plot_detector_analysis()
    
    print("All visualizations complete!")