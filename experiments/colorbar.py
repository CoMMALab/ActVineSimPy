import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def create_colorbar(output_path='epsilon_colorbar.png'):
    """
    Creates and saves a vertical colorbar for epsilon values.
    """
    # Define the colors for the colormap, matching fast_render.py, normalized for matplotlib
    blue = np.array([100, 100, 255, 255]) / 255.0
    orange = np.array([255, 150, 100, 255]) / 255.0
    
    # Create a colormap from blue to orange
    cmap = mcolors.LinearSegmentedColormap.from_list("epsilon_cmap", [blue, orange])
    
    # Create a figure and axes for the colorbar
    fig, ax = plt.subplots(figsize=(0.5, 6))
    fig.subplots_adjust(left=0.5)
    
    # Define the normalizer for the colorbar based on the input range
    norm = mcolors.Normalize(vmin=-0.125, vmax=0.125)
    
    # Create the colorbar
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                      cax=ax, orientation='vertical', ticks=np.linspace(-0.125, 0.125, 11))
    cb.ax.yaxis.set_ticks_position('left')
    # cb.set_ticklabels(['-1', '0', '1'])
    
    # Save the figure
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Colorbar saved to {output_path}")
    plt.close(fig)

if __name__ == '__main__':
    create_colorbar()
