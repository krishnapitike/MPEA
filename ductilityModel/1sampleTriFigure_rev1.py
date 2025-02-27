import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.tri as tri
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

def get_contrasting_color(color):
    # Convert RGB to HSV
    hsv = rgb_to_hsv(color)
    # Adjust hue by 180 degrees (0.5 in [0, 1] scale) and invert value
    hsv[0] = (hsv[0] + 0.5) % 1.0
    hsv[2] = 1.0 - hsv[2]
    # Convert back to RGB
    return hsv_to_rgb(hsv)

# Function to convert ternary composition to Cartesian coordinates
def ternary_to_cartesian(A, B, C):
    x = 0.5 * (2 * B + C) / (A + B + C)
    y = np.sqrt(3) / 2 * C / (A + B + C)
    return x, y

# Heat values for each composition
values = np.array([0, 0, 0, 0.5, 0.5, 0.5, 1])

def generate_ternary_compositions(steps):
    """
    Generate a grid of ternary compositions with specified steps between points.
    """
    compositions = []
    for i in range(steps + 1):
        for j in range(steps + 1 - i):
            k = steps - i - j
            A = i / steps
            B = j / steps
            C = k / steps
            compositions.append([A, B, C])
    return np.array(compositions)

def calculateVEC(c,v1,v2,v3):
    chemSpecies=[v1,v2,v3]
    valenceElectrons=dict({\
        ('Cr',6), ('Ti',4), ('V',5), ('W',6), \
        ('Nb',5), ('Zr',4), ('Ta',5), ('Mo',6),\
        ('Al',3)\
        }) #valence electrons
    vec=[]
    for j, val in enumerate(c):
        sum=0
        for i, chem in enumerate(chemSpecies):
            sum += valenceElectrons[chem]*c[j,i]
        vec.append(sum)
    vec=np.array(vec)
    bsd=0.1914*vec-0.1089
    epsilon_p=-13706.12*bsd**3+38196.22*bsd**2-35519.70*bsd+11025.99
    return(epsilon_p)

def calculateVEC_rev1(c, v1, v2, v3):
    chemSpecies = [v1, v2, v3]
    valenceElectrons = {
        'Cr': 6, 'Ti': 4, 'V': 5, 'W': 6,
        'Nb': 5, 'Zr': 4, 'Ta': 5, 'Mo': 6,
        'Al': 3
    }  # valence electrons
    
    vec = []
    epsilon_p = []
    
    for j, val in enumerate(c):
        # Compute VEC
        vec_sum = 0
        for i, chem in enumerate(chemSpecies):
            vec_sum += valenceElectrons[chem] * c[j, i]
        vec.append(vec_sum)

        # Compute bsd
        bsd = 0.1914 * vec_sum - 0.1089

        # Compute epsilon_p
        if c[j, chemSpecies.index('W')] >= 2 / 3:  # Mask values if W < 1/3
            epsilon_p_val = -13706.12 * bsd**3 + 38196.22 * bsd**2 - 35519.70 * bsd + 11025.99
        else:
            epsilon_p_val = np.nan  # Mask value as NaN
        epsilon_p.append(epsilon_p_val)
    
    return np.array(epsilon_p)

# Create the figure
fig = plt.figure(figsize=(12, 10))

# Define axes for the plots, manual adjustments for custom layout
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(212)

axes = [ax1, ax2, ax3]


def plot_ternary_heatmap_masked(ax, v1, v2, v3):
    # Define the number of steps between 0 and 1 (inclusive)
    steps = 100  # Smaller step size means more detailed grid

    compositions = generate_ternary_compositions(steps)
    xy = np.array([ternary_to_cartesian(*comp) for comp in compositions])
    triangulation = tri.Triangulation(xy[:, 0], xy[:, 1])
    values = calculateVEC(compositions, v1, v2, v3)

    # Determine the boundary midpoint (mask below this line)
    midpoint = ternary_to_cartesian(0.5, 0.5, 0.0)  # Midway between V-W and Ti-W
    mask = xy[:, 1] < midpoint[1]  # Mask points below this Y-coordinate

    # Apply the mask to the triangulation
    triangulation.set_mask(mask[triangulation.triangles].any(axis=1))

    # Create the heatmap
    heatmap = ax.tripcolor(triangulation, values, shading='gouraud', cmap='viridis', clim=(0, 15))

    # Add contour lines
    contours = ax.tricontour(triangulation, values, levels=20, colors='black', linewidths=0.5)
    ax.clabel(contours, inline=True, fontsize=8, fmt="%.1f")

    # Remove axes
    ax.axis('off')

    # Set axis limits and equal aspect ratio
    ax.axis([0, 1, 0, np.sqrt(3) / 2])
    ax.set_aspect('equal', adjustable='box')

    # Add labels for the corners
    ax.text(0.5, np.sqrt(3) / 2 + 0.02, v3, ha='center', va='bottom', fontsize=10)  # Top vertex label
    ax.text(-0.02, -0.02, v1, ha='right', va='top', fontsize=10)  # Left vertex label
    ax.text(1.02, -0.02, v2, ha='left', va='top', fontsize=10)  # Right vertex label

    return heatmap


# Function to plot the heatmap
def plot_ternary_heatmap(ax,v1,v2,v3):

    # Define the number of steps between 0 and 1 (inclusive)
    steps = 100  # Smaller step size means more detailed grid

    compositions = generate_ternary_compositions(steps)
    xy = np.array([ternary_to_cartesian(*comp) for comp in compositions])
    triangulation = tri.Triangulation(xy[:, 0], xy[:, 1])
    values = calculateVEC_rev1(compositions,v1,v2,v3)
    heatmap = ax.tripcolor(triangulation, values, shading='gouraud', cmap='viridis',clim=(0, 15))
 
    # Add contour lines
    #contours = ax.tricontour(triangulation, values, levels=20, colors='black', linewidths=0.5)
    #ax.clabel(contours, inline=True, fontsize=8, fmt="%.1f")

    # Generate contour lines
    contours = ax.tricontour(triangulation, values, levels=20, linewidths=0.5)
    for i, level in enumerate(contours.levels):
        color = get_contrasting_color(plt.cm.viridis((level - min(values)) / (max(values) - min(values))))
        contours.collections[i].set_color(color)
    ax.clabel(contours, inline=True, fontsize=8, fmt="%.1f")

    # Remove axes
    ax.axis('off')

    # Set axis limits and equal aspect ratio
    ax.axis([0, 1, 0, np.sqrt(3)/2])
    ax.set_aspect('equal', adjustable='box')

    # Add labels for the corners
    ax.text(0.5, np.sqrt(3)/2 + 0.02, v3, ha='center', va='bottom', fontsize=10)  # Top vertex label
    ax.text(-0.02, -0.02, v1, ha='right', va='top', fontsize=10)  # Left vertex label
    ax.text(1.02, -0.02, v2, ha='left', va='top', fontsize=10)  # Right vertex label

    return heatmap

# Plot on each axis and collect handles
#heatmaps = [plot_ternary_heatmap(ax) for ax in axes]

heatmaps = []

heatmaps.append(plot_ternary_heatmap_masked(axes[0],'V','W','Ti'))

heatmaps.append(plot_ternary_heatmap_masked(axes[1],'W','V','Cr'))

heatmaps.append(plot_ternary_heatmap_masked(axes[2],'Cr','Ti','W'))

# Assuming all heatmaps share the same color limits
common_clim = (0, 15)  # Define based on your data's range
for heatmap in heatmaps:
    heatmap.set_clim(common_clim)

# Create a common colorbar
cbar_ax = fig.add_axes([0.92, 0.1, 0.03, 0.8])  # Adjust these values for your layout
fig.colorbar(heatmaps[1], cax=cbar_ax)

# Adjust the layout manually
plt.tight_layout(rect=[0, 0, 0.93, 0.96])  # Adjust to make room for colorbar
plt.suptitle('Peak true strain')
plt.savefig('epsilon_p.png')
plt.show()