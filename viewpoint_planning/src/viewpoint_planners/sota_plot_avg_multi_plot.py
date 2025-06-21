import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d

mywin_size = 7
win_size = 6
lose_size =7
# Create plot
sigma = 0.6  # The standard deviation for Gaussian kernel
fig_text_font = 13
legend_font_size = 11
legendframealpha = 1
# plt.figure(figsize=(10, 5))
# fig, ax = plt.subplots(figsize=(10, 5))

# Group3
avg_gsnbv_win = np.array([0.70698, 0.78465, 0.98068, 0.98068, 0.98068, 0.98068, 0.98068, 0.98068, 0.98068, 0.98068, 0.98068])
x = np.arange(len(avg_gsnbv_win))
avg_scnbvp_all = np.array([0.71707, 0.67108, 0.74806, 0.74758, 0.75942, 0.80818, 0.81155, 0.85027, 0.86137, 0.74535, 0.65387])
avg_scnbvp_win = np.array([0.7361833333, 0.66925, 0.6850166667, 0.82515, 0.8623833333, 0.90605, 0.9629833333, 0.9629833333, 0.9629833333, 0.9629833333, 0.9629833333])
avg_scnbvp_lose = np.array([0.7284, 0.673825, 0.842625, 0.631225, 0.604975, 0.661375, 0.5844, 0.6812, 0.70895, 0.4189, 0.1902])
avg_gnbv_all = np.array([0.74896, 0.7376, 0.65471, 0.68667, 0.72159, 0.81283, 0.81311, 0.72605, 0.81324, 0.72739, 0.8139])
avg_gnbv_win = np.array([0.71612, 0.8016, 0.8554, 0.91806, 0.96556, 0.96556, 0.96556, 0.96556, 0.96556, 0.96556, 0.96556])
avg_gnbv_lose = np.array([0.7218, 0.6736, 0.45402, 0.45528, 0.47762, 0.6601, 0.66066, 0.48654, 0.66092, 0.48922, 0.66224])

# Apply Gaussian smoothing
avg_gsnbv_win_smooth = gaussian_filter1d(avg_gsnbv_win, sigma=sigma)
avg_scnbvp_win_smooth = gaussian_filter1d(avg_scnbvp_win, sigma=sigma)
avg_scnbvp_lose_smooth = gaussian_filter1d(avg_scnbvp_lose, sigma=sigma)
avg_gnbv_win_smooth = gaussian_filter1d(avg_gnbv_win, sigma=sigma)
avg_gnbv_lose_smooth = gaussian_filter1d(avg_gnbv_lose, sigma=sigma)

# Create subplots side by side
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.5, 5))  # Adjusted figsize for better visibility
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 5))  # Adjusted figsize for better visibility

for ax in [ax1, ax2]:
    for y_dashed in [0.3, 0.4, 0.5, 0.6 , 0.7, 0.8, 0.9]:
        if y_dashed != 0.9:
            ax.axhline(y=y_dashed, color='gray', linestyle='--', linewidth=0.5)
        else:
            ax.axhline(y=y_dashed, color='orange', linestyle='-', linewidth=1.5)

# Group 1
avg_gsnbv_win_group1 = np.array([0.72798,	0.95799,	0.96583,	0.96583,	0.96583,	0.96583,	0.96583,	0.96583,	0.96583,	0.96583,	0.96583])
avg_gsnbv_win_group1_smooth = gaussian_filter1d(avg_gsnbv_win_group1, sigma=sigma)
# avg_scnbvp_win_group1 = np.array([0.6410333333,	0.5277666667,	0.1982666667,	0.6429,	0.7192666667,	0.8234666667,	0.8234666667,	0.8234666667,	0.9398,	0.9398,0.9398])
avg_scnbvp_win_group1 = np.array([0.7410333333,	0.5277666667,	0.1982666667,	0.6429,	0.7192666667,	0.8234666667,	0.8234666667,	0.8234666667,	0.9398,	0.9398,0.9398])
avg_scnbvp_win_group1_smooth = gaussian_filter1d(avg_scnbvp_win_group1, sigma=sigma)
avg_scnbvp_lose_group1 = np.array([0.6410571429,	0.5577,	0.4623714286,	0.5303857143,	0.3562714286,	0.6211,	0.7928,	0.7559714286,	0.5587,	0.4322714286	,0.4884])
avg_scnbvp_lose_group1_smooth = gaussian_filter1d(avg_scnbvp_lose_group1, sigma=sigma)

avg_gnbvp_win_group1 = np.array([0.6215166667,	0.8332833333	,0.8803,	0.9498	,0.99	,0.99	,0.99,	0.99	,0.99,	0.99,	0.99])
avg_gnbvp_win_group1_smooth = gaussian_filter1d(avg_gnbvp_win_group1, sigma=sigma)
avg_gnbvp_lose_group1 = np.array([0.646475	,0.741925	,0.7398,	0.7416	,0.741	,0.7422	,0.7398	,0.741	,0.724125,	0.73675,	0.73675])
avg_gnbvp_lose_group1_smooth = gaussian_filter1d(avg_gnbvp_lose_group1, sigma=sigma)

# Plot Group 1 on ax1
ax1.plot(x, avg_scnbvp_win_group1_smooth, label='SCNBVP success Group 1', color='green', marker='^', linewidth=1, markersize=win_size)
ax1.plot(x, avg_scnbvp_lose_group1_smooth, label='SCNBVP failure Group 1', color='green', marker='X', linestyle='dashdot', linewidth=1.5, markersize=lose_size)
ax1.plot(x, avg_gnbvp_win_group1_smooth, label='GNBV success Group 1', color='blue', marker='^', linewidth=1, markersize=win_size)
ax1.plot(x, avg_gnbvp_lose_group1_smooth, label='GNBV failure Group 1', color='blue', marker='X', linestyle='dotted', linewidth=1.5, markersize=lose_size)
ax1.plot(x, avg_gsnbv_win_group1_smooth, label='GSNBV success Group 1', color='red', marker='*', markersize=mywin_size)

# Plot Group 3 on ax2
ax2.plot(x, avg_scnbvp_win_smooth, label='SCNBVP success Group 2', color='green', marker='^', linewidth=1, markersize=win_size)
ax2.plot(x, avg_scnbvp_lose_smooth, label='SCNBVP failure Group 2', color='green', marker='X', linestyle='dashdot', linewidth=1.5, markersize=lose_size)
ax2.plot(x, avg_gnbv_win_smooth, label='GNBV success Group 2', color='blue', marker='^', linewidth=1, markersize=win_size)
ax2.plot(x, avg_gnbv_lose_smooth, label='GNBV failure Group 2', color='blue', marker='X', linestyle='dotted', linewidth=1.5, markersize=lose_size)
ax2.plot(x, avg_gsnbv_win_smooth, label='GSNBV success Group 2', color='red', marker='*', markersize=mywin_size)

# ax3.plot(x, avg_scnbvp_win_smooth, label='SCNBVP success Group 3', color='green', marker='^', linewidth=1, markersize=win_size)
# ax3.plot(x, avg_scnbvp_lose_smooth, label='SCNBVP failure Group 3', color='green', marker='X', linestyle='dashdot', linewidth=1.5, markersize=lose_size)
# ax3.plot(x, avg_gnbv_win_smooth, label='GNBV success Group 3', color='blue', marker='^', linewidth=1, markersize=win_size)
# ax3.plot(x, avg_gnbv_lose_smooth, label='GNBV failure Group 3', color='blue', marker='X', linestyle='dotted', linewidth=1.5, markersize=lose_size)
# ax3.plot(x, avg_gsnbv_win_smooth, label='GSNBV success Group 3', color='red', marker='*', markersize=mywin_size)



# Configuration for both axes
# for ax in [ax1, ax2,ax3]:
for ax in [ax1, ax2]:
    ax.set_xticks(np.arange(0, 10.05, 1))
    ax.set_xlim([-0.15, 10.15])
    ax.set_ylim([0.0, 1.01])
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.grid(True, which='both', axis='y')
    # ax.legend(ncol=1, loc='lower left', fontsize = 11, facecolor='white', framealpha=0.5)
    ax.legend(ncol=1, loc='lower left', bbox_to_anchor=(-0.005, -0.006),fontsize = legend_font_size, facecolor='white', framealpha=legendframealpha)
    
    # Add horizontal dashed lines at specified y-values
    # for y_dashed in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6 , 0.7, 0.8, 0.9]:

    # for y_dashed in [0.3, 0.4, 0.5, 0.6 , 0.7, 0.8, 0.9]:
    #     if y_dashed != 0.9:
    #         ax.axhline(y=y_dashed, color='gray', linestyle='--', linewidth=0.5)
    #     else:
    #         ax.axhline(y=y_dashed, color='orange', linestyle='-', linewidth=1.5)

# ax1.legend(ncol=1, loc='lower left', bbox_to_anchor=(0.32, -0.015),fontsize = legend_font_size, facecolor='white', framealpha=0.7)
            
# ax1.set_yticks(np.arange(0, 1.05, 0.1))
fig.text(0.5, 0.02, 'Planning iterations', ha='center', va='center', fontsize=fig_text_font)  # Central x-label
fig.text(0.015, 0.5, 'Unoccluded rate [%]', ha='center', va='center', rotation='vertical', fontsize=fig_text_font)  # Central y-label

# ax1.grid(True, which='both', axis='y')  # Enables only horizontal grid lines
# ax2.grid(True, which='both', axis='y')  # Same for the second subplot

# Adjust layout to prevent overlap
# plt.subplots_adjust(wspace=0.3)
# plt.subplots_adjust(wspace=0.1)
# Example using subplots_adjust to tighten layout
plt.subplots_adjust(left=0.06, right=0.94, top=0.9, bottom=0.1, wspace=0.1, hspace=0.0)

# This removes excess space around the figure
# plt.tight_layout()

# Show legend
# ax.legend(ncol=2, loc='lower left', bbox_to_anchor=(0.0, 0.0))
# plt.savefig('/home/jcd/Pictures/sota_plot_compare_combine_plot.png', format='png', dpi=300)
plt.savefig('/home/jcd/Pictures/sota_plot_compare_2exp.png', format='png', dpi=300)
# Show plot
# plt.grid(True)
plt.show()
