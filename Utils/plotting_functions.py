# Author: Andrea Vaiuso
# Version: 2.1
# Date: 31.07.2025
# Description: This module provides functions for plotting simulation data, saving log data to CSV,
# and creating 3D animations of a drone's trajectory and attitude over time.

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

import numpy as np
from Utils.utils import euler_to_rot
from Drone.Simulation import Simulation
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime
from Worlds.World import World

def plotLogData(log_dict, time, waypoints=None, ncols=2):
    """
    Dynamically plot simulation data with per-plot legend/grid flags and configurable columns.
    Supports per-series custom time vectors when a spec contains key 'time'.
    Parameters:
        log_dict (dict):
            Keys are subplot titles (str).
            Values are dict with:
                - 'data': numpy array (1D or 2D) or dict of 1D arrays
                - 'ylabel': str for the Y-axis label
                - Optional styling for 1D/2D arrays:
                    - 'color', 'linestyle', 'label'
                    - OR 'colors', 'linestyles', 'labels'
                - Optional styling for dict-of-series:
                    - 'styles': { series_label: {'color','linestyle','label'} }
                - 'showlegend': bool (default True)
                - 'showgrid':  bool (default False)
        time (array-like): 1D array of time stamps (s).
        waypoints (list of dict, optional):
            If provided, and subplot title contains 'Position',
            draws horizontal lines at each wp['x'], wp['y'], wp['z'].
        ncols (int): number of columns for subplot grid (default 2).
    """
    n_plots = len(log_dict)
    nrows = (n_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = axes.flatten()

    for ax, (title, spec) in zip(axes, log_dict.items()):
        data = spec['data']
        ylabel = spec.get('ylabel', '')
        showleg = spec.get('showlegend', True)
        showgrid = spec.get('showgrid', False)
        t_series = np.asarray(spec.get('time', time))

        def _plot_series(x, y, lbl=None, color=None, ls=None):
            ax.plot(x, y, label=lbl, color=color, linestyle=ls)

        if isinstance(data, dict):
            styles = spec.get('styles', {})
            for lbl, series in data.items():
                series_time = np.asarray(styles.get(lbl, {}).get('time', t_series))
                _plot_series(series_time, series,
                             styles.get(lbl, {}).get('label', lbl),
                             styles.get(lbl, {}).get('color'),
                             styles.get(lbl, {}).get('linestyle'))
        else:
            arr = np.array(data)
            if arr.ndim == 1:
                _plot_series(t_series, arr,
                             spec.get('label', title),
                             spec.get('color'),
                             spec.get('linestyle'))
            elif arr.ndim == 2:
                n = arr.shape[1]
                colors = spec.get('colors', [None] * n)
                linestyles = spec.get('linestyles', [None] * n)
                labels = spec.get('labels', [f"{title} {i+1}" for i in range(n)])
                # Optional per-column times: 'times': list/tuple of arrays
                col_times = spec.get('times', [t_series] * n)
                for i in range(n):
                    _plot_series(np.asarray(col_times[i]), arr[:, i],
                                 labels[i], colors[i], linestyles[i])

        if waypoints and 'Position' in title:
            axis_key = title[0].lower()
            for i, wp in enumerate(waypoints):
                if axis_key in wp:
                    ax.axhline(y=wp[axis_key],
                               linestyle='--',
                               color='r',
                               label=(f"Waypoint {axis_key.upper()}" if i == 0 else None))
                    ax.text(t_series[-1], wp[axis_key], f"WP{i+1}",
                            color='r', fontsize=10, ha='right', va='center')

        if spec.get('calc_average', False):
            if isinstance(data, dict):
                vals = []
                for v in data.values():
                    v_arr = np.asarray(v)
                    if v_arr.ndim == 1:
                        vals.append(v_arr)
                    else:
                        vals.append(v_arr.reshape(-1))
                avg_val = np.mean(np.concatenate(vals)) if vals else float('nan')
            else:
                arr = np.asarray(data)
                avg_val = float(np.mean(arr))
            ax.set_title(f"{title} (Avg: {avg_val:.2f})")
        else:
            ax.set_title(title)

        ax.set_ylabel(ylabel)
        ylim = spec.get('ylim')
        if ylim:
            ax.set_ylim(ylim)
        ax.set_xlabel("Time (s)")
        if showgrid:
            ax.grid(True)
        if showleg:
            ax.legend()

    for j in range(n_plots, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    plt.savefig(f'Plots/{datetime.now().strftime("%Y%m%d_%H%M%S")}_log_data_plot.png', dpi=300)
    plt.show()


def saveLogData(positions, angles_history, rpms_history, time_history, horiz_speed_history, vertical_speed_history, spl_history, swl_history, waypoints, filename):
    """
    Save the drone simulation data to a CSV file.
    """
    import pandas as pd

    data = {
        'Time': time_history,
        'X Position': positions[:, 0],
        'Y Position': positions[:, 1],
        'Z Position': positions[:, 2],
        'Pitch': angles_history[:, 0],
        'Roll': angles_history[:, 1],
        'Yaw': angles_history[:, 2],
        'RPM1': rpms_history[:, 0],
        'RPM2': rpms_history[:, 1],
        'RPM3': rpms_history[:, 2],
        'RPM4': rpms_history[:, 3],
        'Horizontal Speed': horiz_speed_history,
        'Vertical Speed': vertical_speed_history,
        'SPL': spl_history,
        'SWL': swl_history,
        'Waypoints': [f"{wp['x']},{wp['y']},{wp['z']}" for wp in waypoints]
    }

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")


def plot3DAnimation(sim: Simulation, window=(100, 100, 100)):
    """
    Plot a 3D animation of the drone's trajectory and attitude over time.
    """
    # Extract data from the simulation object
    positions = np.array(sim.positions)
    angles_history = np.array(sim.angles_history)
    rpms_history = np.array(sim.rpms_history)
    horiz_speed_history = np.array(sim.horiz_speed_history)
    vertical_speed_history = np.array(sim.vertical_speed_history)
    targets = np.array(sim.targets)
    waypoints = sim.waypoints
    start_position = sim.drone.init_state['pos']
    dt = sim.dt
    frame_skip = sim.frame_skip

    # --- Animation: 3D Trajectory of the Drone ---
    fig_anim = plt.figure(figsize=(10, 8))
    ax_anim = fig_anim.add_subplot(111, projection='3d')
    ax_anim.set_xlim(0, window[0])
    ax_anim.set_ylim(0, window[1])
    ax_anim.set_zlim(0, window[2])
    ax_anim.set_xlabel('X')
    ax_anim.set_ylabel('Y')
    ax_anim.set_zlabel('Z')
    ax_anim.set_title('Quadcopter Animation')

    trajectory_line, = ax_anim.plot([], [], [], 'b--', lw=2, label='Trajectory')
    drone_scatter = ax_anim.scatter([], [], [], color='red', s=50, label='Drone')
    # Add a scatter for the dynamic target
    target_scatter = ax_anim.scatter([], [], [], marker='*', color='magenta', s=100, label='Target')
    
    time_text = ax_anim.text2D(0.05, 0.05, "", transform=ax_anim.transAxes, fontsize=10,
                               bbox=dict(facecolor='white', alpha=0.8))
    
    # Display the starting point
    ax_anim.scatter(start_position[0], start_position[1], start_position[2],
                    marker='o', color='green', s=100, label='Start')
    # Display waypoints
    for i, wp in enumerate(waypoints, start=1):
        ax_anim.scatter(wp['x'], wp['y'], wp['z'], marker='X', color='purple', s=100,
                        label=f'Waypoint {i}' if i == 1 else None)
        ax_anim.text(wp['x'], wp['y'], wp['z'] + 2, f'{i}', color='black', fontsize=12, ha='center')

    def init_anim():
        trajectory_line.set_data([], [])
        trajectory_line.set_3d_properties([])
        drone_scatter._offsets3d = ([], [], [])
        target_scatter._offsets3d = ([], [], [])
        time_text.set_text("")
        return trajectory_line, drone_scatter, target_scatter, time_text

    def update_anim(frame):
        nonlocal targets
        xdata = positions[:frame, 0]
        ydata = positions[:frame, 1]
        zdata = positions[:frame, 2]
        trajectory_line.set_data(xdata, ydata)
        trajectory_line.set_3d_properties(zdata)

        pos = positions[frame]
        drone_scatter._offsets3d = ([pos[0]], [pos[1]], [pos[2]])
        
        # Update the dynamic target marker
        targ = targets[frame]
        target_scatter._offsets3d = ([targ[0]], [targ[1]], [targ[2]])

        # Update the arrows indicating the drone's attitude
        for q in current_quivers:
            q.remove()
        current_quivers.clear()

        phi, theta, psi = angles_history[frame]
        R = euler_to_rot(phi, theta, psi)
        arrow_len = 4
        x_body = R @ np.array([1, 0, 0])
        y_body = R @ np.array([0, 1, 0])
        z_body = R @ np.array([0, 0, 1])
        qx = ax_anim.quiver(pos[0], pos[1], pos[2],
                            arrow_len * x_body[0], arrow_len * x_body[1], arrow_len * x_body[2],
                            color='r')
        qy = ax_anim.quiver(pos[0], pos[1], pos[2],
                            arrow_len * y_body[0], arrow_len * y_body[1], arrow_len * y_body[2],
                            color='g')
        qz = ax_anim.quiver(pos[0], pos[1], pos[2],
                            arrow_len * z_body[0], arrow_len * z_body[1], arrow_len * z_body[2],
                            color='b')
        current_quivers.extend([qx, qy, qz])

        current_time = frame * dt * frame_skip
        current_rpm = rpms_history[frame]
        text_str = (f"Time: {current_time:.2f} s\n"
                    f"RPM: [{current_rpm[0]:.2f}, {current_rpm[1]:.2f}, {current_rpm[2]:.2f}, {current_rpm[3]:.2f}]\n"
                    f"Vertical Speed (km/h): {vertical_speed_history[frame] * 3.6:.2f} km/h\n"
                    f"Horizontal Speed (km/h): {horiz_speed_history[frame] * 3.6:.2f} km/h\n"
                    f"Pitch: {angles_history[frame][0]:.4f} rad\n"
                    f"Roll: {angles_history[frame][1]:.4f} rad\n"
                    f"Yaw: {angles_history[frame][2]:.4f} rad")
        time_text.set_text(text_str)

        return trajectory_line, drone_scatter, target_scatter, time_text, *current_quivers

    # List to manage attitude arrow objects
    current_quivers = []

    ani = animation.FuncAnimation(fig_anim, update_anim, frames=len(positions),
                                  init_func=init_anim, interval=50, blit=False, repeat=True)
    plt.show()


def plotNoiseEmissionMap(sim: Simulation, window=(100, 100), upper_limit=None, param='spl', label='Noise Level (dB / s) per second'):
    """
    Plot a 2D noise emission map of the drone overlaid with its trajectory and waypoints.

    Parameters:
        sim (Simulation): Simulation object containing positions, noise map, and waypoints.
        window (tuple): (x_max, y_max) display limits.
        upper_limit (float): Optional upper limit for the noise level color scale.
        param (str): Parameter to plot from the noise emission map, e.g., 'spl' for sound pressure level, 'PA' for psychoacoustic annoyance.
    """
    noise_emission_map = sim.noise_emission_map
    if len(noise_emission_map) == 0:
        print("No noise emission data available to plot.")
        return

    # Extract noise map coordinates and SPL values
    x_coords = []
    y_coords = []
    spl_values = []
    for (x, y), value in noise_emission_map.items():
        x_coords.append(x)
        y_coords.append(y)
        spl_values.append(value[param] / sim.simulation_time)

    # Set up figure
    plt.figure(figsize=(10, 6))

    colors = ['white', 'orange', 'red', 'darkred', 'black']
    cmap = LinearSegmentedColormap.from_list(
        'white_yellow_red_violet_black', colors)

    # Plot noise emission scatter
    scatter = plt.scatter(x_coords, y_coords, c=spl_values, cmap=cmap, marker='o', alpha=0.7)
    if upper_limit is not None:
        scatter.set_clim(0, upper_limit)
    plt.colorbar(scatter, label=label)

    # Overlay drone trajectory
    positions = np.array(sim.positions)
    if positions.size > 0:
        plt.plot(positions[:, 0], positions[:, 1], '--', lw=2, color='blue', label='Trajectory')

    # Overlay waypoints
    for i, wp in enumerate(sim.waypoints, start=1):
        plt.scatter(wp['x'], wp['y'], marker='X', color='purple', s=100,
                    label='Waypoint' if i == 1 else None)
        plt.text(wp['x'], wp['y'] + 2, f'{i}', color='black', fontsize=12,
                 ha='center', va='bottom')

    # Final plot adjustments
    plt.xlim(0, window[0])
    plt.ylim(0, window[1])
    if param == 'spl':
        plt.title('Drone Noise Emission Map. Average SPL: {:.2f} dB'.format(np.mean(sim.spl_history)))
    elif param == 'PA':
        plt.title('Drone Noise Emission Map. Total Psychoacoustic Annoyance: {:.2f}'.format(get_total_PA(sim)))
    else:
        plt.title('Drone Noise Emission Map')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'Plots/{datetime.now().strftime("%Y%m%d_%H%M%S")}_noise_emission_map_{param}.png', dpi=300)
    plt.show()


def plotNoiseEmissionHistogram(sims: list, bins=50, upper_limit=None, param='spl', label='Noise Level (dB / s) per second'):
    """
    Plot a histogram of the noise emission levels recorded during multiple simulations.
    The histogram shows in different colors the distribution of sound pressure levels (SPL) across all simulations.
    The bars of different simulations are stacked side by side for comparison.

    Parameters:
        sims (list[Simulation]): List of Simulation objects (or a single simulation) containing noise emission data.
        bins (int): Number of bins for the histogram.
        upper_limit (float): Optional upper limit for the histogram x-axis.
        param (str): Parameter to plot from the noise emission map, e.g., 'spl' for sound pressure level.
    """
    if type(sims) is Simulation:
        sims = [sims]
    # Extract SPL values from each simulation's noise emission map
    all_spl_data = []
    for idx, sim in enumerate(sims):
        spl_vals = [v[param] / sim.simulation_time for v in sim.noise_emission_map.values()]
        all_spl_data.append(spl_vals)

    # Check for data
    if all(len(data) == 0 for data in all_spl_data):
        print("No noise emission data available for histogram.")
        return

    # Determine common bin edges
    combined = np.hstack([np.array(data) for data in all_spl_data if len(data) > 0])
    if upper_limit is not None:
        combined = combined[combined <= upper_limit]
    bin_edges = np.linspace(combined.min(), combined.max(), bins + 1)

    # Plot bars side by side for each simulation
    plt.figure(figsize=(10, 6))
    width = (bin_edges[1] - bin_edges[0]) / (len(sims) + 1)

    for i, spl_data in enumerate(all_spl_data):
        # Compute histogram counts
        counts, _ = np.histogram(spl_data, bins=bin_edges)
        # Compute bar centers
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        # Offset for side-by-side bars
        offset = (i - (len(sims) - 1) / 2) * width
        plt.bar(centers + offset, counts, width=width, alpha=0.7, label=f'Simulation {i + 1}')

    # Final plot adjustments
    plt.xlabel(label)
    plt.ylabel('Count')
    if upper_limit is not None:
        plt.xlim(0, upper_limit)
    plt.title('Noise Emission Histogram Across Simulations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'Plots/{datetime.now().strftime("%Y%m%d_%H%M%S")}_noise_emission_histogram_{param}.png', dpi=300)
    plt.show()

def get_total_PA(sim: Simulation) -> float:
    """
    Calculate the total Psychoacoustic Annoyance (PA) from the simulation's noise emission map.
    
    Parameters:
        sim (Simulation): Simulation object containing noise emission data.
    
    Returns:
        float: Total PA across all noise emission points.
    """
    total_PA = 0.0
    for value in sim.noise_emission_map.values():
        total_PA += value.get('PA', 0.0) / sim.simulation_time
    return total_PA

def show2DWorld(world: World, trajectory=None, A=None, B=None, targets=None,
                image_alpha=0.7, save=False, save_folder="OptimizedTrajectory",
                horiz_speed_history=None, vertical_speed_history=None):
    """
    Visualize a 2D representation of the world with one drone:
    - trajectory: list of (x, y) or array of shape [N, 2]
    - A: dict {'x': ..., 'y': ...} or tuple (x, y)
    - B: dict {'x': ..., 'y': ...} or tuple (x, y)
    - targets: list of dicts or tuples, each as (x, y) or {'x': ..., 'y': ...}

    Parameters:
        world (World): World object containing grid and background image.
        trajectory (list or np.ndarray): Drone trajectory points.
        A (dict or tuple): Start point.
        B (dict or tuple): End point.
        targets (list): List of target points.
        image_alpha (float): Transparency for background image.
        save (bool): Whether to save the plot as a PNG file.
        save_folder (str): Folder to save the plot if save is True.
        horiz_speed_history (list): History of horizontal speeds.
        vertical_speed_history (list): History of vertical speeds.
    """
    speed_magnitudes = None
    if all([
            vertical_speed_history,
            horiz_speed_history,
            len(vertical_speed_history) == len(horiz_speed_history) == len(trajectory)
        ]):
        raw_speed_magnitudes = np.sqrt(np.array(vertical_speed_history)**2 + np.array(horiz_speed_history)**2)
        # Normalize speeds for color mapping
        speed_magnitudes = (raw_speed_magnitudes) / (raw_speed_magnitudes.max() + 1e-6)

    fig, ax = plt.subplots(figsize=(10, 6))
    grid_size = world.grid_size

    # Background image
    if getattr(world, "background_image", None) is not None:
        bg_img = np.array(world.background_image)
        ax.imshow(
            bg_img,
            extent=[0, world.max_world_size, 0, world.max_world_size],
            origin="lower",
            alpha=image_alpha,
            zorder=-1,
        )

    # Grid
    for (x, y, z), params in world.grid.items():
        if z == 0:
            rect = plt.Rectangle(
                (x * grid_size, y * grid_size),
                grid_size,
                grid_size,
                color=world.AREA_PARAMS[params]["color"],
                alpha=world.AREA_PARAMS[params]["alpha"],
            )
            ax.add_patch(rect)

    # Helper to normalize point input
    def as_xy(p):
        if p is None:
            return None
        if isinstance(p, dict):
            return float(p["x"]), float(p["y"])
        return float(p[0]), float(p[1])

    plotted_anything = False
    color = "blue"

    offset = 10 * world.max_world_size * grid_size / 1000  # 1% offset for text

    # Start point
    A_xy = as_xy(A)
    if A_xy is not None:
        ax.scatter(A_xy[0], A_xy[1], color="red", s=60, marker="o", label="Start A")
        ax.text(A_xy[0] + offset, A_xy[1] + offset, "A", color="red", fontsize=10)
        plotted_anything = True

    # End point
    B_xy = as_xy(B)
    if B_xy is not None:
        ax.scatter(B_xy[0], B_xy[1], color="green", s=60, marker="o", label="Goal B")
        ax.text(B_xy[0] + offset, B_xy[1] + offset, "B", color="green", fontsize=10)
        plotted_anything = True

    # Targets
    if targets:
        pts = np.array([as_xy(t) for t in targets if as_xy(t) is not None], dtype=float)
        if pts.size > 0:
            ax.scatter(pts[:, 0], pts[:, 1], color=color, s=30, label="Targets")
            for j, pt in enumerate(pts, 1):
                ax.text(pt[0] + offset, pt[1] + offset, f"{j}", color=color, fontsize=8)
            plotted_anything = True

    # Trajectory
    if trajectory is not None:
        traj_arr = np.asarray(trajectory, dtype=float)
        if traj_arr.ndim == 2 and traj_arr.shape[0] > 1:
            xy = traj_arr[:, :2]  # keep X,Y only

            if speed_magnitudes is None:
                ax.plot(xy[:, 0], xy[:, 1], linestyle="--", lw=1.5,
                        color="blue", label="Trajectory")
            else:
                seg_speeds = 0.5 * (speed_magnitudes[:-1] + speed_magnitudes[1:])
                segments = np.stack([xy[:-1], xy[1:]], axis=1)  # (M, 2, 2)

                cmap = LinearSegmentedColormap.from_list(
                    "speedmap", ["green", "yellow", "orange", "red", "violet"], N=256
                )
                vmin, vmax = float(seg_speeds.min()), float(seg_speeds.max())
                if vmax == vmin:
                    vmax = vmin + 1.0
                norm = Normalize(vmin=vmin, vmax=vmax)

                lc = LineCollection(segments, cmap=cmap, norm=norm,
                                    linestyles="dashed", linewidth=1.5)
                lc.set_array(seg_speeds)
                ax.add_collection(lc)

                ax.add_line(Line2D([0], [0], linestyle="--", color="k",
                                label="Trajectory (speed)"))

            plotted_anything = True

    cbar = fig.colorbar(lc, ax=ax, orientation="vertical")
    cbar.set_label("Drone speed [m/s]")
    # Label ticks with physical (non-normalized) values
    cbar.set_ticks(np.linspace(0, 1, 5))
    cbar.set_ticklabels([
        f"{v:.2f}" for v in np.linspace(
            raw_speed_magnitudes.min(),
            raw_speed_magnitudes.max(),
            5
        )
    ])

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"2D World '{world.world_name}' XY")
    ax.set_xlim(0, world.max_world_size * grid_size)
    ax.set_ylim(0, world.max_world_size * grid_size)
    ax.grid(True)

    if plotted_anything:
        # Set legend location to downright if anything is plotted
        ax.legend(loc="lower right")

    if save:
        plt.savefig(f"{save_folder}/trajectory_{world.world_name}.png", dpi=300, bbox_inches="tight")
    plt.show()


from typing import Iterable, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

def show3DWorld(world,
                trajectory: Optional[Iterable]=None,
                A=None, B=None, targets=None,
                image_alpha: float=0.7,  # kept for API parity, unused in 3D
                save: bool=False, save_folder: str="OptimizedTrajectory",
                horiz_speed_history: Optional[Iterable]=None,
                vertical_speed_history: Optional[Iterable]=None,
                view_pov: tuple=(30, 45)):
    """
    3D view with XY ground grid colored by area params and a speed-colored dashed trajectory.

    Inputs:
      - trajectory: list/array of shape [N, 3] (x, y, z).
      - A, B: dict {'x','y','z'} or tuple (x, y, z). If z missing, defaults to 0.
      - targets: list of dicts/tuples. If z missing, defaults to 0.
      - horiz_speed_history: horizontal speed magnitude per step (m/s).
      - vertical_speed_history: vertical speed magnitude per step (m/s).
      - view_pov: tuple (elevation, azimuth) angles for 3D view.
    """
    # --- speeds
    raw_speed_magnitudes = None
    speed_magnitudes = None
    if (horiz_speed_history is not None and vertical_speed_history is not None
        and trajectory is not None):
        n = len(trajectory)
        if len(horiz_speed_history) == n and len(vertical_speed_history) == n and n > 1:
            h = np.asarray(horiz_speed_history, dtype=float)
            v = np.asarray(vertical_speed_history, dtype=float)
            raw_speed_magnitudes = np.sqrt(h*h + v*v)  # physical units
            denom = float(raw_speed_magnitudes.max()) + 1e-6
            speed_magnitudes = raw_speed_magnitudes / denom  # [0,1]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d", elev=view_pov[0], azim=view_pov[1])
    grid_size = world.grid_size
    L = world.max_world_size * grid_size

    # --- XY ground cells colored by area
    # draw filled quads at z=0 for each grid cell having z==0 in world.grid
    ground_quads = []
    face_colors = []
    face_alphas = []
    for (gx, gy, gz), params in world.grid.items():
        if gz != 0:
            continue
        x0, y0 = gx * grid_size, gy * grid_size
        x1, y1 = x0 + grid_size, y0 + grid_size
        quad = [(x0, y0, 0.0), (x1, y0, 0.0), (x1, y1, 0.0), (x0, y1, 0.0)]
        ground_quads.append(quad)
        face_colors.append(world.AREA_PARAMS[params]["color"])
        face_alphas.append(world.AREA_PARAMS[params]["alpha"])
    if ground_quads:
        pc = Poly3DCollection(ground_quads, facecolors=face_colors, linewidths=0.2, edgecolors="k")
        pc.set_alpha(None)  # per-face alpha handled below
        # Matplotlib does not support per-face alpha directly on Poly3DCollection.
        # Workaround: set a single alpha and encode translucency into colors if needed.
        # If AREA_PARAMS already uses RGBA, prefer that. Otherwise use average alpha:
        if isinstance(face_colors[0], (tuple, list)) and len(face_colors[0]) == 4:
            pass  # RGBA provided
        else:
            pc.set_alpha(float(np.mean(face_alphas)) if face_alphas else 0.5)
        ax.add_collection3d(pc)

    # --- helpers
    def as_xyz(p):
        if p is None:
            return None
        if isinstance(p, dict):
            x = float(p["x"])
            y = float(p["y"])
            z = float(p.get("z", 0.0))
            return x, y, z
        # tuple/list/np array
        if len(p) == 2:
            return float(p[0]), float(p[1]), 0.0
        return float(p[0]), float(p[1]), float(p[2])

    plotted_anything = False
    offset = 0.01 * L

    # --- Start point
    A_xyz = as_xyz(A)
    if A_xyz is not None:
        ax.scatter([A_xyz[0]], [A_xyz[1]], [A_xyz[2]], color="red", s=40, marker="o", label="Start A")
        ax.text(A_xyz[0] + offset, A_xyz[1] + offset, A_xyz[2] + offset, "A", color="red", fontsize=9)
        plotted_anything = True

    # --- End point
    B_xyz = as_xyz(B)
    if B_xyz is not None:
        ax.scatter([B_xyz[0]], [B_xyz[1]], [B_xyz[2]], color="green", s=40, marker="o", label="Goal B")
        ax.text(B_xyz[0] + offset, B_xyz[1] + offset, B_xyz[2] + offset, "B", color="green", fontsize=9)
        plotted_anything = True

    # --- Targets
    if targets:
        pts = [as_xyz(t) for t in targets if as_xyz(t) is not None]
        if len(pts):
            pts = np.asarray(pts, dtype=float)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color="blue", s=20, label="Targets")
            for j, pt in enumerate(pts, 1):
                ax.text(pt[0] + offset, pt[1] + offset, pt[2] + offset, f"{j}", color="blue", fontsize=8)
            plotted_anything = True

    # --- Trajectory (speed-colored dashed 3D line)
    lc = None
    if trajectory is not None:
        traj = np.asarray(trajectory, dtype=float)
        if traj.ndim == 2 and traj.shape[0] > 1:
            # segments (M, 2, 3)
            segments3d = np.stack([traj[:-1], traj[1:]], axis=1)
            if speed_magnitudes is None:
                # fallback dashed line
                ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                        linestyle="--", linewidth=1.5, color="blue", label="Trajectory")
            else:
                seg_speeds = 0.5 * (speed_magnitudes[:-1] + speed_magnitudes[1:])
                cmap = LinearSegmentedColormap.from_list(
                    "speedmap", ["green", "yellow", "orange", "red", "violet"], N=256
                )
                vmin, vmax = float(seg_speeds.min()), float(seg_speeds.max())
                if vmax == vmin:
                    vmax = vmin + 1.0
                norm = Normalize(vmin=vmin, vmax=vmax)
                lc = Line3DCollection(segments3d, cmap=cmap, norm=norm,
                                      linestyles="dashed", linewidth=1.5)
                lc.set_array(seg_speeds)
                ax.add_collection3d(lc)
                # legend proxy
                ax.add_line(Line2D([0], [0], linestyle="--", color="k", label="Trajectory (speed)"))

            ax.plot(
                traj[:, 0], traj[:, 1], np.zeros_like(traj[:, 0]),
                linestyle="--", linewidth=1.5, color="black", alpha=0.8,
                label="XY projection"
            )
            plotted_anything = True

    # --- Colorbar with physical units (if speeds available)
    if lc is not None and raw_speed_magnitudes is not None:
        cbar = fig.colorbar(lc, ax=ax, orientation="vertical", pad=0.07, shrink=0.8)
        cbar.set_label("Drone speed [m/s]")
        cbar.set_ticks(np.linspace(0, 1, 5))
        cbar.set_ticklabels([
            f"{v:.2f}" for v in np.linspace(raw_speed_magnitudes.min(),
                                            raw_speed_magnitudes.max(), 5)
        ])

    # --- axes, limits, grid
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"3D World '{world.world_name}'")

    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    # Z range from data if available, else [0, L/4]
    if trajectory is not None and len(np.asarray(trajectory)) > 0 and np.asarray(trajectory).ndim == 2:
        zmin = float(np.nanmin(np.asarray(trajectory)[:, 2]))
        zmax = float(np.nanmax(np.asarray(trajectory)[:, 2]))
        if not np.isfinite(zmin) or not np.isfinite(zmax) or zmax <= zmin:
            ax.set_zlim(0, max(L * 0.25, 1.0))
        else:
            pad = 0.05 * max(1.0, zmax - zmin)
            ax.set_zlim(max(0.0, zmin - pad), zmax + pad)
    else:
        ax.set_zlim(0, max(L * 0.25, 1.0))

    if plotted_anything:
        ax.legend(loc="upper left")

    if save:
        out = f"{save_folder}/trajectory3D_{world.world_name}.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()
