import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from Worlds.World import World

# ----------------- controls -----------------
mode = "3d"           # "2d", "3d", or "both"
fps  = 550            # frames per second
save = False          # True to save mp4/gif
out_prefix = "trajectories"
frame_skip = 1
# --------------------------------------------

log_file_path   = r"RL\TD3\20251128_151725\optimization_history_log.json"
world_file_path = "Worlds/training_world.pkl"

interval_ms = max(1, int(1000 // max(1, fps)))

world = World.load_world(world_file_path)
g = world.grid_size
W = world.max_world_size * g

# ---- load trajectories ----
params_all = []
with open(log_file_path, "r") as f:
    for line in f:
        s = line.strip()
        if not s:
            continue
        rec = json.loads(s)
        rec["params"].insert(0, {"x": 5, "y": 50, "z": 1, "v": 0})
        params_all.append(rec["params"])

traj_xyz = [np.array([(p["x"], p["y"], p["z"]) for p in t], float) for t in params_all if t]
if not traj_xyz:
    raise ValueError("No trajectories found.")

# frame skipping (1 = no skip)

frame_indices = range(0, len(traj_xyz), max(1, int(frame_skip)))

xs = np.concatenate([t[:, 0] for t in traj_xyz])
ys = np.concatenate([t[:, 1] for t in traj_xyz])
zs = np.concatenate([t[:, 2] for t in traj_xyz])
pad_xy = 0.02 * max(W, 1.0)
pad_z  = 0.05 * max(zs.max() - zs.min(), 1.0)
xlim = (max(0, xs.min()-pad_xy), min(W, xs.max()+pad_xy))
ylim = (max(0, ys.min()-pad_xy), min(W, ys.max()+pad_xy))
zlim = (0, max(1.0, zs.max()+pad_z))

# ------------ helpers ------------
def draw_world_top(ax, image_alpha=0.5):
    if getattr(world, "background_image", None) is not None:
        bg_img = np.array(world.background_image)
        ax.imshow(
            bg_img,
            extent=[0, world.max_world_size, 0, world.max_world_size],
            origin="lower",
            alpha=image_alpha,
            zorder=-1,
        )
    for (gx, gy, gz), area_id in world.grid.items():
        if gz != 0:
            continue
        rect = plt.Rectangle(
            (gx * g, gy * g), g, g,
            color=world.AREA_PARAMS[area_id]["color"],
            alpha=world.AREA_PARAMS[area_id]["alpha"],
            zorder=0
        )
        ax.add_patch(rect)
    ax.set_xlim(0, W)
    ax.set_ylim(0, W)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

def draw_world_ground_3d(ax3d):
    quads = []
    colors = []
    for (gx, gy, gz), area_id in world.grid.items():
        if gz != 0:
            continue
        x0, y0 = gx * g, gy * g
        quads.append([(x0, y0, 0), (x0+g, y0, 0), (x0+g, y0+g, 0), (x0, y0+g, 0)])
        colors.append(world.AREA_PARAMS[area_id]["color"])
    if quads:
        coll = Poly3DCollection(quads, facecolors=colors, edgecolors='none', alpha=0.25)
        ax3d.add_collection3d(coll)
    ax3d.set_xlim(0, W)
    ax3d.set_ylim(0, W)
    ax3d.set_zlim(zlim)
    ax3d.set_box_aspect((W, W, zlim[1]))
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")

# ------------ 2D animation ------------
ani2d = ani3d = None

if mode in ("2d", "both"):
    fig2d, ax2d = plt.subplots(figsize=(7, 7))
    draw_world_top(ax2d)
    line2d, = ax2d.plot([], [], lw=2, zorder=5)
    inter2d, = ax2d.plot([], [], "x", color="black", ms=5, alpha=0.6, zorder=6)
    start2d, = ax2d.plot([], [], "o", color="blue", ms=6, zorder=7)
    end2d,   = ax2d.plot([], [], "s", color="red", ms=5, zorder=7)

    def init2d():
        for artist in (line2d, inter2d, start2d, end2d):
            artist.set_data([], [])
        return line2d, inter2d, start2d, end2d

    def update2d(i):
        xyz = traj_xyz[i]
        ax2d.set_title(f"Top view — frame {i+1}/{len(traj_xyz)}")
        x, y = xyz[:, 0], xyz[:, 1]
        line2d.set_data(x, y)
        inter2d.set_data(x[1:-1], y[1:-1])  # intermediate points
        start2d.set_data(x[0], y[0])
        end2d.set_data(x[-1], y[-1])
        return line2d, inter2d, start2d, end2d

    ani2d = FuncAnimation(fig2d, update2d, frames=frame_indices,
                          init_func=init2d, interval=interval_ms, blit=True, repeat=True)

# ------------ 3D animation ------------
if mode in ("3d", "both"):
    fig3d = plt.figure(figsize=(8, 7))
    ax3d = fig3d.add_subplot(111, projection='3d')
    draw_world_ground_3d(ax3d)
    line3d, = ax3d.plot([], [], [], lw=2)
    inter3d, = ax3d.plot([], [], [], "x", color="black", alpha=0.6)
    proj2d, = ax3d.plot([], [], [], linestyle="--", alpha=0.8)
    start3d, = ax3d.plot([], [], [], "o", color="blue", ms=6)
    end3d,   = ax3d.plot([], [], [], "s", color="red", ms=5)

    def init3d():
        for artist in (line3d, inter3d, proj2d, start3d, end3d):
            artist.set_data([], [])
            artist.set_3d_properties([])
        return line3d, inter3d, proj2d, start3d, end3d

    def update3d(i):
        xyz = traj_xyz[i]
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        ax3d.set_title(f"3D view — frame {i+1}/{len(traj_xyz)}")
        line3d.set_data(x, y); line3d.set_3d_properties(z)
        inter3d.set_data(x[1:-1], y[1:-1]); inter3d.set_3d_properties(z[1:-1])
        proj2d.set_data(x, y); proj2d.set_3d_properties(np.zeros_like(z))
        start3d.set_data([x[0]], [y[0]]); start3d.set_3d_properties([z[0]])
        end3d.set_data([x[-1]], [y[-1]]); end3d.set_3d_properties([z[-1]])
        return line3d, inter3d, proj2d, start3d, end3d

    ani3d = FuncAnimation(fig3d, update3d, frames=frame_indices,
                          init_func=init3d, interval=interval_ms, blit=False, repeat=True)

# ------------ display/save ------------
if save:
    if ani2d is not None:
        ani2d.save(f"{out_prefix}_top.mp4", writer="ffmpeg", fps=fps)
    if ani3d is not None:
        ani3d.save(f"{out_prefix}_3d.mp4", writer="ffmpeg", fps=fps)

plt.show()
