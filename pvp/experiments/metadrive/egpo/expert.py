# mpc_expert.py
from dataclasses import dataclass
import numpy as np
import casadi as cs  # kept for compatibility with old code that imports this

import os
from typing import Optional

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


@dataclass
class MPCConfig:
    # Time step used by the controller (should match env dt)
    dt: float = 0.1

    # Speed limits & reference
    v_min: float = 0.0
    v_max: float = 22.2  # ~80 km/h
    v_ref: float = 5.0   # nominal target speed in m/s

    # Vehicle wheelbase (kept for compatibility, not strictly needed)
    L: float = 1.05234 + 1.4166

    # --- PID gains ---
    # Longitudinal (speed) PID: pedal_norm in [-1, 1]
    kp_v: float = 0.1
    ki_v: float = 0.0
    kd_v: float = 0.05

    # Lateral PID on heading-to-target: steering_norm in [-1, 1]
    kp_steer: float = 2.0
    ki_steer: float = 0.0
    kd_steer: float = 0.001

    # Integral windup limits
    int_v_limit: float = 20.0
    int_steer_limit: float = 5.0

    # --- Occupancy grid parameters (local frame) ---
    # Grid spans x in [0, grid_x_max] forward, y in [-grid_y_max, grid_y_max] lateral
    grid_x_max: float = 30.0   # meters forward from ego
    grid_y_max: float = 10.0   # meters left/right from ego
    grid_resolution: float = 0.05  # meters per cell

    # Extra safety margin around vehicle for inflation [m]
    obstacle_margin: float = 0.5

    # Lane-keeping margin [m] inside lane edges
    lane_keep_margin: float = 2

    # --- RRT* parameters ---
    rrt_max_iters: int = 200         # iterations per planning step
    rrt_step_size: float = 2.0        # step length when extending tree [m]
    rrt_goal_stop_searching_radius: float = 5      

    rrt_goal_radius: float = 1      # how close to goal to consider success [m]
    rrt_goal_sample_prob: float = 0.01 # probability of sampling around goal
    rrt_star_radius: float = 3.0      # neighbor radius for RRT* rewiring [m]

    # Distance along the path to pick tracking point [m]
    path_lookahead: float = 2.0

    # Path smoothing
    path_smooth_iters: int = 5
    path_smooth_alpha: float = 0.5

    # Minimum distance at which we start slowing down based on LIDAR [m]
    slowdown_dist: float = 0.0

    # Minimal speed (so the car doesn’t completely stall unless braking) [m/s]
    v_min_running: float = 0.0

    # --- Steering smoothing ---
    # Exponential smoothing factor on steering command (0..1); higher = less smoothing
    steer_smooth_alpha: float = 0.3

    # Max change in steering_norm per step (rate limit)
    max_steer_delta: float = 0.35


class LinearizedKinematicMPC:
    """
    RRT* (RRT-star) local planner on a 2D occupancy grid built from LIDAR,
    with lane boundaries encoded as obstacles via vehicle._dist_to_route_left_right(),
    plus PID tracking and smoothed steering.

    Interface remains:

        from mpc_expert import LinearizedKinematicMPC, MPCConfig

    Output:
        np.array([steering_norm, pedal_norm]) in [-1, 1]^2
    """

    def __init__(self, cfg: MPCConfig):
        self.cfg = cfg
        
        self._last_target_global = None 

        # PID internal states
        self._prev_speed_error = 0.0
        self._int_speed_error = 0.0

        self._prev_heading_error = 0.0
        self._int_heading_error = 0.0

        # For steering smoothing
        self._prev_steering_norm = 0.0

        # Debug visualization storage (for last planning step)
        self._last_grid = None
        self._last_nodes = None
        self._last_parents = None
        self._last_path = None
        self._last_goal_vec = None

        # Debug plotting state
        self._debug_step = 0
        self._debug_dir = "rrt_save"

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    @staticmethod
    def _wrap_angle(angle_rad: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return (angle_rad + np.pi) % (2.0 * np.pi) - np.pi

    def reset(self):
        """Optional: call this when an episode resets."""
        self._prev_speed_error = 0.0
        self._int_speed_error = 0.0
        self._prev_heading_error = 0.0
        self._int_heading_error = 0.0
        self._prev_steering_norm = 0.0

        self._last_grid = None
        self._last_nodes = None
        self._last_parents = None
        self._last_path = None
        self._last_goal_vec = None
        self._debug_step = 0

    # ---------------------- LIDAR ----------------------
    def _get_lidar_scan(self, vehicle):
        """
        Get raw LIDAR scan from MetaDrive vehicle.

        Returns:
            cloud_points: list of length num_lasers, normalized distances in [0, 1].
            detected_objects: list of objects hit (MetaDrive objects).
        """
        sensor = vehicle.engine.get_sensor("lidar")
        cloud_points, detected_objects = sensor.perceive(
            vehicle,
            physics_world=vehicle.engine.physics_world.dynamic_world,
            num_lasers=vehicle.config["lidar"]["num_lasers"],
            distance=vehicle.config["lidar"]["distance"],  # max LIDAR range (e.g. 50 m)
            show=vehicle.config["show_lidar"],
        )
        return cloud_points, detected_objects

    # ------------------ Occupancy grid ------------------
    def _get_inflation_radius(self, vehicle):
        """
        Effective collision radius around ego: vehicle size + safety margin.
        """
        length = getattr(vehicle, "LENGTH", 4.5)
        width = getattr(vehicle, "WIDTH", 2.0)
        veh_radius = 0.5 * np.sqrt(length ** 2 + width ** 2)
        return float(veh_radius + self.cfg.obstacle_margin)

    def _local_to_grid(self, x_local, y_local):
        """
        Map local coordinates (x,y) to occupancy grid indices (ix, iy).

        Local frame:
          - x forward, in [0, grid_x_max]
          - y lateral, in [-grid_y_max, grid_y_max]

        Grid:
          - ix in [0, nx-1] corresponds to x in [0, grid_x_max]
          - iy in [0, ny-1] corresponds to y in [-grid_y_max, grid_y_max]
        """
        cfg = self.cfg
        res = cfg.grid_resolution
        X_MAX = cfg.grid_x_max
        Y_MAX = cfg.grid_y_max

        if x_local < 0.0 or x_local >= X_MAX:
            return None
        if y_local < -Y_MAX or y_local >= Y_MAX:
            return None

        ix = int(x_local / res)
        iy = int((y_local + Y_MAX) / res)

        return ix, iy

    def _build_occupancy_grid(self, vehicle):
        """
        Build a 2D occupancy grid from LIDAR in ego local frame,
        and encode lane boundaries as obstacles using
        vehicle._dist_to_route_left_right().

        Returns:
            grid: 2D bool array, True = occupied (inflated + lane bounds)
            min_obstacle_dist: minimal distance to any LIDAR obstacle (meters)
        """
        cfg = self.cfg
        res = cfg.grid_resolution
        X_MAX = cfg.grid_x_max
        Y_MAX = cfg.grid_y_max

        num_lasers = vehicle.config["lidar"]["num_lasers"]
        max_range = vehicle.config["lidar"]["distance"]
        nx = int(np.ceil(X_MAX / res))
        ny = int(np.ceil(2 * Y_MAX / res))
        grid = np.zeros((nx, ny), dtype=bool)

        if num_lasers <= 0 or max_range <= 0:
            min_dist = None
        else:
            cloud_points, _ = self._get_lidar_scan(vehicle)
            min_dist = None

            # Rasterize raw obstacle points into grid (un-inflated)
            for i, d_norm in enumerate(cloud_points):
                # d_norm in [0,1]; treat near 1 as "no hit"
                if d_norm >= 0.99:
                    continue

                r = float(d_norm) * float(max_range)
                if r <= 0.0:
                    continue

                # Angle: starting from vehicle head, clockwise
                angle = 2.0 * np.pi * float(i) / float(num_lasers)
                x_local = r * np.cos(angle)
                y_local = r * np.sin(angle)

                if min_dist is None or r < min_dist:
                    min_dist = r

                idx = self._local_to_grid(x_local, y_local)
                if idx is None:
                    continue
                ix, iy = idx
                grid[ix, iy] = True

            # Inflate obstacles by vehicle radius + margin
            inflate_r = self._get_inflation_radius(vehicle)
            if inflate_r > 0.0:
                cell_radius = int(np.ceil(inflate_r / res))
                inflated = grid.copy()
                occ_cells = np.argwhere(grid)
                for ix, iy in occ_cells:
                    for dx in range(-cell_radius, cell_radius + 1):
                        for dy in range(-cell_radius, cell_radius + 1):
                            jx = ix + dx
                            jy = iy + dy
                            if jx < 0 or jx >= nx or jy < 0 or jy >= ny:
                                continue
                            # approximate circular inflation
                            if dx * dx + dy * dy <= cell_radius * cell_radius:
                                inflated[jx, jy] = True
                grid = inflated

        # ---- Lane boundaries as obstacles via _dist_to_route_left_right ----
        lane_margin = cfg.lane_keep_margin
        try:
            dist_left, dist_right = vehicle._dist_to_route_left_right()
            dist_left = float(dist_left)
            dist_right = float(dist_right)
        except Exception:
            dist_left = None
            dist_right = None

        if dist_left is not None and dist_right is not None:
            # Allowed lateral corridor (in ego frame, y):
            #   y in [ right_limit, left_limit ]
            #
            # where lane edges are at:
            #   y = +dist_left       (left edge)
            #   y = -dist_right      (right edge)
            #
            # We shrink this by lane_margin to keep a buffer from edges.
            left_limit = dist_left - lane_margin
            right_limit = -dist_right + lane_margin

            # Clamp to grid extents
            left_limit = np.clip(left_limit, 0.0, Y_MAX)
            right_limit = np.clip(right_limit, -Y_MAX, 0.0)

            # Mark all cells outside [right_limit, left_limit] as occupied
            for ix in range(nx):
                for iy in range(ny):
                    # cell center y coordinate
                    y_center = (iy + 0.5) * res - Y_MAX
                    if y_center > left_limit or y_center < right_limit:
                        grid[ix, iy] = True

        return grid, (float(min_dist) if min_dist is not None else None)
    
    
    def _local_to_global(self, vehicle, x_local, y_local):
        """
        Transform a local point (x_local, y_local) in the ego frame
        back to global coordinates.

        Local frame:
        - origin at vehicle.position
        - x-axis along vehicle.heading
        """
        import numpy as np

        pos = np.array(vehicle.position, dtype=np.float64)
        heading_vec = np.array(vehicle.heading, dtype=np.float64)
        heading_rad = np.arctan2(heading_vec[1], heading_vec[0])

        c = np.cos(heading_rad)
        s = np.sin(heading_rad)

        x_global = pos[0] + c * x_local - s * y_local
        y_global = pos[1] + s * x_local + c * y_local
        return x_global, y_global







    def _point_in_collision_grid(self, p, grid):
        """
        Check if a local point p = (x,y) is in collision (occupied or out of bounds).
        """
        cfg = self.cfg
        X_MAX = cfg.grid_x_max
        Y_MAX = cfg.grid_y_max

        x_local, y_local = float(p[0]), float(p[1])

        # Outside grid region -> treat as collision
        if x_local < 0.0 or x_local >= X_MAX or y_local < -Y_MAX or y_local >= Y_MAX:
            return True

        idx = self._local_to_grid(x_local, y_local)
        if idx is None:
            return True

        ix, iy = idx
        nx, ny = grid.shape
        if ix < 0 or ix >= nx or iy < 0 or iy >= ny:
            return True

        return bool(grid[ix, iy])

    def _segment_in_collision_grid(self, p0, p1, grid):
        """
        Check collision of a segment [p0, p1] against occupancy grid.

        Strategy:
          - Sample along the segment with roughly cell-sized steps.
          - Any sample that lies in an occupied cell or outside grid => collision.
        """
        cfg = self.cfg
        res = cfg.grid_resolution

        p0 = np.asarray(p0, dtype=np.float64)
        p1 = np.asarray(p1, dtype=np.float64)
        seg = p1 - p0
        length = float(np.linalg.norm(seg))
        if length < 1e-6:
            return self._point_in_collision_grid(p0, grid)

        # Step size ~ half a cell
        step = res * 0.5
        num_steps = int(np.ceil(length / step))

        for i in range(num_steps + 1):
            t = i / max(1, num_steps)
            p = p0 + t * seg
            if self._point_in_collision_grid(p, grid):
                return True
        return False

    # ------------------ Frames & goal ------------------
    def _global_to_local(self, vehicle, x_global, y_global):
        """
        Transform a global point (x_global, y_global) into the ego local frame.
        Local frame:
          - origin at vehicle.position
          - x-axis along vehicle.heading
        """
        pos = np.array(vehicle.position, dtype=np.float64)
        heading_vec = np.array(vehicle.heading, dtype=np.float64)
        heading_rad = np.arctan2(heading_vec[1], heading_vec[0])

        dx = x_global - pos[0]
        dy = y_global - pos[1]

        c = np.cos(-heading_rad)
        s = np.sin(-heading_rad)
        x_local = c * dx - s * dy
        y_local = s * dx + c * dy
        return x_local, y_local

    def _compute_local_goal(self, vehicle):
        """
        Choose a local goal for the planner based on navigation checkpoint,
        and clamp it into the occupancy grid region.
        """
        cfg = self.cfg

        if vehicle.navigation is not None and vehicle.navigation.get_checkpoints():
            x_goal_g, y_goal_g = vehicle.navigation.get_checkpoints()[0]
            gx_local, gy_local = self._global_to_local(vehicle, x_goal_g, y_goal_g)
        else:
            gx_local, gy_local = cfg.grid_x_max * 0.8, 0.0

        # Clamp into grid region
        X_MAX = cfg.grid_x_max
        Y_MAX = cfg.grid_y_max
        gx_local = np.clip(gx_local, 0.0, X_MAX * 0.95)
        gy_local = np.clip(gy_local, -Y_MAX * 0.95, Y_MAX * 0.95)

        return np.array([gx_local, gy_local], dtype=np.float64)

    # ------------------- Path smoothing -------------------
    def _smooth_path(self, path):
        """
        Simple iterative smoothing of a path (list of np.array points).
        Keeps endpoints fixed, nudges interior points towards neighbors.
        """
        cfg = self.cfg
        if path is None or len(path) <= 2:
            return path
        
        return path

        pts = [np.array(p, dtype=np.float64) for p in path]
        alpha = cfg.path_smooth_alpha
        n = len(pts)

        for _ in range(cfg.path_smooth_iters):
            for i in range(1, n - 1):
                p_prev = pts[i - 1]
                p_next = pts[i + 1]
                p = pts[i]
                midpoint = 0.5 * (p_prev + p_next)
                pts[i] = p + alpha * (midpoint - p)

        return pts

    # ----------------------- RRT* core -----------------------
    # def _sample_point(self, goal_vec):
    #     """
    #     Sample a point in local frame for RRT*.
    #     With some probability, sample near goal (goal bias).
    #     """
    #     cfg = self.cfg
    #     if np.random.rand() < cfg.rrt_goal_sample_prob:
    #         noise = np.random.normal(loc=0.0, scale=0.5, size=2)
    #         return goal_vec + noise

    #     X_MAX = cfg.grid_x_max
    #     Y_MAX = cfg.grid_y_max
    #     x = np.random.uniform(0.0, X_MAX)
    #     y = np.random.uniform(-Y_MAX, Y_MAX)
    #     return np.array([x, y], dtype=np.float64)

    def _sample_point(self, goal_vec):
        """
        Sample a point in local frame for RRT*.
        With some probability, sample near goal (goal bias).
        """
        cfg = self.cfg

        # Goal bias sampling (unchanged)
        if np.random.rand() < cfg.rrt_goal_sample_prob:
            noise = np.random.normal(loc=0.0, scale=0.5, size=2)
            return goal_vec + noise

        X_MAX = cfg.grid_x_max
        Y_MAX = cfg.grid_y_max

        # In local frame, start is (0, 0)
        x_start = 0.0
        x_goal  = float(goal_vec[0])

        # Range between current and goal
        x_min = min(x_start, x_goal)
        x_max = max(x_start, x_goal)

        # Optionally also clamp to [0, X_MAX] if you only want forward sampling
        x_min = max(0.0, x_min)
        x_max = min(X_MAX, x_max)

        # Fallback if range is degenerate (goal at/behind start or weird cases)
        if x_max <= x_min + 1e-3:
            x = np.random.uniform(0.0, X_MAX)
        else:
            x = np.random.uniform(x_min, x_max)

        y = np.random.uniform(-Y_MAX, Y_MAX)
        return np.array([x, y], dtype=np.float64)














    def _plan_rrt_star_local(self, vehicle):
        """
        Run a 2D RRT* in ego local frame using the occupancy grid.

        Returns:
            path_local: list of 2D points from start (0,0) to goal in local frame,
                        or None if planning failed.
            min_obstacle_dist: minimal obstacle distance from ego (for speed logic)
        """
        cfg = self.cfg

        grid, min_dist = self._build_occupancy_grid(vehicle)
        goal_vec = self._compute_local_goal(vehicle)

        start = np.array([0.0, 0.0], dtype=np.float64)
        if self._point_in_collision_grid(start, grid):
            self._last_grid = grid
            self._last_nodes = [start]
            self._last_parents = [-1]
            self._last_path = None
            self._last_goal_vec = goal_vec
            return None, min_dist
        




        dist_to_goal = float(np.linalg.norm(goal_vec - start))
        if dist_to_goal <= cfg.rrt_goal_stop_searching_radius:
            # Simple path: from (0,0) to the goal in local frame
            path = [start, goal_vec]

            # Store for visualization/debug, like in the normal success case
            self._last_grid = grid
            self._last_nodes = path
            self._last_parents = [-1, 0]  # start has no parent, goal's parent is start
            self._last_path = path
            self._last_goal_vec = goal_vec

            return path, min_dist











        nodes = [start]
        parents = [-1]
        costs = [0.0]  # cost-to-come

        goal_node_idx = None
        goal_node_cost = float("inf")

        for _ in range(cfg.rrt_max_iters):
            q_rand = self._sample_point(goal_vec)

            # --- Nearest node ---
            dists_sq = [np.dot(q - q_rand, q - q_rand) for q in nodes]
            nearest_idx = int(np.argmin(dists_sq))
            q_near = nodes[nearest_idx]

            # --- Steer towards q_rand by rrt_step_size ---
            direction = q_rand - q_near
            dist = float(np.linalg.norm(direction))
            if dist < 1e-6:
                continue
            direction = direction / dist
            step = min(cfg.rrt_step_size, dist)
            q_new = q_near + step * direction

            # --- Collision check ---
            if self._segment_in_collision_grid(q_near, q_new, grid):
                continue

            # --- RRT* choose best parent among neighbors ---
            dists = [np.linalg.norm(q - q_new) for q in nodes]
            neighbor_indices = [
                i for i, d in enumerate(dists) if d <= cfg.rrt_star_radius
            ]

            best_parent = nearest_idx
            best_cost = costs[nearest_idx] + np.linalg.norm(nodes[nearest_idx] - q_new)

            for i in neighbor_indices:
                cand_q = nodes[i]
                cand_cost = costs[i] + np.linalg.norm(cand_q - q_new)
                if cand_cost < best_cost:
                    if not self._segment_in_collision_grid(cand_q, q_new, grid):
                        best_parent = i
                        best_cost = cand_cost

            # --- Add new node ---
            new_idx = len(nodes)
            nodes.append(q_new)
            parents.append(best_parent)
            costs.append(best_cost)

            # --- Rewiring ---
            for i in neighbor_indices:
                if i == best_parent:
                    continue
                cand_q = nodes[i]
                new_cost = best_cost + np.linalg.norm(cand_q - q_new)
                if new_cost + 1e-6 < costs[i]:
                    if not self._segment_in_collision_grid(q_new, cand_q, grid):
                        parents[i] = new_idx
                        costs[i] = new_cost

            # --- Check goal region ---
            if np.linalg.norm(q_new - goal_vec) <= cfg.rrt_goal_radius:
                if best_cost < goal_node_cost:
                    goal_node_idx = new_idx
                    goal_node_cost = best_cost

        # If we never reached goal region, pick best node towards goal as fallback
        if goal_node_idx is None:
            dists_to_goal = [np.linalg.norm(q - goal_vec) for q in nodes]
            best_idx = int(np.argmin(dists_to_goal))
        else:
            best_idx = goal_node_idx

        if best_idx == 0:
            self._last_grid = grid
            self._last_nodes = nodes
            self._last_parents = parents
            self._last_path = None
            self._last_goal_vec = goal_vec
            return None, min_dist

        # Reconstruct path
        path = []
        idx = best_idx
        while idx != -1:
            path.append(nodes[idx])
            idx = parents[idx]
        path.reverse()

        path = self._smooth_path(path)

        # Store for visualization
        self._last_grid = grid
        self._last_nodes = nodes
        self._last_parents = parents
        self._last_path = path
        self._last_goal_vec = goal_vec

        return path, min_dist

    def _pick_target_from_path(self, path_local):
        """
        Given a local-frame path (list of 2D points starting at (0,0)),
        pick a waypoint at least path_lookahead away if possible.
        """
        cfg = self.cfg
        if path_local is None or len(path_local) == 0:
            return np.array([cfg.path_lookahead, 0.0], dtype=np.float64)

        pts = [np.array(p, dtype=np.float64) for p in path_local]

        for p in pts[1:]:
            d = float(np.linalg.norm(p))
            if d >= cfg.path_lookahead:
                return p

        return pts[-1]

    # ---------------- Steering smoothing ----------------
    def _smooth_and_limit_steering(self, raw_steer):
        """
        Apply exponential smoothing and rate limit to steering_norm.
        """
        cfg = self.cfg
        raw_steer = float(np.clip(raw_steer, -1.0, 1.0))

        alpha = cfg.steer_smooth_alpha
        smoothed = (1.0 - alpha) * self._prev_steering_norm + alpha * raw_steer

        delta = smoothed - self._prev_steering_norm
        max_delta = cfg.max_steer_delta
        if delta > max_delta:
            smoothed = self._prev_steering_norm + max_delta
        elif delta < -max_delta:
            smoothed = self._prev_steering_norm - max_delta

        smoothed = float(np.clip(smoothed, -1.0, 1.0))
        self._prev_steering_norm = smoothed
        return smoothed

    # ---------------- Debug visualization ----------------
    def debug_plot_rrt_grid(self, show=True, save_path: Optional[str] = None):
        """
        Visualize the last occupancy grid and RRT tree/path in ego local frame.

        - Occupied cells (including lane mask) shown as dark pixels.
        - RRT nodes and edges overlaid.
        - Final path highlighted.
        - Ego at (0,0) and goal point shown.
        """
        if self._last_grid is None:
            print("[RRT DEBUG] No grid to visualize yet (call action() first).")
            return

        if plt is None:
            print("[RRT DEBUG] matplotlib is not available. Install it to see plots.")
            grid_int = self._last_grid.astype(int)
            print("[RRT DEBUG] Occupancy grid (1=occ, 0=free), y from top to bottom:")
            print(grid_int.T[::-1])
            return

        grid = self._last_grid
        nodes = self._last_nodes or []
        parents = self._last_parents or []
        path = self._last_path
        goal = self._last_goal_vec

        cfg = self.cfg
        X_MAX = cfg.grid_x_max
        Y_MAX = cfg.grid_y_max

        fig, ax = plt.subplots(figsize=(6, 8))

        ax.imshow(
            grid.astype(float).T,
            origin="lower",
            cmap="Greys",
            extent=[0.0, X_MAX, -Y_MAX, Y_MAX],
            interpolation="nearest",
        )

        # Tree edges
        for i, p in enumerate(parents):
            if p < 0:
                continue
            q = nodes[i]
            q_parent = nodes[p]
            ax.plot(
                [q_parent[0], q[0]],
                [q_parent[1], q[1]],
                linewidth=0.5,
            )

        # Nodes
        if len(nodes) > 0:
            xs = [n[0] for n in nodes]
            ys = [n[1] for n in nodes]
            ax.scatter(xs, ys, s=5)

        # Path
        if path is not None and len(path) > 1:
            px = [p[0] for p in path]
            py = [p[1] for p in path]
            ax.plot(px, py, linewidth=2.5)

        # Ego
        ax.scatter([0.0], [0.0], s=80, marker="o")
        ax.text(0.0, 0.0, " ego", fontsize=8)

        # Goal
        if goal is not None:
            ax.scatter([goal[0]], [goal[1]], s=80, marker="*",)
            ax.text(goal[0], goal[1], " goal", fontsize=8)

        ax.set_xlabel("x (forward) [m]")
        ax.set_ylabel("y (lateral) [m]")
        ax.set_title("Occupancy grid + RRT* (ego frame)")

        ax.set_xlim(0.0, X_MAX)
        ax.set_ylim(-Y_MAX, Y_MAX)
        ax.set_aspect("equal", adjustable="box")

        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight")
            print("[RRT DEBUG] Saved plot to:", save_path)

        if show:
            plt.show()
        else:
            plt.close(fig)

    # ------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------
    def action(self, vehicle, obs=None) -> np.ndarray:
        """
        Compute control action from vehicle state using:
          - LIDAR -> 2D occupancy grid (inflated)
          - Lane distances -> mark outside lane as occupied
          - RRT* path planning in local frame
          - PID tracking + steering smoothing

        Returns:
            np.array([steering_norm, pedal_norm]) in [-1, 1]^2
        """
        cfg = self.cfg
        dt = float(cfg.dt)

        # --- RRT* planning ---
        path_local, min_obstacle_dist = self._plan_rrt_star_local(vehicle)


        # Tracking target
        target_local = self._pick_target_from_path(path_local)


        xg, yg = self._local_to_global(vehicle, target_local[0], target_local[1])
        self._last_target_global = (float(xg), float(yg)) 
        print(f"RRT target (global): X = {xg:.2f}, Y = {yg:.2f}")



        # Longitudinal state
        v = float(vehicle.speed)

        # Lateral control
        tx, ty = float(target_local[0]), float(target_local[1])
        heading_error = np.arctan2(ty, tx)
        heading_error = self._wrap_angle(heading_error)

        d_heading_error = (heading_error - self._prev_heading_error) / dt

        self._int_heading_error += heading_error * dt
        self._int_heading_error = float(np.clip(
            self._int_heading_error,
            -cfg.int_steer_limit,
            cfg.int_steer_limit,
        ))

        raw_steer_cmd = (
            cfg.kp_steer * heading_error
            + cfg.kd_steer * d_heading_error
            + cfg.ki_steer * self._int_heading_error
        )

        steering_norm = self._smooth_and_limit_steering(raw_steer_cmd)

        # Longitudinal control with slowdown
        v_ref = float(cfg.v_ref)

        if min_obstacle_dist is not None and min_obstacle_dist < cfg.slowdown_dist:
            scale = max(0.0, min_obstacle_dist / cfg.slowdown_dist)
            v_ref = v_ref * scale
            if v_ref < cfg.v_min_running and min_obstacle_dist > 2.0:
                v_ref = cfg.v_min_running

        v_ref = float(np.clip(v_ref, cfg.v_min, cfg.v_max))

        speed_error = v_ref - v
        d_speed_error = (speed_error - self._prev_speed_error) / dt

        self._int_speed_error += speed_error * dt
        self._int_speed_error = float(np.clip(
            self._int_speed_error,
            -cfg.int_v_limit,
            cfg.int_v_limit,
        ))

        pedal_cmd = (
            cfg.kp_v * speed_error
            + cfg.kd_v * d_speed_error
            + cfg.ki_v * self._int_speed_error
        )

        pedal_norm = float(np.clip(pedal_cmd, -1.0, 1.0))

        # Store errors
        self._prev_heading_error = heading_error
        self._prev_speed_error = speed_error

        # Debug prints
        print("RRT* target_local =", target_local)
        print("heading_error =", heading_error,
              "raw_steer_cmd =", raw_steer_cmd,
              "steering_norm (smoothed) =", steering_norm)
        print("v =", v, "v_ref =", v_ref,
              "speed_error =", speed_error,
              "pedal_norm =", pedal_norm)
        if min_obstacle_dist is not None:
            print("min_obstacle_dist_from_lidar =", min_obstacle_dist)

        u_out = np.array([steering_norm, pedal_norm], dtype=np.float32)
        print("[OccGrid + Lane RRT*] u_norm_for_env (steer, accel) in [-1,1]:", u_out)

        # ---------- DEBUG: save an RRT+grid image to rrt_save/ ----------
        self._debug_step += 10
        if self._debug_step % 2 == 0:  # save every 10 control steps
            os.makedirs(self._debug_dir, exist_ok=True)
            fname = os.path.join(self._debug_dir, "rrt_debug_step_%05d.png" % self._debug_step)
            self.debug_plot_rrt_grid(show=False, save_path=fname)
            print("[RRT DEBUG] Saved visualization to", fname)
        # # ---------------------------------------------------------------

        return u_out
    


