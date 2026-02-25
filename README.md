# Advanced Drone Trajectory Optimization

A multi-objective trajectory optimization framework for quadrotor UAVs that jointly minimizes navigation time, energy consumption, and community noise annoyance while respecting airspace constraints. The system integrates a high-fidelity 6-DoF flight dynamics simulator, data-driven rotor aeroacoustic models, psychoacoustic annoyance metrics, and four distinct optimization backends—Particle Swarm Optimization (PSO), Grey Wolf Optimizer (GWO), Soft Actor–Critic (SAC), and Twin Delayed DDPG (TD3).

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Installation](#3-installation)
4. [Configuration](#4-configuration)
5. [Quadrotor Dynamics Model](#5-quadrotor-dynamics-model)
   - 5.1 [Translational Dynamics](#51-translational-dynamics)
   - 5.2 [Rotational Dynamics](#52-rotational-dynamics)
   - 5.3 [Motor Mixer](#53-motor-mixer)
   - 5.4 [Numerical Integration](#54-numerical-integration)
   - 5.5 [Rotor Aerodynamics Model](#55-rotor-aerodynamics-model)
6. [Flight Controller](#6-flight-controller)
   - 6.1 [PID Architecture](#61-pid-architecture)
   - 6.2 [Cascaded Control Loop](#62-cascaded-control-loop)
7. [Wind & Turbulence Model](#7-wind--turbulence-model)
8. [Aeroacoustic Noise Models](#8-aeroacoustic-noise-models)
   - 8.1 [DNN Lookup-Table Model](#81-dnn-lookup-table-model)
   - 8.2 [EMPA Regression Model](#82-empa-regression-model)
   - 8.3 [Sound Propagation](#83-sound-propagation)
   - 8.4 [Psychoacoustic Annoyance](#84-psychoacoustic-annoyance)
9. [World Environment](#9-world-environment)
10. [Cost Function](#10-cost-function)
11. [Optimization Methods](#11-optimization-methods)
    - 11.1 [Waypoint Parameterization](#111-waypoint-parameterization)
    - 11.2 [Particle Swarm Optimization (PSO)](#112-particle-swarm-optimization-pso)
    - 11.3 [Grey Wolf Optimizer (GWO)](#113-grey-wolf-optimizer-gwo)
    - 11.4 [Soft Actor–Critic (SAC)](#114-soft-actorcritic-sac)
    - 11.5 [Twin Delayed DDPG (TD3)](#115-twin-delayed-ddpg-td3)
12. [Usage Guide](#12-usage-guide)
    - 12.1 [World Creation](#121-world-creation)
    - 12.2 [Running Metaheuristic Optimization](#122-running-metaheuristic-optimization)
    - 12.3 [Running Reinforcement Learning](#123-running-reinforcement-learning)
    - 12.4 [Evaluating a Trajectory](#124-evaluating-a-trajectory)
    - 12.5 [Visualizing Training Progress](#125-visualizing-training-progress)
13. [Simulation Pipeline](#13-simulation-pipeline)
14. [References](#14-references)
15. [Author](#15-author)

---

## 1. Project Overview

Urban Air Mobility (UAM) demands trajectory planners that balance competing objectives: fast point-to-point transit, low energy consumption, and minimal acoustic impact on ground communities. This framework casts the problem as a waypoint perturbation optimization over a heterogeneous ground environment. A baseline straight-line path between origin $A$ and destination $B$ is augmented with $n$ intermediate waypoints whose 3-D positions and cruise speeds are decision variables. Each candidate trajectory is evaluated in closed-loop simulation using a full 6-DoF quadrotor model, a cascaded PID controller, data-driven rotor noise predictions, and ISO-based psychoacoustic metrics.

Four optimization backends are provided:

| Method | Type | Decision Space |
|--------|------|---------------|
| PSO | Metaheuristic (swarm) | $4n$-dimensional continuous |
| GWO | Metaheuristic (pack) | $4n$-dimensional continuous |
| SAC | Model-free RL (off-policy, max-entropy) | Sequential per-waypoint |
| TD3 | Model-free RL (off-policy, deterministic) | Sequential per-waypoint |

---

## 2. Repository Structure

```
├── main.py                          # Factory functions and simulation orchestration
├── test.py                          # Evaluation & visualization of optimized trajectories
├── show_ani_train.py                # Training trajectory animation
├── world_creation_gui.py            # Tkinter-based world editor GUI
├── requirements.txt                 # Python dependencies
│
├── Drone/
│   ├── Drone.py                     # 6-DoF quadrotor dynamics (QuadcopterModel)
│   ├── Controller.py                # Cascaded PID controller (QuadCopterController)
│   ├── Simulation.py                # Time-stepping simulation loop
│   └── Wind.py                      # Dryden turbulence model
│
├── Noise/
│   ├── AIModel.py                   # PyTorch DNN for rotor noise regression
│   ├── DNNModel.py                  # Lookup-table rotor sound model
│   ├── EmpaModel.py                 # EMPA parametric regression model
│   └── Psychoacoustic.py            # Zwicker loudness & psychoacoustic annoyance
│
├── Optimizations/
│   ├── optimizer.py                 # Base optimizer: cost function, logging, persistence
│   ├── PSO_optimizer.py             # Particle Swarm Optimization
│   └── GWO_optimizer.py             # Grey Wolf Optimizer
│
├── RL/
│   ├── environment.py               # Gymnasium DroneTrajectoryEnv
│   ├── trainer.py                   # SAC / TD3 training pipelines
│   └── callbacks.py                 # SB3 episode logger callback
│
├── Rotor/
│   ├── TorchRotorModel.py           # PyTorch BEMT surrogate for rotor aero
│   ├── rotor_model.pth              # Trained rotor model weights
│   ├── normalization_params.pth     # Z-score normalization parameters
│   └── rotor_config.ini             # Rotor geometry (root/tip radii)
│
├── Settings/
│   ├── simulation_parameters.yaml   # Drone physics, PID gains, world config
│   ├── SAC_parameters.yaml          # SAC RL hyperparameters
│   ├── TD3_parameters.yaml          # TD3 RL hyperparameters
│   └── metaheuristic_parameters.yaml# PSO & GWO parameters
│
├── Utils/
│   ├── utils.py                     # Angle wrapping, rotation matrices
│   └── plotting_functions.py        # Log plotting, 3-D animation, CSV export
│
├── Worlds/
│   └── World.py                     # Grid-based world with typed areas
│
├── Outputs/                         # Sensitivity analysis datasets
├── Plots/                           # Generated figures
└── Journal/                         # Paper-related materials
```

---

## 3. Installation

### Prerequisites

- Python ≥ 3.9
- CUDA-capable GPU (recommended for RL training and rotor model inference)
- Conda (recommended) or pip

### Setup

```bash
# Clone the repository
git clone https://github.com/<user>/Advanced-Drone-Trajectory-Optimization.git
cd Advanced-Drone-Trajectory-Optimization

# Create and activate a conda environment
conda create -n drone-opt python=3.10
conda activate drone-opt

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install remaining dependencies
pip install -r requirements.txt

# Additional dependencies (not listed in requirements.txt)
pip install mosqito joblib
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical computation |
| `matplotlib` | Plotting and animation |
| `pandas` | Data manipulation |
| `pillow` | Image loading for world backgrounds |
| `torch` | Neural network models (rotor, noise DNN) |
| `scipy` | Signal processing (Dryden model), optimization |
| `scikit-learn` | Scaling, preprocessing |
| `bayesian-optimization` | Hyperparameter tuning |
| `pyyaml` | Configuration parsing |
| `stable-baselines3` | SAC/TD3 reinforcement learning |
| `gymnasium` | RL environment interface |
| `mosqito` | Psychoacoustic metrics (Zwicker loudness, sharpness, roughness) |
| `joblib` | Model serialization |

---

## 4. Configuration

All tunable parameters are stored in YAML files under `Settings/`.

### `simulation_parameters.yaml`

| Section | Parameter | Default | Unit | Description |
|---------|-----------|---------|------|-------------|
| **Simulation** | `dt` | 0.008 | s | Integration time step |
| | `simulation_time` | 100 | s | Maximum simulation duration |
| | `frame_skip` | 1 | — | Logging decimation factor |
| | `threshold` | 2.0 | m | Waypoint arrival radius |
| | `target_shift_threshold_distance` | 5.0 | m | Segment transition distance |
| | `noise_annoyance_radius` | 15 | m | Noise evaluation ground radius |
| **Drone** | `m` | 5.2 | kg | Total mass |
| | `I` | $[3.8,\;3.8,\;7.1]\times 10^{-3}$ | kg·m² | Moments of inertia $[I_{xx}, I_{yy}, I_{zz}]$ |
| | `d` | $7.5\times 10^{-7}$ | — | Drag factor |
| | `l` | 0.32 | m | Arm length (center to rotor) |
| | `Cd` | $[0.1,\;0.1,\;0.15]$ | — | Translational drag coefficients |
| | `Ca` | $[0.1,\;0.1,\;0.15]$ | — | Aerodynamic friction coefficients |
| | `Jr` | $6\times 10^{-5}$ | kg·m² | Rotor inertia |
| | `max_rpm` | 3000 | rpm | Motor speed limit |
| | `n_rotors` | 4 | — | Number of rotors |
| **PID Gains** | `k_pid_pos` | $(0.5912,\;0.00623,\;0.000520)$ | — | Position loop $(K_p, K_i, K_d)$ |
| | `k_pid_alt` | $(1.406,\;0.00439,\;0.00862)$ | — | Altitude loop |
| | `k_pid_att` | $(47.85,\;0.304,\;0.9377)$ | — | Attitude loop |
| | `k_pid_yaw` | $(0.5,\;10^{-6},\;0.1)$ | — | Yaw loop |
| | `k_pid_hsp` | $(0.0958,\;0.00847,\;0.0)$ | — | Horizontal speed loop |
| | `k_pid_vsp` | $(120.36,\;148.59,\;7.37)$ | — | Vertical speed loop |

### `metaheuristic_parameters.yaml`

| Section | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| **Shared** | `n_points` | 5 | Number of intermediate waypoints |
| | `max_perturbation_offset` | 250.0 m | Maximum perturbation per axis |
| | `max_velocity` | 20.0 m/s | Maximum cruise speed |
| **PSO** | `swarm_size` | 30 | Swarm population |
| | `n_generations` | 100 | Iterations |
| | `inertia` ($\omega$) | 0.5 | Velocity inertia weight |
| | `cognitive_coeff` ($c_1$) | 1.5 | Personal best attraction |
| | `social_coeff` ($c_2$) | 1.5 | Global best attraction |
| **GWO** | `pack_size` | 30 | Wolf pack population |
| | `n_generations` | 100 | Iterations |

### `SAC_parameters.yaml` / `TD3_parameters.yaml`

| Parameter | SAC Default | TD3 Default | Description |
|-----------|-------------|-------------|-------------|
| `total_timesteps` | 100,000 | 100,000 | Training budget |
| `learning_rate` | $10^{-3}$ | $10^{-3}$ | Adam learning rate |
| `buffer_size` | 50,000 | 50,000 | Replay buffer capacity |
| `batch_size` | 256 | 256 | Mini-batch size |
| `gamma` | 0.99 | 0.99 | Discount factor |
| `tau` | 0.005 | 0.005 | Polyak averaging coefficient |
| `ent_coef` | `auto_0.1` | — | Entropy coefficient (SAC) |
| `target_entropy` | $-2$ | — | Target entropy (SAC) |
| `policy_delay` | — | 2 | Policy update delay (TD3) |
| `target_policy_noise` | — | 0.2 | Target smoothing noise (TD3) |
| `target_noise_clip` | — | 0.5 | Noise clip range (TD3) |

---

## 5. Quadrotor Dynamics Model

The quadrotor is modeled as a rigid body with four independent rotors arranged in an X-configuration. The state vector is:

$$\mathbf{x} = \begin{bmatrix} x & y & z & \dot{x} & \dot{y} & \dot{z} & \phi & \theta & \psi & \dot{\phi} & \dot{\theta} & \dot{\psi} \end{bmatrix}^\top \in \mathbb{R}^{12}$$

where $(x,y,z)$ is position in the inertial frame, $(\phi, \theta, \psi)$ are roll, pitch, and yaw Euler angles (ZYX convention), and dots denote time derivatives.

### 5.1 Translational Dynamics

The translational equations of motion in the inertial frame are:

$$\ddot{x} = \frac{T}{m}\left(\cos\psi\sin\theta\cos\phi + \sin\psi\sin\phi\right) - \frac{C_{d_x}}{m}\,\dot{x}$$

$$\ddot{y} = \frac{T}{m}\left(\sin\psi\sin\theta\cos\phi - \cos\psi\sin\phi\right) - \frac{C_{d_y}}{m}\,\dot{y}$$

$$\ddot{z} = \frac{T}{m}\cos\theta\cos\phi - \frac{C_{d_z}}{m}\,\dot{z} - g$$

where:
- $T = \sum_{i=1}^{4} T_i$ is the total thrust (sum of individual rotor thrusts),
- $m$ is the total mass,
- $g = 9.81\;\text{m/s}^2$ is gravitational acceleration,
- $C_{d_x}, C_{d_y}, C_{d_z}$ are translational drag coefficients.

### 5.2 Rotational Dynamics

The control moments are derived from the individual rotor thrusts $T_i$ and torques $Q_i$:

$$u_2 = l\,(T_4 - T_2) \qquad \text{(roll moment)}$$

$$u_3 = l\,(T_3 - T_1) \qquad \text{(pitch moment)}$$

$$u_4 = Q_1 - Q_2 + Q_3 - Q_4 \qquad \text{(yaw moment)}$$

where $l$ is the arm length. The gyroscopic residual angular velocity is:

$$\Omega_r = \omega_1 - \omega_2 + \omega_3 - \omega_4$$

The rotational equations of motion are:

$$\ddot{\phi} = \frac{u_2}{I_{xx}} - \frac{C_{a_\phi}\,\mathrm{sgn}(\dot{\phi})\,\dot{\phi}^2}{I_{xx}} - \frac{J_r\,\Omega_r}{I_{xx}}\,\dot{\theta} - \frac{(I_{zz} - I_{yy})}{I_{xx}}\,\dot{\theta}\,\dot{\psi}$$

$$\ddot{\theta} = \frac{u_3}{I_{yy}} - \frac{C_{a_\theta}\,\mathrm{sgn}(\dot{\theta})\,\dot{\theta}^2}{I_{yy}} + \frac{J_r\,\Omega_r}{I_{yy}}\,\dot{\phi} - \frac{(I_{xx} - I_{zz})}{I_{yy}}\,\dot{\phi}\,\dot{\psi}$$

$$\ddot{\psi} = \frac{u_4}{I_{zz}} - \frac{C_{a_\psi}\,\mathrm{sgn}(\dot{\psi})\,\dot{\psi}^2}{I_{zz}} - \frac{(I_{yy} - I_{xx})}{I_{zz}}\,\dot{\phi}\,\dot{\theta}$$

where $J_r$ is the rotor inertia and $C_{a_\phi}, C_{a_\theta}, C_{a_\psi}$ are aerodynamic friction coefficients.

### 5.3 Motor Mixer

The control inputs $(u_1, u_2, u_3, u_4)$ are mapped to individual motor angular velocities via the allocation matrix. With thrust coefficient $b = 7 \times 10^{-4}$ and drag coefficient $d = 7.5 \times 10^{-7}$:

$$\omega_1^2 = \frac{u_1}{4b} - \frac{u_3}{2bl} + \frac{u_4}{4d}$$

$$\omega_2^2 = \frac{u_1}{4b} - \frac{u_2}{2bl} - \frac{u_4}{4d}$$

$$\omega_3^2 = \frac{u_1}{4b} + \frac{u_3}{2bl} + \frac{u_4}{4d}$$

$$\omega_4^2 = \frac{u_1}{4b} + \frac{u_2}{2bl} - \frac{u_4}{4d}$$

All values are clipped to $[0,\;\omega_{\max}^2]$ before taking the square root, then converted from rad/s to RPM.

### 5.4 Numerical Integration

State propagation uses the classical **4th-order Runge–Kutta** (RK4) scheme:

$$\mathbf{k}_1 = f(\mathbf{x}_t)$$
$$\mathbf{k}_2 = f\!\left(\mathbf{x}_t + \frac{\Delta t}{2}\,\mathbf{k}_1\right)$$
$$\mathbf{k}_3 = f\!\left(\mathbf{x}_t + \frac{\Delta t}{2}\,\mathbf{k}_2\right)$$
$$\mathbf{k}_4 = f\!\left(\mathbf{x}_t + \Delta t\,\mathbf{k}_3\right)$$
$$\mathbf{x}_{t+1} = \mathbf{x}_t + \frac{\Delta t}{6}\left(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4\right)$$

where the state derivative $f(\mathbf{x})$ combines:
- $\dot{\mathbf{p}} = \mathbf{v}$ (position rate = velocity),
- $\dot{\mathbf{v}} = \mathbf{a}_{\text{trans}}$ (translational dynamics),
- $\dot{\boldsymbol{\Theta}} = \boldsymbol{\omega}$ (angle rate = angular velocity),
- $\dot{\boldsymbol{\omega}} = \boldsymbol{\alpha}_{\text{rot}}$ (rotational dynamics).

Post-integration, Euler angles are wrapped to $[-\pi, \pi]$ and angular velocities are clipped to $[-10, 10]$ rad/s.

### 5.5 Rotor Aerodynamics Model

Individual rotor aerodynamics are predicted by a PyTorch neural network surrogate trained on Blade Element Momentum Theory (BEMT) data.

**Architecture:**

$$\text{RPM} \;\xrightarrow{\;\text{Linear}(1 \to 16)\;}\; \text{ReLU} \;\xrightarrow{\;\text{Linear}(16 \to 16)\;}\; \text{ReLU} \;\xrightarrow{\;\text{Linear}(16 \to 6)\;}\; (T,\, Q,\, P,\, C_T,\, C_Q,\, C_P)$$

Inputs and outputs are Z-score normalized:

$$\hat{x} = \frac{x - \mu_x}{\sigma_x}, \qquad y = \hat{y}\,\sigma_y + \mu_y$$

The model outputs per-rotor thrust $T_i$, torque $Q_i$, power $P_i$, and their non-dimensional coefficients.

**Wind-induced thrust perturbation** per rotor:

$$\Delta T = \frac{\rho}{2}\left(k_{v_z}\,\omega\,(R_{\text{tip}}^2 - R_{\text{root}}^2) + \frac{k_{v_z}}{\omega}\,(V_x^2 + V_y^2)\,\ln\frac{R_{\text{tip}}}{R_{\text{root}}}\right)$$

where $k_{v_z} = 2\pi^2 V_z$ and $\rho = 1.225\;\text{kg/m}^3$.

---

## 6. Flight Controller

### 6.1 PID Architecture

Each control channel uses a standard PID controller with anti-windup:

$$e(t) = r(t) - y(t)$$

$$u(t) = K_p\,e(t) + K_i\int_0^t e(\tau)\,d\tau + K_d\,\frac{de}{dt}$$

The integral term is clamped to $[-L,\;L]$ where $L$ is the anti-windup limit (typically 40% of the corresponding actuator limit). The derivative is computed via backward difference:

$$\frac{de}{dt} \approx \frac{e(t) - e(t - \Delta t)}{\Delta t}$$

### 6.2 Cascaded Control Loop

The controller uses an 8-PID cascade architecture with three hierarchical loops:

```
                    ┌─────────────────────────────────────────────────┐
                    │              OUTER LOOP (Position)              │
  Waypoint ───────►│  PID_x ──► vx_des    PID_y ──► vy_des          │
  (x,y,z)          │  PID_z ──► vz_des                               │
                    └───────────────┬────────────────┬────────────────┘
                                    │                │
                    ┌───────────────▼────────────────▼────────────────┐
                    │           MIDDLE LOOP (Speed → Angle)           │
                    │  PID_hspeed(vx, vx_des) ──► θ_des (pitch)      │
                    │  PID_hspeed(vy, vy_des) ──► φ_des (roll)       │
                    │  PID_vspeed(vz, vz_des) ──► Δu₁  (thrust adj.) │
                    └───────────────┬────────────────┬────────────────┘
                                    │                │
                    ┌───────────────▼────────────────▼────────────────┐
                    │            INNER LOOP (Attitude)                │
                    │  PID_roll(φ, φ_des)   ──► u₂                   │
                    │  PID_pitch(θ, θ_des)  ──► u₃                   │
                    │  PID_yaw(ψ, 0)        ──► u₄                   │
                    └─────────────────────────────────────────────────┘
```

**Thrust computation** (altitude channel):

$$T_{\text{hover}} = mg \cdot \min\!\left(\frac{1}{\cos\theta\cos\phi},\;1.5\right)$$

$$u_1 = \text{clip}\!\left(T_{\text{hover}} + \Delta u_1,\; 0,\; u_{1,\max}\right)$$

**Speed-to-angle mapping** (horizontal channels):

$$\theta_{\text{des}} = \text{clip}\!\left(\text{PID}_{\text{hspeed}}(v_x, v_{x,\text{des}}),\;-\theta_{\max},\;\theta_{\max}\right)$$

$$\phi_{\text{des}} = \text{clip}\!\left(-\text{PID}_{\text{hspeed}}(v_y, v_{y,\text{des}}),\;-\phi_{\max},\;\phi_{\max}\right)$$

---

## 7. Wind & Turbulence Model

Atmospheric disturbances are generated using the **Dryden continuous turbulence model** (MIL-HDBK-1797). Wind gusts along the longitudinal ($u$), lateral ($v$), and vertical ($w$) axes are modeled as the output of linear transfer functions driven by white Gaussian noise.

**Turbulence parameters:**

$$\sigma_w = 0.1 \cdot L_{\text{turb}}, \qquad c_h = 0.177 + 0.000823\,h$$

$$L_v = \frac{h}{c_h^{0.2}}, \qquad \sigma = \frac{\sigma_w}{c_h^{0.4}}$$

where $h$ is the flight altitude and $L_{\text{turb}}$ is the turbulence level parameter.

**Longitudinal transfer function:**

$$H_u(s) = \sigma\sqrt{\frac{2L_v}{\pi V_a}} \cdot \frac{V_a}{L_v\,s + V_a}$$

**Lateral/vertical transfer functions:**

$$H_{v,w}(s) = c\,\frac{\sqrt{3}\,\frac{L_v}{V_a}\,s + 1}{\left(\frac{L_v}{V_a}\right)^2 s^2 + 2\frac{L_v}{V_a}\,s + 1}, \qquad c = \sigma\sqrt{\frac{L_v}{\pi V_a}}$$

where $V_a$ is the nominal airspeed. The wind signals are generated via `scipy.signal.lsim` over 10-second windows.

---

## 8. Aeroacoustic Noise Models

### 8.1 DNN Lookup-Table Model

A precomputed 2-D lookup table maps the radiation angle $\zeta$ (elevation angle from the rotor plane) to Sound Power Level (SWL) spectra in 1/3-octave frequency bands. Per-rotor SWL is scaled with RPM:

$$L_{w,i}(f) = L_{w,\text{ref}}(f,\zeta) + 10\,\alpha\,\log_{10}\!\left(\frac{\text{RPM}_i}{\text{RPM}_{\text{ref}}}\right)$$

where $\alpha = 3$ is the RPM exponent and the reference is subtracted for single-rotor operation:

$$L_{w,\text{ref,rotor}}(f) = L_{w,\text{drone}}(f) - 10\log_{10}(N_{\text{rotors}})$$

Total SWL is obtained by incoherent summation:

$$L_{w,\text{total}}(f) = 10\log_{10}\!\left(\sum_{i=1}^{N_{\text{rotors}}} 10^{L_{w,i}(f)/10}\right)$$

### 8.2 EMPA Regression Model

A parametric regression model with per-band coefficients $(a, b, c, d)$:

$$L_{w,i}(f) = L_{w,\text{ref}}(f) + a\,\zeta^2 + b\,|\zeta| + c\,\text{RPM} + d\,\text{RPM}^2 + C_{\text{proc}} - 10\log_{10}(N_{\text{rotors}})$$

Coefficients are fitted via L-BFGS-B minimization of MSE against training data. Input features are normalized with a `MinMaxScaler`.

### 8.3 Sound Propagation

The Sound Pressure Level (SPL) at a ground receiver at distance $r$ from the source is:

$$L_p(f) = L_w(f) - 10\log_{10}(4\pi r^2) - \alpha_{\text{atm}}(f)\,r + DI(f)$$

where:
- $10\log_{10}(4\pi r^2)$ is geometric (spherical) spreading loss,
- $\alpha_{\text{atm}}(f)$ is atmospheric absorption per ISO 9613-1 (function of temperature, humidity, pressure),
- $DI(f)$ is the directivity index.

**Broadband SPL** is computed by power summation over all frequency bands:

$$L_{p,\text{total}} = 10\log_{10}\!\left(\sum_k 10^{L_{p,k}/10}\right)$$

### 8.4 Psychoacoustic Annoyance

Psychoacoustic metrics are computed using the [MoSQITo](https://github.com/Eomys/MoSQITo) library following established standards:

| Metric | Standard | Unit |
|--------|----------|------|
| **Loudness** $N$ | ISO 532-1 (Zwicker, stationary, frequency-domain) | sone |
| **Sharpness** $S$ | DIN 45692 | acum |
| **Roughness** $R$ | Daniel & Weber | asper |
| **Fluctuation Strength** $F$ | — (placeholder) | vacil |

The **Zwicker Psychoacoustic Annoyance** is:

$$PA = N_5 \left(1 + \sqrt{\omega_S^2 + \omega_{FR}^2}\right)$$

where $N_5$ is the 95th percentile of the time-varying loudness, and:

$$\omega_S = \max(S - 1.75,\; 0) \cdot \ln(N_5 + 10)$$

$$\omega_{FR} = \frac{2.18}{N_5^{0.4}}\,(0.4\,F + 0.6\,R)$$

**Band-to-fine-spectrum expansion:** 1/3-octave SPL bands are expanded to a fine-resolution spectrum (default 10 Hz spacing, 24–24,000 Hz) for Zwicker loudness computation. Band edges are:

$$f_{\text{lo}} = f_c / 2^{1/6}, \qquad f_{\text{hi}} = f_c \cdot 2^{1/6}$$

Energy is distributed uniformly across $K$ fine bins within each band:

$$p_{\text{bin}} = \frac{p_{\text{rms,band}}}{\sqrt{K}}, \qquad p_{\text{rms,band}} = 2\times10^{-5} \cdot 10^{L_p/20}$$

---

## 9. World Environment

The ground environment is discretized into a 2-D grid of cells, each assigned an area type with associated constraints and noise penalties:

| ID | Area Type | Min Altitude (m) | Max Altitude (m) | Noise Penalty | Color |
|----|-----------|:-----------------:|:-----------------:|:-------------:|:-----:|
| 1 | Housing Estate | 150 | 1000 | 1.6 | Blue |
| 2 | Industrial Area | 70 | 1000 | 1.2 | Yellow |
| 3 | Open Field (default) | 5 | 1000 | 0.0 | Green |
| 4 | Forbidden Area | 0 | 0 | 100.0 | Red |

**Altitude penalty** for a drone at altitude $z$ over an area with bounds $[z_{\min}, z_{\max}]$:

$$P_{\text{alt}} = \begin{cases}
\sqrt{z_{\min}^2 - z^2} & \text{if } z < z_{\min} \\
\sqrt{z^2 - z_{\max}^2} & \text{if } z > z_{\max} \\
0 & \text{otherwise}
\end{cases}$$

At each simulation step, all ground cells within a configurable radius of the drone's ground projection are evaluated for noise emissions and altitude violations.

---

## 10. Cost Function

The multi-objective cost function aggregates five terms with configurable weights:

$$J = w_t \cdot t_{\text{nav}} + w_p \cdot \sum_{k} P_k + w_n \cdot \overline{L_p \cdot \pi_{\text{noise}}} + w_a \cdot \sum_k P_{\text{alt},k} + w_c \cdot (1 - \eta_{\text{comp}})$$

| Term | Symbol | Weight | Default | Description |
|------|--------|--------|---------|-------------|
| Navigation Time | $t_{\text{nav}}$ | $w_t$ | 1.0 | Total flight time (s) |
| Energy | $\sum P_k$ | $w_p$ | $10^{-4}$ | Cumulative power consumption (W) |
| Noise | $\overline{L_p \cdot \pi_{\text{noise}}}$ | $w_n$ | 15.0 | Mean SPL weighted by area noise penalties |
| Altitude Violation | $\sum P_{\text{alt},k}$ | $w_a$ | $10^{-2}$ | Cumulative altitude constraint violations |
| Completion | $1 - \eta_{\text{comp}}$ | $w_c$ | 1000.0 | Fraction of route not completed |

where $\eta_{\text{comp}} \in [0, 1]$ is the percentage of waypoints successfully reached, and $\pi_{\text{noise}}$ is the noise penalty factor of the ground area beneath the drone.

---

## 11. Optimization Methods

### 11.1 Waypoint Parameterization

Given start point $A \in \mathbb{R}^3$ and end point $B \in \mathbb{R}^3$, $n$ reference waypoints are placed at equally-spaced interior positions on the line segment $\overline{AB}$:

$$\mathbf{r}_i = A + \frac{i}{n+1}(B - A), \qquad i = 1, \ldots, n$$

Each optimization variable is a perturbation vector $\boldsymbol{\delta}_i = (\delta x_i, \delta y_i, \delta z_i, v_i) \in \mathbb{R}^4$ such that the actual waypoint is:

$$\mathbf{w}_i = \mathbf{r}_i + (\delta x_i, \delta y_i, \delta z_i)^\top, \quad \text{cruise speed} = v_i$$

The full decision vector is $\boldsymbol{\xi} = (\boldsymbol{\delta}_1, \ldots, \boldsymbol{\delta}_n) \in \mathbb{R}^{4n}$, bounded per-component to the world geometry and a configurable maximum perturbation offset.

### 11.2 Particle Swarm Optimization (PSO)

A swarm of $S$ particles explores the $4n$-dimensional search space. Each particle $i$ has position $\mathbf{x}_i$ and velocity $\mathbf{v}_i$, updated at each generation $t$:

$$\mathbf{v}_i^{(t+1)} = \omega\,\mathbf{v}_i^{(t)} + c_1\,r_1\,(\mathbf{p}_i - \mathbf{x}_i^{(t)}) + c_2\,r_2\,(\mathbf{g} - \mathbf{x}_i^{(t)})$$

$$\mathbf{x}_i^{(t+1)} = \mathbf{x}_i^{(t)} + \mathbf{v}_i^{(t+1)}$$

where:
- $\omega$ is the inertia weight,
- $c_1, c_2$ are cognitive and social coefficients,
- $r_1, r_2 \sim \mathcal{U}(0,1)$ are random vectors,
- $\mathbf{p}_i$ is the personal best position of particle $i$,
- $\mathbf{g}$ is the global best position.

Velocities are clipped to $[-v_{\max}, v_{\max}]$ and positions to the feasible bounds. One particle is initialized at the zero-perturbation (straight-line baseline) solution.

### 11.3 Grey Wolf Optimizer (GWO)

The GWO models the social hierarchy and hunting behavior of grey wolves. A pack of $P$ wolves is partitioned into three leaders—$\alpha$ (best), $\beta$ (second), $\delta$ (third)—and $\omega$-wolves (rest).

The exploration–exploitation parameter decays linearly:

$$a = 2 - t \cdot \frac{2}{T_{\max}}$$

At each generation, every wolf position is updated by encircling the three leaders:

$$\mathbf{D}_\alpha = \left|\mathbf{C}_1 \cdot \mathbf{X}_\alpha - \mathbf{X}_i\right|, \qquad \mathbf{X}_1 = \mathbf{X}_\alpha - \mathbf{A}_1 \cdot \mathbf{D}_\alpha$$

$$\mathbf{D}_\beta = \left|\mathbf{C}_2 \cdot \mathbf{X}_\beta - \mathbf{X}_i\right|, \qquad \mathbf{X}_2 = \mathbf{X}_\beta - \mathbf{A}_2 \cdot \mathbf{D}_\beta$$

$$\mathbf{D}_\delta = \left|\mathbf{C}_3 \cdot \mathbf{X}_\delta - \mathbf{X}_i\right|, \qquad \mathbf{X}_3 = \mathbf{X}_\delta - \mathbf{A}_3 \cdot \mathbf{D}_\delta$$

$$\mathbf{X}_i^{(t+1)} = \frac{\mathbf{X}_1 + \mathbf{X}_2 + \mathbf{X}_3}{3}$$

where $\mathbf{A}_k = 2a\,\mathbf{r}_1 - a$ and $\mathbf{C}_k = 2\,\mathbf{r}_2$ with $\mathbf{r}_1, \mathbf{r}_2 \sim \mathcal{U}(0,1)$.

### 11.4 Soft Actor–Critic (SAC)

SAC is a maximum-entropy off-policy RL algorithm that augments the standard RL objective with an entropy bonus, encouraging exploration:

$$J(\pi) = \sum_{t=0}^{T} \mathbb{E}\left[r(\mathbf{s}_t, \mathbf{a}_t) + \alpha\,\mathcal{H}\!\left(\pi(\cdot|\mathbf{s}_t)\right)\right]$$

where $\alpha$ is the (auto-tuned) temperature parameter and $\mathcal{H}$ is the policy entropy.

**Environment interface (`DroneTrajectoryEnv`):**

- **Observation space** $\mathcal{S} \subset \mathbb{R}^{22}$:

$$\mathbf{o} = \left(\mathbf{p},\;\mathbf{v},\;\boldsymbol{\Theta},\;\dot{\boldsymbol{\Theta}},\;\boldsymbol{\omega}_{\text{RPM}},\;T_{\Sigma},\;P_{\Sigma},\;n_{\text{remain}},\;\Delta\mathbf{p}_{\text{target}},\;d_{\text{target}}\right)$$

- **Action space** $\mathcal{A} \subset \mathbb{R}^{4}$: $(\delta x,\;\delta y,\;\delta z,\;v)$ — perturbation of the next reference waypoint and cruise speed. An optional 5th dimension acts as a skip gate.

- **Reward**: $r_t = -J_{\text{segment}}$ (negative segment cost).

- **Episode termination**: the agent has placed all $n$ waypoints or reached the final target within the termination distance.

The implementation uses Stable-Baselines3 with `MlpPolicy` and a `DummyVecEnv` wrapper.

### 11.5 Twin Delayed DDPG (TD3)

TD3 extends DDPG with three key modifications to address overestimation bias:

1. **Clipped Double-Q Learning**: two critic networks, take the minimum,
2. **Delayed Policy Updates**: update actor every `policy_delay` critic updates,
3. **Target Policy Smoothing**: add clipped noise to target actions.

$$\tilde{a} = \text{clip}\!\left(\mu_{\theta'}(\mathbf{s}') + \text{clip}(\epsilon,\,-c,\,c),\;a_{\min},\;a_{\max}\right), \qquad \epsilon \sim \mathcal{N}(0, \sigma)$$

The same `DroneTrajectoryEnv` and cost function are used, with optional `NormalActionNoise` for exploration.

---

## 12. Usage Guide

### 12.1 World Creation

Create a custom ground environment using the Tkinter-based GUI:

```bash
python world_creation_gui.py
```

1. Enter the **world size** (meters) and **grid resolution** (meters per cell).
2. Optionally load a background satellite/map image.
3. Select an area type from the dropdown (Housing Estate, Industrial, Open Field, Forbidden).
4. Click and drag to paint rectangular zones.
5. Use **Ctrl+Z** to undo, or **Set Full Area** to fill the entire world.
6. Click **Save World** to export as `.pkl` file to `Worlds/`.

### 12.2 Running Metaheuristic Optimization

**PSO** — via the Jupyter notebook:

```bash
jupyter notebook start_PSO.ipynb
```

Or programmatically:

```python
from main import *
from Optimizations.PSO_optimizer import PSOOptimizer

# Load configuration
params = load_parameters("Settings/simulation_parameters.yaml")

# Build simulation components
init_state = create_initial_state(*params['start_point'])
pid_gains = load_pid_gains(params)
t_max = get_max_thrust_from_rotor_model(params)
controller = create_quadcopter_controller(init_state, pid_gains, t_max, params)
drone = create_quadcopter_model(init_state, controller, params)
world = World.load_world(params['world_data_path'])
noise_model = load_dnn_noise_model(params)
sim = Simulation(drone, world, [], dt=params['dt'], noise_model=noise_model,
                 max_simulation_time=params['simulation_time'])

# Run PSO
A = params['start_point']
B = params['end_point']
optimizer = PSOOptimizer(sim, A, B, config_file="Settings/metaheuristic_parameters.yaml")
best_trajectory = optimizer.start_optimization()
```

**GWO** — similarly via `start_GWO.ipynb` or:

```python
from Optimizations.GWO_optimizer import GWOOptimizer
optimizer = GWOOptimizer(sim, A, B, config_file="Settings/metaheuristic_parameters.yaml")
best_trajectory = optimizer.start_optimization()
```

Results (JSON logs, cost trend plots) are saved to `Optimizations/PSO/<timestamp>/` or `Optimizations/GWO/<timestamp>/`.

### 12.3 Running Reinforcement Learning

**SAC** — via the Jupyter notebook:

```bash
jupyter notebook start_SAC.ipynb
```

Or programmatically:

```python
from RL.trainer import SACTrajectoryTrainer

trainer = SACTrajectoryTrainer(
    config_file="Settings/SAC_parameters.yaml",
    verbose=True
)
best_trajectory = trainer.start_optimization()
```

**TD3:**

```python
from RL.trainer import TD3TrajectoryTrainer

trainer = TD3TrajectoryTrainer(
    config_file="Settings/TD3_parameters.yaml",
    verbose=True
)
best_trajectory = trainer.start_optimization()
```

Training artifacts (model weights, logs, evaluation results) are saved to `RL/SAC/<timestamp>/` or `RL/TD3/<timestamp>/`.

### 12.4 Evaluating a Trajectory

Use `test.py` to simulate and visualize an optimized trajectory:

```python
# In test.py, define waypoints (e.g., from SAC optimization output):
waypoints_optimized = [
    {"x": 27.66, "y": 88.30, "z": 10.92, "v": 18.32},
    {"x": 37.11, "y": 85.37, "z": 23.00, "v": 13.51},
    {"x": 74.18, "y": 89.30, "z": 26.84, "v": 15.42},
    {"x": 95.0,  "y": 50.0,  "z": 1.0,   "v": 15.42}
]
```

```bash
python test.py
```

This runs the full simulation, prints the cost breakdown, and generates:
- 2-D top-down trajectory plot with world overlay,
- 3-D trajectory visualization,
- 3-D animation of the flight.

### 12.5 Visualizing Training Progress

Animate the policy evolution across RL training episodes:

```python
# In show_ani_train.py, configure:
mode = "both"    # "2d", "3d", or "both"
fps = 550
save = False     # Set True to export MP4
```

```bash
python show_ani_train.py
```

---

## 13. Simulation Pipeline

The end-to-end simulation pipeline for a single trajectory evaluation:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          INITIALIZATION                                  │
│  YAML config ──► Drone model + Controller + World + Noise model          │
│                  + Waypoint list + Simulation parameters                  │
└─────────────────────────────────┬────────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼────────────────────────────────────────┐
│                     MAIN SIMULATION LOOP                                 │
│  for t = 0 to T_max (step Δt):                                          │
│    1. Compute dynamic target (look-ahead on current segment)             │
│    2. Controller: state + target ──► (u₁, u₂, u₃, u₄)                   │
│    3. Mixer: (u₁..u₄) ──► (ω₁..ω₄) RPM                                 │
│    4. Rotor model: RPM ──► (T_i, Q_i, P_i, C_T, C_Q, C_P) per rotor    │
│    5. Wind model: apply Dryden turbulence perturbation ΔT per rotor      │
│    6. RK4 integration: state(t) ──► state(t+Δt)                         │
│    7. Ground contact check                                               │
│    8. Every frame_skip steps:                                            │
│       a. Log state, RPM, thrust, power                                   │
│       b. For each ground cell within noise_radius:                        │
│          • Compute distance + elevation angle ζ                          │
│          • Noise model: (ζ, RPMs, d) ──► SPL/SWL per band               │
│          • Broadband summation ──► total SPL                             │
│       c. Compute altitude penalties per area                             │
│    9. Check waypoint transition (advance segment if close enough)         │
│   10. Check termination (final waypoint reached or time exceeded)         │
└─────────────────────────────────┬────────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼────────────────────────────────────────┐
│                     POST-PROCESSING                                      │
│  • Cost function evaluation: J = wₜ·t + wₚ·ΣP + wₙ·SPL + wₐ·Σalt      │
│  • Psychoacoustic map: per-cell Zwicker PA from time-series SPL bands    │
│  • Visualization: 2D/3D plots, trajectory animation, noise heatmap       │
└──────────────────────────────────────────────────────────────────────────┘
```

**Dynamic target strategy**: A look-ahead point is computed along the current segment with look-ahead distance $L = k \cdot v_{\text{des}}$. The drone position is projected onto the segment, and the target is placed $L$ meters ahead of the projection. The target is constrained to never move backward along the segment and is clamped at the segment endpoint.

---

## 14. References

1. S. Mirjalili, S. M. Mirjalili, A. Lewis, "Grey Wolf Optimizer," *Advances in Engineering Software*, vol. 69, pp. 46–61, 2014.
2. J. Kennedy, R. Eberhart, "Particle Swarm Optimization," *Proc. IEEE Int. Conf. Neural Networks*, vol. 4, pp. 1942–1948, 1995.
3. T. Haarnoja, A. Zhou, P. Abbeel, S. Levine, "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor," *Proc. ICML*, pp. 1861–1870, 2018.
4. S. Fujimoto, H. van Hoof, D. Meger, "Addressing Function Approximation Error in Actor-Critic Methods," *Proc. ICML*, pp. 1587–1596, 2018.
5. E. Zwicker, H. Fastl, *Psychoacoustics: Facts and Models*, 3rd ed., Springer, 2007.
6. ISO 532-1:2017, "Acoustics — Methods for calculating loudness — Part 1: Zwicker method."
7. DIN 45692:2009, "Measurement technique for the simulation of the auditory sensation of sharpness."
8. P. Daniel, R. Weber, "Psychoacoustical Roughness: Implementation of an Optimized Model," *Acta Acustica*, vol. 83, pp. 113–123, 1997.
9. ISO 9613-1:1993, "Acoustics — Attenuation of sound during propagation outdoors — Part 1: Calculation of the absorption of sound by the atmosphere."
10. MIL-HDBK-1797, "Flying Qualities of Piloted Aircraft," U.S. Department of Defense, 1997 (Dryden turbulence model).
11. R. W. Beard, T. W. McLain, *Small Unmanned Aircraft: Theory and Practice*, Princeton University Press, 2012.

---

## 15. Author

**Andrea Vaiuso**

---

*This project is provided for research and educational purposes.*