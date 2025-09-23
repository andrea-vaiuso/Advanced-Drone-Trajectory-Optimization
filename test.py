from main import *
from Utils.plotting_functions import *
from Worlds.World import World
import Drone.Simulation
from matplotlib import pyplot as plt
from Optimizations.PSO_optimizer import PSOOptimizer

waypoints_optimized = [{"x": 140.40213377112582, "y": 183.7275071814119, "z": 96.5041475402006, "v": 12.483198287987207}, {"x": 311.8147622870915, "y": 88.24761446888775, "z": 135.97085753787363, "v": 14.89378678892961}, {"x": 735.0, "y": 278.9333712238131, "z": 255.0, "v": 17.10721740613379}, {"x": 890.0, "y": 648.9370405977522, "z": 251.6355139697522, "v": 19.343310829474486}, {"x": 952.8509545738897, "y": 802.7353159664835, "z": 172.72840820090582, "v": 11.538788429152518}, {"x": 950.0, "y": 950.0, "z": 10.0, "v": 5.0}]
parameters = load_parameters("Settings/simulation_parameters.yaml")

A = parameters['start_point']
B = parameters['end_point']

init_state = create_initial_state(A[0], A[1], A[2])


thrust_max = get_max_thrust_from_rotor_model(parameters)
pid_gains = load_pid_gains(parameters)
quad_controller = create_quadcopter_controller(init_state, pid_gains, thrust_max, parameters)
drone = create_quadcopter_model(init_state, quad_controller, parameters)
world = World.load_world(parameters['world_data_path'])
noise_model = load_dnn_noise_model(parameters)

sim = Simulation(
    drone,
    world,
    waypoints_optimized, 
    dt=float(parameters['dt']),
    max_simulation_time=float(parameters['simulation_time']),
    frame_skip=int(parameters['frame_skip']),
    target_reached_threshold=float(parameters['threshold']),
    target_shift_threshold_distance=float(parameters['target_shift_threshold_distance']),
    noise_model=noise_model,
    generate_sound_emission_map=True,
    compute_psychoacoustics=False,
    noise_annoyance_radius=9,
)

sim.startSimulation(stop_at_target=True, use_static_target=False, verbose=True)
show2DWorld(
    world, sim.positions, A, B, waypoints_optimized)
plot3DAnimation(sim, window=(sim.world.max_world_size,sim.world.max_world_size,sim.world.max_world_size))