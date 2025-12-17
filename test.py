from main import *
from Utils.plotting_functions import *
from Worlds.World import World
import Drone.Simulation
from matplotlib import pyplot as plt
from Optimizations.PSO_optimizer import PSOOptimizer
from Optimizations.optimizer import Optimizer as CostWrapper


waypoints_optimized_sac = [{"x": 27.664927049116653, "y": 88.29509735107422, "z": 10.923457145690918, "v": 18.322269439697266}, {"x": 37.11072626980868, "y": 85.373291015625, "z": 23.00494384765625, "v": 13.508941650390625}, {"x": 74.18110830133611, "y": 89.29581451416016, "z": 26.83894920349121, "v": 15.422500610351562}, {"x": 95.0, "y": 50.0, "z": 1.0, "v": 15.422500610351562}]

parameters = load_parameters("Settings/Test/simulation_parameters_test_env.yaml")

A = parameters['start_point']
B = parameters['end_point']

init_state = create_initial_state(A[0], A[1], A[2])


waypoints_selected = waypoints_optimized_sac


thrust_max = get_max_thrust_from_rotor_model(parameters)
pid_gains = load_pid_gains(parameters)
quad_controller = create_quadcopter_controller(init_state, pid_gains, thrust_max, parameters)
drone = create_quadcopter_model(init_state, quad_controller, parameters)
world = World.load_world(parameters['world_data_path'])
noise_model = load_dnn_noise_model(parameters)


sim = Simulation(
    drone,
    world,
    waypoints_selected,
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


cw = CostWrapper(sim, "")

cw.simulation_object.startSimulation(stop_at_target=True, use_static_target=False, verbose=True)
print(f"Costs: {cw.calculate_costs()}")
max_world_size = cw.simulation_object.world.max_world_size
show2DWorld(
    world, cw.simulation_object.positions, A, B, waypoints_selected, 
    horiz_speed_history=cw.simulation_object.horiz_speed_history, 
    vertical_speed_history=cw.simulation_object.vertical_speed_history)
show3DWorld(
    world, cw.simulation_object.positions, A, B, waypoints_selected, 
    horiz_speed_history=cw.simulation_object.horiz_speed_history, 
    vertical_speed_history=cw.simulation_object.vertical_speed_history,
    view_pov=(30, 45))