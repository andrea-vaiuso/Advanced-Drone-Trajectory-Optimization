from main import *
from Utils.plotting_functions import *
from Worlds.World import World
import Drone.Simulation
from matplotlib import pyplot as plt
from Optimizations.PSO_optimizer import PSOOptimizer

waypoints_optimized_gwo = [{"x": 140.40213377112582, "y": 183.7275071814119, "z": 96.5041475402006, "v": 12.483198287987207}, {"x": 311.8147622870915, "y": 88.24761446888775, "z": 135.97085753787363, "v": 14.89378678892961}, {"x": 735.0, "y": 278.9333712238131, "z": 255.0, "v": 17.10721740613379}, {"x": 890.0, "y": 648.9370405977522, "z": 251.6355139697522, "v": 19.343310829474486}, {"x": 952.8509545738897, "y": 802.7353159664835, "z": 172.72840820090582, "v": 11.538788429152518}, {"x": 950.0, "y": 950.0, "z": 10.0, "v": 5.0}]
waypoints_optimized_sac = [
        {
            "x": 0.0,
            "y": 0.0,
            "z": 238.09453097256747,
            "v": 19.661514282226562
        },
        {
            "x": 0.0,
            "y": 0.0,
            "z": 243.7912958318537,
            "v": 19.997188568115234
        },
        {
            "x": 23.636363636363626,
            "y": 23.636363636363626,
            "z": 252.72727272727272,
            "v": 19.99966812133789
        },
        {
            "x": 108.18181818181819,
            "y": 108.18181818181819,
            "z": 253.6363331187855,
            "v": 19.999998092651367
        },
        {
            "x": 192.7272727272727,
            "y": 192.7272727272727,
            "z": 254.54545454545453,
            "v": 19.99997329711914
        },
        {
            "x": 277.27272727272725,
            "y": 277.27272727272725,
            "z": 252.84086747602984,
            "v": 19.926530838012695
        },
        {
            "x": 858.6564691716974,
            "y": 361.81818181818176,
            "z": 256.3636363636364,
            "v": 6.316094398498535
        },
        {
            "x": 946.355244029652,
            "y": 446.3636363636364,
            "z": 249.64852072975853,
            "v": 19.991188049316406
        },
        {
            "x": 1000.0,
            "y": 530.909090909091,
            "z": 258.18166559392756,
            "v": 2.0
        },
        {
            "x": 1000.0,
            "y": 615.4545454545454,
            "z": 259.09090909090907,
            "v": 19.96247673034668
        },
        {
            "x": 950.0,
            "y": 950.0,
            "z": 10.0,
            "v": 20.0
        }
    ]
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
    waypoints_optimized_sac, 
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
    world, sim.positions, A, B, waypoints_optimized_sac)
plot3DAnimation(sim, window=(sim.world.max_world_size,sim.world.max_world_size,sim.world.max_world_size))