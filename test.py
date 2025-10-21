from main import *
from Utils.plotting_functions import *
from Worlds.World import World
import Drone.Simulation
from matplotlib import pyplot as plt
from Optimizations.PSO_optimizer import PSOOptimizer
from Optimizations.optimizer import MetaHeuristicOptimizer as CostWrapper

waypoints_optimized_gwo = [
        {
            "x": 202.4473762898855,
            "y": 241.18585339735836,
            "z": 133.9965963325619,
            "v": 5.000508959435863
        },
        {
            "x": 370.53592444302507,
            "y": 391.939053599772,
            "z": 190.6480506456245,
            "v": 14.126968636724628
        },
        {
            "x": 464.68914665275946,
            "y": 438.20038895203527,
            "z": 161.42151740995882,
            "v": 13.08638316190654
        },
        {
            "x": 723.1410408519813,
            "y": 520.6600754915677,
            "z": 184.71164572001788,
            "v": 20.0
        },
        {
            "x": 984.2136899207717,
            "y": 721.1086667123632,
            "z": 138.6151991413981,
            "v": 15.29579473875952
        },
        {
            "x": 950.0,
            "y": 950.0,
            "z": 10.0,
            "v": 5.0
        }
    ]
waypoints_optimized_sac = [
        {
            "x": 212.3166032270952,
            "y": 10.021315141157672,
            "z": 247.8653897372159,
            "v": 19.84721565246582
        },
        {
            "x": 323.6363650235263,
            "y": 84.5454531582919,
            "z": 251.8181818181818,
            "v": 20.0
        },
        {
            "x": 408.1818195689808,
            "y": 169.09090770374644,
            "z": 252.72727272727272,
            "v": 20.0
        },
        {
            "x": 492.7272741144354,
            "y": 253.636362249201,
            "z": 253.63636363636363,
            "v": 20.0
        },
        {
            "x": 577.2727286598899,
            "y": 338.1818167946555,
            "z": 254.54545454545453,
            "v": 20.0
        },
        {
            "x": 661.8181832053444,
            "y": 422.72727134011006,
            "z": 255.45454545454547,
            "v": 20.0
        },
        {
            "x": 746.363637750799,
            "y": 507.27272588556457,
            "z": 256.3636363636364,
            "v": 20.0
        },
        {
            "x": 830.9090922962536,
            "y": 591.8181804310192,
            "z": 257.27272727272725,
            "v": 20.0
        },
        {
            "x": 915.4545468417082,
            "y": 676.3636349764738,
            "z": 258.1818181818182,
            "v": 20.0
        },
        {
            "x": 1000.0000013871626,
            "y": 760.9090895219282,
            "z": 259.09090909090907,
            "v": 20.0
        },
        {
            "x": 950.0,
            "y": 950.0,
            "z": 10.0,
            "v": 20.0
        }
    ]
waypoints_manual = [{
    "x": 450.0,
    "y": 150.0,
    "z": 200.0,
    "v": 20.0
}, {
    "x": 800.0,
    "y": 430.0,
    "z": 200.0,
    "v": 15.0
}, {
    "x": 930.0,
    "y": 440.0,
    "z": 200.0,
    "v": 20.0
}, {
    "x": 950.0,
    "y": 950.0,
    "z": 10.0,
    "v": 20.0
}
]
no_waypoints = [
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
    world, cw.simulation_object.positions, A, B, waypoints_selected)
plot3DAnimation(cw.simulation_object, window=(max_world_size,max_world_size,max_world_size))

 