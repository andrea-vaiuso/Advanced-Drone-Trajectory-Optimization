import numpy as np
from Drone.Drone import QuadcopterModel
from Drone.Controller import QuadCopterController
from Drone.Simulation import Simulation
from Utils.plotting_functions import plot3DAnimation, plotLogData, plotNoiseEmissionMap, plotNoiseEmissionHistogram, get_total_PA
from Worlds.World import World
from Noise.DNNModel import RotorSoundModel as DNNModel
from Noise.EmpaModel import NoiseModel as EmpaModel
from Rotor.TorchRotorModel import RotorModel
import yaml
from Noise.Psychoacoustic import PsychoacousticBackendAdapter as PsLib

def load_parameters(parameters_file) -> dict:
    """
    Load configuration from a YAML file.
    
    Parameters:
        config_file (str): Path to the YAML configuration file.
    
    Returns:
        dict: Configuration parameters.
    """
    with open(parameters_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def create_training_waypoints() -> list:
    """
    Create a set of waypoints for training the drone.
    Returns:
        list: List of waypoints with x, y, z coordinates and desired speed.
    """
    return [
        {'x': 10.0, 'y': 10.0, 'z': 70.0, 'v': 5},  # Start near origin at high altitude
        {'x': 90.0, 'y': 10.0, 'z': 70.0, 'v': 5},  # Far in x, near y, maintaining high altitude
        {'x': 90.0, 'y': 90.0, 'z': 90.0, 'v': 5},   # Far in both x and y with even higher altitude
        {'x': 10.0, 'y': 90.0, 'z': 20.0, 'v': 5},   # Sharp maneuver: near x, far y with dramatic altitude drop
        {'x': 50.0, 'y': 50.0, 'z': 40.0, 'v': 5},   # Central target with intermediate altitude
        {'x': 60.0, 'y': 60.0, 'z': 40.0, 'v': 5},   # Hovering target 1
        {'x': 70.0, 'y': 70.0, 'z': 80.0, 'v': 5},   # Hovering target 2 going higher
        {'x': 80.0, 'y': 80.0, 'z': 40.0, 'v': 5},   # Hovering target 3 going lower
        {'x': 10.0, 'y': 10.0, 'z': 1.0, 'v': 5}    # Final target: near origin at low altitude
    ]

def create_random_waypoints(n: int = 10, x_range: tuple = (0, 100), y_range: tuple = (0, 100), z_range: tuple = (0, 100), v: float = 5) -> list:
    """
    Create a list of random waypoints for the drone.
    
    Parameters:
        n (int): Number of waypoints to generate.
        x_range (tuple): Range for x coordinates.
        y_range (tuple): Range for y coordinates.
        z_range (tuple): Range for z coordinates.
        v (float): Desired speed for each waypoint.
    
    Returns:
        list: List of waypoints with x, y, z coordinates and desired speed.
    """
    return [
        {'x': np.random.uniform(*x_range), 
         'y': np.random.uniform(*y_range), 
         'z': np.random.uniform(*z_range), 
         'v': v} for _ in range(n)
    ]

def create_initial_state(x: float = 0, y: float = 0, z: float = 0) -> dict:
    """
    Create the initial state of the drone.
    Parameters:
        x (float): Initial x position.
        y (float): Initial y position.
        z (float): Initial z position.
    Returns:
        dict: Initial state of the drone.
    """
    return {
        'pos': np.array([x, y, z]),
        'vel': np.array([0.0, 0.0, 0.0]),
        'angles': np.array([0.0, 0.0, 0.0]),  # roll, pitch, yaw
        'ang_vel': np.array([0.0, 0.0, 0.0]),
        'rpm': np.array([0.0, 0.0, 0.0, 0.0]),
        'thrust': 0.0,
        'torque': 0.0,
        'power': 0.0,
    }

def get_max_thrust_from_rotor_model(parameters) -> RotorModel:
    """
    Get the maximum thrust from the rotor model based on the max RPM value and number of rotors.

    Parameters:
        parameters (dict): Configuration parameters including rotor model path and max RPM.
    Returns:
        float: Maximum thrust value.
    """
    rotor_model = RotorModel(norm_params_path=parameters['norm_params_path'])
    rotor_model.load_model(parameters['rotor_model_path'])
    thrust_max, _, _, _, _, _ = rotor_model.predict_aerodynamic(float(parameters['max_rpm']))
    return thrust_max * int(parameters['n_rotors'])

def load_pid_gains(parameters) -> dict:
    """
    Load PID gains from the configuration parameters.

    Parameters:
        parameters (dict): Configuration parameters.
    
    Returns:
        dict: PID gains for position, altitude, attitude, horizontal speed, and vertical speed.
    """
    return {
        'k_pid_pos': tuple(map(float, parameters['k_pid_pos'])),
        'k_pid_alt': tuple(map(float, parameters['k_pid_alt'])),
        'k_pid_att': tuple(map(float, parameters['k_pid_att'])),
        'k_pid_yaw': tuple(map(float, parameters['k_pid_yaw'])),
        'k_pid_hsp': tuple(map(float, parameters['k_pid_hsp'])),
        'k_pid_vsp': tuple(map(float, parameters['k_pid_vsp'])),
    }

def create_quadcopter_controller(init_state, pid_gains, t_max, parameters) -> QuadCopterController:
    """
    Create a quadcopter controller with the given state and PID gains.

    Parameters:
        init_state (dict): Initial state of the drone.
        pid_gains (dict): PID gains for the controller.
        t_max (float): Maximum thrust command limit.
        max_roll_angle (float): Maximum roll angle limit.
        max_pitch_angle (float): Maximum pitch angle limit.
        max_yaw_angle (float): Maximum yaw angle limit.

    Returns:
        QuadCopterController: Initialized quadcopter controller.
    """
    return QuadCopterController(
        init_state, 
        pid_gains,
        thrust_command_limit=t_max, 
        roll_command_limit=float(parameters['max_roll_angle']), 
        pitch_command_limit=float(parameters['max_pitch_angle']), 
        yaw_command_limit=float(parameters['max_yaw_angle']),
        max_h_speed_limit_kmh=float(parameters['max_h_speed_limit_kmh']),
        max_v_speed_limit_kmh=float(parameters['max_v_speed_limit_kmh']),
        max_angle_limit_deg=float(parameters['max_angle_limit_deg']),
        anti_windup_contrib=float(parameters['anti_windup_contrib'])
    )

def create_quadcopter_model(init_state, quad_controller, parameters) -> QuadcopterModel:
    """
    Create a quadcopter model with the given initial state and controller.

    Parameters:
        init_state (dict): Initial state of the drone.
        quad_controller (QuadCopterController): The quadcopter controller.
        parameters (dict): Configuration parameters for the quadcopter model.

    Returns:
        QuadcopterModel: Initialized quadcopter model.
    """
    return QuadcopterModel(
        m=float(parameters['m']),
        I=list(map(float, parameters['I'])),
        d=float(parameters['d']),
        l=float(parameters['l']),
        Cd=list(map(float, parameters['Cd'])),
        Ca=list(map(float, parameters['Ca'])),
        Jr=float(parameters['Jr']),
        init_state=init_state,
        controller=quad_controller,
        n_rotors=int(parameters['n_rotors']),  # Number of rotors
        rotor_model_path=parameters['rotor_model_path'],  # Path to the pre-trained rotor model
        max_rpm=float(parameters['max_rpm']),  # Maximum RPM for the motors
    )

def load_dnn_noise_model(parameters) -> DNNModel:
    """
    Load the DNN noise model for the simulation.

    Parameters:
        parameters (dict): Configuration parameters including DNN model filename and RPM reference.

    Returns:
        DNNModel: Initialized DNN noise model.
    """
    return DNNModel(
        rpm_reference=float(parameters['rpm_reference']),
        lookup_table_filename=parameters['dnn_model_filename']
    )

def load_empa_noise_model(parameters) -> EmpaModel:
    """
    Load the EMPA noise model for the simulation.

    Parameters:
        parameters (dict): Configuration parameters including EMPA model filenames.

    Returns:
        EmpaModel: Initialized EMPA noise model.
    """
    noise_model = EmpaModel(
        scaler_filename=parameters['empa_model_scaling_filename']
    )
    noise_model.load_model(parameters['empa_model_filename'])
    return noise_model

def generate_log_dict(sim: Simulation) -> dict:
    return {
        'X Position': {
            'data': np.array(sim.positions)[:, 0],
            'ylabel': 'X (m)',
            'color': 'tab:blue',
            'linestyle': '-',
            'label': 'X Position',
            'showgrid': True
        },
        'Y Position': {
            'data': np.array(sim.positions)[:, 1],
            'ylabel': 'Y (m)',
            'color': 'tab:orange',
            'linestyle': '-',
            'label': 'Y Position',
            'showgrid': True
        },
        'Z Position': {
            'data': np.array(sim.positions)[:, 2],
            'ylabel': 'Z (m)',
            'color': 'tab:green',
            'linestyle': '-',
            'label': 'Z Position',
            'showgrid': True
        },
        'Attitude (Pitch, Roll, Yaw)': {
            'data': {
                'Pitch': np.array(sim.angles_history)[:, 0],
                'Roll':  np.array(sim.angles_history)[:, 1],
                'Yaw':   np.array(sim.angles_history)[:, 2]
            },
            'ylabel': 'Angle (rad)',
            'ylim': (-1.5, 1.5),
            'styles': {
                'Pitch': {'color': 'tab:blue',   'linestyle': '-',  'label': 'Pitch'},
                'Roll':  {'color': 'tab:orange', 'linestyle': '--', 'label': 'Roll'},
                'Yaw':   {'color': 'tab:green',  'linestyle': '-.', 'label': 'Yaw'}
            }
        },
        'Motor RPMs': {
            'data': np.array(sim.rpms_history),
            'ylabel': 'RPM',
            'ylim': (0, 3000),
            'colors': ['tab:blue', 'tab:orange', 'tab:green', 'tab:red'],
            'linestyles': ['-', '--', '-.', ':'],
            'labels': ['RPM1 (FR)', 'RPM2 (RR)', 'RPM3 (RL)', 'RPM4 (FL)']
        },
        'Speeds': {
            'data': {
                'Horizontal Speed': np.array(sim.horiz_speed_history),
                'Vertical Speed':   np.array(sim.vertical_speed_history)
            },
            'ylabel': 'Speed (m/s)',
            'styles': {
                'Horizontal Speed': {'color': 'r', 'linestyle': '-', 'label': 'Horizontal Speed'},
                'Vertical Speed':   {'color': 'g', 'linestyle': '-', 'label': 'Vertical Speed'}
            }
        },
        'Sound Pressure Level (SPL)': {
            'data': np.array(sim.spl_history),
            'ylabel': 'Level (dB)',
            'color': 'orange',
            'linestyle': '-',
            'ylim': (20, 100),
            'label': 'SPL',
            'calc_average': True,
            'showgrid': True
        },
        'Sound Power Level (SWL)': {
            'data': np.array(sim.swl_history),
            'ylabel': 'Level (dB)',
            'color': 'purple',
            'linestyle': '-',
            'ylim': (20, 130),
            'label': 'SWL',
            'calc_average': True,
            'showgrid': True
        },
        'Thrust': {
            'data': np.array(sim.thrust_history),
            'ylabel': 'Thrust (N)',
            'color': 'brown',
            'linestyle': '-',
            'ylim': (0, 50),
            'label': 'Thrust',
            'showgrid': True
        },
        'Power': {
            'data': np.array(sim.power_history),
            'ylabel': 'Power (W)',
            'color': 'green',
            'linestyle': '-',
            'ylim': (0, 1000),
            'label': 'Power',
            'calc_average': True,
            'showgrid': True
        },
        'Wind Thrust': {
            'data': np.array(sim.delta_t_history),
            'ylabel': 'Wind Thrust (N)',
            'color': 'cyan',
            'linestyle': '-',
            'label': 'Wind Thrust',
            'showgrid': True
        },
    }
