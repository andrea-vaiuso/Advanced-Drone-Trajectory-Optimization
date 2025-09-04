# Author: Andrea Vaiuso
# Version: 2.1
# Date: 24.07.2025
# Description: This module defines the PIDController and QuadCopterController classes for controlling a quadcopter drone.
# It implements PID control for position, altitude, attitude, and yaw, and computes control commands based on the drone's state.
# Includes a simple anti-windup mechanism for the PID controllers.

import numpy as np

class PIDController:
    def __init__(self, gains: tuple, windup_limit: float = 100.0, name: str = "generic"):
        """Initialize the PID controller.

        Parameters:
            name (str): Name of the PID controller.
            gains (tuple[float, float, float]): The ``(kp, ki, kd)`` PID gains.
            windup_limit (float): Maximum absolute value for the integral term
                (anti-windup).
        """
        self.name = name
        self.kp, self.ki, self.kd = gains
        self.error = 0.0
        self.derivative = 0.0
        self.integral = 0.0
        self.prev_error = 0.0
        self.output = 0.0
        self.integral_limit = abs(windup_limit)

    def update(self, current_value: float, target_value: float, dt: float) -> float:
        """
        Compute the PID controller output with anti-windup.

        Parameters:
            current_value (float): The current measurement.
            target_value (float): The desired setpoint.
            dt (float): Time step.

        Returns:
            float: Control output.
        """
        self.error = target_value - current_value
        self.integral += self.error * dt
        # Anti-windup: clamp the integral term
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        self.derivative = (self.error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = self.error
        self.output = self.kp * self.error + self.ki * self.integral + self.kd * self.derivative
        return self.output

    def __str__(self):
        return f"PID {self.name}: (err={self.error}, deriv={self.derivative}, integ={self.integral}, output={self.output})"

class QuadCopterController:
    def __init__(self, state: dict,
                 pid_gains: dict,
                 thrust_command_limit: float = 100.0, roll_command_limit: float = 10.0, 
                 pitch_command_limit: float = 10.0, yaw_command_limit: float = 10.0,
                 max_h_speed_limit_kmh: float = 50.0,
                 max_v_speed_limit_kmh: float = 20.0,
                 max_angle_limit_deg: float = 30.0,
                 anti_windup_contrib: float = 0.4):
        """
        Initialize the quadcopter controller with PID controllers for position, altitude, attitude, and yaw.

        Parameters:
            state (dict): Current state of the drone.
            pid_gains (dict): Dictionary containing PID gains for position, altitude, attitude, and yaw.
            thrust_command_limit, roll_command_limit, pitch_command_limit, yaw_command_limit (float): Saturation limits for the control commands.
            max_h_speed_limit_kmh (float): Maximum horizontal speed limit in km/h.
            max_v_speed_limit_kmh (float): Maximum vertical speed limit in km/h.
            max_angle_limit_deg (float): Maximum angle limit in degrees for roll and pitch commands.
            anti_windup_contrib (float): Contribution factor for anti-windup automatic system.
        """
        self.u1_limit = thrust_command_limit
        self.u2_limit = yaw_command_limit
        self.u3_limit = roll_command_limit
        self.u4_limit = pitch_command_limit

        self.max_h_speed_limit = max_h_speed_limit_kmh / 3.6
        self.max_v_speed_limit = max_v_speed_limit_kmh / 3.6
        
        self.max_angle_limit_rad = np.radians(max_angle_limit_deg)  # Convert to radians
        
        pos, alt, att, yaw_g, hsp, vsp = self.unpack_pid_gains(pid_gains)

        # Position PIDs
        self.pid_x = PIDController(pos, self.u3_limit * anti_windup_contrib, name="pos_x")
        self.pid_y = PIDController(pos, self.u2_limit * anti_windup_contrib, name="pos_y")
        self.pid_z = PIDController(alt, self.u1_limit * anti_windup_contrib, name="pos_z")

        # Attitude PIDs (roll and pitch)
        self.pid_roll  = PIDController(att, self.u2_limit * anti_windup_contrib, name="roll")
        self.pid_pitch = PIDController(att, self.u3_limit * anti_windup_contrib, name="pitch")

        # Yaw PID
        self.pid_yaw   = PIDController(yaw_g, self.u4_limit * anti_windup_contrib, name="yaw")

        # Speed PIDs (horizontal and vertical)
        self.pid_h_speed = PIDController(hsp, self.max_h_speed_limit * anti_windup_contrib, name="h_speed")
        self.pid_v_speed = PIDController(vsp, self.max_v_speed_limit * anti_windup_contrib, name="v_speed")

        self.state = state

    def update(self, state: dict, target: dict, dt: float, m: float, g: float = 9.81, desired_v_speed: float = None, desired_h_speed: float = None) -> tuple:
        """
        Compute the control commands for the quadcopter.

        Parameters:
            state (dict): Current state of the drone.
            target (dict): Target position with keys 'x', 'y', and 'z'.
            dt (float): Time step.
            m (float): Mass of the drone.
            g (float): Gravity acceleration (default 9.81 m/s^2).
            vert_speed_limit (float): Maximum vertical speed limit.
            horiz_speed_limit (float): Maximum horizontal speed limit.

        Returns:
            tuple: (thrust_command, roll_command, pitch_command, yaw_command)
        """
        x, y, z = state['pos']
        roll, pitch, yaw = state['angles'] # In radians
        x_t, y_t, z_t = target['x'], target['y'], target['z']
        v_x, v_y, v_z = state['vel']

        # if desired speeds are None, set to max limits
        if desired_v_speed is None:
            desired_v_speed = self.max_v_speed_limit
        if desired_h_speed is None:
            desired_h_speed = self.max_h_speed_limit

        # clamp desired speeds to limits
        desired_v_speed = np.clip(desired_v_speed, -self.max_v_speed_limit, self.max_v_speed_limit)
        desired_h_speed = np.clip(desired_h_speed, -self.max_h_speed_limit, self.max_h_speed_limit)

        # Outer loop: position control with feed-forward for hover
        compensation = np.clip(1.0 / (np.cos(pitch) * np.cos(roll)), 1.0, 1.5) # Compensation factor for hover thrust
        hover_thrust = m * g * compensation # Hover thrust based on mass and gravity

        vz_t = self.pid_z.update(z, z_t, dt)
        vz_t = np.clip(vz_t, -desired_v_speed, desired_v_speed)

        vz_output = self.pid_v_speed.update(v_z, vz_t, dt)
        thrust_command = np.clip(hover_thrust + vz_output, 0, self.u1_limit)

        # Step 1: Position controllers compute desired velocities (vx_t, vy_t)
        vx_t = self.pid_x.update(x, x_t, dt)
        vy_t = self.pid_y.update(y, y_t, dt)
        # Compute total desired horizontal speed and clip if necessary
        v_horiz = np.array([vx_t, vy_t])
        norm = np.linalg.norm(v_horiz)
        if norm > desired_h_speed and norm > 0:
            v_horiz = v_horiz * (desired_h_speed / norm)
        vx_t, vy_t = v_horiz[0], v_horiz[1]
        # Step 2: Speed controllers compute desired angles from velocity errors
        pitch_des = np.clip(self.pid_h_speed.update(v_x, vx_t, dt), -self.max_angle_limit_rad, self.max_angle_limit_rad)
        roll_des  = np.clip(-self.pid_h_speed.update(v_y, vy_t, dt), -self.max_angle_limit_rad, self.max_angle_limit_rad)
        # Compute desired yaw based on target position
        dx = target['x'] - x
        dy = target['y'] - y
        yaw_des = np.arctan2(dy, dx)
        
        # Inner loop: attitude control
        roll_command = self.pid_roll.update(roll, roll_des, dt) # Roll command based on desired roll
        pitch_command = self.pid_pitch.update(pitch, pitch_des, dt) # Pitch command based on desired pitch
        yaw_command = self.pid_yaw.update(yaw, 0, dt)  # Yaw command based on desired yaw (set to 0 for now)

        # Saturate the commands
        thrust_command = np.clip(thrust_command, 0, self.u1_limit) 
        roll_command = np.clip(roll_command, -self.u2_limit, self.u2_limit) 
        pitch_command = np.clip(pitch_command, -self.u3_limit, self.u3_limit)
        yaw_command = np.clip(yaw_command, -self.u4_limit, self.u4_limit)

        return (thrust_command, roll_command, pitch_command, yaw_command)
    
    def print_pid_status(self):
        """
        Print the status of all PID controllers.
        """
        print(f"QuadCopterController Status:")
        print(f"{self.pid_x}")
        print(f"{self.pid_y}")
        print(f"{self.pid_z}")
        print(f"{self.pid_roll}")
        print(f"{self.pid_pitch}")
        print(f"{self.pid_yaw}")
        print(f"{self.pid_h_speed}")
        print(f"{self.pid_v_speed}")

    @staticmethod
    def unpack_pid_gains(pid_gains: dict) -> tuple:
        """
        Unpack PID gains from a dictionary.

        Parameters:
            pid_gains (dict): Dictionary containing PID gains.

        Returns:
            tuple: Tuples of PID gains for position, altitude, attitude, yaw,
                horizontal speed and vertical speed controllers.
        """
        pos = tuple(pid_gains['k_pid_pos'])
        alt = tuple(pid_gains['k_pid_alt'])
        att = tuple(pid_gains['k_pid_att'])
        yaw = tuple(pid_gains['k_pid_yaw'])
        hsp = tuple(pid_gains['k_pid_hsp'])
        vsp = tuple(pid_gains['k_pid_vsp'])

        return pos, alt, att, yaw, hsp, vsp