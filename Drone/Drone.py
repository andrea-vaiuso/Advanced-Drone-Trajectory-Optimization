# Author: Andrea Vaiuso
# Version: 3.0
# Date: 18.07.2025
# Description: This module defines the QuadcopterModel class, which simulates the dynamics of a quadcopter drone using BEMT.
# It includes methods for translational and rotational dynamics, state updates, and wind effects.

import numpy as np
from Controller import QuadCopterController
from utils import wrap_angle
import numpy as np
from Rotor.TorchRotorModel import RotorModel


class QuadcopterModel:
    def __init__(self, m: float, I: np.ndarray, d: float, l: float, Cd: float, 
                Ca: np.ndarray, Jr: float,
                init_state: dict, controller: QuadCopterController, n_rotors = 4,
                max_rpm: float = 8000.0, rotor_model_path: str = 'Rotor/rotor_model.pth', rotor_data_path: str = 'Rotor/rotor_config.ini'):
        """
        Initialize the physical model of the quadcopter.

        Parameters:
            m (float): Mass.
            I (np.ndarray): Moment of inertia vector.
            d (float): Drag factor.
            l (float): Distance from the center to the rotor.
            Cd (np.ndarray): Traslational drag coefficients.
            Ca (np.ndarray): Aerodynamic friction coefficients.
            Jr (float): Rotor inertia.
            init_state (dict): Initial state of the quadcopter.
            controller (QuadCopterController): Controller for the quadcopter.
            n_rotors (int): Number of rotors. Default is 4 for a quadcopter.
            max_rpm (float): Maximum RPM for the motors.
            rotor_model_path (str): Path to the pre-trained rotor model. Default is 'Rotor/rotor_model.pth'.
            rotor_data_path (str): Path to the rotor configuration file. Default is 'Rotor/rotor_config.ini'.
        """

        self.rho = 1.225  # Air density in kg/m³
        self.g = 9.81

        self.m = m
        self.I = I
        self.d = d
        self.l = l
        self.Cd = Cd
        self.Ca = Ca
        self.Jr = Jr
        self.state = init_state
        self.init_state = init_state.copy()  # Store the initial state for reset
        self.controller = controller
        self.max_rpm = max_rpm
        self.delta_T = np.zeros(n_rotors)
        self.thrust = 0.0
        self.thrust_no_wind = 0.0  # Thrust without wind effect

        # From rotor ini file, read rotor radius
        with open(rotor_data_path, 'r') as f:
            for line in f:
                if line.startswith("radius"):
                    radius_sections = line.split("=")[1].strip()
                    self.R_root = float(radius_sections.split(" ")[-1].strip())
                    self.R_tip = float(radius_sections.split(" ")[0].strip())
        
        self.rotors: list[RotorModel] = []
        for _ in range(n_rotors):
            rotor_model = RotorModel(1,6, norm_params_path="Rotor/normalization_params.pth")
            rotor_model.load_model(rotor_model_path)
            self.rotors.append(rotor_model)
        
        self.max_rpm_sq = (self.max_rpm * 2 * np.pi / 60)**2 # Maximum RPM squared for clipping

        self.c_t = 0
        self.c_q = 0


    def compute_hover_rpm(self, c_t) -> None:
        """
        Compute the RPM value needed for hovering flight nondepending on thrust coefficient.
        """
        T_hover = self.m * self.g  # Hover thrust needed to balance weight
        w_hover = np.sqrt(T_hover / (4 * c_t))
        rpm_hover = w_hover * 60.0 / (2.0 * np.pi)
        return rpm_hover
        # Uncomment the following line for debug information:
        # print(f"[INFO] Hover thrust needed = {T_hover:.2f} N, hover rpm per motor ~ {rpm_hover:.1f} rpm")

    def __str__(self) -> str:
        """
        Return a string representation of the quadcopter model.
        """
        return f"Quadcopter Model: state = {self.state}"

    def _translational_dynamics(self, state: dict) -> np.ndarray:
        """
        Compute the translational accelerations.

        Parameters:
            state (dict): Current state.

        Returns:
            np.ndarray: Acceleration vector [x_ddot, y_ddot, z_ddot].
        """
        x_dot, y_dot, z_dot = state['vel']
        roll, pitch, yaw = state['angles'] # Respect to the world frame in radians
        T = np.sum(state['thrust'])  # Total thrust from all rotors

        x_ddot = (T / self.m *
                  (np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll))
                  - self.Cd[0] / self.m * x_dot)
        y_ddot = (T / self.m *
                  (np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll))
                  - self.Cd[1] / self.m * y_dot)
        z_ddot = (T / self.m *
                  (np.cos(pitch) * np.cos(roll))
                  - self.Cd[2] / self.m * z_dot - self.g)

        return np.array([x_ddot, y_ddot, z_ddot])

    def _rotational_dynamics(self, state: dict) -> np.ndarray:
        """
        Compute the rotational accelerations.

        Parameters:
            state (dict): Current state.

        Returns:
            np.ndarray: Angular acceleration vector [phi_ddot, theta_ddot, psi_ddot].
        """
        omega = self.get_omega()
        phi_dot, theta_dot, psi_dot = state['ang_vel']
        
        u_2 = self.l * (state['thrust'][3] - state['thrust'][1])
        u_3 = self.l * (state['thrust'][2] - state['thrust'][0])
        u_4 = state['torque'][0] - state['torque'][1] + state['torque'][2] - state['torque'][3]

        Omega_r = (omega[0] - omega[1] + omega[2] - omega[3])
        Omega_r_J_r = self.Jr * Omega_r 

        phi_ddot = (u_2 / self.I[0]
                    - self.Ca[0] * np.sign(phi_dot) * phi_dot**2 / self.I[0]
                    - Omega_r_J_r / self.I[0] * theta_dot
                    - (self.I[2] - self.I[1]) / self.I[0] * theta_dot * psi_dot)
        theta_ddot = (u_3 / self.I[1]
                      - self.Ca[1] * np.sign(theta_dot) * theta_dot**2 / self.I[1]
                      + Omega_r_J_r / self.I[1] * phi_dot
                      - (self.I[0] - self.I[2]) / self.I[1] * phi_dot * psi_dot)
        psi_ddot = (u_4 / self.I[2]
                    - self.Ca[2] * np.sign(psi_dot) * psi_dot**2 / self.I[2]
                    - (self.I[1] - self.I[0]) / self.I[2] * phi_dot * theta_dot)

        return np.array([phi_ddot, theta_ddot, psi_ddot])

    def _mixer(self, u1: float, u2: float, u3: float, u4: float) -> tuple:
        """
        Compute the RPM for each motor based on the control inputs.

        Parameters:
            u1, u2, u3, u4 (float): Control inputs.
        Returns:
            tuple: RPM values for each motor.
        """
        b = 0.0007
        d = 7.5e-7
        l = self.l
        
        w1_sq = (u1 / (4 * b)) - (u3 / (2 * b * l)) + (u4 / (4 * d))
        w2_sq = (u1 / (4 * b)) - (u2 / (2 * b * l)) - (u4 / (4 * d))
        w3_sq = (u1 / (4 * b)) + (u3 / (2 * b * l)) + (u4 / (4 * d))
        w4_sq = (u1 / (4 * b)) + (u2 / (2 * b * l)) - (u4 / (4 * d))
        
        w1_sq = np.clip(w1_sq, 0.0, self.max_rpm_sq)
        w2_sq = np.clip(w2_sq, 0.0, self.max_rpm_sq)
        w3_sq = np.clip(w3_sq, 0.0, self.max_rpm_sq)
        w4_sq = np.clip(w4_sq, 0.0, self.max_rpm_sq)
        
        w1 = np.sqrt(w1_sq)
        w2 = np.sqrt(w2_sq)
        w3 = np.sqrt(w3_sq)
        w4 = np.sqrt(w4_sq)
        
        rpm1 = w1 * 60.0 / (2.0 * np.pi)
        rpm2 = w2 * 60.0 / (2.0 * np.pi)
        rpm3 = w3 * 60.0 / (2.0 * np.pi)
        rpm4 = w4 * 60.0 / (2.0 * np.pi)

        return rpm1, rpm2, rpm3, rpm4

    def _rk4_step(self, state_old: dict, dt: float) -> dict: #Check physical meaning
        """
        Performs a single integration step using the classical 4th-order Runge-Kutta (RK4) method.
        The RK4 method is a numerical technique for solving ordinary differential equations (ODEs).
        It estimates the state of the system at the next time step by computing four increments (k1, k2, k3, k4),
        each representing the derivative of the state at different points within the interval. These increments
        are combined to produce a weighted average, providing a more accurate estimate than simpler methods
        like Euler integration.
        The state is represented as a dictionary containing position ('pos'), velocity ('vel'), angles ('angles'),
        angular velocity ('ang_vel'), and motor RPM ('rpm'). The function advances the state by time step `dt`
        using the system's translational and rotational dynamics.

        Parameters:
            state (dict): Current state of the quadcopter.
            dt (float): Time step.

        Returns:
            dict: New state after the integration step.
        """
        def f(s: dict) -> dict:
            return {
                'pos': s['vel'],
                'vel': self._translational_dynamics(s),
                'angles': s['ang_vel'],
                'ang_vel': self._rotational_dynamics(s),
            }
        
        state = state_old.copy()  # Create a copy of the current state to avoid modifying it directly
        
        reminder_data = {
            'thrust': state['thrust'],  # Keep the same thrust for the first half step
            'torque': state['torque'],  # Keep the same torque for the first half step
            'power': state['power'],  # Keep the same power for the first half step
            'rpm': state['rpm']  # Keep the same RPM for the first half step
        }

        k1 = f(state)
        state1 = {key: state[key] + k1[key] * (dt / 2) for key in ['pos', 'vel', 'angles', 'ang_vel']} 
        state1.update(reminder_data)  # Include reminder data in the first half step
        k2 = f(state1)
        
        state2 = {key: state[key] + k2[key] * (dt / 2) for key in ['pos', 'vel', 'angles', 'ang_vel']}
        state2.update(reminder_data)  # Include reminder data in the second half step
        k3 = f(state2)
        
        state3 = {key: state[key] + k3[key] * dt for key in ['pos', 'vel', 'angles', 'ang_vel']}
        state3.update(reminder_data)  # Include reminder data in the final step
        k4 = f(state3)

        state_new = {}
        for key in ['pos', 'vel', 'angles', 'ang_vel']:
            state_new[key] = state[key] + (dt / 6) * (k1[key] + 2 * k2[key] + 2 * k3[key] + k4[key])
        state_new['angles'] = np.array([wrap_angle(a) for a in state_new['angles']])
        state_new['ang_vel'] = np.clip(state_new['ang_vel'], -10, 10)
        state_new.update(reminder_data)
        return state_new

    def update_wind(self, V_components: np.array, simulate_wind=True, rpm_ref: float = 2500) -> None:
        """
        Update the wind signal for the quadcopter model.

        Parameters:
            V (np.array): Wind components in the x, y, and z directions.
            simulate_wind (bool): Whether to simulate wind effects. Default is True.
            rpm_ref (float): Reference RPM for wind effect calculation. Default is 2500.

        This method computes the additional thrust generated by wind and stores
        it in ``self.delta_T``. The value is applied to the rotor thrust during
        ``update_state``.
        """
        V_x, V_y, V_z = V_components
        omega = self.get_omega()
        if simulate_wind:
            kvz = 2 * np.pi ** 2 * V_z
            pT_1 = kvz * omega * (self.R_tip ** 2 - self.R_root ** 2)
            pT_2 = kvz / omega * (V_x ** 2 + V_y ** 2) * np.log(self.R_tip / self.R_root)
            self.delta_T = (pT_1 + pT_2) * self.rho * 0.5
        else:
            self.delta_T = np.zeros(len(self.rotors))

    def reset_state(self) -> None:
        """
        Reset the drone's state to the initial state.
        """
        self.state = self.init_state.copy()

    def get_omega(self) -> np.ndarray:
        """
        Get the current angular velocities of the motors in rad/s.
        Returns:
            np.ndarray: Angular velocities of the motors in rad/s.
        """
        return self._rpm_to_omega(np.array(self.state['rpm']) + 1)

    @staticmethod
    def _rpm_to_omega(rpm: np.ndarray) -> np.ndarray:
        """
        Convert motor RPM to angular velocity (rad/s).

        Parameters:
            rpm (np.ndarray): Array of motor RPMs.

        Returns:
            np.ndarray: Angular velocities in rad/s.
        """
        return rpm * 2 * np.pi / 60
    
    @staticmethod
    def _omega_to_rpm(omega: np.ndarray) -> np.ndarray:
        """
        Convert angular velocity (rad/s) to motor RPM.

        Parameters:
            omega (np.ndarray): Array of angular velocities in rad/s.

        Returns:
            np.ndarray: Motor RPMs.
        """
        return omega * 60 / (2 * np.pi)
    
    def compute_rotor_effects(self):
        # Compute thrust and thrust coefficient
        T = []
        Q = []
        P = []
        CT = 0 ; CQ = 0 ; CP = 0
        for i, rotor in enumerate(self.rotors):
            t, q, p, ct, cq, cp =  rotor.predict_aerodynamic(self.state['rpm'][i])
            T.append(t)
            Q.append(q)
            P.append(p)
            CT += ct ; CQ += cq ; CP += cp
        
        self.state['thrust'] = T  # Thrust from all rotors
        self.thrust = self.state['thrust']
        self.state['torque'] = Q  # Torque from all rotors
        self.state['power'] = P # Power from all rotors
        self.c_t = CT / len(self.rotors)  # Average thrust coefficient from all rotors
        self.c_q = CQ / len(self.rotors)  # Average torque coefficient from all rotors
        self.c_p = CP / len(self.rotors)  # Average power coefficient from all rotors

    
    def update_state(self, target: dict, dt: float, ground_control: bool = True, hit_accel_threshold: float = 1.0, verbose = True) -> None:
        """
        Update the drone's state by computing control commands, mixing motor RPMs,
        and integrating the dynamics.

        Parameters:
            target (dict): Target position with keys 'x', 'y', and 'z'.
            dt (float): Time step.
            ground_control (bool): Whether to apply ground control logic. Default is True.
            hit_accel_threshold (float): Threshold for detecting a hard landing. Default is 1.0 m/s² following MIL-STD-1290A.
            verbose (bool): Whether to print debug information. Default is True.
        """

        # Control inputs from the controller
        u1, u2, u3, u4 = self.controller.update(self.state, target, dt, self.m)
        # print(f"Control inputs: u1={u1:.2f}, u2={u2:.2f}, u3={u3:.2f}, u4={u4:.2f}")
        # Update the state with the new RPMs
        rpm1, rpm2, rpm3, rpm4 = self._mixer(u1, u2, u3, u4) 
        self.state["rpm"] = np.array([rpm1, rpm2, rpm3, rpm4]) 

        self.compute_rotor_effects()  # Compute thrust, torque, and power from the rotors

        self.thrust_no_wind = self.state['thrust'].copy()
        self.state['thrust'] = (np.array(self.state['thrust']) + self.delta_T).tolist()
        self.thrust = self.state['thrust']

        # Perform a Runge-Kutta 4th order integration step to update the state
        self.state = self._rk4_step(self.state, dt)

        #print(f"(t+1): {self.state}", end='\r')

        # Ground control logic
        if self.state['pos'][2] <= 0 and ground_control:
            self.state['pos'][2] = 0
            self.state['vel'][2] = 0  # Reset vertical velocity to zero
            # check if vertical acceleration is too high and differenciate from landing to hit
            if verbose:
                if self.state['vel'][2] < -hit_accel_threshold:  # If the vertical velocity is exceeds the threshold
                    print("[WARNING] Drone has hit the ground")
                else:
                    print("[INFO] Drone has landed.")
        
        # print(f"Thrust: {self.thrust:.2f} N, Thrust Coefficient: {self.c_t:.4f}, RPM to hover: {self.compute_hover_rpm(self.c_t):.1f} rpm")

