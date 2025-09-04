from scipy.optimize import minimize
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd

def convert_rpm_to_scaled_radians(rpm):
    return (rpm * 2 * np.pi / 60) / 10

def calculate_loudness(SWL):
    SWL_rms = np.sqrt(np.mean(SWL**2))
    SWL_loudness_db = 20 * np.log10(SWL_rms)
    return SWL_loudness_db

lw_ref = np.array([
       58.06334401, 63.39435536, 89.61406778, 87.6743877 , 88.47605171,
       67.1866716 , 94.21653901, 91.14444512, 81.34365872, 75.15627989,
       75.39196891, 75.73503128, 75.97300911, 85.36198312, 87.782705  ,
       91.05921408, 92.13047783, 94.3908669 , 95.34185699, 87.87681575,
       87.38338376, 81.02250449, 70.70962634, 57.09210232, 63.03212589,
       66.01390403, 69.07662034, 67.47656277, 66.70210595, 67.02101602,
       65.63809766, 63.19930878, 58.75525069, 52.18803922, 42.67844877,
       57.00634241, 60.76582418, 61.76174003, 57.87361306, 40.01645517,
       49.78732418, 58.98197588, 59.07283618, 46.88958525, 42.68777265,
       56.08423807, 54.37489695, 35.69959248, 50.56571747, 51.21904409,
       35.22702614, 48.30134472, 42.23665432, 43.40865783, 45.17203854,
       40.42100188, 43.71566937, 40.90928369, 38.65975645, 41.53974623,
       36.13367194, 38.50438545, 38.89934827])

class NoiseModel:
    def __init__(self, scaler_filename="scaler.joblib"):
        self.params = None
        self.scaler = joblib.load(scaler_filename) if scaler_filename else None

    def _calculate_predicted_Lw_total(self, data, num_rotors=4):
        """
        Calculate predicted total sound power levels using the regression model.

        Parameters:
        - params (array-like): Flattened array of a, b, c, d coefficients.
        - data (dict): Dataset containing Lw_ref, zeta, RPM, C_proc, and actual Lw_total.
        - num_rotors (int): Number of rotors (default: 4).

        Returns:
        - predicted_Lw_total (array-like): Predicted total sound power levels.
        """
        if self.params is None:
            raise ValueError("Model parameters are not set. Please fit the model first using model_fit().")
        # Reshape params to extract coefficients a, b, c, d
        n_frequencies = len(data['Lw_ref'][0])
        a, b, c, d = self.params[:n_frequencies], self.params[n_frequencies:2 * n_frequencies], self.params[2 * n_frequencies:3 * n_frequencies], self.params[3 * n_frequencies:]

        # Compute predicted Lw_individual for each rotor and then combine them
        predicted_Lw_total = []
        for Lw_ref, zeta, RPMs, C_proc in zip(data['Lw_ref'], data['zeta'], data['RPM'], data['C_proc']):
            Lw_individual_list = [
                Lw_ref + a * (zeta ** 2) + b * np.abs(zeta) + c * RPM + d * (RPM ** 2) + C_proc - 10 * np.log10(num_rotors)
                for RPM in RPMs
            ]
            Lw_total = 10 * np.log10(np.sum([10 ** (Lw / 10) for Lw in Lw_individual_list], axis=0))
            predicted_Lw_total.append(Lw_total)

        return np.array(predicted_Lw_total)

    def _regression_loss(self, data, num_rotors=4):
        """
        Compute the loss (mean squared error) between predicted and actual Lw_total.

        Parameters:
        - params (array-like): Flattened array of a, b, c, d coefficients.
        - data (dict): Dataset containing Lw_ref, zeta, RPM, C_proc, and actual Lw_total.
        - num_rotors (int): Number of rotors (default: 4).

        Returns:
        - mse (float): Mean squared error.
        """
        predicted_Lw_total = self._calculate_predicted_Lw_total(data, num_rotors)
        actual_Lw_total = np.concatenate(data['Lw_total'], axis=0)
        predicted_Lw_total_flat = np.concatenate(predicted_Lw_total, axis=0)  # Flatten to match actual data
        mse = np.mean((predicted_Lw_total_flat - actual_Lw_total) ** 2)
        return mse
    
    def model_fit(self, data, num_rotors=4, seed_value=42):
        """
        Optimize the coefficients a, b, c, d to minimize the regression loss.

        Parameters:
        - data (dict): Dataset containing Lw_ref, zeta, RPM, C_proc, and actual Lw_total.
        - num_rotors (int): Number of rotors (default: 4).
        """
        n_frequencies = len(data['Lw_ref'][0])
        initial_guess = np.zeros(4 * n_frequencies)  # Initial guess for a, b, c, d coefficients
        #np.random.seed(seed_value)
        #initial_guess = np.random.uniform(low=-1, high=1, size=4 * n_frequencies)

        result = minimize(
            self._regression_loss,
            initial_guess,
            args=(data, num_rotors),
            method='L-BFGS-B',
            options={'maxiter': 10000, 'disp': True}
        )

        optimized_params = result.x
        a, b, c, d = (
            optimized_params[:n_frequencies],
            optimized_params[n_frequencies:2 * n_frequencies],
            optimized_params[2 * n_frequencies:3 * n_frequencies],
            optimized_params[3 * n_frequencies:]
        )
        print("Number of iterations:", result.nit)
        print("Final loss (objective function value):", result.fun)
        self.params = np.concatenate([a, b, c, d])

    def predict(self, input, num_rotors=4):
        """
        Execute the model using the optimized coefficients a, b, c, d.

        Parameters:
        - input (dict): Dataset containing Lw_ref, zeta, RPM, C_proc.
        - a, b, c, d (array-like): Optimized coefficients.
        - num_rotors (int): Number of rotors (default: 4).

        Returns:
        - predicted_Lw_total (array-like): Predicted total sound power levels.
        """
        n_frequencies = len(input['Lw_ref'][0])
        predicted_Lw_total = self._calculate_predicted_Lw_total(input, num_rotors)
        return predicted_Lw_total

    def get_noise_emissions(self, zeta_angle, rpms, distance) -> tuple:
        """
        Get the Sound Pressure Level (SPL) based on the zeta angle and RPM.
        The zeta angle is in radians, and the RPM is the rotor speed.
        
        Parameters:
            zeta_angle (float): The angle in radians between 0 and 2π calculated as arctan(height/distance).
            rpms (list): List of RPM values for the rotors.
            distance (float): The distance from the noise source in meters.
        Returns:
            tuple: (SPL, SWL) where SPL is the Sound Pressure Level and SWL is the Sound Power Level.
        """
        # Scale input parameters using the scaler
        rpm1, rpm2, rpm3, rpm4 = rpms
        input_df = pd.DataFrame([{
            'delta_zeta': zeta_angle,
            'RPM1': rpm1,
            'RPM2': rpm2,
            'RPM3': rpm3,
            'RPM4': rpm4,
            'C_proc': 1.0
        }])
        norm_data = self.scaler.transform(input_df)
        
        zeta_angle_scaled = norm_data[0][0]
        rpms_scaled = norm_data[0][1:5]
        rpms_scaled = rpms_scaled / 15
        C_proc_scaled = norm_data[0][5]

        input_data = {
            'Lw_ref': [lw_ref],
            'zeta': [zeta_angle_scaled],
            'RPM': [rpms_scaled],
            'C_proc': [C_proc_scaled]
        }

        predicted_Lw_total = self.predict(input_data)
        swl = calculate_loudness(predicted_Lw_total[0])
        spl =  swl - abs(10 * np.log10(1/(4 * np.pi * ((distance+1e-4)**2))))
        return abs(spl), abs(swl)


    def save_model(self, a, b, c, d, filename):
        """
        Save the model coefficients to a file.
        Parameters:
        - a, b, c, d (array-like): Coefficients to save.
        - filename (str): Name of the file to save the coefficients (npz format).
        """
        np.savez(filename, a=a, b=b, c=c, d=d)
        print(f"Model coefficients saved to {filename}")

    def load_model(self, filename):
        """
        Load the model coefficients from a file.
        Parameters:
        - filename (str): Name of the file to load the coefficients from (npz format).
        Returns:
        - a, b, c, d (array-like): Loaded coefficients.
        """
        data = np.load(filename)
        a, b, c, d = data['a'], data['b'], data['c'], data['d']
        print(f"Model coefficients loaded from {filename}")
        self.params = np.concatenate([a, b, c, d])
        return a, b, c, d


