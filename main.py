import numpy as np
from Drone import QuadcopterModel
from Controller import QuadCopterController
from Simulation import Simulation
from plotting_functions import plot3DAnimation, plotLogData, plotNoiseEmissionMap, plotNoiseEmissionHistogram, get_total_PA
from World import World
from Noise.DNNModel import RotorSoundModel as DNNModel
from Noise.EmpaModel import NoiseModel as EmpaModel
from Rotor.TorchRotorModel import RotorModel
import yaml
from Noise.Psychoacoustic import PsychoacousticBackendAdapter as PsLib
from opt_func import calculate_costs