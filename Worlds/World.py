# Author: Andrea Vaiuso
# Version: 2.0
# Date: 15.07.2025
# Description: This module defines the World class, which represents a simulated environment for a drone.

import json
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ---------------- World Class ----------------
class World:
    # Dizionario statico per mappare l'ID ai parametri dell'area
    AREA_PARAMS = {
        1: {
            "id": 1,
            "name": "Housing Estate",
            "min_altitude": 150, 
            "max_altitude": 1000, 
            "noise_penalty": 1.6,
            "color": "blue",
            "alpha": 0.2
        },
        2: {
            "id": 2,
            "name": "Industrial Area",
            "min_altitude": 70, 
            "max_altitude": 1000, 
            "noise_penalty": 1.2,
            "color": "yellow",
            "alpha": 0.2
        },
        3: {
            "id": 3,
            "name": "Open Field",
            "min_altitude": 5, 
            "max_altitude": 1000, 
            "noise_penalty": 0,
            "color": "green",
            "alpha": 0.1
        },
        4: {
            "id": 4,
            "name": "Forbidden Area",
            "min_altitude": 0, 
            "max_altitude": 0, 
            "noise_penalty": 100,
            "color": "red",
            "alpha": 0.2
        }
    }
    
    DEFAULT_AREA_ID = 3

    def __init__(self, grid_size, world_size, world_name="World", background_image_path=None):
        """
        Initializes a World instance.
        Args:
            grid_size (int): Size of each grid cell in the world. This parameter defines the resolution of the world grid for area classification and noise receivers placement.
            world_size (int): Total size of the world (assumed square) in meters.
            world_name (str): Name of the world.
            background_image_path (str): Path to the background image file.
        """

        self.grid_size = grid_size
        self.max_world_size = world_size // grid_size
        self.grid = {}  # mapping: area coordinate tuple -> area_id
        self.world_name = world_name
        self.background_image = None
        if background_image_path:
            # Check if the background is squared and save it
            self.background_image = np.array(Image.open(background_image_path).convert('RGB'))
            if self.background_image.shape[0] != self.background_image.shape[1]:
                print("Warning: the background image is not squared. Cropping it...")
                min_dim = min(self.background_image.shape[0], self.background_image.shape[1])
                self.background_image = self.background_image[:min_dim, :min_dim]

    def get_area(self, x, y, z):
        return (x // self.grid_size, y // self.grid_size, z // self.grid_size)

    def set_area_parameters(self, x_1, x_2, y_1, y_2, parameters):
        """
        Imposta l'ID dell'area nelle coordinate specificate.
        I parametri vengono presi dal dizionario 'parameters', che deve contenere almeno la chiave "id".
        """
        area_id = parameters["id"]
        for x in range(x_1, x_2 + 1, self.grid_size):
            for y in range(y_1, y_2 + 1, self.grid_size):
                for z in range(0, self.max_world_size * self.grid_size, self.grid_size):
                    area = self.get_area(x, y, z)
                    self.grid[area] = area_id

    def get_area_parameters(self, x, y, z):
        area = self.get_area(x, y, z)
        area_id = self.grid.get(area, World.DEFAULT_AREA_ID)
        return World.AREA_PARAMS.get(area_id, {})

    def get_area_center_point(self, x, y, z):
        area = self.get_area(x, y, z)
        return ((area[0] + 0.5) * self.grid_size, 
                (area[1] + 0.5) * self.grid_size, 
                (area[2] + 0.5) * self.grid_size)

    def get_areas_in_circle(self, x: float, y: float, height: float, radius: float, include_areas_out_of_bounds: bool = False) -> tuple:
        """
        Returns a list of area center points and their parameters within a circle of given radius.
        The circle is defined by its center (x, y) and height, and the radius is in grid units.
        If include_areas_out_of_bounds is True, areas outside the world bounds are also included.

        Parameters:
            x (float): X coordinate of the circle center.
            y (float): Y coordinate of the circle center.
            height (float): Height at which to check the areas.
            radius (float): Radius of the circle in grid units.
            include_areas_out_of_bounds (bool): If True, includes areas outside the world bounds.

        Returns:
            tuple: A tuple containing two lists:
                - areas_in_circle: List of area center points (x, y, z).
                - parameters_in_circle: List of dictionaries with area parameters.
        """
        areas_in_circle = []
        parameters_in_circle = []
        radius_squared = radius ** 2

        # Calculate the bounds of the circle
        if include_areas_out_of_bounds:
            min_x = x - radius
            max_x = x + radius
            min_y = y - radius
            max_y = y + radius
        else:
            min_x = max(0, x - radius)
            max_x = min(self.max_world_size * self.grid_size, x + radius)
            min_y = max(0, y - radius)
            max_y = min(self.max_world_size * self.grid_size, y + radius)

        # Create ranges for x, y, and z coordinates based on the grid size
        x_range = np.arange(min_x, max_x + 1, self.grid_size)
        y_range = np.arange(min_y, max_y + 1, self.grid_size)
        z_range = np.arange(0, height * self.grid_size, self.grid_size)

        # Create a meshgrid for x and y coordinates
        x_mesh, y_mesh = np.meshgrid(x_range, y_range, indexing='ij')
        x_flat, y_flat = x_mesh.ravel(), y_mesh.ravel()
        distances_sq = (x_flat - x) ** 2 + (y_flat - y) ** 2
        valid_indices = np.where(distances_sq <= radius_squared)[0]

        for idx in valid_indices:
            i, j = x_flat[idx], y_flat[idx]
            for z in z_range:
                area_center = self.get_area_center_point(i, j, z)
                # Get the area parameters for the center point
                area_params = self.get_area_parameters(i, j, z)
                areas_in_circle.append(area_center)
                parameters_in_circle.append(area_params)
        return areas_in_circle, parameters_in_circle

    def save_world(self, filename):
        data = {
            'grid_size': self.grid_size,
            'max_world_size': self.max_world_size,
            'grid': self.grid,
            'world_name': self.world_name,
            'background_image': self.background_image
        }
        with open(filename, 'wb') as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_world(cls, filename):
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        world = cls(data['grid_size'], data['max_world_size'] * data['grid_size'], data['world_name'])
        world.grid = data['grid']
        if data['background_image'] is not None:
            world.background_image = data['background_image']
        return world
    
    def plot_world_from_top(self, image_alpha=0.5, A=None, B=None):
        fig, ax = plt.subplots(figsize=(8, 6))
        grid_size = self.grid_size

        # Background image
        if getattr(self, "background_image", None) is not None:
            bg_img = np.array(self.background_image)
            ax.imshow(
                bg_img,
                extent=[0, self.max_world_size, 0, self.max_world_size],
                origin="lower",
                alpha=image_alpha,
                zorder=-1,
            )

        # Grid
        for (x, y, z), params in self.grid.items():
            if z == 0:
                rect = plt.Rectangle(
                    (x * grid_size, y * grid_size),
                    grid_size,
                    grid_size,
                    color=self.AREA_PARAMS[params]["color"],
                    alpha=self.AREA_PARAMS[params]["alpha"],
                )
                ax.add_patch(rect)
        
        # Plot points A and B if provided
        if A is not None:
            ax.plot(A[0], A[1], marker='o', color='cyan', markersize=10, label='Point A')
            ax.text(A[0] + 1, A[1] + 1, 'A', color='cyan', fontsize=12, weight='bold')
        if B is not None:
            ax.plot(B[0], B[1], marker='o', color='magenta', markersize=10, label='Point B')
            ax.text(B[0] + 1, B[1] + 1, 'B', color='magenta', fontsize=12, weight='bold')
        
        ax.set_xlim(0, self.max_world_size * grid_size)
        ax.set_ylim(0, self.max_world_size * grid_size)
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        ax.set_title(f"World: {self.world_name}")
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    world = World(grid_size=5, world_size=100, world_name="Test World")
    world.set_area_parameters(0, 100, 0, 100, World.AREA_PARAMS[3])  # Open Field as base
    world.set_area_parameters(30, 69, 30, 70, World.AREA_PARAMS[4])  # Forbidden Area
    world.set_area_parameters(30, 69, 0, 29, World.AREA_PARAMS[1])   # Housing Estate
    world.set_area_parameters(30, 69, 71, 99, World.AREA_PARAMS[2])  # Industrial Area
    world.set_area_parameters(30, 69, 86, 89, World.AREA_PARAMS[3])  # Tunnel of Open Field
    world.save_world("Worlds/training_world.pkl")
    world.plot_world_from_top(image_alpha=0.5)