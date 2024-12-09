import random
import numpy as np
from typing import Optional
import pygame

def mod2pi(theta):
    """
    Returns angle from 0 to 2pi

    Args:
        theta (float): angle
    Returns:
        (float): wrapped angle between 0 to 2pi rad
    """
    return (theta + np.pi) % 2*np.pi - np.pi

class Bird2D:
    def __init__(self, x: float, y: float, theta: Optional[float]=None, speed:Optional[float]=2.0) -> None:
        """
        Initialises bird with state [x,y,theta,v]. All units are SI units, and metres and pixels are equivalent (these are giant birds)
        """
        self.position = np.array([x, y], dtype=np.float32)
        self.theta = theta if theta is not None else np.random.uniform(0,2*np.pi)
        self.theta = mod2pi(self.theta)
        self.velocity = np.array([np.cos(self.theta), np.sin(self.theta)]) * speed
        # angular velocity is unnecessary (birds dont spin anyways)
        self.acceleration = np.zeros(2, dtype=np.float32) # acceleration is initialised as 0

        # limits
        self.max_speed = 5.0
        self.max_force = 0.1

        # flocking parameters
        self.perception_radius = 50  # Radius to detect neighbors
        self.separation_radius = 25  # Radius for separation (avoidance)

        # flocking weights
        self.alignment_weight = 1.0
        self.cohesion_weight = 1.0
        self.separation_weight = 1.5

    def flock(self, birds):
        """
        Flocking rules on alignment, cohesion and seperation.
        Alignment: Each bird steers towards the average velocity of its neighbours
        Cohesion: Each bird steers towards the centre of mass of its neighbours
        Seperation: Each bird steers away from its neighbours to avoid collision.
        
        Each force would be normalised, limited and weighted. 
        """
        alignment = np.zeros(2, dtype=np.float32)
        cohesion = np.zeros(2, dtype=np.float32)
        separation = np.zeros(2, dtype=np.float32)
        total_neighbours = 0 

        for other_bird in birds:
            if other_bird == self:
                continue
            distance_between = np.linalg.norm(other_bird.position - self.position)
            if distance_between < self.perception_radius:
                alignment += other_bird.velocity
                cohesion += other_bird.position

                if distance_between < self.separation_radius:
                    # normalised vector to neighbour
                    separation += (self.position - other_bird.position) / distance_between
                
                total_neighbours += 1
        
        if total_neighbours > 0:
            alignment /= total_neighbours
            alignment = self._steer_towards(alignment)

            cohesion = cohesion / total_neighbours - self.position
            cohesion = self._steer_towards(cohesion)

            separation = self._steer_towards(separation)
        
        # Apply weights to each behavior
        self.apply_force(alignment * self.alignment_weight)
        self.apply_force(cohesion * self.cohesion_weight)
        self.apply_force(separation * self.separation_weight)

    def _steer_towards(self, target):
        """
        Calculate a steering force towards the target vector
        """
        if np.linalg.norm(target) == 0:
            return np.zeros(2, dtype=np.float32)
        # normalise and clip
        target = (target / np.linalg.norm(target)) * self.max_speed
        steer = target - self.velocity # error in speed
        if np.linalg.norm(steer) > self.max_force:
            steer = (steer / np.linalg.norm(steer)) * self.max_force
        return steer

    def update(self):
        """
        Updates bird's state. Each update call lasts dt in time.
        """
        self.velocity += self.acceleration
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed: # clip speed
            self.velocity = (self.velocity / speed) * self.max_speed

        self.position += self.velocity 

        self.theta = np.arctan2(self.velocity[1], self.velocity[0])

        self.acceleration.fill(0) # reset acceleration

    def apply_force(self, force):
        """
        Apply a force/acceleration (mass is unit)
        """
        self.acceleration += force

    def edges(self, width, height):
        """
        Screen wrapping
        """
        if self.position[0] > width:
            self.position[0] = 0
        elif self.position[0] < 0:
            self.position[0] = width
        if self.position[1] > height:
            self.position[1] = 0
        elif self.position[1] < 0:
            self.position[1] = height

    def draw(self, window):
        """
        Draw the bird as a triangle facing its velocity direction.
        """
        size = 5  # Triangle size
        points = [
            self.position + self._rotate(np.array([0, -size]), self.theta+np.pi/2),  # Tip
            self.position + self._rotate(np.array([size / 2, size]), self.theta+np.pi/2),  # Right wing
            self.position + self._rotate(np.array([-size / 2, size]), self.theta+np.pi/2),  # Left wing
        ]
        # Convert NumPy arrays to lists for pygame
        points = [point.tolist() for point in points]
        pygame.draw.polygon(window, (255, 255, 255), points)

    def _rotate(self, point, angle):
        """
        Rotate a 2D point around the origin by the given angle.
        """
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ])
        return np.dot(rotation_matrix, point)