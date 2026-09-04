import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def variance_of_distances(center, points):
    distances = np.linalg.norm(points - center, axis=1)
    return np.var(distances)

def circumcenter(points):
    def objective(center):
        return variance_of_distances(center, points)

    initial_guess = np.mean(points, axis=0)

    result = minimize(objective, initial_guess, method='BFGS')

    return result.x

def model_func(vertices):
    center = circumcenter(vertices)
    v1 = vertices[1] - vertices[0]
    v2 = vertices[2] - vertices[0]
    normal = np.cross(v1, v2)
    radius = np.linalg.norm(center - vertices[0])
    return center, normal, radius

def distance_func(model, point):
    center, normal, radius = model
    
    normal = normal / np.linalg.norm(normal)
    
    point_to_plane_dist = np.abs(np.dot(point - center, normal))
    
    point_proj = point - np.dot(point - center, normal) * normal
    
    point_proj_to_center_dist = np.linalg.norm(point_proj - center)
    
    point_proj_to_circle_dist = np.abs(point_proj_to_center_dist - radius)
    
    distance = np.sqrt(point_to_plane_dist**2 + point_proj_to_circle_dist**2)
    
    return distance

class RANSAC:
    def __init__(self, num_iterations, sample_size, distance_threshold):
        self.num_iterations = num_iterations
        self.sample_size = sample_size
        self.distance_threshold = distance_threshold

    def fit(self, data, model_func=model_func, distance_func=distance_func):
        best_model = None
        best_inliers = []
        
        for _ in range(self.num_iterations):
            sample_indices = np.random.choice(len(data), self.sample_size, replace=False)
            sample = data[sample_indices]
            
            candidate_model = model_func(sample)
            
            distances = np.array([distance_func(candidate_model, point) for point in data])
            
            inliers = np.where(distances < self.distance_threshold)[0]
            
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_model = candidate_model

        return best_model, best_inliers