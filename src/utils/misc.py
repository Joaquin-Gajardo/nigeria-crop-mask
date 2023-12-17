import numpy as np
from sklearn.metrics.pairwise import haversine_distances


EARTH_RADIUS = 6371000 # in meters


def get_great_circle_distance(lat, lon):
    """Get the great circle distance in meters between two points on Earth."""
    
    A = [lon * np.pi / 180., lat * np.pi / 180.] 
    B = [(lon + 1) * np.pi / 180., lat * np.pi / 180.] 
    dx = (EARTH_RADIUS) * haversine_distances([A, B])[0, 1]
    
    return dx