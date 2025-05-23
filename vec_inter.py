import numpy as np
import math

def vector_inter(p1, p2, V1, V2, coord, dim):
    p = np.zeros(2)

    if dim == 1:
        if abs(p1[0] - p2[0]) > 1e-6:
            lamda = - (p2[0] - coord) / (p1[0] - p2[0])
        else:
            lamda = 0
        p[1] = lamda * p1[1] + (1-lamda) * p2[1]
        p[0] = coord

    elif dim == 2:
        if abs(p1[1] - p2[1]) > 1e-6:
            lamda = - (p2[1] - coord) / (p1[1] - p2[1])
        else:
            lamda = 0
        p[0] = lamda * p1[0] + (1-lamda) * p2[0]
        p[1] = coord

    #print('lamda = ', lamda)
    V = lamda * V1 + (1-lamda) * V2
    return V, p, lamda



def vector_inter2(p1, p2, V1, V2, coord, dim):
    p = np.zeros(2)

    if dim == 1:  # Intersect with vertical line (x = coord)
        # Check if the edge is vertical (same x coordinates)
        if abs(p1[0] - p2[0]) < 1e-6:
            # Edge is vertical, cannot intersect with another vertical line
            # unless they're the same line
            if abs(p1[0] - coord) < 1e-6:
                # It's the same line, use the top point
                lamda = 0.0
                p[0] = coord
                p[1] = p1[1]
            else:
                # No intersection
                lamda = float('inf')
                p[0] = coord
                p[1] = 0  # Default value
        else:
            # Normal case - calculate intersection
            lamda = (coord - p2[0]) / (p1[0] - p2[0])
            p[0] = coord
            p[1] = lamda * p1[1] + (1-lamda) * p2[1]

    elif dim == 2:  # Intersect with horizontal line (y = coord)
        # Check if the edge is horizontal (same y coordinates)
        if abs(p1[1] - p2[1]) < 1e-6:
            # Edge is horizontal, cannot intersect with another horizontal line
            # unless they're the same line
            if abs(p1[1] - coord) < 1e-6:
                # It's the same line, use the leftmost point
                lamda = 0.0
                p[0] = p1[0]
                p[1] = coord
            else:
                # No intersection
                lamda = float('inf')
                p[0] = 0  # Default value
                p[1] = coord
        else:
            # Normal case - calculate intersection
            lamda = (coord - p2[1]) / (p1[1] - p2[1])
            p[0] = lamda * p1[0] + (1-lamda) * p2[0]
            p[1] = coord

    # Calculate interpolated value
    if lamda == float('inf'):
        V = V1  # Default to the first value in case of no intersection
    else:
        # Clamp lambda to [0,1] to avoid extrapolation
        lamda_clamped = max(0, min(1, lamda))
        V = lamda_clamped * V1 + (1-lamda_clamped) * V2

    return V, p, lamda
"""
demo
"""

p1 = np.array([5, 8])
p2 = np.array([7, 4])
V1 = np.array([1, -2])
V2 = np.array([-1, 3])
coord = 4
dim = 1
p = vector_inter(p1, p2, V1, V2, coord, dim)

#print(p)
