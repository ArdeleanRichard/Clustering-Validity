import numpy as np


def diagonal_line(points):
    if len(points) == 0:
        return None, None, None, None

    min_x = max_x = points[0][0]
    min_y = max_y = points[0][1]

    for point in points[1:]:
        x, y = point
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)

    return ((min_x, min_y), (max_x, max_y))


def vertical_line(x):
    return ((x, -1), (x, 1))

def horizontal_line(x):
    return ((-1, x), (1, x))


def assign_labels_by_given_line(points, line):
    # Calculate the slope and intercept of the line (y = mx + b)
    (x1, y1), (x2, y2) = line

    if x1 == x2:  # Vertical line (undefined slope)
        labels = [1 if x < x1 else 0 for x, _ in points]
    elif y1 == y2:  # Horizontal line (slope = 0)
        labels = [1 if y < y1 else 0 for _, y in points]
    else:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1

        # Assign labels to points based on their position relative to the line
        labels = []
        for point in points:
            # Unpack the coordinates of the point
            x, y = point

            # Calculate the value of y for the given x on the line
            line_y = m * x + b

            # Check if the point is above or below the line
            if y > line_y:
                labels.append(1)  # Point is above the line, assign label 1
            else:
                labels.append(0)  # Point is below the line, assign label 0

    return np.array(labels)



def assign_labels_by_given_plane(points, plane):
    # Calculate the slope and intercept of the line (y = mx + b)


    if plane=="vertical":  # Vertical line (undefined slope)
        labels = [1 if x < 0 else 0 for x, _, _ in points]
    elif plane=="horizontal":  # Horizontal line (slope = 0)
        labels = [1 if z < 0 else 0 for _, _, z in points]
    else:
        p1, p2, p3 = plane
        # Calculate two vectors lying on the plane
        v1 = np.array(p2) - np.array(p1)
        v2 = np.array(p3) - np.array(p1)

        # Calculate the normal vector to the plane
        normal_vector = np.cross(v1, v2)

        # Assign labels to points based on their position relative to the plane
        labels = []
        for point in points:
            # Calculate vector from the point to any point on the plane
            vec_to_point = np.array(point) - np.array(p1)

            # Calculate dot product between the vector to the point and the normal vector
            dot_product = np.dot(vec_to_point, normal_vector)

            # Check if the point is above or below the plane
            if dot_product > 0:
                labels.append(1)  # Point is above the plane, assign label 1
            else:
                labels.append(0)  # Point is below or on the plane, assign label 0

    return np.array(labels)