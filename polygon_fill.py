import vec_inter
import numpy as np
import cv2 as cv
import math

def render_img(faces, vertices, vcolors, uvs, depth, shading):
    # initialize blank image
    img = np.zeros((513, 513, 3))
    img = img.astype('uint8')
    img.fill(255)

    L = vertices.shape[0]
    K = faces.shape[0]


    depth2 = np.zeros(faces.shape[0])
    # calculate weighted center of depth for each triangle
    for l in range(0, faces.shape[0]):
        depth2[l] = (depth[faces[l][0]] + depth[faces[l][1]] + depth[faces[l][2]]) / 3
    print('depth2 = ', depth2)
    # depth follows indices of vertices array. depth[0] is the depth of the 
    # vertex vertices[0]. 
    # depth2 follows indices of faces array. depth2[0] is the center of mass
    # of the first triangle in faces, faces[0]
    # get sorted indices of the depth array, flip them to descending order. 
    # the resulting array's indices will indicating the order of the 
    # triangles to be drawn.
    # these indices will index faces which in turn will index vertices
    for l in np.flip(depth2.argsort()):
        img = polygon_fill(img, np.array([vertices[faces[l][0]], vertices[faces[l][1]], vertices[faces[l][2]]]),
                                np.array([vcolors[faces[l][0]], vcolors[faces[l][1]], vcolors[faces[l][2]]]),
                                np.array([uvs[faces[l][0]], uvs[faces[l][1]], uvs[faces[l][2]]]), 
                                shading)
    return img

def find_active_edges(active_edges, vertices, K, xmin, xmax, ymin, ymax, y):
    for k in range(0, K):
        if ymin[k] == y + 1 and ymax[k] != ymin[k]:
            active_edges[k][0][1] = ymax[k]
            active_edges[k][1][1] = ymin[k]
            # pws kseroume poia tetagmenh antistoixei se poia tetmhmenh; grammikh paremvolh
            # "metaksy" dyo shmeiwn, alla h paremvolh ginetai panw se ena apo ta dyo shmeia,
            # etsi vriskoume poies htan oi arxikes tetmhmenes
            _, p, __ = vec_inter.vector_inter(vertices[k-1], vertices[k], 0, 0, ymax[k], 2)
            active_edges[k][0][0] = p[0] 
            _, p, __ = vec_inter.vector_inter(vertices[k-1], vertices[k], 0, 0, ymin[k], 2)
            active_edges[k][1][0] = p[0] 
            
        if ymax[k] == y + 1:
            active_edges[k][0][1] = -1
            active_edges[k][1][1] = -1
            active_edges[k][0][0] = -1 
            active_edges[k][1][0] = -1 



def find_active_points(active_points, active_edges, vertices, m, K, xmin, xmax, ymin, ymax, y):
    #y_total_min = np.astype(np.min(ymin), int)
    #x_total_min = np.astype(np.min(xmin), int)
    for k in range(0, K):
        # protash 1
        if ymin[k] == y + 1:
            active_points[k][1] = ymin[k]
            _, p, __ = vec_inter.vector_inter(vertices[k-1], vertices[k], 0, 0, ymin[k], 2)
            active_points[k][0] = p[0]
        # protash 3
        elif active_points[k][0] != -1 and m[k] != math.inf:
            active_points[k][0] = active_points[k][0] + 1/m[k]
        # exclude horizontal lines, protash 2
        if ymin[k] == ymax[k] or ymax[k] == y + 1:
            active_points[k][1] = -1
            active_points[k][0] = -1

def polygon_fill(img, vertices, vcolors, uv, shading):
    """
    fill a single triangle
    """
    M = img.shape[0] 
    N = img.shape[1] 
    K = 3 # a triangle has 3 vertices
    ymax = np.zeros(3)
    ymin = np.zeros(3)
    xmax = np.zeros(3)
    xmin = np.zeros(3)
    # prwth diastash gia plevres. max 2 plevres mporoun na einai energes th fora logw
    # kyrtothtas. defterh diastash einai ta shmeia pou orizoun tis plevres. 
    # trith diastash einai oi syntagmenes tou kathe shmeiou. 0 gia tetmhmenh, 1 gia 
    # tetagmenh.
    active_edges = np.zeros((3,2,2))
    active_edges.fill(-1)
    active_edges_idx = []
    active_points = np.zeros((3,2))
    active_points.fill(-1)
    m = np.zeros(3)

    for k in range(0, K): 
        #print(k)
        ymax[k] = max(vertices[k-1][1], vertices[k][1])
        ymin[k] = min(vertices[k-1][1], vertices[k][1])
        xmax[k] = max(vertices[k-1][0], vertices[k][0])
        xmin[k] = min(vertices[k-1][0], vertices[k][0])
        if vertices[k][0] != vertices[k-1][0]:
            m[k] = (vertices[k][1] - vertices[k-1][1]) / (vertices[k][0] - vertices[k-1][0])
        else:
            m[k] = math.inf

    
    y_total_min = np.astype(np.min(ymin), int)
    y_total_max = np.astype(np.max(ymax), int)
    y = y_total_min - 1
    x_total_min = np.astype(np.min(xmin), int)
    x_total_max = np.astype(np.max(xmax), int)

    # vriskoume lista energwn akmwn
    find_active_edges(active_edges, vertices, K, xmin, xmax, ymin, ymax, y)

    # vriskoume lista energwn oriakwn shmeiwn
    find_active_points(active_points, active_edges, vertices, m, K, xmin, xmax, ymin, ymax, y)

    print('mk = ', m)
    print('active edges = ', active_edges)
    print('active points = ', active_points)

    ### polygon fill loop ###
    for y in range(y_total_min, y_total_max):
        # sort lista oriakwn shmeiwn kata x  ;;__;;
        sorted_active_points = np.sort(active_points, 0)
        sorted_active_points = np.astype(sorted_active_points, 'int')
        print('sorted active points = ', sorted_active_points)
        
        # fast pixel drawing
        if shading == "t":
            # only two points will be active at a time, the inactive point 
            # with negative values will be at the beginning, so we only need
            # the range of the other two points' x coordinate
            img = t_shading(img, vertices, uv, y, range(sorted_active_points[1][0], 
                            sorted_active_points[2][0]+1), 
                            cv.imread('fresque-saint-georges-2452226686.jpg'))
        elif shading == "f":
            img = f_shading(img, vertices, vcolors, y - 1, 
                            range(sorted_active_points[1][0], sorted_active_points[2][0]+1))

        # enhmerwnoume lista energwn akmwn
        find_active_edges(active_edges, vertices, K, xmin, xmax, ymin, ymax, y)
        
        # enhmerwnoume lista energwn oriakwn shmeiwn
        find_active_points(active_points, active_edges, vertices, m, K, xmin, xmax, ymin, ymax, y)
                
        print('y = ', y+1)
        print('active edges = ', active_edges)
        print('active points = ', active_points)
    return img


def f_shading(img, vertices, vcolors, rows, cols):
    K = vertices.shape[0]
    vcolors = np.multiply(vcolors, [255])
    mean_color = np.zeros(3)
    for i in range(0, 3):
        for k in range(0, K):
            mean_color[K-1-i] = mean_color[K-1-i] + vcolors[k][i]
    mean_color = np.multiply(mean_color, [1/K])
    mean_color = np.rint(mean_color)
    mean_color = np.astype(mean_color, 'uint8')
    cols = np.array(cols)
    #print('mean color = ', mean_color)
    #print('cols = ', cols)
    #print('cols type = ', cols.dtype)
    #print('cols size = ', cols.size)

    if cols.size > 1:
        img[rows][cols[0]:cols[-1]] = mean_color
    elif cols.size == 1:
        img[rows][cols] = mean_color
    #print('img[', rows, '][', cols,'] = ', img[img.shape[0] - rows][cols])
    return img


def t_shading(img, vertices, uv, y, cols, textImg):
    ## Get texture dimensions
    tex_height, tex_width = textImg.shape[0], textImg.shape[1]

    # Sort vertices by y-coordinate to identify top, middle, bottom
    sorted_indices = np.argsort([v[1] for v in vertices])
    v_top, v_mid, v_bot = vertices[sorted_indices]
    uv_top, uv_mid, uv_bot = uv[sorted_indices]

    # Find intersection points on each edge with current scanline
    def find_intersection(v1, v2, uv1, uv2, y_val):
        # If edge is horizontal, no meaningful intersection
        if abs(v1[1] - v2[1]) < 1e-6:
            return None, None

        # Calculate interpolation factor t
        t = (y_val - v2[1]) / (v1[1] - v2[1])

        # If t is outside [0,1], the scanline doesn't intersect this edge
        if t < 0 or t > 1:
            return None, None

        # Calculate intersection point x-coordinate
        x = v2[0] + t * (v1[0] - v2[0])

        # Calculate corresponding texture coordinate
        tex_coord = uv2 + t * (uv1 - uv2)

        return x, tex_coord

    # Find intersections with all three edges
    intersections = []

    # Edge: top to middle
    x1, tex1 = find_intersection(v_top, v_mid, uv_top, uv_mid, y)
    if x1 is not None:
        intersections.append((x1, tex1))

    # Edge: top to bottom
    x2, tex2 = find_intersection(v_top, v_bot, uv_top, uv_bot, y)
    if x2 is not None:
        intersections.append((x2, tex2))

    # Edge: middle to bottom
    x3, tex3 = find_intersection(v_mid, v_bot, uv_mid, uv_bot, y)
    if x3 is not None:
        intersections.append((x3, tex3))

    # Need exactly 2 intersections to draw a scanline
    if len(intersections) != 2:
        return img

    # Sort intersections by x-coordinate
    intersections.sort(key=lambda i: i[0])

    # Get left and right edge points
    x_left, tex_left = intersections[0]
    x_right, tex_right = intersections[1]

    # Convert cols to list for easier processing
    cols = list(cols)

    # Process each pixel in the scanline
    for x in cols:
        # Skip if outside the range of intersection points
        if x < x_left or x > x_right:
            continue

        # Calculate interpolation factor for this pixel
        if abs(x_right - x_left) < 1e-6:  # Avoid division by zero
            kappa = 0
        else:
            kappa = (x - x_left) / (x_right - x_left)

        # Interpolate texture coordinate
        tex_coord = tex_left + kappa * (tex_right - tex_left)
        # Convert to texture pixel coordinates
        tx = int(tex_coord[0] * (tex_width - 1))
        ty = int(tex_coord[1] * (tex_height - 1))

        # Clamp to texture boundaries
        tx = max(0, min(tex_width - 1, tx))
        ty = max(0, min(tex_height - 1, ty))

        # Sample texture and set pixel
        img[y][x] = textImg[ty][tx]

    return img



