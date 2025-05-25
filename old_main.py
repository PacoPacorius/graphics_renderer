import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from typing import Tuple
import math
import polygon_fill
import vec_inter

def translate(t_vec: np.ndarray) -> np.ndarray:
    """
    Create an affine transformation matrix w.r.t. the
    specified translation vector.
    """
    xform = np.eye(4)
    xform[0:3, 3] = t_vec.flatten()
    return xform

def rotate(axis: np.ndarray, angle: float, center: np.ndarray) -> np.ndarray:
    """
    Create an affine transformation matrix w.r.t. the
    specified rotation.
    """
    # Normalize the axis vector
    axis = axis / np.linalg.norm(axis)

    # Rodrigues' rotation formula for rotation matrix
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    # Cross product matrix of axis
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])

    # Rotation matrix using Rodrigues' formula
    R = np.eye(3) + sin_a * K + (1 - cos_a) * np.dot(K, K)

    # Create 4x4 homogeneous transformation matrix
    xform = np.eye(4)
    xform[0:3, 0:3] = R

    # Apply translation to rotate around center point
    # T = T(center) * R * T(-center)
    T_to_origin = translate(-center)
    T_back = translate(center)

    # Rotation around center
    R_homogeneous = np.eye(4)
    R_homogeneous[0:3, 0:3] = R

    xform = T_back @ R_homogeneous @ T_to_origin

    return xform

def compose(mat1: np.ndarray, mat2: np.ndarray) -> np.ndarray:
    """
    Combine two transformation matrices into one.
    """
    return mat1 @ mat2

def world2view(pts: np.ndarray, R: np.ndarray, c0: np.ndarray) -> np.ndarray:
    """
    Implements a world-to-view transform, i.e. transforms the specified
    points to the coordinate frame of a camera.
    """
    # Convert points to homogeneous coordinates
    if pts.shape[0] == 3:
        pts_homo = np.vstack([pts, np.ones(pts.shape[1])])
    else:
        pts_homo = pts

    # Create transformation matrix: T = [R | -R*c0; 0 0 0 1]
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = -R @ c0.flatten()

    # Transform points
    pts_view = T @ pts_homo

    return pts_view[0:3, :]

def lookat(eye: np.ndarray, up: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the camera's view matrix (i.e., its coordinate frame transformation specified
    by a rotation matrix R, and a translation vector t).
    """
    # Forward vector (camera looks in negative z direction)
    forward = target.flatten() - eye.flatten()
    forward = forward / np.linalg.norm(forward)

    # Right vector
    right = np.cross(forward, up.flatten())
    right = right / np.linalg.norm(right)

    # Up vector (recompute to ensure orthogonality)
    up_new = np.cross(right, forward)
    up_new = up_new / np.linalg.norm(up_new)

    # Rotation matrix (camera coordinate system)
    R = np.array([right, up_new, -forward]).T

    # Translation vector
    t = eye.flatten()

    return R, t

def perspective_project(pts: np.ndarray, focal: float, R: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project the specified 3d points pts on the image plane, according to a pinhole
    perspective projection model.
    """
    # Transform to camera coordinates
    pts_cam = world2view(pts, R, t)

    # Perspective projection
    x_proj = focal * pts_cam[0, :] / pts_cam[2, :]
    y_proj = focal * pts_cam[1, :] / pts_cam[2, :]

    pts_2d = np.vstack([x_proj, y_proj])
    depths = pts_cam[2, :]

    return pts_2d, depths

def rasterize(pts_2d: np.ndarray, plane_w: int, plane_h: int, res_w: int, res_h: int) -> np.ndarray:
    """
    Rasterize the incoming 2d points from the camera plane to image pixel coordinates.
    """
    # Convert from camera plane coordinates to pixel coordinates
    # Camera plane center is at (0, 0), image origin is bottom-left

    # Scale and translate
    scale_x = res_w / plane_w
    scale_y = res_h / plane_h

    # Convert to pixel coordinates (origin at bottom-left)
    pixel_x = (pts_2d[0, :] + plane_w/2) * scale_x
    pixel_y = (pts_2d[1, :] + plane_h/2) * scale_y

    pts_raster = np.vstack([pixel_x, pixel_y])

    return pts_raster

def render_object(v_pos, v_clr, t_pos_idx, plane_h, plane_w, res_h, res_w, focal, eye, up, target, v_uvs=None, shading='t', texture_img=None) -> np.ndarray:
    """
    Render the specified object with perspective-correct texture mapping (no polygon_fill).
    """
    # Prepare empty image
    img = np.ones((res_h, res_w, 3), dtype=np.uint8) * 255
    z_buffer = np.full((res_h, res_w), np.inf)

    # Camera transform
    R, t = lookat(eye, up, target)
    pts_2d, depths = perspective_project(v_pos.T, focal, R, t)
    pts_raster = rasterize(pts_2d, plane_w, plane_h, res_w, res_h).T

    # Default UVs
    if v_uvs is None:
        v_uvs = np.zeros((v_pos.shape[0], 2))

    # Perspective divide values
    inv_z = 1.0 / depths
    uvs_proj = v_uvs * inv_z[:, None]

    # Loop over triangles
    for tri in t_pos_idx:
        idx0, idx1, idx2 = tri
        verts = np.array([pts_raster[idx0], pts_raster[idx1], pts_raster[idx2]])
        z_vals = np.array([depths[idx0], depths[idx1], depths[idx2]])
        uvs = np.array([uvs_proj[idx0], uvs_proj[idx1], uvs_proj[idx2]])
        invzs = inv_z[[idx0, idx1, idx2]]

        # Bounding box
        min_x = max(int(np.floor(np.min(verts[:, 0]))), 0)
        max_x = min(int(np.ceil(np.max(verts[:, 0]))), res_w - 1)
        min_y = max(int(np.floor(np.min(verts[:, 1]))), 0)
        max_y = min(int(np.ceil(np.max(verts[:, 1]))), res_h - 1)

        # Edge functions
        def edge_func(v0, v1, p):
            return (p[0] - v0[0]) * (v1[1] - v0[1]) - (p[1] - v0[1]) * (v1[0] - v0[0])

        area = edge_func(verts[0], verts[1], verts[2])
        if abs(area) < 1e-10:
            continue

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                p = np.array([x + 0.5, y + 0.5])
                w0 = edge_func(verts[1], verts[2], p)
                w1 = edge_func(verts[2], verts[0], p)
                w2 = edge_func(verts[0], verts[1], p)

                if (w0 >= 0 and w1 >= 0 and w2 >= 0) or (w0 <= 0 and w1 <= 0 and w2 <= 0):
                    w0 /= area
                    w1 /= area
                    w2 /= area

                    z_interp = 1.0 / (w0 * invzs[0] + w1 * invzs[1] + w2 * invzs[2])
                    if z_interp < z_buffer[y, x]:
                        uv_interp = (w0 * uvs[0] + w1 * uvs[1] + w2 * uvs[2]) * z_interp
                        tex_u = int(np.clip(uv_interp[0] * texture_img.shape[1], 0, texture_img.shape[1] - 1))
                        tex_v = int(np.clip((1 - uv_interp[1]) * texture_img.shape[0], 0, texture_img.shape[0] - 1))  # flip Y

                        color = texture_img[tex_v, tex_u]
                        img[y, x] = (color * 255).astype(np.uint8)
                        z_buffer[y, x] = z_interp

    return img


#def render_object(v_pos, v_clr, t_pos_idx, plane_h, plane_w, res_h, res_w, focal, eye, up, target, v_uvs=None, shading='t') -> np.ndarray:
    #"""
    #Render the specified object from the specified camera using polygon_fill functions.
    #"""
    ## Get camera transformation
    #R, t = lookat(eye, up, target)
#
    ## Project vertices to 2D
    #pts_2d, depths = perspective_project(v_pos.T, focal, R, t)
#
    ## Rasterize to pixel coordinates
    #pts_raster = rasterize(pts_2d, plane_w, plane_h, res_w, res_h)
#
    ## Convert to format expected by polygon_fill.render_img
    ## polygon_fill expects vertices as (N, 2) array
    #vertices_2d = pts_raster.T
#
    ## Ensure we have UV coordinates
    #if v_uvs is None:
        ## Create default UV coordinates if not provided
        #v_uvs = np.random.rand(v_pos.shape[0], 2)
#
    ## Call the polygon_fill.render_img function
    ## Note: render_img expects specific parameter order and formats
    #img = polygon_fill.render_img(
        #faces=t_pos_idx,
        #vertices=vertices_2d,
        #vcolors=v_clr,
        #uvs=v_uvs,
        #depth=depths,
        #shading=shading
    #)
#
    #return img

def load_data():
    """
    Load and extract data from hw2.npy file.
    """
    data = np.load('hw2.npy', allow_pickle=True).item()

    # Extract parameters
    k_cam_up = data['k_cam_up']
    k_sensor_height = data['k_sensor_height']
    k_sensor_width = data['k_sensor_width']
    k_f = data['k_f']
    car_velocity = data['car_velocity']
    k_road_radius = data['k_road_radius']
    k_road_center = data['k_road_center']
    k_cam_car_rel_pos = data['k_cam_car_rel_pos']
    k_duration = data['k_duration']
    k_fps = data['k_fps']
    v_pos = data['v_pos'].T
    v_uvs = data['v_uvs']
    t_pos_idx = data['t_pos_idx']
    k_cam_target = data['k_cam_target']

    return {
        'k_cam_up': k_cam_up,
        'k_sensor_height': k_sensor_height,
        'k_sensor_width': k_sensor_width,
        'k_f': k_f,
        'car_velocity': car_velocity,
        'k_road_radius': k_road_radius,
        'k_road_center': k_road_center,
        'k_cam_car_rel_pos': k_cam_car_rel_pos,
        'k_duration': k_duration,
        'k_fps': k_fps,
        'v_pos': v_pos,
        'v_uvs': v_uvs,
        't_pos_idx': t_pos_idx,
        'k_cam_target': k_cam_target
    }

def demo_case1():
    """
    Case 1: Static camera with z-axis parallel to car's speed vector.
    Creates an animated version with 25 fps for 5 seconds.
    """
    print("Running Animated Case 1: Static camera")

    # Load data
    data = load_data()
    texImg = plt.imread('stone-72_diffuse.jpg')

    # Extract parameters
    k_road_radius = data['k_road_radius']
    k_road_center = data['k_road_center']
    car_velocity = data['car_velocity']
    k_cam_car_rel_pos = data['k_cam_car_rel_pos']
    k_cam_up = data['k_cam_up']
    v_pos = data['v_pos']
    v_uvs = data['v_uvs']
    t_pos_idx = data['t_pos_idx']
    k_f = data['k_f']
    k_sensor_width = data['k_sensor_width']
    k_sensor_height = data['k_sensor_height']

    print(f"DEBUG - Road radius: {k_road_radius}")
    print(f"DEBUG - Road center: {k_road_center}")
    print(f"DEBUG - Car velocity: {car_velocity}")
    print(f"DEBUG - Camera relative position: {k_cam_car_rel_pos}")
    print(f"DEBUG - Object vertices shape: {v_pos.shape}")
    print(f"DEBUG - Object faces shape: {t_pos_idx.shape}")
    print(f"DEBUG - Focal length: {k_f}")
    print(f"DEBUG - Sensor size: {k_sensor_width}x{k_sensor_height}")
    print(f"DEBUG - Object bounds: min={np.min(v_pos, axis=0)}, max={np.max(v_pos, axis=0)}")

    # Create simple vertex colors
    v_clr = np.random.rand(v_pos.shape[0], 3)

    # Animation parameters - 25 fps for 5 seconds = 125 frames
    fps = 25
    duration = 5
    num_frames = fps * duration

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 512)  # Updated to match polygon_fill expected resolution
    ax.set_ylim(0, 512)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Case 1: Static Camera Animation (Using polygon_fill)')

    # Initialize empty image
    im = ax.imshow(np.ones((512, 512, 3), dtype=np.uint8) * 255, animated=True)

    def animate_frame(frame):
        # Calculate car position on circular path
        t = frame / num_frames * 2 * np.pi
        car_pos = k_road_center + k_road_radius * np.array([np.cos(t), np.sin(t), 0])

        # Car's speed vector (tangent to circle)
        speed_vector = np.array([-np.sin(t), np.cos(t), 0])

        # Camera position (static relative to car)
        cam_pos = car_pos + k_cam_car_rel_pos

        # Camera target (along speed vector)
        cam_target = car_pos + speed_vector

        # Render frame using polygon_fill functions
        img = render_object(
            v_pos + car_pos,  # Translate object to car position
            v_clr,
            t_pos_idx,
            k_sensor_height,
            k_sensor_width,
            512,  # res_h (match polygon_fill expected size)
            512,  # res_w
            k_f,
            cam_pos,
            k_cam_up,
            cam_target,
            v_uvs,
            shading='t',
            texture_img=texImg
        )

        im.set_array(img)
        return [im]

    # Create and run animation
    anim = animation.FuncAnimation(
        fig, animate_frame, frames=num_frames,
        interval=1000/fps, blit=True, repeat=True
    )

    plt.tight_layout()
    plt.show()

    return anim

def demo_case2():
    """
    Case 2: Camera spins around and always points to k_cam_target.
    Creates an animated version with 25 fps for 5 seconds.
    """
    print("Running Animated Case 2: Spinning camera")

    # Load data
    data = load_data()
    texImg = plt.imread('stone-72_diffuse.jpg')

    # Extract parameters
    k_road_radius = data['k_road_radius']
    k_road_center = data['k_road_center']
    car_velocity = data['car_velocity']
    k_cam_car_rel_pos = data['k_cam_car_rel_pos']
    k_cam_up = data['k_cam_up']
    k_cam_target = data['k_cam_target']
    v_pos = data['v_pos']
    v_uvs = data['v_uvs']
    t_pos_idx = data['t_pos_idx']
    k_f = data['k_f']
    k_sensor_width = data['k_sensor_width']
    k_sensor_height = data['k_sensor_height']

    # Create simple vertex colors
    v_clr = np.random.rand(v_pos.shape[0], 3)

    # Animation parameters - 25 fps for 5 seconds = 125 frames
    fps = 25
    duration = 5
    num_frames = fps * duration

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 512)  # Updated to match polygon_fill expected resolution
    ax.set_ylim(0, 512)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Case 2: Spinning Camera Animation (Using polygon_fill)')

    # Initialize empty image
    im = ax.imshow(np.ones((512, 512, 3), dtype=np.uint8) * 255, animated=True)

    def animate_frame(frame):
        # Calculate car position on circular path
        t = frame / num_frames * 2 * np.pi
        car_pos = k_road_center + k_road_radius * np.array([np.cos(t), np.sin(t), 0])

        # Camera rotates around the car
        cam_angle = t * 2  # Camera spins faster than car moves
        cam_rel_pos = np.array([
            k_cam_car_rel_pos[0] * np.cos(cam_angle) - k_cam_car_rel_pos[1] * np.sin(cam_angle),
            k_cam_car_rel_pos[0] * np.sin(cam_angle) + k_cam_car_rel_pos[1] * np.cos(cam_angle),
            k_cam_car_rel_pos[2]
        ])
        cam_pos = car_pos + cam_rel_pos

        # Camera always points to target
        cam_target = k_cam_target

        # Render frame using polygon_fill functions
        img = render_object(
            v_pos + car_pos,  # Translate object to car position
            v_clr,
            t_pos_idx,
            k_sensor_height,
            k_sensor_width,
            512,  # res_h (match polygon_fill expected size)
            512,  # res_w
            k_f,
            cam_pos,
            k_cam_up,
            cam_target,
            v_uvs,
            shading='t',
            texture_img=texImg
        )

        im.set_array(img)
        return [im]

    # Create and run animation
    anim = animation.FuncAnimation(
        fig, animate_frame, frames=num_frames,
        interval=1000/fps, blit=True, repeat=True
    )

    plt.tight_layout()
    plt.show()

    return anim

def run_both_animations():
    """
    Run both animations simultaneously in separate windows.
    """
    print("3D Graphics Animation Demo")
    print("=" * 40)
    print("Each animation runs at 25 fps for 5 seconds (125 frames total)")
    print("Close the animation windows to continue...")

    # Run Case 1 animation
    print("\nStarting Case 1 Animation...")
    anim1 = demo_case1()

    # Run Case 2 animation
    print("\nStarting Case 2 Animation...")
    anim2 = demo_case2()

    return anim1, anim2

def save_animations():
    """
    Optional function to save animations as MP4 files.
    Requires ffmpeg to be installed.
    """
    print("Saving animations as MP4 files...")

    # You can uncomment and modify this section to save animations
    # Note: Requires ffmpeg to be installed on your system

    anim1 = demo_case1()
    anim1.save('case1_animation.mp4', writer='ffmpeg', fps=25)

    anim2 = demo_case2()
    anim2.save('case2_animation.mp4', writer='ffmpeg', fps=25)

    print("Animations saved!")

def main():
    """
    Main function to run both animated demos.
    """
    # Run the animations
    #animations = run_both_animations()

    save_animations()
    print("\nAnimations completed!")

    #return animations

if __name__ == "__main__":
    animations = main()


