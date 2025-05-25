import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from typing import Tuple
import math
import polygon_fill
import vec_inter
import argparse
import time

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

def render_object(v_pos, v_clr, t_pos_idx, plane_h, plane_w, res_h, res_w, focal, eye, up, target, v_uvs=None, shading='t') -> np.ndarray:
    """
    Render the specified object from the specified camera using polygon_fill functions.
    """
    # Get camera transformation
    R, t = lookat(eye, up, target)

    # Project vertices to 2D
    pts_2d, depths = perspective_project(v_pos.T, focal, R, t)

    # Rasterize to pixel coordinates
    pts_raster = rasterize(pts_2d, plane_w, plane_h, res_w, res_h)

    # Convert to format expected by polygon_fill.render_img
    # polygon_fill expects vertices as (N, 2) array
    vertices_2d = pts_raster.T

    # Ensure we have UV coordinates
    if v_uvs is None:
        # Create default UV coordinates if not provided
        v_uvs = np.random.rand(v_pos.shape[0], 2)

    # Call the polygon_fill.render_img function
    # Note: render_img expects specific parameter order and formats
    img = polygon_fill.render_img(
        faces=t_pos_idx,
        vertices=vertices_2d,
        vcolors=v_clr,
        uvs=v_uvs,
        depth=depths,
        shading=shading
    )

    return img

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

def precompute_animation_data(data, case_type='case1', fps=25, duration=5):
    """
    Precompute all animation frame data to avoid redundant calculations during animation.
    """
    num_frames = fps * duration
    
    # Extract common parameters
    k_road_radius = data['k_road_radius']
    k_road_center = data['k_road_center']
    k_cam_car_rel_pos = data['k_cam_car_rel_pos']
    k_cam_target = data.get('k_cam_target', np.array([0, 0, 0]))
    
    # Precompute trigonometric values
    t_values = np.linspace(0, 2 * np.pi, num_frames)
    cos_values = np.cos(t_values)
    sin_values = np.sin(t_values)
    
    # Precompute car positions
    car_positions = k_road_center + k_road_radius * np.column_stack([cos_values, sin_values, np.zeros(num_frames)])
    
    # Precompute speed vectors (tangent to circle)
    speed_vectors = np.column_stack([-sin_values, cos_values, np.zeros(num_frames)])
    
    # Precompute camera data based on case type
    if case_type == 'case1':
        # Static camera relative to car
        cam_positions = car_positions + k_cam_car_rel_pos
        cam_targets = car_positions + speed_vectors
    elif case_type == 'case2':
        # Camera rotates around car
        cam_angles = t_values * 2  # Camera spins faster
        cos_cam = np.cos(cam_angles)
        sin_cam = np.sin(cam_angles)
        
        cam_rel_positions = np.column_stack([
            k_cam_car_rel_pos[0] * cos_cam - k_cam_car_rel_pos[1] * sin_cam,
            k_cam_car_rel_pos[0] * sin_cam + k_cam_car_rel_pos[1] * cos_cam,
            np.full(num_frames, k_cam_car_rel_pos[2])
        ])
        
        cam_positions = car_positions + cam_rel_positions
        cam_targets = np.tile(k_cam_target, (num_frames, 1))
    
    return {
        'car_positions': car_positions,
        'cam_positions': cam_positions,
        'cam_targets': cam_targets,
        'num_frames': num_frames
    }

def create_animation(case_type='case1', save_to_disk=False, show_on_screen=True, fps=25, duration=5):
    """
    Create optimized animation with option to save or display.
    """
    print(f"Running Optimized {case_type.title()}: {'Saving to disk' if save_to_disk else 'Showing on screen'}")
    
    # Load data
    data = load_data()
    
    # Extract parameters
    v_pos = data['v_pos']
    v_uvs = data['v_uvs']
    t_pos_idx = data['t_pos_idx']
    k_f = data['k_f']
    k_sensor_width = data['k_sensor_width']
    k_sensor_height = data['k_sensor_height']
    k_cam_up = data['k_cam_up']
    
    # Create optimized vertex colors (single computation)
    np.random.seed(42)  # For reproducible colors
    v_clr = np.random.rand(v_pos.shape[0], 3)
    
    # Precompute all frame data
    print("Precomputing animation data...")
    start_time = time.time()
    anim_data = precompute_animation_data(data, case_type, fps, duration)
    precompute_time = time.time() - start_time
    print(f"Precomputation completed in {precompute_time:.2f} seconds")
    
    # Set up the figure and axis
    if show_on_screen:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, 512)
        ax.set_ylim(0, 512)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'{case_type.title()}: {"Static" if case_type == "case1" else "Spinning"} Camera Animation (Optimized)')
        
        # Initialize empty image
        im = ax.imshow(np.ones((512, 512, 3), dtype=np.uint8) * 255, animated=True)
    else:
        fig, ax, im = None, None, None
    
    def animate_frame(frame):
        # Get precomputed data for this frame
        car_pos = anim_data['car_positions'][frame]
        cam_pos = anim_data['cam_positions'][frame]
        cam_target = anim_data['cam_targets'][frame]
        
        # Render frame using polygon_fill functions
        img = render_object(
            v_pos + car_pos,  # Translate object to car position
            v_clr,
            t_pos_idx,
            k_sensor_height,
            k_sensor_width,
            512,  # res_h
            512,  # res_w
            k_f,
            cam_pos,
            k_cam_up,
            cam_target,
            v_uvs,
            shading='t' if case_type == 'case1' else 'f'
        )
        
        if show_on_screen:
            im.set_array(img)
            return [im]
        else:
            return img
    
    # Create animation
    if show_on_screen and not save_to_disk:
        # Show animation on screen only
        anim = animation.FuncAnimation(
            fig, animate_frame, frames=anim_data['num_frames'],
            interval=1000/fps, blit=True, repeat=True
        )
        plt.tight_layout()
        plt.show()
        return anim
    
    elif save_to_disk and not show_on_screen:
        # Save animation to disk only (no display)
        print("Generating frames for video...")
        start_time = time.time()
        
        # Generate all frames without displaying
        frames = []
        for frame in range(anim_data['num_frames']):
            if frame % 25 == 0:  # Progress indicator
                print(f"Rendering frame {frame}/{anim_data['num_frames']}")
            img = animate_frame(frame)
            frames.append(img)
        
        render_time = time.time() - start_time
        print(f"Frame rendering completed in {render_time:.2f} seconds")
        
        # Create a minimal animation for saving
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, 512)
        ax.set_ylim(0, 512)
        ax.set_aspect('equal')
        ax.axis('off')
        im = ax.imshow(frames[0], animated=True)
        
        def save_animate_frame(frame):
            im.set_array(frames[frame])
            return [im]
        
        anim = animation.FuncAnimation(
            fig, save_animate_frame, frames=len(frames),
            interval=1000/fps, blit=True, repeat=False
        )
        
        # Save animation
        filename = f'{case_type}_animation_optimized.mp4'
        print(f"Saving animation to {filename}...")
        start_time = time.time()
        anim.save(filename, writer='ffmpeg', fps=fps, bitrate=1800)
        save_time = time.time() - start_time
        print(f"Animation saved in {save_time:.2f} seconds")
        
        plt.close(fig)  # Close figure to free memory
        return anim
    
    else:
        # Both save and show (original behavior but optimized)
        anim = animation.FuncAnimation(
            fig, animate_frame, frames=anim_data['num_frames'],
            interval=1000/fps, blit=True, repeat=True
        )
        
        if save_to_disk:
            filename = f'{case_type}_animation_optimized.mp4'
            print(f"Saving animation to {filename}...")
            anim.save(filename, writer='ffmpeg', fps=fps, bitrate=1800)
        
        if show_on_screen:
            plt.tight_layout()
            plt.show()
        
        return anim

def demo_case1(save_to_disk=False, show_on_screen=True):
    """
    Case 1: Static camera with z-axis parallel to car's speed vector.
    """
    return create_animation('case1', save_to_disk, show_on_screen)

def demo_case2(save_to_disk=False, show_on_screen=True):
    """
    Case 2: Camera spins around and always points to k_cam_target.
    """
    return create_animation('case2', save_to_disk, show_on_screen)

def main():
    """
    Main function with command line arguments to control save/display behavior.
    """
    parser = argparse.ArgumentParser(description='3D Graphics Animation Demo')
    parser.add_argument('--mode', choices=['show', 'save', 'both'], default='show',
                       help='Mode: show on screen, save to disk, or both (default: show)')
    parser.add_argument('--case', choices=['case1', 'case2', 'both'], default='both',
                       help='Which case to run: case1, case2, or both (default: both)')
    parser.add_argument('--fps', type=int, default=25,
                       help='Frames per second (default: 25)')
    parser.add_argument('--duration', type=int, default=5,
                       help='Duration in seconds (default: 5)')
    
    args = parser.parse_args()
    
    save_to_disk = args.mode in ['save', 'both']
    show_on_screen = args.mode in ['show', 'both']
    
    print("3D Graphics Animation Demo (Optimized)")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    print(f"Cases: {args.case}")
    print(f"Animation: {args.fps} fps for {args.duration} seconds ({args.fps * args.duration} frames total)")
    
    if args.case in ['case1', 'both']:
        print(f"\n{'='*20} CASE 1 {'='*20}")
        demo_case1(save_to_disk, show_on_screen)
    
    if args.case in ['case2', 'both']:
        print(f"\n{'='*20} CASE 2 {'='*20}")
        demo_case2(save_to_disk, show_on_screen)
    
    print("\nAnimations completed!")

if __name__ == "__main__":
    main()
