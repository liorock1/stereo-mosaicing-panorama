import cv2
import numpy as np
import imageio
# import matplotlib.pyplot as plt
from PIL import Image
import os


def get_video_frames(video_path):
    """
    Reads a video file and returns a 4D NumPy array of BGR frames.
    Shape: (Number of frames, Height, Width, 3)
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return np.array(frames)


def get_translation_matrices(frames):
    matrices = []
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Pre-calculate for the very first frame
    kp_prev, des_prev = orb.detectAndCompute(frames[0], None)

    for i in range(1, len(frames)):
        # Calculate for the current frame
        kp_curr, des_curr = orb.detectAndCompute(frames[i], None)

        # Match with the previous frame's data
        matches = bf.match(des_prev, des_curr)

        src_pts = np.float32([kp_prev[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([kp_curr[m.trainIdx].pt for m in matches])

        # PLOT HERE - for the report
        # if i == len(frames) // 2:
        #     plot_matches(frames[i - 1], frames[i], src_pts, dst_pts)

        tx, ty = np.median(dst_pts - src_pts, axis=0)

        # Create the matrix
        M = np.eye(3, dtype=np.float32)
        M[0, 2], M[1, 2] = tx, ty
        matrices.append(M)

        # Move current to previous for the next iteration
        kp_prev, des_prev = kp_curr, des_curr

    return np.array(matrices)

# === need to stabilize? ===

def get_global_transforms(frames, rel_matrices):
    """
    Computes global transformation matrices and calculates canvas size
    """
    num_frames = len(frames)
    # unpacking for Height, Width
    height, width = frames.shape[1:3]

    mid_idx = num_frames // 2

    H = [None] * num_frames
    H[mid_idx] = np.eye(3)

    # Propagate backwards
    for i in range(mid_idx - 1, -1, -1):
        H[i] = H[i + 1] @ rel_matrices[i]

    # Propagate forwards
    for i in range(mid_idx + 1, num_frames):
        H[i] = H[i - 1] @ np.linalg.inv(rel_matrices[i - 1])

    # Determine Canvas Boundaries
    corners = np.array([
        [0, 0, 1],
        [width, 0, 1],
        [0, height, 1],
        [width, height, 1]
    ]).T

    all_x = []
    for i in range(num_frames):
        projected_corners = H[i] @ corners
        projected_corners /= projected_corners[2, :]
        all_x.extend(projected_corners[0, :])

    min_x = min(all_x)
    max_x = max(all_x)

    # Apply Offset Translation
    offset_val = -min_x
    translation_offset = np.array([
        [1, 0, offset_val],
        [0, 1, 0],
        [0, 0, 1]
    ])

    global_matrices = [translation_offset @ h for h in H]

    canvas_width = int(np.ceil(max_x - min_x))
    canvas_height = height

    return global_matrices, (canvas_width, canvas_height)


def create_mosaic(frames, global_matrices, canvas_size, slit_offset=0):
    canvas_width, canvas_height = canvas_size
    panorama = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    num_frames = len(frames)
    frame_height, frame_width = frames.shape[1:3]
    slit_x_in_frame = (frame_width // 2) + slit_offset

    for i in range(num_frames):
        # 1. Project the current slit center
        slit_point = np.array([slit_x_in_frame, 0, 1]).reshape(3, 1)
        projected_point = global_matrices[i] @ slit_point
        canvas_x = int((projected_point[0] / projected_point[2]).item())

        # 2. Define start_x and end_x
        if i < num_frames - 1:
            next_projected = global_matrices[i + 1] @ slit_point
            next_canvas_x = int((next_projected[0] / next_projected[2]).item())
            start_x = min(canvas_x, next_canvas_x)
            end_x = max(canvas_x, next_canvas_x)
        else:
            # For the last frame
            start_x = canvas_x
            end_x = canvas_x + (frame_width // 10)

        # Clamp boundaries to prevent errors
        start_x = max(0, start_x)
        end_x = min(canvas_width, end_x + 1)

        # 3. Apply Backward Warping
        inv_H = np.linalg.inv(global_matrices[i])

        # Pull matrix values into local variables for faster Python access
        h00, h01, h02 = inv_H[0]
        h10, h11, h12 = inv_H[1]
        h20, h21, h22 = inv_H[2]

        for x in range(start_x, end_x):
            # Pre-calculate X-parts that don't change in the 'y' loop
            x0 = h00 * x + h02
            x1 = h10 * x + h12
            x2 = h20 * x + h22

            for y in range(canvas_height):
                # Manual Dot Product
                z = x2 + h21 * y
                if z == 0: z = 1.0

                src_x = int(round((x0 + h01 * y) / z))
                src_y = int(round((x1 + h11 * y) / z))

                if 0 <= src_x < frame_width and 0 <= src_y < frame_height:
                    panorama[y, x] = frames[i, src_y, src_x]

    return panorama


# ========================
# VISUALIZATION FUNCTIONS FOR THE REPORT
# ========================

# visualize ORB matches
# def plot_matches(f1, f2, src_pts, dst_pts):
#     # 1. Convert BGR to RGB if the images have 3 channels (Color)
#     if len(f1.shape) == 3:
#         f1_rgb = cv2.cvtColor(f1, cv2.COLOR_BGR2RGB)
#         f2_rgb = cv2.cvtColor(f2, cv2.COLOR_BGR2RGB)
#     else:
#         f1_rgb, f2_rgb = f1, f2  # Grayscale stays the same
#
#     # 2. Stack the RGB frames
#     vis = np.concatenate((f1_rgb, f2_rgb), axis=1)
#     offset = f1.shape[1]
#
#     plt.figure(figsize=(15, 7))
#
#     # 3. Explicitly set vmin/vmax if images are grayscale to prevent scaling
#     if len(vis.shape) == 2:
#         plt.imshow(vis, cmap='gray', vmin=0, vmax=255)
#     else:
#         plt.imshow(vis)
#
#     # Plotting points/lines0
#     for i in range(min(50, len(src_pts))):
#         x1, y1 = src_pts[i]
#         x2, y2 = dst_pts[i]
#         plt.plot([x1, x2 + offset], [y1, y2], 'r-', alpha=0.5, linewidth=1)
#         plt.plot(x1, y1, 'go', markersize=3)
#         plt.plot(x2 + offset, y2, 'bo', markersize=3)
#
#     plt.axis('off')
#     plt.title("Feature Matching Visualization")
#     plt.show()


# create video
def panoramas_to_video(panoramas, output_path, fps=10):

    # Imageio expects RGB. If your images are BGR (from OpenCV), we must flip them to RGB first.
    processed_frames = []
    for frame in panoramas:
        # Convert to uint8 if not already
        frame_uint8 = frame.astype(np.uint8)
        # OpenCV BGR to RGB conversion
        frame_rgb = frame_uint8[:, :, ::-1]
        processed_frames.append(frame_rgb)

    imageio.mimsave(output_path, processed_frames, fps=fps, codec='libx264')


# ========================
# Submission API
# ========================
def generate_panorama(input_frames_path, n_out_frames):
    """
    Main entry point for ex4. Generates a list of panorama frames.
    """
    # 1. Load images from the directory in the correct sorted order
    # Format: "frame_00000.jpg", "frame_00001.jpg", etc.
    all_files = sorted([f for f in os.listdir(input_frames_path) if f.endswith('.jpg')])

    frames_list = []
    for file_name in all_files:
        img_path = os.path.join(input_frames_path, file_name)
        img = cv2.imread(img_path)
        if img is not None:
            frames_list.append(img)

    # Convert list to 4D NumPy array (N, H, W, 3) as required by your functions
    frames = np.array(frames_list)

    # 2. Compute relative translation matrices
    rel_matrices = get_translation_matrices(frames)

    # 3. Compute global transforms and determine canvas size
    global_matrices, canvas_size = get_global_transforms(frames, rel_matrices)

    # 4. Generate n_out_frames panoramas by varying the slit_offset
    # We create a range of offsets (e.g., from -width/4 to +width/4)
    # to create different perspectives of the panorama.
    frame_width = frames.shape[2]
    max_offset = frame_width // 4  # Adjust this value to control perspective shift
    offsets = np.linspace(-max_offset, max_offset, n_out_frames)

    panorama_frames = []
    for offset in offsets:
        # Create the mosaic (BGR format)
        mosaic_bgr = create_mosaic(frames, global_matrices, canvas_size, slit_offset=int(offset))

        # 5. Convert BGR to RGB and then to a PIL Image
        mosaic_rgb = cv2.cvtColor(mosaic_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(mosaic_rgb)

        panorama_frames.append(pil_img)

    return panorama_frames

if __name__=="__main__":
    # 1. Load video
    frames_array = get_video_frames('exrecise inputs/banana.mp4')

    # 2. Get relative transformations (Step 1)
    rel_matrices = get_translation_matrices(frames_array)

    # 3. skipped stabilization

    # 4. Get Global Matrices and Canvas Size (Step 3)
    global_matrices, canvas_size = get_global_transforms(frames_array, rel_matrices)
#     # === create classic panorama ===
#     # 5. Create the Classic Panorama (center slit)
#     print("Generating classic panorama...")
#     panorama = create_mosaic(frames_array, global_matrices, canvas_size, slit_offset=0)
#
#     # # Save and Show
#     cv2.imwrite('outputs/classic_panorama_kessaria_fast.jpg', panorama)
#     cv2.imshow('Classic Panorama', panorama)
#
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

    # === create panorama video ===
    slits = [-200,0,200]
    panoramas = []
    for slit in slits:
        panoramas.append(create_mosaic(frames_array, global_matrices, canvas_size, slit))
        # save viewpoints
        if (slit in [-200,0,200]):
            cv2.imwrite(f'outputs/banana/banana {slit} panorama.jpg', panoramas[-1])

    # panoramas_to_video(panoramas,"video output/boat_video_new.mp4")
