def create_mosaic(frames, global_matrices, canvas_size, slit_offset=0):
    """
    Creates a static panorama by sampling vertical strips from each frame.

    Args:
        frames: Stabilized BGR frames (N, H, W, 3).
        global_matrices: Matrices mapping each frame to the canvas.
        canvas_size: (Width, Height) of the canvas.
        slit_offset: Pixels away from the center of the frame (0 for classic).
    """
    canvas_width, canvas_height = canvas_size
    panorama = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    num_frames = len(frames)
    frame_height, frame_width = frames.shape[1:3]

    # Slit is at center (frame_width // 2) + offset
    slit_x_in_frame = (frame_width // 2) + slit_offset

    for i in range(num_frames):
        # 1. Project the current slit center
        slit_point = np.array([slit_x_in_frame, 0, 1]).reshape(3, 1)
        projected_point = global_matrices[i] @ slit_point
        canvas_x = int((projected_point[0] / projected_point[2]).item())

        # 2. Determine the boundaries (Handling Left and Right movement)
        if i < num_frames - 1:
            next_projected = global_matrices[i + 1] @ slit_point
            next_canvas_x = int((next_projected[0] / next_projected[2]).item())

            # The strip should cover the space between the two centers
            # This handles the 'moving left' case where next_canvas_x < canvas_x
            start_x = min(canvas_x, next_canvas_x)
            end_x = max(canvas_x, next_canvas_x)
        else:
            # For the last frame, just use a default width
            start_x = canvas_x
            end_x = canvas_x + (frame_width // 10)

        # 3. Apply Backward Warping
        inv_H = np.linalg.inv(global_matrices[i])

        # Clamp to canvas boundaries
        start_x = max(0, start_x)
        end_x = min(canvas_width, end_x + 1)  # +1 overlap prevents 1px black gaps

        for x in range(start_x, end_x):
            for y in range(canvas_height):
                src_pt = inv_H @ np.array([x, y, 1])
                # Check for division by zero just in case
                z = src_pt[2] if src_pt[2] != 0 else 1.0
                src_x = int(round(src_pt[0] / z))
                src_y = int(round(src_pt[1] / z))

                if 0 <= src_x < frame_width and 0 <= src_y < frame_height:
                    panorama[y, x] = frames[i, src_y, src_x]

    return panorama