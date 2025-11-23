import math
import numpy as np

# Image size (portrait)
img_width = 1080
img_height = 1920

# 35mm equivalent focal length
f_35mm_equiv = 26.0  # mm

# Full-frame width in mm
FF_WIDTH = 36.0

# Horizontal field of view (radians)
hfov_rad = 2 * math.atan(FF_WIDTH / (2 * f_35mm_equiv))

# Focal length in pixels
fx = (img_width / 2) / math.tan(hfov_rad / 2)
fy = fx  # assume square pixels

# Principal point (center of image)
cx = img_width / 2
cy = img_height / 2

# Intrinsic matrix
K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
])

print("Intrinsic matrix K:\n", K)
print(f"fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
print(f"Horizontal FOV ≈ {math.degrees(hfov_rad):.1f}°")
