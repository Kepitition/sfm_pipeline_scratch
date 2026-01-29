#######################################################################
################## CHALMERS UNIVERSITY OF TECHNOLOGY ##################
####################### COMPUTER VISION PROJECT #######################
#################### AYBERK TUNCA - 11 JANUARY 2026 ###################
#######################################################################

import numpy as np
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
from project_helpers import homography_to_RT, correct_H_sign, get_dataset_info
import matplotlib.cm

def pflat(x):
    # Normalizes (m,n)-array x of n homogeneous points
    # to last coordinate 1.
    if x.ndim == 1:
        return x / x[-1]
    return x / x[-1:, :]

def skew(t):
    t = np.asarray(t).flatten()
    return np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0]
    ])


def project_rotation(R):
    U, _, Vt = np.linalg.svd(R)
    R_proj = U @ Vt
    if np.linalg.det(R_proj) < 0:
        R_proj = U @ np.diag([1, 1, -1]) @ Vt
    return R_proj


# We used triangulate function previous assignment so I just copied and pasted. Also changed from cartesian to 3D as input
def triangulate_3D_point_DLT(x1, x2, P1, P2):

    """
    Triangulate a single 3D point from two views using DLT.

    Parameters
    ----------
        x1, x2 : ndarray, (2, )
            2D points in the first/second image. Also it can be homogenous as well.

        P1, P2 : ndarray, (3, 4)
            Camera matrices of the first/second image

    Returns
    -------
        X : ndarray, (4, )
            Estimated 3D point in homogeneous coordinates.
    """
    # Your code here
    if x1.shape[0] == 3:
        u1, v1 = x1[0]/x1[2], x1[1]/x1[2]
    else:
        u1, v1 = x1[0], x1[1]

    if x2.shape[0] == 3:
        u2, v2 = x2[0]/x2[2], x2[1]/x2[2]
    else:
        u2, v2 = x2[0], x2[1]

    if np.ndim(u1) == 0:
        u1 = np.array([u1])
        v1 = np.array([v1])
        u2 = np.array([u2])
        v2 = np.array([v2])

    N = u1.shape[0]
    X = np.zeros((4, N))

    for i in range(N):
        A = np.array([
            u1[i] * P1[2, :] - P1[0, :],  # x from 1  u1(P1_3^T X) - (P1_1^T X) = 0
            v1[i] * P1[2, :] - P1[1, :],  # y from 1  v1(P1_3^T X) - (P1_2^T X) = 0
            u2[i] * P2[2, :] - P2[0, :],  # x from 2  u2(P2_3^T X) - (P2_1^T X) = 0
            v2[i] * P2[2, :] - P2[1, :]  # y from 2  v2(P2_3^T X) - (P2_2^T X) = 0
        ])

        U, S, Vt = np.linalg.svd(A)
        X[:, i] = Vt[-1, :]

    if N == 1:
        return X[:, 0]

    return X


def depth_check(P, X):

    X_norm = X/X[3,:]

    x_proj = P @ X_norm

    M = P[:,:3]

    # depth = det(M) * z_cam / ||m3||
    det_M = np.linalg.det(M)
    norm_m3 = np.linalg.norm(M[2,:])

    depths = (det_M * x_proj[2,:]) / norm_m3

    front_ids = depths > 0
    return depths, front_ids


def triangulate_points(x1, x2, P1, P2, filter_behind=True):
    if x1.ndim == 1:
        x1 = x1.reshape(3, 1)
    if x2.ndim == 1:
        x2 = x2.reshape(3, 1)

    X_all = triangulate_3D_point_DLT(x1, x2, P1, P2)

    if X_all.ndim == 1:
        X_all = X_all.reshape(4, 1)

    _, front1 = depth_check(P1, X_all)
    _, front2 = depth_check(P2, X_all)

    valid_mask = front1 & front2

    if filter_behind:
        return X_all[:, valid_mask], valid_mask
    else:
        return X_all, np.ones(X_all.shape[1], dtype=bool)

def estimate_F_DLT(x1s, x2s):
    """
    x1s and x2s contain matching points
    x1s - 2D image points in the first image in homogenous coordinates (3xN)
    x2s - 2D image points in the second image in homogenous coordinates (3xN)
    """
    # Your code here
    x1_flat = pflat(x1s)
    x2_flat = pflat(x2s)

    u1,v1,w1 = x1_flat[0],x1_flat[1],x1_flat[2]
    u2,v2,w2 = x2_flat[0],x2_flat[1],x2_flat[2]

    M = np.array([u2*u1, u2*v1, u2*w1,
                  v2*u1, v2*v1, v2*w1,
                  w2*u1, w2*v1, w2*w1]).T
    U, S, Vt = np.linalg.svd(M)

    v = Vt[-1]

    F = v.reshape(3,3)

    # print(f"Smallest singular value: {S[-1]}")
    # print(f"||Mv||: {np.linalg.norm(M@v)}")
    return F


def enforce_essential(E_approx):
    """
    E_approx - Approximate Essential matrix (3x3)
    """
    U , S, Vt = np.linalg.svd(E_approx)
    S_new = np.array([1,1,0])

    E = U @ np.diag(S_new) @ Vt

    _, S_check, _ = np.linalg.svd(E)
    # print(f"New singular check (1,1,0): {S_check}")
    # print(f"det(E) = {np.linalg.det(E)}")
    return E

# Apply enforce_essential and estimate_F_DLT (maybe combine later)
def estimate_E_8point(x1, x2):
    E_approx = estimate_F_DLT(x1, x2)
    E = enforce_essential(E_approx)
    return E


def extract_P_from_E(E):
    """
    A function that extract the four P2 solutions given above
    E - Essential matrix (3x3)
    P - Array containing all four P2 solutions (4x3x4) (i.e. P[i,:,:] is the ith solution)
    """
    U, S, Vt = np.linalg.svd(E)

    if np.linalg.det(U@Vt) < 0:
        Vt = -Vt

    W = np.array([[0,-1,0],
                  [1,0,0],
                  [0,0,1]])
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt

    u3 = U[:,2].reshape(3,1)

    # print(u3.shape)
    P1 = np.hstack((R1,u3))
    P2 = np.hstack((R1,-u3))
    P3 = np.hstack((R2,u3))
    P4 = np.hstack((R2,-u3))

    P = np.array([P1,P2,P3,P4])

    return P

# Extract E basically
def decompose_E(E):
    P_all = extract_P_from_E(E)

    solutions = []
    for i in range(4):
        R = P_all[i, :, :3]
        t = P_all[i, :, 3]
        R = project_rotation(R)
        solutions.append((R, t))

    return solutions

def find_visible_camera(P1,P2_all, x1, x2):
    total_points = x1.shape[1]
    best_id = -1
    max_inline = -1

    all_results = []

    for i in range(4):
        P2_curr = P2_all[i]

        X_all = triangulate_3D_point_DLT(x1,x2,P1,P2_curr)

        _, front1 = depth_check(P1,X_all)
        _, front2 = depth_check(P2_curr, X_all)

        total_valid = np.sum(front1 & front2)

        all_results.append({
            'P2': P2_curr,
            'X': X_all,
            'total_valid': total_valid,
            'valid_masked': front1 & front2,
        })

        print(f"Solution {i+1}: {total_valid}/{total_points} points are in front")

        if total_valid > max_inline:
            max_inline = total_valid
            best_id = i

    return best_id, all_results

# find_visible_camera wrapper
def select_pose_cheirality(solutions, x1, x2):
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])

    # Convert (R,t) solutions to P2 array format
    P2_all = np.zeros((4, 3, 4))
    for i, (R, t) in enumerate(solutions):
        P2_all[i] = np.hstack([R, t.reshape(3, 1)])

    best_id, all_results = find_visible_camera(P1, P2_all, x1, x2)

    best_P2 = all_results[best_id]['P2']
    best_R = best_P2[:, :3]
    best_t = best_P2[:, 3]
    best_count = all_results[best_id]['total_valid']

    return best_R, best_t, best_count

def compute_epipolar_errors(F, x1s, x2s):
    '''
    x1s and x2s contain matching points
    x1s - 2D image points in the first image in homogenous coordinates (3xN)
    x2s - 2D image points in the second image in homogenous coordinates (3xN)
    F - Fundamental matrix (3x3)
    '''

    epipolar_lines = F @ x1s

    a = epipolar_lines[0]
    b = epipolar_lines[1]
    c = epipolar_lines[2]

    epipolar_errors = (np.abs(np.sum(x2s * epipolar_lines, axis=0)))/(np.sqrt(a**2 + b**2))

    return epipolar_errors

# Again compute_epipolar_errors wrapper (worse naming sorry :/)
def compute_epipolar_error(E, x1, x2):
    x1 = x1 / x1[2:3, :]
    x2 = x2 / x2[2:3, :]

    d1 = compute_epipolar_errors(E.T, x2, x1)
    d2 = compute_epipolar_errors(E, x1, x2)

    # basically mean-ish
    return 0.5 * (d1 + d2)

# Estimate E RANSAC
def estimate_E_ransac(x1, x2, threshold, max_iter=3000, seed=None):
    if seed is not None:
        np.random.seed(seed)

    x1 = x1 / x1[2:3, :]
    x2 = x2 / x2[2:3, :]

    N = x1.shape[1]

    # best placeholders
    best_E, best_R, best_t = None, None, None
    best_inliers = np.zeros(N, dtype=bool)
    best_num = 0

    for _ in range(max_iter):
        idx = np.random.choice(N, 8, replace=False)

        E = estimate_E_8point(x1[:, idx], x2[:, idx])
        errors = compute_epipolar_error(E, x1, x2)
        inliers = errors < threshold
        num = np.sum(inliers)

        if num > best_num:
            solutions = decompose_E(E)
            R, t, front_count = select_pose_cheirality(
                solutions, x1[:, inliers], x2[:, inliers]
            )
            # Play with percents (0.7 works okay now)
            if front_count > 0.7 * num:
                best_E, best_R, best_t = E, R, t
                best_inliers = inliers
                best_num = num

    if best_num >= 8:
        E = estimate_E_8point(x1[:, best_inliers], x2[:, best_inliers])
        solutions = decompose_E(E)
        R, t, _ = select_pose_cheirality(
            solutions, x1[:, best_inliers], x2[:, best_inliers]
        )
        best_E, best_R, best_t = E, R, t

    return best_E, best_R, best_t, best_inliers


# Homography estimation
def estimate_H_DLT(x1, x2):
    x1 = x1 / x1[2:3, :]
    x2 = x2 / x2[2:3, :]

    N = x1.shape[1]
    A = np.zeros((2 * N, 9))

    for i in range(N):
        u1, v1 = x1[0, i], x1[1, i]
        u2, v2 = x2[0, i], x2[1, i]
        A[2 * i] = [-u1, -v1, -1, 0, 0, 0, u2 * u1, u2 * v1, u2]
        A[2 * i + 1] = [0, 0, 0, -u1, -v1, -1, v2 * u1, v2 * v1, v2]

    _, _, Vt = np.linalg.svd(A)

    return Vt[-1].reshape(3, 3)

# Same thing with epipolar and E error calculation
def compute_H_error(H, x1, x2):
    x1 = x1 / x1[2:3, :]
    x2 = x2 / x2[2:3, :]

    x2p = H @ x1
    x2p = x2p / (x2p[2:3, :] + 1e-10)
    e_fwd = np.sqrt(np.sum((x2[:2] - x2p[:2]) ** 2, axis=0))

    Hi = np.linalg.inv(H)
    x1p = Hi @ x2
    x1p = x1p / (x1p[2:3, :] + 1e-10)
    e_bwd = np.sqrt(np.sum((x1[:2] - x1p[:2]) ** 2, axis=0))

    return 0.5 * (e_fwd + e_bwd)

# Estimate H RANSAC
def estimate_H_ransac(x1, x2, threshold, max_iter=3000, seed=None):
    if seed is not None:
        np.random.seed(seed)

    x1 = x1 / x1[2:3, :]
    x2 = x2 / x2[2:3, :]

    N = x1.shape[1]

    best_H = None
    best_inliers = np.zeros(N, dtype=bool)
    best_num = 0

    for _ in range(max_iter):
        idx = np.random.choice(N, 4, replace=False)

        H = estimate_H_DLT(x1[:, idx], x2[:, idx])
        errors = compute_H_error(H, x1, x2)
        inliers = errors < threshold
        num = np.sum(inliers)

        if num > best_num:
            best_H = H
            best_inliers = inliers
            best_num = num

    if best_num >= 4:
        best_H = estimate_H_DLT(x1[:, best_inliers], x2[:, best_inliers])

    return best_H, best_inliers


# Parallel RANSAC for E and H
# Wanted in report pdf so it might be wrong
# Need parameter adjustments here and there
def estimate_pose_ransac(x1_px, x2_px, K, pixel_thresh, max_iter=3000, seed=None):
    if seed is not None:
        np.random.seed(seed)

    K_inv = np.linalg.inv(K)
    x1_n = K_inv @ x1_px
    x2_n = K_inv @ x2_px

    focal = K[0, 0]
    thresh_E = pixel_thresh / focal
    thresh_H = 3 * pixel_thresh / focal

    E, R_E, t_E, inl_E = estimate_E_ransac(x1_n, x2_n, thresh_E, max_iter, seed)
    num_E = np.sum(inl_E) if inl_E is not None else 0

    H, inl_H = estimate_H_ransac(x1_n, x2_n, thresh_H, max_iter, seed)
    num_H = np.sum(inl_H) if inl_H is not None else 0

    R_H, t_H = None, None
    if H is not None and num_H > 0:
        H = correct_H_sign(H, x1_n, x2_n)
        RT_sols = homography_to_RT(H)

        P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
        best_cnt = 0

        for i in range(2):
            # Get the last element
            R = RT_sols[i, :, :3]
            t = RT_sols[i, :, 3]
            P2 = np.hstack([R, t.reshape(3, 1)])

            _, valid = triangulate_points(
                x1_n[:, inl_H], x2_n[:, inl_H], P1, P2, filter_behind=True
            )
            cnt = np.sum(valid)

            if cnt > best_cnt:
                best_cnt = cnt
                R_H, t_H = R, t

    if R_E is not None and (R_H is None or num_E >= 0.8 * num_H):
        return R_E, t_E, inl_E, 'E'
    elif R_H is not None:
        return R_H, t_H, inl_H, 'H'
    elif R_E is not None:
        return R_E, t_E, inl_E, 'E'
    else:
        return np.eye(3), np.array([0, 0, 1]), np.ones(x1_px.shape[1], dtype=bool), 'FAIL'


# Translation estimation with known R
def estimate_T_DLT(x_px, X_world, R, K):
    K_inv = np.linalg.inv(K)
    x_n = K_inv @ x_px
    x_n = x_n / x_n[2:3, :]

    X = X_world[:3, :] / X_world[3:4, :]

    N = x_n.shape[1]

    A = []
    b = []

    for i in range(N):
        xi = x_n[:, i]
        Xi = X[:, i]
        RXi = R @ Xi

        sk = skew(xi)
        A.append(sk[:2, :])
        b.append(-sk[:2, :] @ RXi)

    A = np.vstack(A)
    b = np.hstack(b)

    t, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return t


def estimate_T_ransac(x_px, X_world, R, K, threshold, max_iter=1000, seed=None):
    if seed is not None:
        np.random.seed(seed)

    N = x_px.shape[1]

    best_t = np.zeros(3)
    best_inl = np.zeros(N, dtype=bool)
    best_num = 0

    for _ in range(max_iter):
        idx = np.random.choice(N, 2, replace=False)

        t = estimate_T_DLT(x_px[:, idx], X_world[:, idx], R, K)

        P = np.hstack([R, t.reshape(3, 1)])
        x_proj = K @ P @ X_world

        z = x_proj[2:3, :]
        valid_z = np.abs(z) > 1e-10
        x_proj[:, valid_z.flatten()] = x_proj[:, valid_z.flatten()] / z[:, valid_z.flatten()]

        errors = np.full(N, np.inf)
        errors[valid_z.flatten()] = np.sqrt(np.sum(
            (x_px[:2, valid_z.flatten()] - x_proj[:2, valid_z.flatten()]) ** 2, axis=0
        ))

        inl = errors < threshold
        num = np.sum(inl)

        if num > best_num:
            best_t = t
            best_inl = inl
            best_num = num

    if best_num >= 2:
        best_t = estimate_T_DLT(x_px[:, best_inl], X_world[:, best_inl], R, K)

    return best_t, best_inl


# Feature extraction and matching
def extract_sift(images):
    detector = cv2.SIFT_create(
        nfeatures=0,
        nOctaveLayers=3,
        contrastThreshold=0.01,
        edgeThreshold=20,
        sigma=1.6
    )

    kps, descs = [], []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
        kp, des = detector.detectAndCompute(gray, None)
        kps.append(kp)
        descs.append(des)

    return kps, descs

# Ratio test for sift
def match_sift(des1, des2, ratio=0.8):
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return []

    if des1.dtype == np.uint8:
        norm = cv2.NORM_HAMMING
    else:
        norm = cv2.NORM_L2

    bf = cv2.BFMatcher(norm)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m in matches:
        if len(m) == 2:
            if m[0].distance < ratio * m[1].distance:
                good.append(m[0])

    return good


def get_matched_points(kp1, kp2, matches):
    if len(matches) == 0:
        return np.zeros((3, 0)), np.zeros((3, 0))

    p1 = np.array([kp1[m.queryIdx].pt for m in matches]).T
    p2 = np.array([kp2[m.trainIdx].pt for m in matches]).T

    return np.vstack([p1, np.ones(p1.shape[1])]), np.vstack([p2, np.ones(p2.shape[1])])


# Relative rotations to absolute rotation conversion
def chain_rotations(rel_R):
    abs_R = [np.eye(3)]

    # Seach between cam rotations
    for R in rel_R:
        R_new = R @ abs_R[-1]
        R_new = project_rotation(R_new)
        abs_R.append(R_new)
    return abs_R


# Filter outliers
def filter_points_distance(X, factor=5, quantile=0.9):
    # Check shape
    if X.shape[1] == 0:
        return X, np.array([], dtype=bool)

    Xc = X[:3, :] / X[3:4, :]
    cent = np.mean(Xc, axis=1, keepdims=True)
    dists = np.linalg.norm(Xc - cent, axis=0)
    # TODO: test other quantiles sometime (now 0.9)
    thresh = factor * np.quantile(dists, quantile)
    mask = dists <= thresh

    return X[:, mask], mask

# Front check
def filter_points_reprojection(X, cameras, K, threshold=10.0):
    if X.shape[1] == 0:
        return X, np.array([], dtype=bool)

    N = X.shape[1]
    valid = np.zeros(N, dtype=bool)

    for P in cameras:
        if P is None:
            continue

        x = K @ P @ X
        z = x[2, :]

        # Front check
        in_front = z > 0.1
        valid = valid | in_front

    return X[:, valid], valid


# Visualization starts here
# Instead of matplotlib open3d library used because of the fps problems and lag
def create_camera_frustum(R, t, K, scale=0.5, color=[1, 0, 0]):
    C = -R.T @ t

    w, h = 1.0, 0.75
    corners = np.array([
        [w, h, 1],
        [-w, h, 1],
        [-w, -h, 1],
        [w, -h, 1]
    ]) * scale

    corners_world = (R.T @ corners.T).T + C

    points = np.vstack((C, corners_world))
    lines = [[0, 1], [0, 2], [0, 3], [0, 4],[1, 2], [2, 3], [3, 4], [4, 1]]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])

    return line_set

# Instead of matplotlib open3d library used because of the fps problems and lag
def visualize(points_3D, cameras, colors=None, title="Reconstruction", K=None, point_size=2.0):
    vis_geometries = []

    points_euclidean = (points_3D[:3, :] / points_3D[3:4, :]).T

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_euclidean)

    if colors is not None and len(colors) == len(points_euclidean):
        c = colors / 255.0 if np.max(colors) > 1.0 else colors
        if c.ndim == 1:
            c = np.stack([c, c, c], axis=1)
        pcd.colors = o3d.utility.Vector3dVector(c)

    vis_geometries.append(pcd)
    vis_geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))

    if K is None:
        K = np.eye(3)

    cmap = matplotlib.cm.get_cmap('rainbow')

    for i, P in enumerate(cameras):
        if P is None:
            continue
        R = P[:, :3]
        t = P[:, 3]
        rgba = cmap(i / max(1, len(cameras) - 1))
        cam_color = rgba[:3]
        frustum = create_camera_frustum(R, t, K, scale=0.1, color=cam_color)
        vis_geometries.append(frustum)

    # Internal config (from web)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1280, height=720)

    for geom in vis_geometries:
        vis.add_geometry(geom)

    opt = vis.get_render_option()
    opt.point_size = point_size
    opt.background_color = np.asarray([0, 0, 0]) # Black to see it better

    vis.run()
    vis.destroy_window()


# SFM RUN PIPELINE STARTS HERE
def run_sfm(dataset_id=None, images=None, K=None, init_pair=None, pixel_threshold=10.0, seed=42):
    # Reset seed
    np.random.seed(seed)

    if dataset_id is not None:
        K, img_names, init_pair_def, thresh_def = get_dataset_info(dataset_id)

        if init_pair is None:
            init_pair = tuple(init_pair_def)
        if pixel_threshold == 1.0:
            pixel_threshold = thresh_def

        images = []
        for name in img_names:
            img = plt.imread(name)
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            images.append(img)

    n_img = len(images)
    K_inv = np.linalg.inv(K)

    # Extract SIFT features
    kps, descs = extract_sift(images)

    # Blanks to match consecutive pairs
    all_matches, all_x1, all_x2 = [], [], []

    for i in range(n_img - 1):
        m = match_sift(descs[i], descs[i + 1])
        x1, x2 = get_matched_points(kps[i], kps[i + 1], m)
        all_matches.append(m)
        all_x1.append(x1)
        all_x2.append(x2)

    # Blanks to estimate relative poses
    rel_R, rel_t, inl_masks = [], [], []

    for i in range(n_img - 1):
        R, t, inl, src = estimate_pose_ransac(
            all_x1[i], all_x2[i], K, pixel_threshold, seed=seed + i
        )

        rel_R.append(R)
        rel_t.append(t)
        inl_masks.append(inl)

    # Relative to absolute rotation
    abs_R = chain_rotations(rel_R)

    # Common 3d points
    i1, i2 = init_pair

    m_init = match_sift(descs[i1], descs[i2])
    x1_init, x2_init = get_matched_points(kps[i1], kps[i2], m_init)

    R_init, t_init, inl_init, src = estimate_pose_ransac(x1_init, x2_init, K, pixel_threshold, seed=seed)

    # Normalized points
    x1_n = K_inv @ x1_init[:, inl_init]
    x2_n = K_inv @ x2_init[:, inl_init]

    P1_loc = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2_loc = np.hstack([R_init, t_init.reshape(3, 1)])

    X_loc, valid_tri = triangulate_points(x1_n, x2_n, P1_loc, P2_loc, filter_behind=True)

    R_i1 = abs_R[i1]
    X_world = np.vstack([R_i1.T @ (X_loc[:3, :] / X_loc[3:4, :]),np.ones(X_loc.shape[1])])

    X_world, filt_mask = filter_points_distance(X_world)

    inl_idx = np.where(inl_init)[0]
    valid_idx = inl_idx[valid_tri]
    valid_idx = valid_idx[filt_mask]

    desc_3D = descs[i1][[m_init[j].queryIdx for j in valid_idx]]

    # Estimate camera t vector
    translations = [np.zeros(3)]
    cameras = [np.hstack([np.eye(3), np.zeros((3, 1))])]

    trans_thresh = 3 * pixel_threshold

    for i in range(1, n_img):
        R_i = abs_R[i]

        m3d = match_sift(descs[i], desc_3D, ratio=0.8)

        if len(m3d) < 6:
            t_rel = rel_t[i - 1] if i > 0 else np.zeros(3)
            t_i = translations[-1] + abs_R[i - 1].T @ t_rel
        else:
            x_2d = np.array([kps[i][m.queryIdx].pt for m in m3d]).T
            x_2d = np.vstack([x_2d, np.ones(x_2d.shape[1])])
            X_3d = X_world[:, [m.trainIdx for m in m3d]]

            t_i, inl_t = estimate_T_ransac(x_2d, X_3d, R_i, K, trans_thresh, seed=seed + i)

        translations.append(t_i)
        cameras.append(np.hstack([R_i, t_i.reshape(3, 1)]))

    # Triangulate and match colors
    all_pts = [X_world]
    all_colors = []

    for j in range(X_world.shape[1]):
        if j < len(valid_idx):
            pt = kps[i1][m_init[valid_idx[j]].queryIdx].pt
            px = int(np.clip(pt[0], 0, images[i1].shape[1] - 1))
            py = int(np.clip(pt[1], 0, images[i1].shape[0] - 1))
            # For colors
            all_colors.append(images[i1][py, px])

    for i in range(n_img - 1):
        if cameras[i] is None or cameras[i + 1] is None:
            continue

        mask = inl_masks[i]
        if np.sum(mask) == 0:
            continue

        x1_n = K_inv @ all_x1[i][:, mask]
        x2_n = K_inv @ all_x2[i][:, mask]

        X_pair, valid_pair = triangulate_points(
            x1_n, x2_n, cameras[i], cameras[i + 1], filter_behind=True
        )

        X_pair, dist_mask = filter_points_distance(X_pair)

        all_pts.append(X_pair)

        inl_idx_pair = np.where(mask)[0]
        valid_idx_pair = inl_idx_pair[valid_pair][dist_mask]

        for j in range(len(valid_idx_pair)):
            if valid_idx_pair[j] < len(all_matches[i]):
                pt = kps[i][all_matches[i][valid_idx_pair[j]].queryIdx].pt
                px = int(np.clip(pt[0], 0, images[i].shape[1] - 1))
                py = int(np.clip(pt[1], 0, images[i].shape[0] - 1))
                # Again for colors
                all_colors.append(images[i][py, px])

    # Match everything basically
    points_3D = np.hstack(all_pts)
    colors = np.array(all_colors[:points_3D.shape[1]])

    points_3D, valid_final = filter_points_reprojection(points_3D, cameras, K)
    colors = colors[valid_final[:len(colors)]] if len(colors) > 0 else colors

    # open3d vis
    visualize(
        points_3D,
        cameras,
        colors,
        # TODO: title not shown sometimes
        title=f"Dataset {dataset_id}: {points_3D.shape[1]} pts, {n_img} cams",
        K=K
    )

    return {
        'points_3D': points_3D,
        'cameras': cameras,
        'colors': colors,
        'rotations': abs_R,
        'translations': translations,
        'K': K
    }


if __name__ == "__main__":
    results = run_sfm(dataset_id=10, seed=42)