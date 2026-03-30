import json
import os

import cv2
import numpy as np

rgb_path = "output/debug_run/topdown_final.png"
lst_path = "output/debug_run/lst_full.png"
output_path = "output/debug_run/lst_gt.png"

GRID_SIZE = 13
DEFAULT_TRANSFORM_MODE = "affine"
DEFAULT_EDIT_MODE = "pairs"
GRID_CLAMP_MIN = -0.10
GRID_CLAMP_MAX = 1.10
FIT_EPS = 1e-6
POINT_DELETE_RADIUS = 14


class UltimateWarpTool:
    def __init__(self):
        if not os.path.exists(rgb_path):
            raise FileNotFoundError(f"Missing: {rgb_path}")
        if not os.path.exists(lst_path):
            raise FileNotFoundError(f"Missing: {lst_path}")

        print("Initializing Quick Align Tool (Pairs + Grid + Auto ROI)...")
        self.img_rgb = cv2.imread(rgb_path)
        self.img_lst_full = cv2.imread(lst_path)

        self.tgt_h, self.tgt_w = self.img_rgb.shape[:2]
        self.h_raw, self.w_raw = self.img_lst_full.shape[:2]

        gray = cv2.cvtColor(self.img_lst_full, cv2.COLOR_BGR2GRAY)
        coords = cv2.findNonZero(gray)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            pad = 20
            self.roi_x = max(0, x - pad)
            self.roi_y = max(0, y - pad)
            self.roi_w = min(self.w_raw - self.roi_x, w + 2 * pad)
            self.roi_h = min(self.h_raw - self.roi_y, h + 2 * pad)
        else:
            self.roi_x, self.roi_y, self.roi_w, self.roi_h = 0, 0, self.w_raw, self.h_raw

        self.screen_h = 800
        self.scale_disp = min(1.0, self.screen_h / self.h_raw)
        self.w_disp = int(self.w_raw * self.scale_disp)
        self.h_disp = int(self.h_raw * self.scale_disp)
        self.img_lst_disp_base = cv2.resize(self.img_lst_full, (self.w_disp, self.h_disp))

        self.scale_tgt_disp = min(1.0, self.screen_h / self.tgt_h)
        self.w_tgt_disp = int(self.tgt_w * self.scale_tgt_disp)
        self.h_tgt_disp = int(self.tgt_h * self.scale_tgt_disp)
        self.img_rgb_disp_base = cv2.resize(self.img_rgb, (self.w_tgt_disp, self.h_tgt_disp))

        self.prev_scale = 0.5
        self.w_prev = int(self.tgt_w * self.prev_scale)
        self.h_prev = int(self.tgt_h * self.prev_scale)
        self.img_rgb_prev = cv2.resize(self.img_rgb, (self.w_prev, self.h_prev))

        self.grid = self._make_default_grid()
        self.grid0 = self.grid.copy()

        self.alpha = 0.6
        self.edit_mode = DEFAULT_EDIT_MODE
        self.transform_mode = DEFAULT_TRANSFORM_MODE

        self.last_fit_rmse = None
        self.last_transform = None
        self.last_transform_kind = None
        self.last_warp_source = None

        self.selected_pts = set()
        self.is_dragging = False
        self.last_mouse_pos = (0, 0)
        self.drag_start_pos = None
        self.is_box_selecting = False

        self.pair_src_pts = []
        self.pair_dst_pts = []
        self.pending_pair_point = None
        self.pending_pair_side = None

        self.win_src = "1. LST Source"
        self.win_tgt = "2. RGB Reference"
        self.win_res = "3. Realtime Preview"

    def _make_default_grid(self):
        grid = np.zeros((GRID_SIZE, GRID_SIZE, 2), dtype=np.float32)
        ys = np.linspace(0, 1, GRID_SIZE, dtype=np.float32)
        xs = np.linspace(0, 1, GRID_SIZE, dtype=np.float32)
        for iy in range(GRID_SIZE):
            for ix in range(GRID_SIZE):
                grid[iy, ix] = [xs[ix], ys[iy]]
        return grid

    def reset_grid(self):
        self.grid = self.grid0.copy()
        self.selected_pts.clear()
        self.is_dragging = False
        self.drag_start_pos = None
        self.is_box_selecting = False

    def clear_pairs(self):
        self.pair_src_pts.clear()
        self.pair_dst_pts.clear()
        self.pending_pair_point = None
        self.pending_pair_side = None

    def _grid_src_points(self):
        pts = self.grid.copy()
        pts[..., 0] = self.roi_x + pts[..., 0] * self.roi_w
        pts[..., 1] = self.roi_y + pts[..., 1] * self.roi_h
        return pts

    def _grid_dst_points(self, target_w, target_h):
        xs = np.linspace(0, target_w - 1, GRID_SIZE, dtype=np.float32)
        ys = np.linspace(0, target_h - 1, GRID_SIZE, dtype=np.float32)
        dst = np.zeros((GRID_SIZE, GRID_SIZE, 2), dtype=np.float32)
        for iy in range(GRID_SIZE):
            for ix in range(GRID_SIZE):
                dst[iy, ix] = [xs[ix], ys[iy]]
        return dst

    def _src_full_to_screen(self, pt):
        return int(round(pt[0] * self.scale_disp)), int(round(pt[1] * self.scale_disp))

    def _src_screen_to_full(self, x, y):
        px = np.clip(x / self.scale_disp, 0, self.w_raw - 1)
        py = np.clip(y / self.scale_disp, 0, self.h_raw - 1)
        return float(px), float(py)

    def _tgt_full_to_screen(self, pt):
        return int(round(pt[0] * self.scale_tgt_disp)), int(round(pt[1] * self.scale_tgt_disp))

    def _tgt_screen_to_full(self, x, y):
        px = np.clip(x / self.scale_tgt_disp, 0, self.tgt_w - 1)
        py = np.clip(y / self.scale_tgt_disp, 0, self.tgt_h - 1)
        return float(px), float(py)

    def _get_corner_mask(self):
        mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        mask[0, 0] = True
        mask[0, -1] = True
        mask[-1, 0] = True
        mask[-1, -1] = True
        return mask

    def _get_fit_mask(self):
        moved = np.linalg.norm(self.grid - self.grid0, axis=2) > FIT_EPS
        return moved | self._get_corner_mask()

    def _project_points(self, src_pts, transform, kind):
        if kind == "perspective":
            src_h = np.concatenate([src_pts, np.ones((src_pts.shape[0], 1), dtype=np.float32)], axis=1)
            proj = src_h @ transform.T
            return proj[:, :2] / np.maximum(proj[:, 2:3], 1e-6)
        return (src_pts @ transform[:, :2].T) + transform[:, 2]

    def _fit_error(self, src_pts, dst_pts, transform, kind):
        pred = self._project_points(src_pts, transform, kind)
        return float(np.sqrt(np.mean(np.sum((pred - dst_pts) ** 2, axis=1))))

    def _fallback_transform(self, target_w, target_h, mode):
        src = self._grid_src_points()
        dst = self._grid_dst_points(target_w, target_h)
        mask = self._get_corner_mask()
        src_pts = src[mask].reshape(-1, 2).astype(np.float32)
        dst_pts = dst[mask].reshape(-1, 2).astype(np.float32)

        if mode == "similarity":
            transform, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
            kind = "affine"
        elif mode == "affine":
            transform, _ = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.LMEDS)
            kind = "affine"
        else:
            transform = cv2.getPerspectiveTransform(src_pts, dst_pts)
            kind = "perspective"

        if transform is None:
            transform = cv2.getPerspectiveTransform(src_pts, dst_pts)
            kind = "perspective"

        rmse = self._fit_error(src_pts, dst_pts, transform, kind)
        return transform, kind, rmse

    def estimate_global_transform(self, target_w, target_h, mode=None):
        mode = mode or self.transform_mode
        src = self._grid_src_points()
        dst = self._grid_dst_points(target_w, target_h)
        mask = self._get_fit_mask()
        src_pts = src[mask].reshape(-1, 2).astype(np.float32)
        dst_pts = dst[mask].reshape(-1, 2).astype(np.float32)

        transform = None
        kind = "affine"

        if mode == "similarity":
            transform, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
        elif mode == "affine":
            transform, _ = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.LMEDS)
        elif mode == "perspective":
            transform, _ = cv2.findHomography(src_pts, dst_pts, method=0)
            kind = "perspective"
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        if transform is None:
            return self._fallback_transform(target_w, target_h, mode)

        rmse = self._fit_error(src_pts, dst_pts, transform, kind)
        return transform, kind, rmse

    def pair_min_points(self, mode=None):
        mode = mode or self.transform_mode
        if mode == "similarity":
            return 2
        if mode == "affine":
            return 3
        if mode == "perspective":
            return 4
        return 0

    def _pair_mode_transform(self):
        if self.transform_mode == "elastic":
            return "affine"
        return self.transform_mode

    def can_estimate_pairs(self, mode=None):
        mode = mode or self._pair_mode_transform()
        return len(self.pair_src_pts) >= self.pair_min_points(mode)

    def estimate_pair_transform(self, target_w, target_h, mode=None):
        mode = mode or self._pair_mode_transform()
        src_pts = np.asarray(self.pair_src_pts, dtype=np.float32)
        dst_pts = np.asarray(self.pair_dst_pts, dtype=np.float32)
        if src_pts.shape[0] < self.pair_min_points(mode):
            return None, None, None

        dst_scaled = dst_pts.copy()
        dst_scaled[:, 0] *= target_w / float(self.tgt_w)
        dst_scaled[:, 1] *= target_h / float(self.tgt_h)

        transform = None
        kind = "affine"
        if mode == "similarity":
            transform, _ = cv2.estimateAffinePartial2D(src_pts, dst_scaled, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        elif mode == "affine":
            transform, _ = cv2.estimateAffine2D(src_pts, dst_scaled, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        elif mode == "perspective":
            method = cv2.RANSAC if src_pts.shape[0] >= 5 else 0
            transform, _ = cv2.findHomography(src_pts, dst_scaled, method=method, ransacReprojThreshold=5.0)
            kind = "perspective"

        if transform is None:
            return None, None, None

        rmse = self._fit_error(src_pts, dst_scaled, transform, kind)
        return transform, kind, rmse

    def get_real_coords_map(self, target_w, target_h):
        map_sparse = self.grid.copy()
        map_sparse[..., 0] = self.roi_x + map_sparse[..., 0] * self.roi_w
        map_sparse[..., 1] = self.roi_y + map_sparse[..., 1] * self.roi_h
        map_dense = cv2.resize(map_sparse, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        return map_dense[..., 0].astype(np.float32), map_dense[..., 1].astype(np.float32)

    def warp_current(self, target_w, target_h):
        if self.edit_mode == "pairs":
            transform, kind, rmse = self.estimate_pair_transform(target_w, target_h)
            if transform is not None:
                if kind == "perspective":
                    warped = cv2.warpPerspective(self.img_lst_full, transform, (target_w, target_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                else:
                    warped = cv2.warpAffine(self.img_lst_full, transform, (target_w, target_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                self.last_transform = transform
                self.last_transform_kind = kind
                self.last_fit_rmse = rmse
                self.last_warp_source = "pairs"
                return warped

            fallback_mode = "affine" if self.transform_mode == "elastic" else self.transform_mode
            transform, kind, rmse = self.estimate_global_transform(target_w, target_h, mode=fallback_mode)
            if kind == "perspective":
                warped = cv2.warpPerspective(self.img_lst_full, transform, (target_w, target_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            else:
                warped = cv2.warpAffine(self.img_lst_full, transform, (target_w, target_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            self.last_transform = transform
            self.last_transform_kind = kind
            self.last_fit_rmse = rmse
            self.last_warp_source = "grid-fallback"
            return warped

        if self.transform_mode == "elastic":
            map_x, map_y = self.get_real_coords_map(target_w, target_h)
            warped = cv2.remap(self.img_lst_full, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            self.last_transform = None
            self.last_transform_kind = "elastic"
            self.last_fit_rmse = None
            self.last_warp_source = "grid-elastic"
            return warped

        transform, kind, rmse = self.estimate_global_transform(target_w, target_h)
        if kind == "perspective":
            warped = cv2.warpPerspective(self.img_lst_full, transform, (target_w, target_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        else:
            warped = cv2.warpAffine(self.img_lst_full, transform, (target_w, target_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        self.last_transform = transform
        self.last_transform_kind = kind
        self.last_fit_rmse = rmse
        self.last_warp_source = "grid"
        return warped

    def _draw_grid_overlay(self, disp):
        grid_screen_x = (self.roi_x + self.grid[..., 0] * self.roi_w) * self.scale_disp
        grid_screen_y = (self.roi_y + self.grid[..., 1] * self.roi_h) * self.scale_disp
        disp_pts = np.stack([grid_screen_x, grid_screen_y], axis=2).astype(np.int32)

        for i in range(GRID_SIZE):
            pts = np.ascontiguousarray(disp_pts[i], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(disp, [pts], False, (100, 100, 100), 1, cv2.LINE_AA)
        for i in range(GRID_SIZE):
            pts = np.ascontiguousarray(disp_pts[:, i], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(disp, [pts], False, (100, 100, 100), 1, cv2.LINE_AA)

        moved_mask = np.linalg.norm(self.grid - self.grid0, axis=2) > FIT_EPS
        for iy in range(GRID_SIZE):
            for ix in range(GRID_SIZE):
                pt = tuple(disp_pts[iy, ix])
                if (iy, ix) in self.selected_pts:
                    cv2.circle(disp, pt, 5, (0, 255, 0), -1)
                    cv2.circle(disp, pt, 7, (255, 255, 255), 1)
                else:
                    is_corner = (ix in [0, GRID_SIZE - 1] and iy in [0, GRID_SIZE - 1])
                    if is_corner:
                        color = (0, 0, 255)
                    elif moved_mask[iy, ix]:
                        color = (0, 200, 255)
                    else:
                        color = (255, 100, 100)
                    cv2.circle(disp, pt, 3, color, -1)

        if self.is_box_selecting and self.drag_start_pos:
            cv2.rectangle(disp, self.drag_start_pos, self.last_mouse_pos, (0, 255, 255), 1)

    def _draw_pairs_on_image(self, img, points, to_screen_fn, side):
        for idx, pt in enumerate(points, start=1):
            px, py = to_screen_fn(pt)
            cv2.circle(img, (px, py), 5, (0, 255, 255), -1)
            cv2.circle(img, (px, py), 7, (0, 0, 0), 1)
            cv2.putText(img, str(idx), (px + 6, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        if self.pending_pair_side == side and self.pending_pair_point is not None:
            px, py = to_screen_fn(self.pending_pair_point)
            cv2.circle(img, (px, py), 6, (255, 255, 0), 1)
            cv2.drawMarker(img, (px, py), (255, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)
            cv2.putText(img, "pending", (px + 6, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    def update_display(self):
        src_disp = self.img_lst_disp_base.copy()
        tgt_disp = self.img_rgb_disp_base.copy()

        if self.edit_mode == "grid":
            self._draw_grid_overlay(src_disp)
        else:
            self._draw_pairs_on_image(src_disp, self.pair_src_pts, self._src_full_to_screen, "src")

        self._draw_pairs_on_image(tgt_disp, self.pair_dst_pts, self._tgt_full_to_screen, "dst")

        mode_text = f"Edit: {self.edit_mode.upper()} | Transform: {self.transform_mode.upper()}"
        cv2.rectangle(src_disp, (0, 0), (src_disp.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(src_disp, mode_text, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 1)

        pair_need = self.pair_min_points(self._pair_mode_transform())
        pair_info = f"Pairs: {len(self.pair_src_pts)}/{pair_need}"
        if self.edit_mode == "pairs" and self.pending_pair_side is not None:
            pair_info += f" | Waiting: {self.pending_pair_side.upper()}"
        cv2.rectangle(tgt_disp, (0, 0), (tgt_disp.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(tgt_disp, pair_info, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 1)

        cv2.imshow(self.win_src, src_disp)
        cv2.imshow(self.win_tgt, tgt_disp)

        warped_prev = self.warp_current(self.w_prev, self.h_prev)
        blended = cv2.addWeighted(self.img_rgb_prev, 1.0 - self.alpha, warped_prev, self.alpha, 0)

        fit_text = "Fit RMSE: N/A" if self.last_fit_rmse is None else f"Fit RMSE: {self.last_fit_rmse:.2f}px"
        source_text = f"Warp: {self.last_warp_source or 'n/a'}"
        info = f"{mode_text} | {pair_info} | Alpha: {self.alpha:.2f} | {fit_text} | {source_text}"
        cv2.rectangle(blended, (0, 0), (self.w_prev, 34), (0, 0, 0), -1)
        cv2.putText(blended, info, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1)
        cv2.imshow(self.win_res, blended)

    def _clamp_grid(self):
        self.grid[..., 0] = np.clip(self.grid[..., 0], GRID_CLAMP_MIN, GRID_CLAMP_MAX)
        self.grid[..., 1] = np.clip(self.grid[..., 1], GRID_CLAMP_MIN, GRID_CLAMP_MAX)

    def apply_move(self, dx, dy):
        if not self.selected_pts:
            return

        if self.transform_mode == "elastic":
            is_elastic = False
            if len(self.selected_pts) == 1:
                sy, sx = list(self.selected_pts)[0]
                is_corner = (sx in [0, GRID_SIZE - 1] and sy in [0, GRID_SIZE - 1])
                if is_corner:
                    is_elastic = True

            if is_elastic:
                sy, sx = list(self.selected_pts)[0]
                for y in range(GRID_SIZE):
                    for x in range(GRID_SIZE):
                        wx = 1.0 - (x / (GRID_SIZE - 1)) if sx == 0 else x / (GRID_SIZE - 1)
                        wy = 1.0 - (y / (GRID_SIZE - 1)) if sy == 0 else y / (GRID_SIZE - 1)
                        w = wx * wy
                        self.grid[y, x][0] += dx * w
                        self.grid[y, x][1] += dy * w
            else:
                for iy, ix in self.selected_pts:
                    self.grid[iy, ix][0] += dx
                    self.grid[iy, ix][1] += dy
        else:
            for iy, ix in self.selected_pts:
                self.grid[iy, ix][0] += dx
                self.grid[iy, ix][1] += dy

        self._clamp_grid()

    def select_points_in_box(self, pt1, pt2, add_to_selection=False):
        x1, x2 = min(pt1[0], pt2[0]), max(pt1[0], pt2[0])
        y1, y2 = min(pt1[1], pt2[1]), max(pt1[1], pt2[1])

        grid_screen_x = (self.roi_x + self.grid[..., 0] * self.roi_w) * self.scale_disp
        grid_screen_y = (self.roi_y + self.grid[..., 1] * self.roi_h) * self.scale_disp

        new_selection = set()
        for iy in range(GRID_SIZE):
            for ix in range(GRID_SIZE):
                px, py = grid_screen_x[iy, ix], grid_screen_y[iy, ix]
                if x1 <= px <= x2 and y1 <= py <= y2:
                    new_selection.add((iy, ix))

        if add_to_selection:
            self.selected_pts.update(new_selection)
        else:
            self.selected_pts = new_selection

    def add_pair_click(self, side, point_full):
        if self.pending_pair_side is None:
            self.pending_pair_side = side
            self.pending_pair_point = point_full
            return

        if self.pending_pair_side == side:
            self.pending_pair_point = point_full
            return

        if self.pending_pair_side == "src":
            self.pair_src_pts.append(self.pending_pair_point)
            self.pair_dst_pts.append(point_full)
        else:
            self.pair_src_pts.append(point_full)
            self.pair_dst_pts.append(self.pending_pair_point)

        self.pending_pair_side = None
        self.pending_pair_point = None

    def undo_last_pair(self):
        if self.pending_pair_side is not None:
            self.pending_pair_side = None
            self.pending_pair_point = None
            return True
        if self.pair_src_pts:
            self.pair_src_pts.pop()
            self.pair_dst_pts.pop()
            return True
        return False

    def _remove_nearest_pair(self, side, x, y):
        points = self.pair_src_pts if side == "src" else self.pair_dst_pts
        if not points:
            return False

        if side == "src":
            screen_pts = [self._src_full_to_screen(pt) for pt in points]
        else:
            screen_pts = [self._tgt_full_to_screen(pt) for pt in points]

        best_idx = None
        best_dist = POINT_DELETE_RADIUS ** 2
        for idx, (px, py) in enumerate(screen_pts):
            dist = (x - px) ** 2 + (y - py) ** 2
            if dist <= best_dist:
                best_dist = dist
                best_idx = idx

        if best_idx is None:
            return False

        self.pair_src_pts.pop(best_idx)
        self.pair_dst_pts.pop(best_idx)
        return True

    def handle_pair_mouse(self, side, event, x, y, flags):
        if event == cv2.EVENT_MOUSEWHEEL:
            delta = 0.05 if flags > 0 else -0.05
            self.alpha = max(0.0, min(1.0, self.alpha + delta))
            self.update_display()
            return

        if event == cv2.EVENT_RBUTTONDOWN:
            removed = self._remove_nearest_pair(side, x, y)
            if not removed and self.pending_pair_side == side:
                self.pending_pair_side = None
                self.pending_pair_point = None
            self.update_display()
            return

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        point_full = self._src_screen_to_full(x, y) if side == "src" else self._tgt_screen_to_full(x, y)
        self.add_pair_click(side, point_full)
        self.update_display()

    def source_mouse_callback(self, event, x, y, flags, param):
        if self.edit_mode == "pairs":
            self.handle_pair_mouse("src", event, x, y, flags)
            return

        ctrl_pressed = (flags & cv2.EVENT_FLAG_CTRLKEY)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.last_mouse_pos = (x, y)
            self.drag_start_pos = (x, y)

            clicked_idx = None
            min_dist = 10000
            grid_screen_x = (self.roi_x + self.grid[..., 0] * self.roi_w) * self.scale_disp
            grid_screen_y = (self.roi_y + self.grid[..., 1] * self.roi_h) * self.scale_disp
            for iy in range(GRID_SIZE):
                for ix in range(GRID_SIZE):
                    px, py = grid_screen_x[iy, ix], grid_screen_y[iy, ix]
                    dist = (x - px) ** 2 + (y - py) ** 2
                    if dist < 100 and dist < min_dist:
                        min_dist = dist
                        clicked_idx = (iy, ix)

            if clicked_idx:
                self.is_dragging = True
                self.is_box_selecting = False
                if ctrl_pressed:
                    if clicked_idx in self.selected_pts:
                        self.selected_pts.remove(clicked_idx)
                    else:
                        self.selected_pts.add(clicked_idx)
                elif clicked_idx not in self.selected_pts:
                    self.selected_pts = {clicked_idx}
            else:
                self.is_dragging = False
                self.is_box_selecting = True
                if not ctrl_pressed:
                    self.selected_pts.clear()

            self.update_display()

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_dragging:
                dx_screen = x - self.last_mouse_pos[0]
                dy_screen = y - self.last_mouse_pos[1]
                dx_norm = (dx_screen / self.scale_disp) / self.roi_w
                dy_norm = (dy_screen / self.scale_disp) / self.roi_h
                self.apply_move(dx_norm, dy_norm)
                self.last_mouse_pos = (x, y)
                self.update_display()
            elif self.is_box_selecting:
                self.last_mouse_pos = (x, y)
                self.update_display()

        elif event == cv2.EVENT_LBUTTONUP:
            if self.is_box_selecting:
                self.select_points_in_box(self.drag_start_pos, (x, y), add_to_selection=ctrl_pressed)
                self.is_box_selecting = False
                self.update_display()
            self.is_dragging = False

        elif event == cv2.EVENT_MOUSEWHEEL:
            delta = 0.05 if flags > 0 else -0.05
            self.alpha = max(0.0, min(1.0, self.alpha + delta))
            self.update_display()

    def target_mouse_callback(self, event, x, y, flags, param):
        if self.edit_mode != "pairs":
            if event == cv2.EVENT_MOUSEWHEEL:
                delta = 0.05 if flags > 0 else -0.05
                self.alpha = max(0.0, min(1.0, self.alpha + delta))
                self.update_display()
            return
        self.handle_pair_mouse("dst", event, x, y, flags)

    def handle_keyboard(self, key_char, key_code):
        if key_char == ord("p"):
            self.edit_mode = "pairs"
            if self.transform_mode == "elastic":
                self.transform_mode = "affine"
            return True
        if key_char == ord("g"):
            self.edit_mode = "grid"
            return True
        if key_char == ord("1"):
            self.transform_mode = "similarity"
            return True
        if key_char == ord("2"):
            self.transform_mode = "affine"
            return True
        if key_char == ord("3"):
            self.transform_mode = "perspective"
            return True
        if key_char == ord("4") and self.edit_mode == "grid":
            self.transform_mode = "elastic"
            return True
        if key_char == ord("u"):
            return self.undo_last_pair()
        if key_char == ord("x"):
            if self.edit_mode == "pairs":
                self.clear_pairs()
            else:
                self.reset_grid()
            return True
        if key_char == ord("r"):
            self.reset_grid()
            self.clear_pairs()
            return True

        if self.edit_mode != "grid" or not self.selected_pts:
            return False

        step_x = 0.5 / self.roi_w
        step_y = 0.5 / self.roi_h
        dx, dy = 0, 0
        if key_char == ord("w") or key_code == 2490368:
            dy = -step_y
        elif key_char == ord("s") or key_code == 2621440:
            dy = step_y
        elif key_char == ord("a") or key_code == 2424832:
            dx = -step_x
        elif key_char == ord("d") or key_code == 2555904:
            dx = step_x

        if dx != 0 or dy != 0:
            self.apply_move(dx, dy)
            return True
        return False

    def _save_metadata(self):
        meta_path = os.path.splitext(output_path)[0] + "_transform.json"
        moved_mask = np.linalg.norm(self.grid - self.grid0, axis=2) > FIT_EPS
        moved_points = []
        for iy, ix in np.argwhere(moved_mask):
            moved_points.append({"grid_index": [int(iy), int(ix)], "grid_uv": [float(self.grid[iy, ix, 0]), float(self.grid[iy, ix, 1])]})

        payload = {
            "edit_mode": self.edit_mode,
            "transform_mode": self.transform_mode,
            "warp_source": self.last_warp_source,
            "fit_rmse_px": self.last_fit_rmse,
            "roi": {"x": int(self.roi_x), "y": int(self.roi_y), "w": int(self.roi_w), "h": int(self.roi_h)},
            "moved_grid_points": moved_points,
            "pair_src_points": [[float(x), float(y)] for x, y in self.pair_src_pts],
            "pair_dst_points": [[float(x), float(y)] for x, y in self.pair_dst_pts],
        }
        if self.last_transform is not None:
            payload["transform_kind"] = self.last_transform_kind
            payload["transform_matrix"] = self.last_transform.tolist()

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Saved transform metadata: {meta_path}")

    def save_result(self):
        print("Saving final aligned GT...")
        final_img = self.warp_current(self.tgt_w, self.tgt_h)
        final_gray = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, final_gray)
        self._save_metadata()
        print(f"[OK] GT saved: {output_path}")

    def run(self):
        cv2.namedWindow(self.win_src)
        cv2.namedWindow(self.win_tgt)
        cv2.namedWindow(self.win_res)
        cv2.setMouseCallback(self.win_src, self.source_mouse_callback)
        cv2.setMouseCallback(self.win_tgt, self.target_mouse_callback)
        self.update_display()

        print("==================================================")
        print("  Quick Align Tool")
        print("==================================================")
        print("  Recommended workflow:")
        print("    1) Start in pair mode [P].")
        print("    2) Click source point on LST, then matching point on RGB.")
        print("    3) Use 4-6 pairs for perspective, 6-10 pairs if you want more stability.")
        print("    4) If tiny local residual remains, switch to grid mode [G] for micro-fix.")
        print("  Controls:")
        print("    [P] pair mode  [G] grid mode")
        print("    [1] similarity  [2] affine  [3] perspective  [4] elastic(grid only)")
        print("    [Left Click] add pair or select grid point")
        print("    [Right Click] delete nearest pair")
        print("    [Ctrl+Click] multi-select grid points")
        print("    [Drag]/[WASD] move selected grid points")
        print("    [U] undo last pair or pending point")
        print("    [X] clear current mode points  [R] full reset")
        print("    [Mouse Wheel] overlay alpha  [Space] save  [Q] quit")
        print("==================================================")

        while True:
            key = cv2.waitKeyEx(30)
            if key == -1:
                continue
            key_char = key & 0xFF

            if key_char == ord("q"):
                break
            if key_char == ord(" "):
                self.save_result()
                break
            if key_char == ord("z"):
                self.alpha = max(0.0, self.alpha - 0.05)
                self.update_display()
                continue
            if key_char == ord("c"):
                self.alpha = min(1.0, self.alpha + 0.05)
                self.update_display()
                continue

            if self.handle_keyboard(key_char, key):
                self.update_display()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = UltimateWarpTool()
    app.run()
