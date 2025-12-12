import cv2
import numpy as np
import os

# ================= 配置区域 =================
# 1. RGB 基准图
rgb_path = "output/debug_run/topdown_final.png"
# 2. LST 全尺寸图
lst_path = "output/debug_run/lst_full.png"
# 3. 输出路径
output_path = "output/debug_run/lst_gt.png"

# 网格密度 (13x13 = 169个点)
GRID_SIZE = 13


# ===========================================

class UltimateWarpTool:
    def __init__(self):
        if not os.path.exists(rgb_path): raise FileNotFoundError(f"Missing: {rgb_path}")
        if not os.path.exists(lst_path): raise FileNotFoundError(f"Missing: {lst_path}")

        print("初始化 Ultimate Tool (Multi-select + Elastic + Auto ROI)...")
        self.img_rgb = cv2.imread(rgb_path)
        self.img_lst_full = cv2.imread(lst_path)

        self.tgt_h, self.tgt_w = self.img_rgb.shape[:2]
        self.h_raw, self.w_raw = self.img_lst_full.shape[:2]

        # --- 1. 自动计算有效区域 (ROI) ---
        gray = cv2.cvtColor(self.img_lst_full, cv2.COLOR_BGR2GRAY) if len(
            self.img_lst_full.shape) == 3 else self.img_lst_full
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

        # --- 2. 显示适配 ---
        self.screen_h = 800
        self.scale_disp = min(1.0, self.screen_h / self.h_raw)
        self.w_disp, self.h_disp = int(self.w_raw * self.scale_disp), int(self.h_raw * self.scale_disp)

        self.img_lst_disp_base = cv2.resize(self.img_lst_full, (self.w_disp, self.h_disp))

        # --- 3. 预览适配 ---
        self.prev_scale = 0.5
        self.w_prev, self.h_prev = int(self.tgt_w * self.prev_scale), int(self.tgt_h * self.prev_scale)
        self.img_rgb_prev = cv2.resize(self.img_rgb, (self.w_prev, self.h_prev))

        # --- 4. 网格初始化 ---
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE, 2), dtype=np.float32)
        ys = np.linspace(0, 1, GRID_SIZE)
        xs = np.linspace(0, 1, GRID_SIZE)
        for iy in range(GRID_SIZE):
            for ix in range(GRID_SIZE):
                self.grid[iy, ix] = [xs[ix], ys[iy]]

        # --- 5. 交互状态 ---
        self.alpha = 0.6
        self.selected_pts = set()  # 存储 (iy, ix) 集合
        self.is_dragging = False
        self.last_mouse_pos = (0, 0)
        self.drag_start_pos = None  # 用于框选
        self.is_box_selecting = False

        self.win_src = "1. Ultimate Control (LST)"
        self.win_res = "2. Realtime Preview"

    def get_real_coords_map(self, target_w, target_h):
        map_sparse = self.grid.copy()
        # 映射回全图坐标
        map_sparse[..., 0] = self.roi_x + map_sparse[..., 0] * self.roi_w
        map_sparse[..., 1] = self.roi_y + map_sparse[..., 1] * self.roi_h
        # 双三次插值生成 Dense Map (平滑保证)
        map_dense = cv2.resize(map_sparse, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        return map_dense[..., 0].astype(np.float32), map_dense[..., 1].astype(np.float32)

    def update_display(self):
        # --- 左侧窗口 ---
        disp = self.img_lst_disp_base.copy()

        grid_screen_x = (self.roi_x + self.grid[..., 0] * self.roi_w) * self.scale_disp
        grid_screen_y = (self.roi_y + self.grid[..., 1] * self.roi_h) * self.scale_disp
        disp_pts = np.stack([grid_screen_x, grid_screen_y], axis=2).astype(np.int32)

        # 1. 绘制网格线
        for i in range(GRID_SIZE):
            pts = np.ascontiguousarray(disp_pts[i], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(disp, [pts], False, (100, 100, 100), 1, cv2.LINE_AA)
        for i in range(GRID_SIZE):
            pts = np.ascontiguousarray(disp_pts[:, i], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(disp, [pts], False, (100, 100, 100), 1, cv2.LINE_AA)

        # 2. 绘制点
        for iy in range(GRID_SIZE):
            for ix in range(GRID_SIZE):
                pt = tuple(disp_pts[iy, ix])
                if (iy, ix) in self.selected_pts:
                    # 选中：高亮绿圈
                    cv2.circle(disp, pt, 5, (0, 255, 0), -1)
                    cv2.circle(disp, pt, 7, (255, 255, 255), 1)
                else:
                    # 未选中：根据位置给颜色
                    is_corner = (ix in [0, GRID_SIZE - 1] and iy in [0, GRID_SIZE - 1])
                    if is_corner:
                        color = (0, 0, 255)  # 角点红
                    else:
                        color = (255, 100, 100)  # 普通点浅蓝
                    cv2.circle(disp, pt, 3, color, -1)

        # 3. 绘制框选矩形
        if self.is_box_selecting and self.drag_start_pos:
            cv2.rectangle(disp, self.drag_start_pos, self.last_mouse_pos, (0, 255, 255), 1)

        cv2.imshow(self.win_src, disp)

        # --- 右侧预览 ---
        map_x, map_y = self.get_real_coords_map(self.w_prev, self.h_prev)
        warped_prev = cv2.remap(self.img_lst_full, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)
        blended = cv2.addWeighted(self.img_rgb_prev, 1.0 - self.alpha, warped_prev, self.alpha, 0)

        # UI Info
        info = f"Selected: {len(self.selected_pts)} | Alpha: {self.alpha:.2f}"
        cv2.rectangle(blended, (0, 0), (self.w_prev, 30), (0, 0, 0), -1)
        cv2.putText(blended, info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        cv2.imshow(self.win_res, blended)

    def apply_move(self, dx, dy):
        """
        核心移动逻辑：
        1. 如果只选中了1个点，且是角点 -> 触发弹性联动 (Elastic)。
        2. 如果选中多个点，或内部点 -> 刚性平移 (Rigid Move)，不影响未选中的点。
        """
        # 判断是否触发弹性模式
        is_elastic = False
        if len(self.selected_pts) == 1:
            (sy, sx) = list(self.selected_pts)[0]
            is_corner = (sx in [0, GRID_SIZE - 1] and sy in [0, GRID_SIZE - 1])
            if is_corner: is_elastic = True

        if is_elastic:
            # === 弹性模式 (Rubber Sheet) ===
            (sy, sx) = list(self.selected_pts)[0]
            for y in range(GRID_SIZE):
                for x in range(GRID_SIZE):
                    # 计算权重
                    if sx == 0:
                        wx = 1.0 - (x / (GRID_SIZE - 1))
                    else:
                        wx = x / (GRID_SIZE - 1)

                    if sy == 0:
                        wy = 1.0 - (y / (GRID_SIZE - 1))
                    else:
                        wy = y / (GRID_SIZE - 1)

                    # 混合权重
                    w = wx * wy
                    self.grid[y, x][0] += dx * w
                    self.grid[y, x][1] += dy * w
        else:
            # === 刚性模式 (Local Rigid) ===
            # 只移动选中的点，其他点(包括角点)纹丝不动
            for (iy, ix) in self.selected_pts:
                self.grid[iy, ix][0] += dx
                self.grid[iy, ix][1] += dy

    def select_points_in_box(self, pt1, pt2, add_to_selection=False):
        """计算框选范围内的点"""
        x1, x2 = min(pt1[0], pt2[0]), max(pt1[0], pt2[0])
        y1, y2 = min(pt1[1], pt2[1]), max(pt1[1], pt2[1])

        # 获取当前所有点的屏幕坐标
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

    def mouse_callback(self, event, x, y, flags, param):
        ctrl_pressed = (flags & cv2.EVENT_FLAG_CTRLKEY)

        if event == cv2.EVENT_LBUTTONDOWN:
            self.last_mouse_pos = (x, y)
            self.drag_start_pos = (x, y)

            # 1. 检测点击了哪个点
            clicked_idx = None
            min_dist = 10000

            grid_screen_x = (self.roi_x + self.grid[..., 0] * self.roi_w) * self.scale_disp
            grid_screen_y = (self.roi_y + self.grid[..., 1] * self.roi_h) * self.scale_disp

            for iy in range(GRID_SIZE):
                for ix in range(GRID_SIZE):
                    px, py = grid_screen_x[iy, ix], grid_screen_y[iy, ix]
                    dist = (x - px) ** 2 + (y - py) ** 2
                    if dist < 100:  # 10px radius
                        if dist < min_dist:
                            min_dist = dist
                            clicked_idx = (iy, ix)

            if clicked_idx:
                # 点击了点
                self.is_dragging = True
                self.is_box_selecting = False
                if ctrl_pressed:
                    # Ctrl: 切换选中状态
                    if clicked_idx in self.selected_pts:
                        self.selected_pts.remove(clicked_idx)
                    else:
                        self.selected_pts.add(clicked_idx)
                else:
                    # 单选：如果点击的点不在当前选中集合里，清空并单选
                    if clicked_idx not in self.selected_pts:
                        self.selected_pts = {clicked_idx}
                    # 如果已经在集合里，保持集合不变(准备整体拖动)
            else:
                # 点击空白处 -> 准备框选
                self.is_dragging = False
                self.is_box_selecting = True
                if not ctrl_pressed:
                    self.selected_pts.clear()

            self.update_display()

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_dragging:
                # 拖动点
                dx_screen = x - self.last_mouse_pos[0]
                dy_screen = y - self.last_mouse_pos[1]
                dx_norm = (dx_screen / self.scale_disp) / self.roi_w
                dy_norm = (dy_screen / self.scale_disp) / self.roi_h

                self.apply_move(dx_norm, dy_norm)
                self.last_mouse_pos = (x, y)
                self.update_display()

            elif self.is_box_selecting:
                # 拖动框
                self.last_mouse_pos = (x, y)
                self.update_display()  # 绘制框

        elif event == cv2.EVENT_LBUTTONUP:
            if self.is_box_selecting:
                # 结算框选
                self.select_points_in_box(self.drag_start_pos, (x, y), add_to_selection=ctrl_pressed)
                self.is_box_selecting = False
                self.update_display()
            self.is_dragging = False

        # 滚轮调节透明度
        elif event == cv2.EVENT_MOUSEWHEEL:
            delta = 0.05 if flags > 0 else -0.05
            self.alpha = max(0.0, min(1.0, self.alpha + delta))
            self.update_display()

    def handle_keyboard(self, key_char, key_code):
        if not self.selected_pts: return False

        step_x = 0.5 / self.roi_w  # 微调步长
        step_y = 0.5 / self.roi_h
        dx, dy = 0, 0

        # 上
        if key_char == ord('w') or key_code == 2490368:
            dy = -step_y
        # 下
        elif key_char == ord('s') or key_code == 2621440:
            dy = step_y
        # 左
        elif key_char == ord('a') or key_code == 2424832:
            dx = -step_x
        # 右
        elif key_char == ord('d') or key_code == 2555904:
            dx = step_x

        if dx != 0 or dy != 0:
            self.apply_move(dx, dy)
            return True
        return False

    def run(self):
        cv2.namedWindow(self.win_src)
        cv2.namedWindow(self.win_res)
        cv2.setMouseCallback(self.win_src, self.mouse_callback)
        self.update_display()

        print("==================================================")
        print("  Ultimate Warp Tool V6")
        print("==================================================")
        print("  1. [左键点] 选择点 (角点红，内部浅蓝，选中绿)")
        print("  2. [Ctrl+点] 多选 / 反选")
        print("  3. [空白处拖拽] 框选 (配合 Ctrl 可加选)")
        print("  4. [拖动选中点] 或 [WASD] 移动")
        print("     - 拖动单个角点 -> 弹性变形 (影响全局)")
        print("     - 拖动内部/多个点 -> 局部移动 (角点固定)")
        print("  5. [滚轮] 透明度 | [空格] 保存")
        print("==================================================")

        while True:
            key = cv2.waitKeyEx(30)
            if key == -1: continue
            key_char = key & 0xFF

            if key_char == ord('q'):
                break
            elif key_char == ord(' '):
                print("渲染最终结果...")
                map_x, map_y = self.get_real_coords_map(self.tgt_w, self.tgt_h)
                final_img = cv2.remap(self.img_lst_full, map_x, map_y, interpolation=cv2.INTER_CUBIC)
                if len(final_img.shape) == 3:
                    final_gray = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
                else:
                    final_gray = final_img
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(output_path, final_gray)
                print(f"【GT Saved】: {output_path}")
                break
            elif key_char == ord('z'):
                self.alpha = max(0.0, self.alpha - 0.05);
                self.update_display()
            elif key_char == ord('c'):
                self.alpha = min(1.0, self.alpha + 0.05);
                self.update_display()
            else:
                if self.handle_keyboard(key_char, key):
                    self.update_display()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = UltimateWarpTool()
    app.run()