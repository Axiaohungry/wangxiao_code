from pathlib import Path

import cv2
import numpy as np


def draw_colorbar(
    out_path: Path,
    height: int = 720,
    bar_width: int = 90,
    label_width: int = 120,
    ticks: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25, 0.0),
) -> None:
    """Generate the same 0..1 JET thermal colorbar used by view_phase5_roi_splat.py."""
    gradient = np.linspace(1.0, 0.0, height, dtype=np.float32)[:, None]
    gray_u8 = np.clip(gradient * 255.0, 0, 255).astype(np.uint8)
    bar = cv2.applyColorMap(gray_u8, cv2.COLORMAP_JET)
    bar = cv2.resize(bar, (bar_width, height), interpolation=cv2.INTER_NEAREST)

    canvas = np.full((height, bar_width + label_width, 3), 255, dtype=np.uint8)
    canvas[:, :bar_width] = bar

    for value in ticks:
        y = int(round((1.0 - value) * (height - 1)))
        cv2.line(canvas, (bar_width, y), (bar_width + 12, y), (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"{value:.2f}",
            (bar_width + 20, y + 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


if __name__ == "__main__":
    draw_colorbar(Path("output/thermal_colorbar_0_1_jet.png"))
