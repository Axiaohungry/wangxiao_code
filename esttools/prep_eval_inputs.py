# esttools/prep_eval_inputs.py
import argparse, json
from pathlib import Path
import cv2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt = cv2.imread(args.gt, cv2.IMREAD_GRAYSCALE)
    pr = cv2.imread(args.pred, cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise FileNotFoundError(args.gt)
    if pr is None:
        raise FileNotFoundError(args.pred)

    Hg, Wg = gt.shape[:2]
    Hp, Wp = pr.shape[:2]

    gt_r = cv2.resize(gt, (Wp, Hp), interpolation=cv2.INTER_AREA)
    out_gt = out_dir / "gt_train.png"
    cv2.imwrite(str(out_gt), gt_r)

    scale = {"gt_hw": [Hg, Wg], "pred_hw": [Hp, Wp], "sx": Wp / float(Wg), "sy": Hp / float(Hg)}
    (out_dir / "scale.json").write_text(json.dumps(scale, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[OK] wrote:", out_gt)
    print("[OK] wrote:", out_dir / "scale.json", scale)

if __name__ == "__main__":
    main()