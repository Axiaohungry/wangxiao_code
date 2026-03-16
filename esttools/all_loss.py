# -*- coding: utf-8 -*-
"""
批量汇总各组别 loss 表格前两列，并横向拼接成一个总表。

依赖：
    pip install pandas openpyxl xlrd

说明：
1. 默认只扫描“根目录下的一级子文件夹”中的“直接文件”，不递归扫描更深层目录。
2. 候选 loss 文件筛选规则：
   - 文件名中包含 "loss"（不区分大小写）
   - 扩展名属于：.csv / .tsv / .txt / .xlsx / .xls
3. 如果一个组别中存在多个 loss 文件，选择规则为：
   - 优先选择“修改时间最新”的候选文件
   - 若修改时间相同，优先级：.csv > .tsv > .txt > .xlsx > .xls
   - 若还相同，再按文件名字典序
4. 横向拼接时按“行号”对齐，不按第一列的数值键对齐。
   如果各组行数不同，较短组别末尾会自动补 NaN。
"""

import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd


# =========================
# 1. 根目录与输出文件配置
# =========================
ROOT_DIR = Path(r"D:\PycharmProjects\wangxiao_code\output\runs\2026-01-20_phase5_sunfix_el75\phase5")
OUTPUT_FILE = ROOT_DIR / "merged_loss_first2cols.csv"

# 支持的表格扩展名
SUPPORTED_EXTS = {".csv", ".tsv", ".txt", ".xlsx", ".xls"}
TEXT_TABLE_EXTS = {".csv", ".tsv", ".txt"}
EXCEL_TABLE_EXTS = {".xlsx", ".xls"}

# 文本表格可能出现的编码
ENCODINGS_TO_TRY = [
    "utf-8-sig",
    "utf-8",
    "gb18030",
    "gbk",
    "utf-16",
    "latin1",
]

# 尝试的分隔符（None 表示自动推断）
SEPARATORS_TO_TRY = [None, ",", "\t", ";", "|"]


def print_line(char: str = "=", n: int = 100) -> None:
    """打印分隔线。"""
    print(char * n)


def get_group_dirs(root_dir: Path) -> List[Path]:
    """获取根目录下所有一级子文件夹。"""
    return sorted([p for p in root_dir.iterdir() if p.is_dir()], key=lambda x: x.name.lower())


def get_loss_candidates(group_dir: Path) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    在组别目录中查找文件。
    返回：
        all_files           : 该组别目录下所有直接文件
        supported_candidates: 文件名含 loss 且扩展名支持的候选文件
        unsupported_loss    : 文件名含 loss 但扩展名不支持的文件
    """
    all_files = sorted([p for p in group_dir.iterdir() if p.is_file()], key=lambda x: x.name.lower())

    loss_named_files = [p for p in all_files if "loss" in p.name.lower()]
    supported_candidates = [p for p in loss_named_files if p.suffix.lower() in SUPPORTED_EXTS]
    unsupported_loss = [p for p in loss_named_files if p.suffix.lower() not in SUPPORTED_EXTS]

    return all_files, supported_candidates, unsupported_loss


def select_best_candidate(candidates: List[Path]) -> Path:
    """
    多个 loss 文件时的选择策略：
    1) 修改时间最新优先
    2) 若相同，扩展名优先级：csv > tsv > txt > xlsx > xls
    3) 若还相同，文件名字典序
    """
    ext_priority = {
        ".csv": 0,
        ".tsv": 1,
        ".txt": 2,
        ".xlsx": 3,
        ".xls": 4,
    }

    selected = sorted(
        candidates,
        key=lambda p: (
            -p.stat().st_mtime,                      # 修改时间越新越优先
            ext_priority.get(p.suffix.lower(), 99), # 扩展名优先级
            p.name.lower(),                         # 文件名字典序
        )
    )[0]
    return selected


def read_text_table(file_path: Path) -> Tuple[pd.DataFrame, str]:
    """
    读取 csv/tsv/txt。
    尝试多种编码、多种分隔符。
    返回：
        df
        read_mode（描述最终用什么方式读取成功）
    """
    last_error = None

    for encoding in ENCODINGS_TO_TRY:
        for sep in SEPARATORS_TO_TRY:
            try:
                if sep is None:
                    # 自动推断分隔符
                    df = pd.read_csv(
                        file_path,
                        sep=None,
                        engine="python",
                        encoding=encoding
                    )
                    return df, f"read_csv(auto-sep, encoding={encoding})"
                else:
                    df = pd.read_csv(
                        file_path,
                        sep=sep,
                        encoding=encoding
                    )
                    return df, f"read_csv(sep={repr(sep)}, encoding={encoding})"
            except Exception as e:
                last_error = e

    raise RuntimeError(f"文本表格读取失败：{last_error}")


def read_excel_table(file_path: Path) -> Tuple[pd.DataFrame, str]:
    """
    读取 xlsx/xls。
    """
    try:
        df = pd.read_excel(file_path)
        return df, "read_excel(default)"
    except ImportError as e:
        raise RuntimeError(
            f"读取 Excel 失败，可能缺少依赖。请安装：pip install openpyxl xlrd\n原始错误：{e}"
        )
    except Exception as e:
        raise RuntimeError(f"Excel 表格读取失败：{e}")


def read_table_safely(file_path: Path) -> Tuple[pd.DataFrame, str]:
    """
    根据扩展名安全读取表格。
    """
    suffix = file_path.suffix.lower()

    if suffix in TEXT_TABLE_EXTS:
        return read_text_table(file_path)
    elif suffix in EXCEL_TABLE_EXTS:
        return read_excel_table(file_path)
    else:
        raise RuntimeError(f"不支持的文件类型：{suffix}")


def main() -> None:
    # 1. 检查根目录
    if not ROOT_DIR.exists():
        print(f"[错误] 根目录不存在：{ROOT_DIR}")
        sys.exit(1)

    if not ROOT_DIR.is_dir():
        print(f"[错误] ROOT_DIR 不是文件夹：{ROOT_DIR}")
        sys.exit(1)

    # 2. 找到所有组别目录
    group_dirs = get_group_dirs(ROOT_DIR)
    print_line()
    print(f"根目录：{ROOT_DIR}")
    print(f"共找到 {len(group_dirs)} 个组别文件夹：{[p.name for p in group_dirs]}")
    print_line()

    if not group_dirs:
        print("[错误] 根目录下没有找到任何一级子文件夹。")
        sys.exit(1)

    merged_frames = []
    success_groups = []

    # 3. 遍历每个组别
    for group_dir in group_dirs:
        group_name = group_dir.name
        print(f"\n组别：{group_name}")

        all_files, supported_candidates, unsupported_loss = get_loss_candidates(group_dir)

        # 打印当前组别中所有直接文件
        if all_files:
            print("  该组别中的文件：")
            for f in all_files:
                print(f"    - {f.name}")
        else:
            print("  [警告] 该组别文件夹为空，已跳过。")
            continue

        # 提示 loss 命名但不支持的文件
        if unsupported_loss:
            print("  [提示] 以下文件名包含 loss，但不是支持的表格类型，已跳过：")
            for f in unsupported_loss:
                print(f"    - {f.name} ({f.suffix})")

        # 没有可用候选
        if not supported_candidates:
            print("  [警告] 未找到可用的 loss 表格文件，已跳过该组别。")
            continue

        # 多候选时打印规则
        if len(supported_candidates) > 1:
            print("  检测到多个 loss 候选文件，将按以下规则选取：")
            print("    1) 修改时间最新")
            print("    2) 若时间相同：.csv > .tsv > .txt > .xlsx > .xls")
            print("    3) 若仍相同：按文件名字典序")
            print("  当前候选文件：")
            for f in supported_candidates:
                print(f"    - {f.name} | 修改时间: {pd.to_datetime(f.stat().st_mtime, unit='s')}")

        # 选中文件
        selected_file = select_best_candidate(supported_candidates)
        print(f"  选中的 loss 文件：{selected_file.name}")
        print(f"  完整路径：{selected_file}")

        # 4. 读取文件
        try:
            df, read_mode = read_table_safely(selected_file)
            print(f"  读取方式：{read_mode}")
        except Exception as e:
            print(f"  [错误] 读取失败：{e}")
            continue

        # 5. 打印列名
        col_names = [str(c) for c in df.columns.tolist()]
        print(f"  读取到的列名：{col_names}")

        # 6. 至少要有两列
        if df.shape[1] < 2:
            print(f"  [警告] 文件列数不足 2 列，实际只有 {df.shape[1]} 列，已跳过。")
            continue

        # 7. 取前两列，并重置索引，确保横向拼接按行号对齐
        first_two = df.iloc[:, :2].copy().reset_index(drop=True)

        original_first_two_cols = [str(c) for c in first_two.columns.tolist()]
        print(f"  前两列原始列名：{original_first_two_cols}")

        # 8. 重命名为 “组别名_col1 / 组别名_col2”
        first_two.columns = [f"{group_name}_col1", f"{group_name}_col2"]
        print(f"  重命名后列名：{list(first_two.columns)}")

        # 9. 收集待拼接数据
        merged_frames.append(first_two)
        success_groups.append(group_name)

    # 10. 没有任何成功数据时退出
    if not merged_frames:
        print_line()
        print("[错误] 没有任何组别成功读取到可用的前两列数据，未生成输出文件。")
        sys.exit(1)

    # 11. 横向拼接
    merged_df = pd.concat(merged_frames, axis=1)

    # 12. 保存结果
    try:
        merged_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    except PermissionError as e:
        print_line()
        print(f"[错误] 输出文件保存失败，可能被 Excel 占用：{OUTPUT_FILE}")
        print(f"详细错误：{e}")
        sys.exit(1)
    except Exception as e:
        print_line()
        print(f"[错误] 输出文件保存失败：{e}")
        sys.exit(1)

    # 13. 最终打印
    print_line()
    print(f"成功合并的组别：{success_groups}")
    print(f"最终拼接后的表格维度：{merged_df.shape[0]} 行 x {merged_df.shape[1]} 列")
    print(f"输出文件保存路径：{OUTPUT_FILE}")
    print("处理完成。")


if __name__ == "__main__":
    main()