"""read_file 工具处理器。"""

import os


async def _handle_read_file(args: dict, work_dir: str) -> str:
    """read_file 工具处理器。"""
    import pandas as pd
    filename = args.get("filename", "")
    n_rows = args.get("n_rows", 20)
    sheet_name = args.get("sheet_name") or 0
    filepath = os.path.join(work_dir, filename)

    if not os.path.exists(filepath):
        return f"错误: 文件 '{filename}' 不存在于工作目录"

    try:
        if filename.lower().endswith((".xlsx", ".xls")):
            xl = pd.ExcelFile(filepath)
            # 列出所有sheet信息
            sheets_info = [f"Sheet {i}: '{s}' ({xl.parse(s).shape[0]}行 x {xl.parse(s).shape[1]}列)" for i, s in enumerate(xl.sheet_names)]
            # 选择指定sheet
            if isinstance(sheet_name, str) and sheet_name.isdigit():
                sheet_name = int(sheet_name)
            if isinstance(sheet_name, int):
                sname = xl.sheet_names[sheet_name]
            else:
                sname = sheet_name if sheet_name in xl.sheet_names else xl.sheet_names[0]
            df = xl.parse(sname).head(n_rows)
            col_names = list(df.columns)
            return f"Excel文件, {len(xl.sheet_names)}个sheet:\n" + "\n".join(sheets_info) + f"\n\n读取 sheet '{sname}' 前{n_rows}行 (共{df.shape[0]}行 x {df.shape[1]}列):\n列名: {col_names}\n\n" + df.to_string()
        else:
            df = pd.read_csv(filepath, nrows=n_rows) if filename.endswith(".csv") else pd.read_table(filepath, nrows=n_rows)
            return f"文件前{n_rows}行 ({df.shape[0]}行 x {df.shape[1]}列):\n" + df.to_string()
    except Exception as e:
        return f"读取失败: {e}"
