# -*- coding: utf-8 -*-
from kg_agent import KGAgent
from mkg_agent import MechanismKGAgent
import os
import re

def safe_filename(text: str) -> str:
    """把中文/特殊字符的 LostType 转成安全文件名"""
    text = str(text)
    return re.sub(r'[\\/:*?"<>|]', "_", text)

if __name__ == "__main__":
    # ===== 1. 现场数据知识图谱（保留） =====
    kg = KGAgent(csv_path="E:/pycharm_project/lost-circ-rag/data/raw/data/WellData_with_MLR.csv")
    kg.visualize_examples_side_by_side(
        mlr_threshold=0.55,
        save_path="E:/pycharm_project/lost-circ-rag/outputs/KG_Example_Compare.png"
    )

    # ===== 2. 机理知识图谱（A/B/C/D × 所有 LostType × 多视图） =====
    out_dir = "E:/pycharm_project/lost-circ-rag/outputs/MKG_all"  # ✅ 改成正斜杠
    os.makedirs(out_dir, exist_ok=True)

    kg_mech = MechanismKGAgent(
        csv_path="E:/pycharm_project/lost-circ-rag/data/raw/data/MechanismRules_300.csv",
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_pwd="12345678"
    )

    # 如果你不想每次清库，改成 False
    kg_mech.build_graph(clear_old=True)

    # LostType 列表
    lost_types = sorted(
        set(
            lt for lt in kg_mech.df.get("LostType", []).dropna().tolist()
            if str(lt).strip()
        )
    )
    print("[TEST] 检测到 LostType 列表：", lost_types)

    layout_modes = ["A", "B", "C"]
    views = ["full", "mechanism", "evidence", "constraint", "action"]

    for lost_type in lost_types:
        safe_lt = safe_filename(lost_type)
        print(f"\n[TEST] ==== LostType: {lost_type} ====")

        # ----- A/B/C：静态布局 × 多视图 -----
        for mode in layout_modes:
            for view in views:
                save_path = os.path.join(out_dir, f"MKG_{safe_lt}_{view}_layout_{mode}.png")
                try:
                    kg_mech.visualize_multiview(
                        lost_type=lost_type,
                        view=view,
                        layout_mode=mode,
                        interactive=False,     # ✅ 显式
                        save_path=save_path,
                        return_buffer=False,
                        limit_rules=150         # ✅ 建议加：避免太大
                    )
                    print(f"[OK] {lost_type} - view={view} - layout={mode} → {save_path}")
                except Exception as e:
                    print(f"[ERR] {lost_type} - view={view} - layout={mode} 失败：{e}")

        # ----- D：交互式 HTML（多视图） -----
        for view in views:
            html_path = os.path.join(out_dir, f"MKG_{safe_lt}_{view}_interactive.html")
            try:
                # ✅ 先验证 return_html=True（给 Gradio 用的链路）
                raw_html = kg_mech.visualize_multiview(
                    lost_type=lost_type,
                    view=view,
                    layout_mode="LAYER",     # ✅ 让 D 也使用多维立体布局语义
                    interactive=True,
                    save_path=None,
                    return_html=True,
                    limit_rules=80
                )
                assert isinstance(raw_html, str) and "<html" in raw_html.lower()

                # ✅ 再落盘保存，便于本地双击打开
                dir_ = os.path.dirname(html_path)
                if dir_:
                    os.makedirs(dir_, exist_ok=True)
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(raw_html)

                print(f"[OK] {lost_type} - view={view} - 方案 D (交互式) → {html_path}")
            except Exception as e:
                print(f"[ERR] {lost_type} - view={view} - 方案 D 失败：{e}")

    print("\n[TEST] 全部 LostType + A/B/C/D 模式批量绘图完成。")
