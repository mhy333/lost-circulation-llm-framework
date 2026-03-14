# -*- coding: utf-8 -*-
"""
kg_agent.py —— 井漏知识图谱模块（增强版）
----------------------------------------------------------
功能：
1. 从 CSV 构建知识图谱（带 MLR、漏失标签）
2. 节点颜色：红=漏失，蓝=未漏失
3. 关系类型：AFFECTS / CONSTRAINS / REPRESENTS
4. 关系粗细 ∝ MLR 值
5. 支持 MLR 阈值筛选，只显示高风险样本
"""

from py2neo import Graph, Node, Relationship
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import textwrap

class KGAgent:
    def __init__(self,
                 csv_path="E:/pycharm_project/lost-circ-rag/data/raw/数据/WellData_with_MLR.csv",
                 neo4j_uri="bolt://localhost:7687",
                 neo4j_user="neo4j",
                 neo4j_pwd="12345678"):
        """初始化 Neo4j 连接"""
        self.graph = Graph(neo4j_uri, auth=(neo4j_user, neo4j_pwd))
        self.csv_path = csv_path
        print("[KG] 已连接 Neo4j")

        # 参数字段
        self.param_cols = [
            "FormationPressureGradient", "FormationRupturePressureGradient",
            "WOB", "RPM", "PumpPressure", "HookLoad", "ROP", "Displacement",
            "Density", "ECD", "OutletFlow", "LagTime", "TheoreticalMaximumDisplacement"
        ]

        # 机理关系分类
        self.relations = {
            "AFFECTS": ["ECD", "Density", "PumpPressure", "OutletFlow", "WOB", "RPM", "ROP"],
            "CONSTRAINS": ["FormationPressureGradient", "FormationRupturePressureGradient"],
            "REPRESENTS": ["MLR"]
        }

    # ======================================================
    # 1️⃣ 构建知识图谱（支持按井号分组）
    # ======================================================
    def build_graph(self, clear_old=True, sample_limit=None):
        df = pd.read_csv(self.csv_path)
        if clear_old:
            self.graph.run("MATCH (n) DETACH DELETE n")
            print("[KG] 已清空旧图谱")

        if sample_limit:
            df = df.head(sample_limit)

        tx = self.graph.begin()
        risk_node = Node("Risk", name="LostCirculation", desc="井漏风险")
        tx.merge(risk_node, "Risk", "name")

        # ✅ 按 WellID 分组建图
        for well_id, group in df.groupby("WellID"):
            print(f"[KG] 正在构建井号 {well_id} 的知识子图，共 {len(group)} 条记录")

            for i, row in group.iterrows():
                sample_id = f"{well_id}_{i}"
                mlr_val = float(row["MLR"])
                label = int(row["LostCirculation"])
                color = "red" if label == 1 else "blue"

                # 井节点
                well_node = Node("Well",
                                 id=well_id,
                                 depth=float(row["WellDepth"]),
                                 mlr=mlr_val,
                                 label=label,
                                 color=color)
                tx.merge(well_node, "Well", "id")

                # MLR 节点
                mlr_node = Node("Index", name=f"MLR_{sample_id}", value=mlr_val)
                tx.merge(mlr_node, "Index", "name")
                tx.merge(Relationship(mlr_node, "REPRESENTS", risk_node))
                tx.merge(Relationship(well_node, "HAS_INDEX", mlr_node))

                # 参数节点
                for p in self.param_cols:
                    val = float(row[p])
                    param_node = Node("Parameter", name=f"{p}_{sample_id}", param=p, value=val)
                    tx.merge(param_node, "Parameter", "name")

                    rel_type = next((k for k, v in self.relations.items() if p in v), "AFFECTS")
                    rel = Relationship(param_node, rel_type, risk_node)
                    rel["weight"] = mlr_val
                    tx.merge(rel)

                    tx.merge(Relationship(well_node, "HAS_PARAMETER", param_node))

        self.graph.commit(tx)
        print(f"[KG] 按井号构建图谱完成，共 {df['WellID'].nunique()} 口井。")

    def _layout_community(self, G, seed=7,
                          inter_radius=3.5,
                          intra_k=0.35,
                          intra_iter=120):
        """
        B 模式：社区簇状布局（Louvain + 社区中心偏移）
        """
        try:
            import community as community_louvain
        except ImportError:
            print("[KG] 未安装 python-louvain，退回 spring_layout")
            return nx.spring_layout(G, seed=seed)

        import math
        from collections import defaultdict

        U = G.to_undirected()
        partition = community_louvain.best_partition(U, random_state=seed)

        # community -> nodes
        comm2nodes = defaultdict(list)
        for n, c in partition.items():
            comm2nodes[c].append(n)

        # 给每个社区一个中心（圆形排布）
        comm_ids = sorted(comm2nodes.keys())
        m = max(len(comm_ids), 1)
        centers = {}
        for i, cid in enumerate(comm_ids):
            ang = 2 * math.pi * i / m
            centers[cid] = (
                inter_radius * math.cos(ang),
                inter_radius * math.sin(ang)
            )

        pos = {}
        for cid, nodes in comm2nodes.items():
            sub = U.subgraph(nodes)

            if sub.number_of_nodes() == 1:
                pos[nodes[0]] = centers[cid]
                continue

            sub_pos = nx.spring_layout(
                sub, k=intra_k, iterations=intra_iter, seed=seed
            )

            cx, cy = centers[cid]
            for n in nodes:
                x, y = sub_pos[n]
                pos[n] = (x + cx, y + cy)

        # 写回社区编号（可用于调试/着色）
        for n in G.nodes():
            G.nodes[n]["community"] = partition.get(n, 0)

        return pos

    def visualize_examples(self, mlr_threshold=0.55, save_path=None):
        """
        绘制两组典型样本（1漏 + 1不漏）—— 美化版
        """
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
        matplotlib.rcParams['axes.unicode_minus'] = False

        df = pd.read_csv(self.csv_path)

        leak_case = df[df["LostCirculation"] == 1].head(1)
        normal_case = df[df["LostCirculation"] == 0].head(1)

        if leak_case.empty or normal_case.empty:
            print("⚠️ 数据中缺少漏失或未漏失样本，无法绘制示例图。")
            return

        selected = pd.concat([leak_case, normal_case])
        print(f"[KG] 已选择 {len(selected)} 条样本（1漏 + 1不漏）进行绘图。")

        G = nx.DiGraph()

        # 构建节点关系
        for i, row in selected.iterrows():
            well_id = str(row["WellID"])
            mlr_val = float(row["MLR"])
            leak_flag = int(row["LostCirculation"])
            color = "#E74C3C" if leak_flag == 1 else "#3498DB"  # 漏失=红，未漏=蓝

            risk_node = f"井号 {well_id}"
            G.add_node(risk_node, color=color, size=1200, type="Risk")

            # 添加参数节点
            for p in self.param_cols:
                val = float(row[p])
                src = f"{p}\n{val:.2f}"
                rel_type = next((k for k, v in self.relations.items() if p in v), "AFFECTS")
                G.add_edge(src, risk_node, label=rel_type, weight=mlr_val)

        # 绘图美化
        plt.figure(figsize=(9, 7))
       # pos = nx.kamada_kawai_layout(G)  # 更均匀布局
        pos = self._layout_community(G)

        # 边的样式（颜色 + 宽度）
        edge_colors, edge_widths = [], []
        for u, v in G.edges():
            rel = G[u][v]["label"]
            w = G[u][v]["weight"]
            if rel == "AFFECTS":
                edge_colors.append("#5DADE2")  # 柔和蓝
            elif rel == "CONSTRAINS":
                edge_colors.append("#F5B041")  # 柔和橙
            elif rel == "REPRESENTS":
                edge_colors.append("#58D68D")  # 柔和绿
            else:
                edge_colors.append("gray")
            edge_widths.append(1.2 + 2.5 * w)  # 控制线条宽度范围

        # 节点颜色
        node_colors = [G.nodes[n].get("color", "#F2F4F4") for n in G.nodes()]
        node_sizes = [G.nodes[n].get("size", 1000) for n in G.nodes()]

        nx.draw_networkx_nodes(G, pos,
                               node_color=node_colors,
                               node_size=node_sizes,
                               alpha=0.9,
                               linewidths=1.5,
                               edgecolors="black")
        nx.draw_networkx_labels(G, pos, font_size=9)

        nx.draw_networkx_edges(G, pos,
                               edge_color=edge_colors,
                               width=edge_widths,
                               alpha=0.7,
                               arrowsize=15,
                               connectionstyle="arc3,rad=0.08")

        edge_labels = nx.get_edge_attributes(G, "label")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                     font_size=8, font_color="#2C3E50")

        plt.title("典型井漏样本知识图谱对比（红=漏失，蓝=正常）", fontsize=12, pad=2 )
        plt.axis("off")
        plt.tight_layout()

        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor="white")
            print(f"[KG] 图谱已保存: {save_path}")

        plt.show()

    def visualize_examples_side_by_side(self, mlr_threshold=0.55, well_id=None, save_path=None):
        """
        并排绘制 1 漏 + 1 不漏 样本的知识图谱对比（增强论文版）
        可指定 well_id，仅绘制该井的样本对比。
        """
        import matplotlib
        import matplotlib.pyplot as plt
        import io
        import os

        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Times New Roman']
        matplotlib.rcParams['font.family'] = ['sans-serif']
        matplotlib.rcParams['axes.unicode_minus'] = False

        df = pd.read_csv(self.csv_path)

        # ✅ 如果指定井号，则只筛选该井
        if well_id:
            df = df[df["WellID"] == well_id]
            if df.empty:
                print(f"⚠️ 未找到井号 {well_id} 的数据。")
                return None

        leak_case = df[(df["LostCirculation"] == 1) & (df["MLR"] > mlr_threshold)]
        normal_case = df[(df["LostCirculation"] == 0) & (df["MLR"] < mlr_threshold)]

        if leak_case.empty or normal_case.empty:
            print("⚠️ 未找到符合条件的样本，请检查阈值或井号。")
            return None

        leak_case = leak_case.sort_values(by="MLR", ascending=False).head(1)
        normal_case = normal_case.sort_values(by="MLR", ascending=True).head(1)
        selected = {"漏失样本（高风险）": leak_case, "未漏失样本（低风险）": normal_case}

        fig, axes = plt.subplots(1, 2, figsize=(14, 8), dpi=300)
        fig.suptitle(f"井号 {well_id} 典型井漏样本知识图谱对比（红=漏失, 蓝=正常）", fontsize=14, y=0.95)

        for idx, (title, data) in enumerate(selected.items()):
            G = nx.DiGraph()
            row = data.iloc[0]

            well_id_str = str(row["WellID"])
            mlr_val = float(row["MLR"])
            leak_flag = int(row["LostCirculation"])
            color = "#E74C3C" if leak_flag == 1 else "#3498DB"

            risk_node = f"井号 {well_id_str}\nMLR={mlr_val:.3f}"
            G.add_node(risk_node, color=color, size=2600, type="Risk")

            def wrap_label(name, width=6):
                return "\n".join(textwrap.wrap(name, width=width))

            for p in self.param_cols:
                val = float(row[p])
                p_wrapped = wrap_label(p, width=6)
                src = f"{p_wrapped}\n{val:.2f}"
                rel_type = next((k for k, v in self.relations.items() if p in v), "AFFECTS")
                G.add_node(src, color="#F9E79F", size=1800, type="Param")
                G.add_edge(src, risk_node, label=rel_type, weight=mlr_val)

            # ✅ B 模式：社区簇状布局
            pos = self._layout_community(G)
            # 轻微整体缩放，适合论文排版
            for k, (x, y) in pos.items():
                pos[k] = (0.85 * float(x), 0.85 * float(y))

            edge_colors, edge_widths = [], []
            for u, v in G.edges():
                rel = G[u][v]["label"]
                w = G[u][v]["weight"]
                edge_colors.append({
                                       "AFFECTS": "#5DADE2",
                                       "CONSTRAINS": "#F5B041",
                                       "REPRESENTS": "#58D68D"
                                   }.get(rel, "gray"))
                edge_widths.append(0.8 + 1.5 * w)

            node_colors = [G.nodes[n].get("color", "#F9E79F") for n in G.nodes()]
            node_sizes = [G.nodes[n].get("size", 1800) for n in G.nodes()]

            nx.draw_networkx_nodes(G, pos, ax=axes[idx],
                                   node_color=node_colors, node_size=node_sizes,
                                   alpha=0.96, edgecolors="black", linewidths=1.3)
            nx.draw_networkx_labels(G, pos, ax=axes[idx],
                                    font_size=9, font_family="SimHei",
                                    verticalalignment="center", horizontalalignment="center")
            nx.draw_networkx_edges(G, pos, ax=axes[idx],
                                   edge_color=edge_colors, width=edge_widths,
                                   alpha=0.8, arrowsize=8, connectionstyle="arc3,rad=0.08")
            nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, "label"),
                                         font_size=8, font_color="#2C3E50", ax=axes[idx])

            axes[idx].set_title(title, fontsize=12, pad=0.5)
            axes[idx].margins(0.05)
            axes[idx].set_aspect('equal', adjustable='datalim')
            axes[idx].axis("off")

        plt.subplots_adjust(wspace=0.1, top=0.88, bottom=0.05)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300, bbox_inches='tight', facecolor="white")
        plt.close()
        buf.seek(0)
        return buf
