# -*- coding: utf-8 -*-
"""
mkg_agent.py —— 井漏机理知识图谱模块
----------------------------------------------------------
功能：
1. 从 MechanismRule_full_MR001_036.csv 构建“机理知识图谱”
2. 核心节点：
   - Concept（LostCirculation）
   - Rule（MR_001~MR_036）
   - LostType / Severity / RiskLevel / TimePattern / FormationType / Parameter
3. 主要关系：
   - (Rule)-[:BELONGS_TO]->(LostType)
   - (Rule)-[:HAS_SEVERITY]->(Severity)
   - (Rule)-[:HAS_RISK]->(RiskLevel)
   - (Rule)-[:HAS_PATTERN]->(TimePattern)
   - (Rule)-[:APPLICABLE_TO]->(FormationType)
   - (Rule)-[:INVOLVES_PARAM]->(Parameter)
   - (Rule)-[:IS_MECHANISM_OF]->(Concept {name:"LostCirculation"})
4. 支持简单可视化：按 LostType 绘制局部机理子图
"""

from py2neo import Graph, Node, Relationship
import pandas as pd
import ast
import networkx as nx
import matplotlib.pyplot as plt
import textwrap
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import textwrap
import io
import os
import json
import re
from collections import defaultdict
from pyvis.network import Network
import numpy as np

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 可选：社区发现算法（方案 B 用），需要 pip install python-louvain
try:
    import community as community_louvain

    HAS_LOUVAIN = True
except Exception:
    HAS_LOUVAIN = False


class MechanismKGAgent:
    # --- Action layer keywords (for Action nodes) ---
    ACTION_KEYWORDS = {
        "LCM": ["堵漏", "桥堵", "颗粒", "纤维", "LCM", "plug", "bridg"],
        "MPD": ["控压", "MPD", "managed pressure"],
        "ECD_CONTROL": ["降低ECD", "控制ECD", "调整排量", "调整密度", "surge", "swab"],
        "CEMENT": ["水泥", "cement", "挤水泥"],
        "SPOT_PILL": ["堵漏浆", "pill", "spot"],
    }

    def __init__(self,
                 csv_path="E:/pycharm_project/lost-circ-rag/data/raw/data/MechanismRules_300.csv",
                 neo4j_uri="bolt://localhost:7687",
                 neo4j_user="neo4j",
                 neo4j_pwd="12345678"):
        """初始化 Neo4j 连接"""
        self.graph = Graph(neo4j_uri, auth=(neo4j_user, neo4j_pwd))
        self.csv_path = csv_path
        self.df = pd.read_csv(self.csv_path, encoding="utf-8")
        print("[MechanismKG] 已连接 Neo4j，并加载规则表：", self.csv_path)

        # ---------------- 通用解析函数（原样保留） ----------------

    @staticmethod
    def _safe_eval_list(value):
        if isinstance(value, list):
            return value
        if not isinstance(value, str) or value.strip() == "":
            return []
        try:
            result = ast.literal_eval(value)
            if isinstance(result, list):
                return result
            return [result]
        except Exception:
            if "," in value:
                return [v.strip() for v in value.split(",") if v.strip()]
            else:
                return [value.strip()]

    @staticmethod
    def _safe_json(value, default=None):
        if default is None:
            default = {}
        if not isinstance(value, str) or value.strip() == "":
            return default
        try:
            return json.loads(value)
        except Exception:
            return default

    @staticmethod
    def _safe_eval_logic_terms(value: str):
        """
        从 ConditionsStructured 抽取逻辑项。
        支持：
        1) {"logic":[...]}  (JSON)
        2) {"logic":[...]}  (literal_eval)
        3) ["...", "..."]   (JSON list)
        """
        if not isinstance(value, str) or value.strip() == "":
            return []

        # JSON 优先
        try:
            obj = json.loads(value)
            if isinstance(obj, dict):
                lg = obj.get("logic", [])
                if isinstance(lg, str):
                    return [lg.strip()] if lg.strip() else []
                if isinstance(lg, list):
                    return [str(x).strip() for x in lg if str(x).strip()]
            if isinstance(obj, list):
                return [str(x).strip() for x in obj if str(x).strip()]
        except Exception:
            pass

        # fallback：literal_eval
        try:
            obj = ast.literal_eval(value)
            if isinstance(obj, dict):
                lg = obj.get("logic", [])
                if isinstance(lg, list):
                    return [str(x).strip() for x in lg if str(x).strip()]
        except Exception:
            return []

        return []

    @staticmethod
    def _safe_eval_params(value):
        if isinstance(value, list):
            return value
        if not isinstance(value, str) or value.strip() == "":
            return []
        try:
            result = ast.literal_eval(value)
            if isinstance(result, list):
                return [str(x).strip() for x in result]
        except Exception:
            pass
        return [v.strip() for v in value.split(",") if v.strip()]

    @staticmethod
    def _safe_eval_conditions(value):
        if not isinstance(value, str) or value.strip() == "":
            return []
        try:
            result = ast.literal_eval(value)
            if isinstance(result, list):
                params = []
                for item in result:
                    if isinstance(item, dict) and "param" in item:
                        params.append(str(item["param"]).strip())
                return params
        except Exception:
            return []
        return []

    # ---------- 新增：文本聚合 & 换行工具 ----------

    @staticmethod
    def _agg_field(series, max_chars=140):
        """把一列文本合并成一段概要：去重 + 截断"""
        vals = [str(v).strip() for v in series.dropna() if str(v).strip()]
        if not vals:
            return "暂无明确描述"
        seen, uniq = set(), []
        for v in vals:
            if v not in seen:
                uniq.append(v)
                seen.add(v)
        text = "；".join(uniq)
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        return text

    @staticmethod
    def _wrap(text, width=16, max_lines=6):
        """节点内文字自动换行"""
        if not text:
            return ""
        lines = textwrap.wrap(text, width=width)
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            lines[-1] = lines[-1] + "..."
        return "\n".join(lines)

    # ======================================================
    # 1️⃣ 构建“井漏机理知识图谱”（Neo4j）—— 原逻辑不动
    # ======================================================
    def build_graph(self, clear_old=True):
        """Build mechanism knowledge graph in Neo4j from self.df.

        Notes
        -----
        - 保持原有节点/关系命名与字段不变，避免 main 侧调用异常。
        - 修复旧版缩进问题：Mechanism/Evidence/Constraint/Action 只处理最后一行的问题。
        """
        df = self.df  # 直接用已读好的

        if clear_old:
            self.graph.run("MATCH (n) DETACH DELETE n")
            print("[MechanismKG] 已清空旧图谱")

        tx = self.graph.begin()

        concept_node = Node("Concept", name="LostCirculation", desc="井漏及其机理")
        tx.merge(concept_node, "Concept", "name")

        for _, row in df.iterrows():
            rule_id = row.get("RuleID")
            rule_name = row.get("RuleName")
            lost_type_name = row.get("LostType")
            severity_name = row.get("SeverityLevel")
            risk_level_name = row.get("RiskLevel")
            time_pattern_name = row.get("TimePattern")

            rule_node = Node(
                "Rule",
                RuleID=rule_id,
                name=rule_name,
                LostType=lost_type_name,
                SeverityLevel=severity_name,
                RiskLevel=risk_level_name,
                TimePattern=time_pattern_name,
                MechanismShort=row.get("MechanismShort", ""),
                MechanismDetail=row.get("MechanismDetail", ""),
                ConditionsText=row.get("ConditionsText", ""),
                ConditionsStructured=row.get("ConditionsStructured", ""),
                IdentificationFeatures=row.get("IdentificationFeatures", ""),
                TypicalCurve=row.get("TypicalCurve", ""),
                TypicalLossRate=row.get("TypicalLossRate", ""),
                LossBehavior=row.get("LossBehavior", ""),
                Source=row.get("Source", "")
            )
            tx.merge(rule_node, "Rule", "RuleID")
            tx.merge(Relationship(rule_node, "IS_MECHANISM_OF", concept_node))

            # ---- Core classification layers ----
            if isinstance(lost_type_name, str) and lost_type_name.strip():
                lt_node = Node("LostType", name=lost_type_name.strip())
                tx.merge(lt_node, "LostType", "name")
                tx.merge(Relationship(rule_node, "BELONGS_TO", lt_node))

            if isinstance(severity_name, str) and severity_name.strip():
                sev_node = Node("Severity", name=severity_name.strip())
                tx.merge(sev_node, "Severity", "name")
                tx.merge(Relationship(rule_node, "HAS_SEVERITY", sev_node))

            if isinstance(risk_level_name, str) and risk_level_name.strip():
                rl_node = Node("RiskLevel", name=risk_level_name.strip())
                tx.merge(rl_node, "RiskLevel", "name")
                tx.merge(Relationship(rule_node, "HAS_RISK", rl_node))

            if isinstance(time_pattern_name, str) and time_pattern_name.strip():
                tp_node = Node("TimePattern", name=time_pattern_name.strip())
                tx.merge(tp_node, "TimePattern", "name")
                tx.merge(Relationship(rule_node, "HAS_PATTERN", tp_node))

            # ---- Context layer: formations ----
            formation_list = self._safe_eval_list(row.get("ApplicableFormationType", ""))
            for f_name in formation_list:
                f_name = str(f_name).strip()
                if not f_name:
                    continue
                f_node = Node("FormationType", name=f_name)
                tx.merge(f_node, "FormationType", "name")
                tx.merge(Relationship(rule_node, "APPLICABLE_TO", f_node))

            # ---- Parameter layer ----
            params_from_key = self._safe_eval_params(row.get("KeyParameters", ""))
            params_from_cond = self._safe_eval_conditions(row.get("ConditionsStructured", ""))
            all_params = sorted(set(params_from_key + params_from_cond))
            for p_name in all_params:
                p_name = str(p_name).strip()
                if not p_name:
                    continue
                p_node = Node("Parameter", name=p_name)
                tx.merge(p_node, "Parameter", "name")
                tx.merge(Relationship(rule_node, "INVOLVES_PARAM", p_node))

            # ======================================================
            # Mechanism layer (FIXED: must be inside loop)
            # ======================================================
            mech_short = str(row.get("MechanismShort", "")).strip()
            mech_detail = str(row.get("MechanismDetail", "")).strip()
            if mech_short:
                mech_node = Node("Mechanism", name=mech_short)
                tx.merge(mech_node, "Mechanism", "name")
                tx.merge(Relationship(rule_node, "INVOLVES_MECHANISM", mech_node))
                if mech_detail:
                    tx.run(
                        """MATCH (m:Mechanism {name:$name})
                           SET m.detail = $detail""",
                        name=mech_short, detail=mech_detail
                    )

            # ======================================================
            # Evidence layer
            # ======================================================
            idf_list = self._safe_eval_list(row.get("IdentificationFeatures", ""))
            for feat in idf_list:
                feat = str(feat).strip()
                if not feat:
                    continue
                idf_node = Node("IdentificationFeature", name=feat)
                tx.merge(idf_node, "IdentificationFeature", "name")
                tx.merge(Relationship(rule_node, "IDENTIFIED_BY", idf_node))

            curve = str(row.get("TypicalCurve", "")).strip()
            if curve:
                curve_node = Node("TypicalCurve", name=curve)
                tx.merge(curve_node, "TypicalCurve", "name")
                tx.merge(Relationship(rule_node, "HAS_CURVE", curve_node))

            mm_list = self._safe_eval_list(row.get("MeasurementMethods", ""))
            for mm in mm_list:
                mm = str(mm).strip()
                if not mm:
                    continue
                mm_node = Node("MeasurementMethod", name=mm)
                tx.merge(mm_node, "MeasurementMethod", "name")
                tx.merge(Relationship(rule_node, "MEASURED_BY", mm_node))

            # ======================================================
            # Constraint layer
            # ======================================================
            logic_terms = self._safe_eval_logic_terms(row.get("ConditionsStructured", ""))
            for term in logic_terms:
                term = str(term).strip()
                if not term:
                    continue
                lt_node = Node("LogicTerm", name=term)
                tx.merge(lt_node, "LogicTerm", "name")
                tx.merge(Relationship(rule_node, "CONSTRAINED_BY", lt_node))

            # ======================================================
            # Extra context (optional but safe): FormationProperties / GeologicalFeatures
            # (字段不存在时 _safe_json/_safe_eval_list 会返回空)
            # ======================================================
            fp = self._safe_json(row.get("FormationProperties", ""), default={})
            if isinstance(fp, dict):
                for k, v in fp.items():
                    name = f"{k}:{v}"
                    fp_node = Node("FormationProperty", name=name, key=str(k), value=str(v))
                    tx.merge(fp_node, "FormationProperty", "name")
                    tx.merge(Relationship(rule_node, "HAS_FORMATION_PROPERTY", fp_node))

            gf_list = self._safe_eval_list(row.get("GeologicalFeatures", ""))
            for gf in gf_list:
                gf = str(gf).strip()
                if not gf:
                    continue
                gf_node = Node("GeologicalFeature", name=gf)
                tx.merge(gf_node, "GeologicalFeature", "name")
                tx.merge(Relationship(rule_node, "HAS_GEO_FEATURE", gf_node))

            # ======================================================
            # Action layer (keyword-based, keeps original logic)
            # ======================================================
            text_blob = " ".join([
                str(row.get("LossBehavior", "")),
                str(row.get("MechanismDetail", "")),
                str(row.get("ConditionsText", "")),
            ]).strip()

            for act, kws in MechanismKGAgent.ACTION_KEYWORDS.items():
                if any(str(kw).lower() in text_blob.lower() for kw in kws):
                    act_node = Node("Action", name=act)
                    tx.merge(act_node, "Action", "name")
                    tx.merge(Relationship(rule_node, "RECOMMENDS_ACTION", act_node))

        tx.commit()
        print(f"[MechanismKG] 图谱构建完成，共写入规则：{len(df)}")

    def visualize_by_lost_type(self,
                               lost_type="裂缝性漏失",
                               layout_mode="A",
                               save_path=None,
                               return_buffer=False):
        """
        参数
        ----
        lost_type : str
            例如 "裂缝性漏失"、"渗透性漏失"、"诱导裂缝漏失" 等
        layout_mode : {"A","B","C"}
            A: 放射型（你当前用的方案）
            B: 社区簇状布局（Louvain）
            C: 左右分层布局（Parameter -> Rule -> LostType/Severity/Risk）
        save_path : str or None
            若不为 None，则保存为 PNG 文件
        return_buffer : bool
            若 True，返回 BytesIO，可直接给 Gradio 的 Image 使用

        返回
        ----
        BytesIO 或 None
        """
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False

        # ---------- 1）从 Neo4j 查询该 LostType 对应子图 ----------
        query = """
        MATCH (lt:LostType {name: $lost_type})
              <-[:BELONGS_TO]-(r:Rule)
        OPTIONAL MATCH (r)-[:INVOLVES_PARAM]->(p:Parameter)
        OPTIONAL MATCH (r)-[:HAS_SEVERITY]->(s:Severity)
        OPTIONAL MATCH (r)-[:HAS_RISK]->(rk:RiskLevel)
        OPTIONAL MATCH (r)-[:HAS_PATTERN]->(tp:TimePattern)
        RETURN lt, r, collect(DISTINCT p) as params,
               collect(DISTINCT s) as sevs,
               collect(DISTINCT rk) as risks,
               collect(DISTINCT tp) as patterns
        """
        result = self.graph.run(query, lost_type=lost_type).data()
        if not result:
            print(f"[MechanismKG] 未找到 LostType = {lost_type} 的规则。")
            return None

        # ---------- 2）构造 NetworkX 图 ----------
        G = nx.DiGraph()

        lt_node = result[0]["lt"]
        lt_name = lt_node["name"]
        lt_label = lt_name

        # 中心节点：LostType
        G.add_node(lt_label, type="LostType", color="#E74C3C", size=2600)

        rule_nodes = []
        severity_nodes = set()
        risk_nodes = set()
        pattern_nodes = set()
        param_nodes = set()

        for record in result:
            r = record["r"]
            if not r:
                continue
            rule_id = r.get("RuleID", "")
            rule_name = r.get("name", "")
            rule_label = f"{rule_id}\n{textwrap.fill(rule_name, 8)}"

            G.add_node(rule_label, type="Rule", color="#3498DB", size=2000)
            G.add_edge(rule_label, lt_label, label="BELONGS_TO")
            rule_nodes.append(rule_label)

            # Severity
            for s in record.get("sevs", []):
                if not s:
                    continue
                s_label = f"Severity:{s['name']}"
                severity_nodes.add(s_label)
                G.add_node(s_label, type="Severity", color="#F5B041", size=1500)
                G.add_edge(rule_label, s_label, label="HAS_SEVERITY")

            # RiskLevel
            for rk in record.get("risks", []):
                if not rk:
                    continue
                rk_label = f"Risk:{rk['name']}"
                risk_nodes.add(rk_label)
                G.add_node(rk_label, type="RiskLevel", color="#C0392B", size=1500)
                G.add_edge(rule_label, rk_label, label="HAS_RISK")

            # TimePattern
            for tp in record.get("patterns", []):
                if not tp:
                    continue
                tp_label = f"Pattern:{tp['name']}"
                pattern_nodes.add(tp_label)
                G.add_node(tp_label, type="TimePattern", color="#58D68D", size=1500)
                G.add_edge(rule_label, tp_label, label="HAS_PATTERN")

            # Parameters
            for p in record.get("params", []):
                if not p:
                    continue
                p_label = f"Param:{p['name']}"
                param_nodes.add(p_label)
                if p_label not in G:
                    G.add_node(p_label, type="Parameter", color="#F9E79F", size=1400)
                G.add_edge(rule_label, p_label, label="INVOLVES_PARAM")

        # ---------- 3）根据 layout_mode 选择布局 ----------
        if layout_mode.upper() == "A":
            pos = self._layout_radial(
                G, lt_label, rule_nodes,
                list(severity_nodes), list(risk_nodes),
                list(pattern_nodes), list(param_nodes)
            )
            title = f"井漏机理知识图谱子图（方案A-放射状）：{lost_type}"

        elif layout_mode.upper() == "B":
            pos = self._layout_community(G)
            title = f"井漏机理知识图谱子图（方案B-社区簇状）：{lost_type}"

        elif layout_mode.upper() == "C":
            pos = self._layout_hierarchical(G)
            title = f"井漏机理知识图谱子图（方案C-左右分层）：{lost_type}"

        else:
            print(f"[MechanismKG] 未知布局模式 {layout_mode}，回退为 A。")
            pos = self._layout_radial(
                G, lt_label, rule_nodes,
                list(severity_nodes), list(risk_nodes),
                list(pattern_nodes), list(param_nodes)
            )
            title = f"井漏机理知识图谱子图（方案A）：{lost_type}"

        # ---------- 4）绘制图像（参考 MLR KG 的论文版风格） ----------
        import matplotlib
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        import io

        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Times New Roman']
        matplotlib.rcParams['font.family'] = ['sans-serif']
        matplotlib.rcParams['axes.unicode_minus'] = False

        fig, ax = plt.subplots(figsize=(14, 9), dpi=300)
        ax.set_facecolor("white")

        # ====== A) 先画边：必须“淡+细”，否则边框永远看不见 ======
        nx.draw_networkx_edges(
            G, pos,
            ax=ax,
            edge_color="#222222",
            width=0.7,
            alpha=0.22,  # ✅ 关键：降低边的存在感
            arrows=True,
            arrowsize=8,
            connectionstyle="arc3,rad=0.06",
            min_source_margin=6,
            min_target_margin=6,
            zorder=1
        )

        # ====== B) 节点：两层描边（外黑内彩），保证任何缩放都能看到边框 ======
        type_styles = {
            "LostType": {"edge": "#1B4F72", "lw": 2.6},
            "Rule": {"edge": "#6E2C00", "lw": 1.8},
            "Parameter": {"edge": "#7E5109", "lw": 1.6},
            "Action": {"edge": "#7D6608", "lw": 1.6},
            "RiskLevel": {"edge": "#922B21", "lw": 1.6},
            "Severity": {"edge": "#7D3C98", "lw": 1.6},
            "FormationType": {"edge": "#1F618D", "lw": 1.5},
            "Mechanism": {"edge": "#7B241C", "lw": 1.5},
            "IdentificationFeature": {"edge": "#1E8449", "lw": 1.4},
            "TypicalCurve": {"edge": "#117A65", "lw": 1.4},
            "LogicTerm": {"edge": "#5B2C6F", "lw": 1.4},
            "TimePattern": {"edge": "#145A32", "lw": 1.4},
        }

        for t, style in type_styles.items():
            nodes = [n for n, d in G.nodes(data=True) if d.get("type") == t]
            if not nodes:
                continue

            sizes = [G.nodes[n].get("size", 1200) for n in nodes]
            colors = [G.nodes[n].get("color", "#D5D8DC") for n in nodes]

            # 外黑描边（稍大一点点）
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=nodes,
                ax=ax,
                node_color="none",
                node_size=[s * 1.06 for s in sizes],
                edgecolors="black",
                linewidths=style["lw"] + 1.2,
                alpha=1.0,
                zorder=2
            )

            # 内彩描边 + 填充
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=nodes,
                ax=ax,
                node_color=colors,
                node_size=sizes,
                edgecolors=style["edge"],
                linewidths=style["lw"],
                alpha=1.0,  # ✅ 关键：密集节点不要透明
                zorder=3
            )

        # ====== C) 标签：Rule 太密时只显示 Top-K（否则糊） ======
        def short_label(n):
            # 规则节点只显示编号，其他显示冒号后内容
            if n.startswith("Rule:"):
                return n.replace("Rule:", "")
            if ":" in n:
                return n.split(":", 1)[-1]
            return n

        labels = {n: short_label(n) for n in G.nodes()}

        # Rule 超过 30 就只标 Top-30（按度数）
        rule_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "Rule"]
        if len(rule_nodes) > 30:
            deg = dict(G.degree(rule_nodes))
            top_rule = set(sorted(rule_nodes, key=lambda x: deg.get(x, 0), reverse=True)[:30])
            labels = {n: lab for n, lab in labels.items() if (n not in rule_nodes) or (n in top_rule)}

        nx.draw_networkx_labels(
            G, pos,
            labels=labels,
            ax=ax,
            font_size=9,
            font_family="SimHei",
            font_color="#111111",
            verticalalignment="center",
            horizontalalignment="center",
            zorder=4
        )

        # ====== D) edge label：默认关（密集图谱开启必糊），做成开关 ======
        show_edge_labels = False
        if show_edge_labels:
            edge_labels = nx.get_edge_attributes(G, "label")
            nx.draw_networkx_edge_labels(
                G, pos,
                edge_labels=edge_labels,
                ax=ax,
                font_size=7,
                font_color="#2C3E50",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7)
            )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axis("off")
        fig.tight_layout()

        buf = None
        if save_path:
            dir_ = os.path.dirname(save_path)
            if dir_:
                os.makedirs(dir_, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"[MechanismKG] 图谱已保存: {save_path}")

        if return_buffer:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", facecolor="white")
            buf.seek(0)

        plt.close(fig)
        return buf

    # ---------- A：放射型布局 ----------
    def _layout_radial(
            self,
            G,
            lt_label=None,
            rule_nodes=None,
            severity_nodes=None,
            risk_nodes=None,
            pattern_nodes=None,
            param_nodes=None
    ):
        """
        方案A：环状/多层投影布局（Radial/Ring）
        允许只传入 G；其余参数缺省时从 G 自动推断。
        """

        # ---------- 1) 自动推断各类节点（尽量兼容不同字段命名） ----------
        def get_type(n, data):
            # 你项目里可能叫 node_type/type/kind/category/label 等，做个兜底
            for k in ("node_type", "type", "kind", "category", "role", "label"):
                if k in data and data[k] is not None:
                    return str(data[k]).lower()
            # 也可能把类型写在节点名里
            return str(n).lower()

        if rule_nodes is None:
            rule_nodes = [n for n, d in G.nodes(data=True) if "rule" in get_type(n, d) or str(n).startswith("MR_")]
        if severity_nodes is None:
            severity_nodes = [n for n, d in G.nodes(data=True) if "severity" in get_type(n, d) or "程度" in str(n)]
        if risk_nodes is None:
            risk_nodes = [n for n, d in G.nodes(data=True) if
                          "risk" in get_type(n, d) or "风险" in str(n) or "mlr" in str(n).lower()]
        if pattern_nodes is None:
            pattern_nodes = [n for n, d in G.nodes(data=True) if
                             "pattern" in get_type(n, d) or "征兆" in str(n) or "表现" in str(n)]
        if param_nodes is None:
            param_nodes = [n for n, d in G.nodes(data=True) if "param" in get_type(n, d) or "参数" in str(n)]

        # LostType 中心节点：优先用 lt_label；否则在图里找 “losttype/漏失类型/裂缝性/渗透性/诱导裂缝…” 之类
        if lt_label is None:
            cand = []
            for n, d in G.nodes(data=True):
                t = get_type(n, d)
                name = str(n)
                if "losttype" in t or "漏失类型" in name or "漏失" in name:
                    cand.append(n)
            lt_label = cand[0] if cand else (next(iter(G.nodes)) if len(G) else None)

        # ---------- 2) 生成环状坐标（示例：中心 + 多圈） ----------
        import numpy as np

        pos = {}
        if lt_label is not None and lt_label in G:
            pos[lt_label] = (0.0, 0.0)

        def place_on_ring(nodes, r, start_angle=0.0):
            if not nodes:
                return
            k = len(nodes)
            angles = np.linspace(0, 2 * np.pi, k, endpoint=False) + start_angle
            for i, n in enumerate(nodes):
                pos[n] = (r * np.cos(angles[i]), r * np.sin(angles[i]))

        # 你可以按你想要的圈层顺序调整半径
        place_on_ring(rule_nodes, r=1.0, start_angle=0.0)
        place_on_ring(pattern_nodes, r=1.8, start_angle=0.3)
        place_on_ring(param_nodes, r=2.6, start_angle=0.6)
        place_on_ring(risk_nodes, r=3.4, start_angle=0.9)
        place_on_ring(severity_nodes, r=4.2, start_angle=1.2)

        # 对于未被分组的节点，丢到一个外圈避免 KeyError
        remaining = [n for n in G.nodes() if n not in pos]
        place_on_ring(remaining, r=5.0, start_angle=0.2)

        return pos

    """
    def _layout_radial(self, G, lt_label, rule_nodes, severity_nodes, risk_nodes,
                       pattern_nodes, param_nodes):

        #方案A：中心 LostType，规则环 + 各类节点半环（和你现在的图类似）

        import math
        pos = {}

        # 中心
        pos[lt_label] = (0.0, 0.0)

        # 规则：内圈圆环
        n_rule = max(len(rule_nodes), 1)
        r_rule = 2.2
        for i, n in enumerate(rule_nodes):
            angle = 2 * math.pi * i / n_rule
            pos[n] = (r_rule * math.cos(angle), r_rule * math.sin(angle))

        # 几个外圈半环：上/右/左/下
        def place_arc(nodes, r, start_deg, end_deg):
            if not nodes:
                return
            m = len(nodes)
            start = math.radians(start_deg)
            end = math.radians(end_deg)
            for i, n in enumerate(nodes):
                if m == 1:
                    angle = (start + end) / 2
                else:
                    angle = start + (end - start) * i / (m - 1)
                pos[n] = (r * math.cos(angle), r * math.sin(angle))

        # Pattern：顶部
        place_arc(pattern_nodes, r=4.0, start_deg=60, end_deg=120)
        # Parameter：右侧
        place_arc(param_nodes, r=4.0, start_deg=-30, end_deg=30)
        # Severity：左侧
        place_arc(severity_nodes, r=4.0, start_deg=150, end_deg=210)
        # Risk：底部
        place_arc(risk_nodes, r=4.3, start_deg=230, end_deg=310)

        # 对于未被放置的节点（理论上不该有），退回 spring_layout
        left_nodes = [n for n in G.nodes() if n not in pos]
        if left_nodes:
            sub_pos = nx.spring_layout(G.subgraph(left_nodes), k=0.8, iterations=50)
            pos.update(sub_pos)

        return pos
"""

    # ---------- B：社区簇状布局 ----------

    """
    def _layout_community(self, G):

        #方案B：基于 Louvain 的社区检测 + spring 布局
        #不依赖节点类型，由结构自动形成簇。

        if not HAS_LOUVAIN:
            print("[MechanismKG] 未安装 python-louvain，社区布局退回 spring_layout。")
            return nx.spring_layout(G, k=1.0, iterations=100)

        undirected = G.to_undirected()
        partition = community_louvain.best_partition(undirected)

        # 根据社区给每个节点一个“偏移”，让社区之间稍微分开
        pos_base = nx.spring_layout(undirected, k=0.9, iterations=120)
        # 可以在这里根据 partition 做一点 cluster 拉远，这里先简单返回
        for n in G.nodes():
            G.nodes[n]["community"] = partition.get(n, 0)

        return pos_base
"""

    def _layout_community(self, G, seed: int = 7,
                          inter_radius: float = 8.0,
                          intra_k: float = 0.45,
                          intra_iter: int = 120):
        """
        方案B：社区簇状布局（Louvain + 社区中心偏移 + 社区内spring）
        - inter_radius: 社区中心分布半径（越大簇之间越分开）
        - intra_k: 社区内部 spring 的 k（越小簇越紧凑）
        """
        import networkx as nx
        import math
        from collections import defaultdict

        if not HAS_LOUVAIN:
            print("[MechanismKG] 未安装 python-louvain，社区布局退回 spring_layout。")
            return nx.spring_layout(G, k=1.0, iterations=150, seed=seed)

        U = G.to_undirected()
        partition = community_louvain.best_partition(U, random_state=seed)

        # community -> nodes
        comm2nodes = defaultdict(list)
        for n, c in partition.items():
            comm2nodes[c].append(n)

        # 给每个社区一个中心（环形/圆形分布）
        comm_ids = sorted(comm2nodes.keys())
        m = max(len(comm_ids), 1)
        comm_center = {}
        for i, cid in enumerate(comm_ids):
            ang = 2 * math.pi * i / m
            comm_center[cid] = (inter_radius * math.cos(ang), inter_radius * math.sin(ang))

        pos = {}

        # 社区内部做 spring_layout，然后整体平移到社区中心
        for cid, nodes in comm2nodes.items():
            sub = U.subgraph(nodes)

            # 单节点社区直接放中心
            if sub.number_of_nodes() <= 1:
                n0 = nodes[0]
                pos[n0] = comm_center[cid]
                continue

            sub_pos = nx.spring_layout(sub, k=intra_k, iterations=intra_iter, seed=seed)

            cx, cy = comm_center[cid]
            for n in nodes:
                x, y = sub_pos[n]
                pos[n] = (x + cx, y + cy)
        # 写回 community 属性（方便上色或图例）
        for n in G.nodes():
            G.nodes[n]["community"] = partition.get(n, 0)
        # 兜底：万一有节点没被分到（极少见）
        for n in G.nodes():
            if n not in pos:
                pos[n] = (0.0, 0.0)
        return pos

    # ---------- C：左右分层布局 ----------
    def _layout_hierarchical(self, G):
        """
        方案C：手动按类型分层：
        Parameter(0) -> Rule(1) -> LostType/Severity/Risk/Pattern(2/3)
        """
        # 按类型分簇
        layers = {
            "Parameter": [],
            "Rule": [],
            "LostType": [],
            "Severity": [],
            "RiskLevel": [],
            "TimePattern": []
        }
        for n, data in G.nodes(data=True):
            t = data.get("type", "Rule")
            layers.setdefault(t, []).append(n)

        pos = {}
        # 各层的 x 坐标
        x_map = {
            "Parameter": 0.0,
            "Rule": 2.0,
            "LostType": 4.0,
            "Severity": 4.0,
            "RiskLevel": 4.0,
            "TimePattern": 4.0
        }

        # 为每一类在 y 方向均匀排布
        def place_layer(nodes, x):
            if not nodes:
                return
            n = len(nodes)
            ys = list(range(n))
            # 居中
            offset = (n - 1) / 2.0
            for i, node in enumerate(sorted(nodes)):
                pos[node] = (x, (ys[i] - offset) * 0.8)

        place_layer(layers["Parameter"], x_map["Parameter"])
        place_layer(layers["Rule"], x_map["Rule"])

        # 将几个“结果类”拼到右侧
        combined_right = layers["LostType"] + layers["Severity"] + layers["RiskLevel"] + layers["TimePattern"]
        place_layer(combined_right, x_map["LostType"])

        # 仍有缺失节点则用 spring_layout 兜底
        left_nodes = [n for n in G.nodes() if n not in pos]
        if left_nodes:
            sub_pos = nx.spring_layout(G.subgraph(left_nodes), k=0.8, iterations=50)
            pos.update(sub_pos)

        return pos

    # ======================================================
    # 2.1 兼容 main.py 的 B / C 包装函数
    # ======================================================
    def visualize_by_lost_type_compact(self,
                                       lost_type="裂缝性漏失",
                                       save_path=None,
                                       return_buffer=False):
        """
        方案B：紧凑聚类布局（社区簇状）
        只是对 visualize_by_lost_type 的一个封装，强制 layout_mode="B"
        """
        return self.visualize_by_lost_type(
            lost_type=lost_type,
            layout_mode="B",
            save_path=save_path,
            return_buffer=return_buffer
        )

    def visualize_layered(self,
                          lost_type="裂缝性漏失",
                          save_path=None,
                          return_buffer=False):
        """
        方案C：左右分层布局
        同样封装到 visualize_by_lost_type，强制 layout_mode="C"
        """
        return self.visualize_by_lost_type(
            lost_type=lost_type,
            layout_mode="C",
            save_path=save_path,
            return_buffer=return_buffer
        )

    # ======================================================
    # 3️⃣ 交互式机理知识图谱（方案 D）：pyvis + HTML
    # ======================================================
    def visualize_interactive(self,
                              lost_type="裂缝性漏失",
                              html_path=None,
                              return_html=False):
        """使用 pyvis 生成交互式机理知识图谱（更清爽 + 悬停展示完整信息 + 自动折叠）。

        - 保持原函数签名不变，避免 main 调用异常
        - 节点 label 仅显示短文本；完整内容放入 title（hover tooltip）
        - 支持自动 clustering 折叠 Param/Evidence/Logic，减少凌乱
        - 支持推理路径高亮：如需高亮，可在调用前设置 self._highlight_rule_ids = set([...])
        """
        highlight_rule_ids = set(getattr(self, "_highlight_rule_ids", set()) or [])

        net = Network(
            height="650px",
            width="100%",
            bgcolor="#ffffff",
            font_color="#2C3E50",
            notebook=False,  # ✅ 一定要 False
            directed=True
        )

        # 物理参数：更“高级”的布局观感
        net.set_options("""
        var options = {
          "interaction": {"hover": true, "tooltipDelay": 120, "hideEdgesOnDrag": true},
          "physics": {
            "enabled": true,
            "stabilization": {"iterations": 180},
            "barnesHut": {"gravitationalConstant": -9000, "springLength": 150, "springConstant": 0.03, "damping": 0.35}
          },
          "edges": {
            "smooth": {"type":"dynamic"},
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.7}},
            "color": {"inherit": false, "color":"#777777"},
            "font": {"size": 10, "align": "middle"}
          },
          "nodes": {
            "font": {"size": 14, "face": "Microsoft YaHei"},
            "borderWidth": 1,
            "shadow": {"enabled": true, "size": 6}
          }
        }
        """)

        # ===== 从 Neo4j 查询 LostType 子图（保持你原来的查询思路，扩展更多层） =====
        query = """
        MATCH (lt:LostType {name: $lost_type})<-[:BELONGS_TO]-(r:Rule)
        WITH lt, r
        LIMIT 120
        OPTIONAL MATCH (r)-[:INVOLVES_PARAM]->(p:Parameter)
        OPTIONAL MATCH (r)-[:HAS_SEVERITY]->(s:Severity)
        OPTIONAL MATCH (r)-[:HAS_RISK]->(rk:RiskLevel)
        OPTIONAL MATCH (r)-[:HAS_PATTERN]->(tp:TimePattern)
        OPTIONAL MATCH (r)-[:INVOLVES_MECHANISM]->(m:Mechanism)
        OPTIONAL MATCH (r)-[:CONSTRAINED_BY]->(lg:LogicTerm)
        OPTIONAL MATCH (r)-[:IDENTIFIED_BY]->(idf:IdentificationFeature)
        OPTIONAL MATCH (r)-[:HAS_CURVE]->(cv:TypicalCurve)
        OPTIONAL MATCH (r)-[:RECOMMENDS_ACTION]->(ac:Action)
        RETURN lt, r,
               collect(DISTINCT p)  as params,
               collect(DISTINCT s)  as sevs,
               collect(DISTINCT rk) as risks,
               collect(DISTINCT tp) as patterns,
               collect(DISTINCT m)  as mechs,
               collect(DISTINCT lg) as logics,
               collect(DISTINCT idf) as idfs,
               collect(DISTINCT cv)  as curves,
               collect(DISTINCT ac)  as actions
        """
        data = self.graph.run(query, lost_type=lost_type).data()

        if not data:
            msg = f"未查询到漏失类型：{lost_type}"
            if return_html:
                return f"<div style='color:red;font-size:14px;'>{msg}</div>"
            return None

        # center node
        lt_name = data[0]["lt"]["name"]
        center_id = f"LT::{lt_name}"
        net.add_node(center_id, label=lt_name, color="#E74C3C", shape="ellipse", size=46,
                     title=f"<b>LostType</b>: {lt_name}")

        def _short(text, n=16):
            text = str(text or "").strip()
            return text if len(text) <= n else text[:n] + "…"

        def _add_node(nid, label, title, color, shape="dot", size=18, group=None, dim=False):
            if dim:
                net.add_node(nid, label=label, title=title, color="#E0E0E0", shape=shape, size=size, group=group)
            else:
                net.add_node(nid, label=label, title=title, color=color, shape=shape, size=size, group=group)

        def _add_edge(u, v, label, hi=False):
            if hi:
                net.add_edge(u, v, label=label, color="#111111", width=3)
            else:
                net.add_edge(u, v, label=label, color="#777777", width=1)

        for rec in data:
            r = rec.get("r")
            if not r:
                continue

            rule_id = r.get("RuleID", "")
            rule_name = r.get("name", "") or r.get("RuleName", "")
            mech_short = r.get("MechanismShort", "") or ""
            cond_text = r.get("ConditionsText", "") or ""
            risk = r.get("RiskLevel", "")
            sev = r.get("SeverityLevel", "")

            rid = f"R::{rule_id}"
            is_hi = (rule_id in highlight_rule_ids) if highlight_rule_ids else False
            dim_rule = bool(highlight_rule_ids) and (not is_hi)

            rule_title = (
                f"<b>{rule_id}</b> — {_short(rule_name, 80)}<br>"
                f"<b>Mechanism</b>: {_short(mech_short, 120)}<br>"
                f"<b>Condition</b>: {_short(cond_text, 180)}<br>"
                f"<b>Risk/Severity</b>: {risk} / {sev}"
            )
            _add_node(rid, label=str(rule_id), title=rule_title, color="#3498DB",
                      shape="box", size=34, group="Rule", dim=dim_rule)
            _add_edge(rid, center_id, "BELONGS_TO", hi=is_hi)

            # Mechanism
            for m in rec.get("mechs", []) or []:
                if not m:
                    continue
                mname = m.get("name", "")
                mid = f"MECH::{mname}"
                mlabel = _short(mname, 14)
                mdetail = m.get("detail", "") or ""
                mtitle = f"<b>Mechanism</b>: {mname}<br>{_short(mdetail, 260)}"
                _add_node(mid, mlabel, mtitle, "#F1948A", "ellipse", 22, "Mechanism", dim=dim_rule)
                _add_edge(rid, mid, "INVOLVES_MECHANISM", hi=is_hi)

            # Severity / Risk / Pattern
            for s in rec.get("sevs", []) or []:
                if not s:
                    continue
                sid = f"SEV::{s.get('name')}"
                _add_node(sid, s.get("name", ""), f"<b>Severity</b>: {s.get('name', '')}",
                          "#F5B041", "dot", 18, "Severity", dim=dim_rule)
                _add_edge(rid, sid, "HAS_SEVERITY", hi=is_hi)

            for rk in rec.get("risks", []) or []:
                if not rk:
                    continue
                rkid = f"RISK::{rk.get('name')}"
                _add_node(rkid, rk.get("name", ""), f"<b>RiskLevel</b>: {rk.get('name', '')}",
                          "#C0392B", "dot", 20, "Risk", dim=dim_rule)
                _add_edge(rid, rkid, "HAS_RISK", hi=is_hi)

            for tp in rec.get("patterns", []) or []:
                if not tp:
                    continue
                tpid = f"TP::{tp.get('name')}"
                _add_node(tpid, _short(tp.get("name", ""), 10), f"<b>TimePattern</b>: {tp.get('name', '')}",
                          "#58D68D", "dot", 18, "Pattern", dim=dim_rule)
                _add_edge(rid, tpid, "HAS_PATTERN", hi=is_hi)

            # Parameters (group=Param for clustering)
            for p in rec.get("params", []) or []:
                if not p:
                    continue
                pname = p.get("name", "")
                pid = f"PARAM::{pname}"
                _add_node(pid, pname, f"<b>Parameter</b>: {pname}", "#F9E79F", "dot", 14, "Param", dim=dim_rule)
                _add_edge(rid, pid, "INVOLVES_PARAM", hi=is_hi)

            # Logic / Evidence / Actions (grouped for clustering)
            for lg in rec.get("logics", []) or []:
                if not lg:
                    continue
                lname = lg.get("name", "")
                lid = f"LOGIC::{lname}"
                _add_node(lid, _short(lname, 14), f"<b>LogicTerm</b>: {lname}", "#D7BDE2", "dot", 14, "Logic",
                          dim=dim_rule)
                _add_edge(rid, lid, "CONSTRAINED_BY", hi=is_hi)

            for idf in rec.get("idfs", []) or []:
                if not idf:
                    continue
                fname = idf.get("name", "")
                fid = f"OBS::{fname}"
                _add_node(fid, _short(fname, 12), f"<b>Identification</b>: {fname}", "#7DCEA0", "dot", 13, "Evidence",
                          dim=dim_rule)
                _add_edge(rid, fid, "IDENTIFIED_BY", hi=is_hi)

            for cv in rec.get("curves", []) or []:
                if not cv:
                    continue
                cname = cv.get("name", "")
                cid = f"CURVE::{cname}"
                _add_node(cid, _short(cname, 12), f"<b>TypicalCurve</b>: {cname}", "#48C9B0", "dot", 13, "Evidence",
                          dim=dim_rule)
                _add_edge(rid, cid, "HAS_CURVE", hi=is_hi)

            for ac in rec.get("actions", []) or []:
                if not ac:
                    continue
                aname = ac.get("name", "")
                aid = f"ACT::{aname}"
                _add_node(aid, aname, f"<b>Action</b>: {aname}", "#F7DC6F", "diamond", 16, "Action", dim=dim_rule)
                _add_edge(rid, aid, "RECOMMENDS_ACTION", hi=is_hi)

        # --- clustering via JS injection (collapse Param/Evidence/Logic) ---
        html = net.generate_html(notebook=False)
        cluster_js = """
        <script type="text/javascript">
        function doClustering() {
          try{
            var groupsToCluster = ["Param","Evidence","Logic"];
            groupsToCluster.forEach(function(g){
              network.cluster({
                joinCondition: function(nodeOptions){
                  return nodeOptions.group === g;
                },
                clusterNodeProperties: {
                  id: "CLUSTER_" + g,
                  label: g.toUpperCase() + "s",
                  borderWidth: 2,
                  shape: "database",
                  color: "#DDDDDD"
                }
              });
            });
          }catch(e){}
        }
        network.once("stabilized", doClustering);
        </script>
        """
        html = html.replace("</body>", cluster_js + "\n</body>")

        if return_html:
            return html

        if html_path:
            dir_ = os.path.dirname(html_path)
            if dir_:
                os.makedirs(dir_, exist_ok=True)
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html)
            return html_path

        return html

    def visualize_multiview(self,
                            lost_type="裂缝性漏失",
                            view="full",
                            layout_mode="LAYER",
                            save_path=None,
                            return_buffer=False,
                            interactive=False,
                            return_html=False,
                            limit_rules: int = 150,
                            dpi: int = 300):
        """多维立体子图：同一底层图谱，多视角投影"""
        view = (view or "full").lower()
        layout_mode = (layout_mode or "LAYER").upper()

        cypher = """
        MATCH (lt:LostType {name:$lost_type})<-[:BELONGS_TO]-(r:Rule)
        WITH lt,r
        LIMIT $limit_rules
        OPTIONAL MATCH (r)-[:HAS_SEVERITY]->(sv:Severity)
        OPTIONAL MATCH (r)-[:HAS_RISK]->(rk:RiskLevel)
        OPTIONAL MATCH (r)-[:HAS_PATTERN]->(tp:TimePattern)
        OPTIONAL MATCH (r)-[:APPLICABLE_TO]->(fm:FormationType)
        OPTIONAL MATCH (r)-[:INVOLVES_PARAM]->(p:Parameter)
        OPTIONAL MATCH (r)-[:INVOLVES_MECHANISM]->(m:Mechanism)
        OPTIONAL MATCH (r)-[:IDENTIFIED_BY]->(idf:IdentificationFeature)
        OPTIONAL MATCH (r)-[:HAS_CURVE]->(cv:TypicalCurve)
        OPTIONAL MATCH (r)-[:CONSTRAINED_BY]->(lg:LogicTerm)
        OPTIONAL MATCH (r)-[:RECOMMENDS_ACTION]->(ac:Action)
        RETURN lt,
                r,
                collect(DISTINCT sv) AS severities,
                collect(DISTINCT rk) AS risks,
                collect(DISTINCT tp) AS patterns,
                collect(DISTINCT fm) AS formations,
                collect(DISTINCT p) AS params,
                collect(DISTINCT m) AS mechs,
                collect(DISTINCT idf) AS idfs,
                collect(DISTINCT cv) AS curves,
                collect(DISTINCT lg) AS logics,
                collect(DISTINCT ac) AS actions
        """
        records = list(self.graph.run(cypher, lost_type=lost_type, limit_rules=limit_rules))

        # ---------- 1) 构图 ----------
        G = nx.DiGraph()
        lt_label = f"LostType:{lost_type}"
        G.add_node(lt_label, type="LostType", color="#2E86C1", size=2600)

        rule_nodes = set()

        for rec in records:
            r = rec["r"]
            rid = r.get("RuleID", "")
            rname = r.get("RuleName", "")
            rule_label = f"Rule:{rid}"
            rule_nodes.add(rule_label)
            G.add_node(rule_label, type="Rule", color="#F5B041", size=2200,
                       title=self._wrap(rname, width=22, max_lines=6))
            G.add_edge(rule_label, lt_label, label="BELONGS_TO")

            def _add_list(node_list, prefix, ntype, color, rel):
                out = []
                for n in node_list or []:
                    if not n:
                        continue
                    name = n.get("name", "")
                    if not name:
                        continue
                    lab = f"{prefix}:{name}"
                    if lab not in G:
                        G.add_node(lab, type=ntype, color=color, size=1600)
                    G.add_edge(rule_label, lab, label=rel)
                    out.append(lab)
                return out

            # 基础维度
            severities = _add_list(rec["severities"], "Severity", "Severity", "#AF7AC5", "HAS_SEVERITY")
            risks = _add_list(rec["risks"], "Risk", "RiskLevel", "#C0392B", "HAS_RISK")
            patterns = _add_list(rec["patterns"], "Pattern", "TimePattern", "#58D68D", "HAS_PATTERN")

            # 多维立体：formation/param/mechanism/evidence/constraint/action
            formations = _add_list(rec["formations"], "Form", "FormationType", "#85C1E9", "APPLICABLE_TO")
            params = _add_list(rec["params"], "Param", "Parameter", "#F9E79F", "INVOLVES_PARAM")
            mechs = _add_list(rec["mechs"], "Mech", "Mechanism", "#F1948A", "INVOLVES_MECHANISM")
            idfs = _add_list(rec["idfs"], "Obs", "IdentificationFeature", "#7DCEA0", "IDENTIFIED_BY")
            curves = _add_list(rec["curves"], "Curve", "TypicalCurve", "#48C9B0", "HAS_CURVE")
            logics = _add_list(rec["logics"], "Logic", "LogicTerm", "#D7BDE2", "CONSTRAINED_BY")
            actions = _add_list(rec["actions"], "Action", "Action", "#F7DC6F", "RECOMMENDS_ACTION")

            # 按 view 做投影（删掉不需要的类型）
            keep_types = {
                "full": {"LostType", "Rule", "Severity", "RiskLevel", "TimePattern", "FormationType", "Parameter",
                         "Mechanism", "IdentificationFeature", "TypicalCurve", "LogicTerm", "Action"},
                "mechanism": {"LostType", "Rule", "Mechanism", "FormationType", "Severity", "RiskLevel"},
                "evidence": {"LostType", "Rule", "IdentificationFeature", "TypicalCurve", "Mechanism"},
                "constraint": {"LostType", "Rule", "LogicTerm", "Mechanism", "Parameter"},
                "action": {"LostType", "Rule", "Action", "Mechanism", "Severity", "RiskLevel"},
            }.get(view, None)

            if keep_types:
                # 延迟到循环后统一清理（避免影响当前 rec 的添加）
                pass

        # 统一按 view 清理
        keep_types = {
            "full": {"LostType", "Rule", "Severity", "RiskLevel", "TimePattern", "FormationType", "Parameter",
                     "Mechanism", "IdentificationFeature", "TypicalCurve", "LogicTerm", "Action"},
            "mechanism": {"LostType", "Rule", "Mechanism", "FormationType", "Severity", "RiskLevel"},
            "evidence": {"LostType", "Rule", "IdentificationFeature", "TypicalCurve", "Mechanism"},
            "constraint": {"LostType", "Rule", "LogicTerm", "Mechanism", "Parameter"},
            "action": {"LostType", "Rule", "Action", "Mechanism", "Severity", "RiskLevel"},
        }.get(view, {"LostType", "Rule", "Mechanism"})

        drop_nodes = [n for n, d in G.nodes(data=True) if d.get("type") not in keep_types]
        G.remove_nodes_from(drop_nodes)

        # ---------- 2) 布局 ----------
        if interactive:
            net = Network(height="650px", width="100%", directed=True, bgcolor="#FFFFFF", font_color="#111111")
            net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=140, spring_strength=0.05,
                                   damping=0.6)

            for n, d in G.nodes(data=True):
                net.add_node(
                    n,
                    label=self._wrap(n.replace("Rule:", "").replace("LostType:", ""), 18, 4),
                    title=d.get("title", ""),
                    color=d.get("color", "#999999"),
                    size=max(8, int(d.get("size", 1400) / 120))
                )

            for u, v, ed in G.edges(data=True):
                net.add_edge(u, v, label=ed.get("label", ""))

            html = net.generate_html(notebook=False)

            # ✅ 1) 给 Gradio：直接返回 HTML 字符串
            if return_html:
                return html

            # ✅ 2) 如果需要落盘：先保证 dirname 非空，避免 WinError 3
            if save_path:
                dir_ = os.path.dirname(save_path)
                if dir_:
                    os.makedirs(dir_, exist_ok=True)
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(html)
                print(f"[MechanismKG] 多维立体交互图已生成：{save_path}")
                return save_path

            # ✅ 3) 兜底：也返回 HTML
            return html

        # 静态布局：支持 A/B/C + LAYER
        layout_mode = (layout_mode or "LAYER").upper()

        if layout_mode in ["LAYER", "A", "RADIAL", "RING"]:
            # 方案A：环状总览（你现在的多层投影圈）
            pos = self._layout_radial(G)

        elif layout_mode in ["B", "COMMUNITY", "CLUSTER"]:
            # 方案B：紧凑聚类（社区簇状）
            pos = self._layout_community(G)

        elif layout_mode in ["C", "HIER", "HIERARCHICAL"]:
            # 方案C：分层展开（左右分层）
            pos = self._layout_hierarchical(G)

        else:
            pos = nx.spring_layout(G, k=0.9, iterations=80, seed=7)

        # ---------- 3) 绘图（论文版：边淡、节点有边框） ----------
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Times New Roman']
        matplotlib.rcParams['axes.unicode_minus'] = False

        fig, ax = plt.subplots(figsize=(14, 9), dpi=dpi)
        ax.set_facecolor("white")
        ax.set_title(f"多维立体机理知识图谱（view={view}）：{lost_type}", fontsize=14, fontweight="bold")

        # 1) 先画边：必须淡，否则边框永远被压没
        nx.draw_networkx_edges(
            G, pos,
            ax=ax,
            edge_color="#222222",
            width=0.8,
            alpha=0.5,
            arrows=True,
            arrowsize=10,
            connectionstyle="arc3,rad=0.06",
        )

        # 2) 再画节点：按 type 分组画，给每类节点统一黑边框（跟你 MLR 一样）
        type_styles = {
            "LostType": {"lw": 1.4},
            "Rule": {"lw": 1.4},
            "Parameter": {"lw": 1.4},
            "FormationType": {"lw": 1.4},
            "Mechanism": {"lw": 1.4},
            "IdentificationFeature": {"lw": 1.4},
            "TypicalCurve": {"lw": 1.4},
            "LogicTerm": {"lw": 1.4},
            "Action": {"lw": 1.4},
            "RiskLevel": {"lw": 1.4},
            "Severity": {"lw": 1.4},
            "TimePattern": {"lw": 1.4},
        }

        # 未覆盖到的 type 也给默认边框
        default_lw = 1.4

        for t in sorted(set(nx.get_node_attributes(G, "type").values())):
            nodes = [n for n, d in G.nodes(data=True) if d.get("type") == t]
            if not nodes:
                continue

            node_colors = [G.nodes[n].get("color", "#D5D8DC") for n in nodes]
            node_sizes = [G.nodes[n].get("size", 1200) for n in nodes]
            lw = type_styles.get(t, {}).get("lw", default_lw)

            nx.draw_networkx_nodes(
                G, pos,
                nodelist=nodes,
                node_color=node_colors,
                node_size=node_sizes,
                alpha=0.98,
                edgecolors="black",  # ✅ 关键：明确指定黑边框
                linewidths=lw,  # ✅ 关键：边框线宽
                ax=ax,
            )

        # 3) 标签：Rule 多时只显示 RuleID（否则糊）
        labels = {}
        for n, d in G.nodes(data=True):
            t = d.get("type", "")
            if t == "Rule":
                labels[n] = n.replace("Rule:", "")
            elif t == "LostType":
                labels[n] = lost_type
            else:
                labels[n] = n.split(":", 1)[-1]

        nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, ax=ax)
        ax.axis("off")

        if return_buffer:
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
            plt.close(fig)
            buf.seek(0)
            return buf

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
            plt.close(fig)
            print(f"[MechanismKG] 多维立体静态图已生成：{save_path}")
            return save_path

        plt.show()
        return None

    def _layout_multilayer(self, G):
        """多层分层布局：把“立体感”显式投影到 X 轴层次"""
        # layer order
        layer_x = {
            "IdentificationFeature": 0,
            "TypicalCurve": 0,
            "LogicTerm": 2,
            "FormationType": 4,
            "Parameter": 4,
            "Rule": 6,
            "Mechanism": 8,
            "LostType": 10,
            "Severity": 10,
            "RiskLevel": 10,
            "TimePattern": 10,
            "Action": 12
        }

        # group nodes by x
        groups = defaultdict(list)
        for n, d in G.nodes(data=True):
            t = d.get("type", "")
            x = layer_x.get(t, 6)
            groups[x].append(n)

        pos = {}
        # within each layer, spread vertically
        for x in sorted(groups.keys()):
            nodes = groups[x]
            # stable order
            nodes = sorted(nodes)
            if len(nodes) == 1:
                pos[nodes[0]] = (x, 0.0)
                continue
            ys = np.linspace(-len(nodes) / 2.0, len(nodes) / 2.0, len(nodes))
            for n, y in zip(nodes, ys):
                pos[n] = (x, float(y))

        # small spring refinement within layers to reduce overlaps
        try:
            pos_ref = nx.spring_layout(G, pos=pos, fixed=list(pos.keys()), iterations=0)
            pos.update(pos_ref)
        except Exception:
            pass
        return pos

    # ======================================================
    # 4️⃣ 约束推理接口：用 LogicTerm 在图谱中筛选一致性规则
    # ======================================================
    def query_consistent_rules(self, logic_terms, expected_lost_type=None, top_k=50):
        """给定观测逻辑项 logic_terms，返回可满足（子集匹配）的规则列表"""
        if not isinstance(logic_terms, list):
            logic_terms = [str(logic_terms)]
        logic_terms = [str(x).strip() for x in logic_terms if str(x).strip()]

        cypher = """
        WITH $logic_terms AS L
        MATCH (r:Rule)
        OPTIONAL MATCH (r)-[:CONSTRAINED_BY]->(lg:LogicTerm)
        WITH r, collect(DISTINCT lg.name) AS RLG, L
        WHERE size(RLG) > 0 AND all(x IN RLG WHERE x IN L)
        RETURN r.RuleID AS RuleID,
                r.RuleName AS RuleName,
                r.LostType AS LostType,
                r.SeverityLevel AS SeverityLevel,
                r.RiskLevel AS RiskLevel,
                RLG AS RuleLogic
        ORDER BY size(RLG) DESC
        LIMIT $top_k
        """
        rows = list(self.graph.run(cypher, logic_terms=logic_terms, top_k=int(top_k)))

        # 可选：按 LostType 再过滤
        if expected_lost_type:
            rows = [r for r in rows if str(r.get("LostType", "")) == str(expected_lost_type)]

        return rows

    # ======================================================
    # Reasoning bundle for LLM (optional helper, does not affect main)
    # ======================================================
    def build_reasoning_bundle(self, logic_terms, expected_lost_type=None, top_k=12, make_graph=True):
        """把“观测逻辑项”转换成可喂给 LLM 的推理包。

        Parameters
        ----------
        logic_terms : list[str] | str
            观测逻辑项（与 LogicTerm 同构），如 ["ECD>FG", "PitLoss", "MW_high"] 等
        expected_lost_type : str | None
            如提供，则只保留该 LostType 下的规则
        top_k : int
            返回规则条数
        make_graph : bool
            是否返回高亮推理路径的交互 HTML（用于 Gradio 的 gr.HTML）

        Returns
        -------
        dict: {rules_topk, rule_ids, llm_context, graph_html}
        """
        rows = self.query_consistent_rules(
            logic_terms=logic_terms,
            expected_lost_type=expected_lost_type,
            top_k=top_k
        )

        rule_ids = [r.get("RuleID") for r in rows if r.get("RuleID")]

        # LLM context（结构化规则卡片）
        cards = []
        for r in rows:
            cards.append(
                f"- Rule {r.get('RuleID')} | LostType={r.get('LostType')} | Risk={r.get('RiskLevel')} | Severity={r.get('SeverityLevel')}\n"
                f"  LogicTerms: {r.get('RuleLogic')}\n"
            )
        llm_context = "Mechanism-KG Consistent Rules (Top):\n" + "\n".join(cards)

        graph_html = None
        if make_graph and rows:
            # 高亮推理路径：通过设置实例变量，不改 visualize_interactive 的函数签名
            self._highlight_rule_ids = set(rule_ids)
            lt = expected_lost_type or rows[0].get("LostType")
            if lt:
                graph_html = self.visualize_interactive(lost_type=lt, return_html=True)
            # 清理，避免影响后续普通可视化
            self._highlight_rule_ids = set()

        return {
            "rules_topk": rows,
            "rule_ids": rule_ids,
            "llm_context": llm_context,
            "graph_html": graph_html
        }
