# -*- coding: utf-8 -*-
"""
hybrid_agent.py —— RAG + MLR + ChatGLM3 多智能体融合（共享模型版）
复用主程序中已加载的 ChatGLM3 模型与 tokenizer
"""
import re, json, html
from retriever import HybridRetriever
from mlr_model import MLRModel
from kg_agent import KGAgent
from mkg_agent import MechanismKGAgent
from rag_chain import RAGChain
from PIL import Image
from lost_type_model import LossTypeModel
from loss_point.pipeline import run_loss_point_from_logs

class HybridAgent:
    def __init__(self, model, tokenizer, retriever=None, rag_chain=None):
        """
        - model / tokenizer: 外部共享的 ChatGLM3 实例
        - retriever: HybridRetriever（检索，可复用主程序实例）
        - rag_chain: RAGChain（可共享 format/clean 逻辑）
        """
        self.model = model
        self.tokenizer = tokenizer
        self.retriever = retriever or HybridRetriever()
        self.rag_chain = rag_chain or RAGChain(model=model, tokenizer=tokenizer)
        print("[*] HybridAgent 使用共享 ChatGLM3 模型 ✅")

        # MLR 小模型
        self.mlr = MLRModel(
            "E:/jupyter/lost_circulation/records/paper-bhyt/MLR/data/Well_B_combined_feature_importance.csv",
            threshold=0.55
        )
        try:
            self.mlr.fit_normalizer_from_csv(
                "E:/jupyter/lost_circulation/records/paper-bhyt/MLR/data/Well_B_IQR.csv"
            )
        except Exception:
            print("[WARN] 未找到 IQR 文件，跳过归一化")

        # 数据知识图谱 + 机理知识图谱
        self.kg = KGAgent(
            csv_path="E:/pycharm_project/lost-circ-rag/data/raw/data/WellData_with_MLR.csv"
        )
        self.mechanism_kg = MechanismKGAgent(
            csv_path="E:/pycharm_project/lost-circ-rag/data/raw/data/MechanismRules.csv",
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_pwd="12345678"
        )

        self.loss_type_model=LossTypeModel()
        self._loss_point_logs = None #缓存测井曲线df，供漏点识别用

    # ========= 数值特征抽取 =========
    def extract_numeric_features(self, text: str):
        """从自然语言中提取可计算参数"""
        feature_map = {
            "井深": "WellDepth", "well depth": "WellDepth",
            "DC": "DC",
            "地层压力梯度": "FormationPressureGradient",
            "地层破裂压力梯度": "FormationRupturePressureGradient",
            "钻压": "WOB", "WOB": "WOB",
            "转速": "RPM", "RPM": "RPM",
            "扭矩": "TOR", "TOR": "TOR",
            "泵压": "PumpPressure", "pump pressure": "PumpPressure",
            "钩载": "HookLoad", "hook load": "HookLoad",
            "机械钻速": "ROP", "ROP": "ROP",
            "排量": "Displacement", "displacement": "Displacement",
            "密度": "Density", "density": "Density",
            "ECD": "ECD",
            "出口流量": "OutletFlow",
            "滞后时间": "LagTime",
            "理论最大排量": "TheoreticalMaximumDisplacement"
        }
        pattern = r"([A-Za-z\u4e00-\u9fff]+)\s*[:：=]?\s*([0-9.]+)"
        matches = re.findall(pattern, text)
        features = {}
        for key, val in matches:
            for k, mapped in feature_map.items():
                if k.lower() in key.lower():
                    features[mapped] = float(val)
                    break
        return features

    # ========= 机理知识图谱相关工具函数 =========
    def _infer_lost_type_from_query(self, q: str) -> str:
        """
        根据用户问句中提到的漏失类型，推断 LostType 名称。
        若未显式提到具体类型，则给一个默认类型（例如：裂缝性漏失），
        以便至少能生成一张机理知识图谱。
        """
        if "渗透性漏失" in q:
            return "渗透性漏失"
        if "多孔" in q or "溶洞" in q or "喀斯特" in q:
            return "多孔-溶洞漏失"
        if "诱导裂缝" in q:
            return "诱导裂缝漏失"
        if "失返" in q:
            return "完全失返"
        if "裂缝性漏失" in q or "fracture" in q.lower():
            return "裂缝性漏失"

        if ("机理" in q) or ("原因" in q) or ("mechanism" in q.lower()):
            return "裂缝性漏失"

        return "裂缝性漏失"

    def _infer_kg_style(self, q: str) -> str:
        """
        根据问句推断机理知识图谱风格（A/B/C/D）
        """
        q_lower = q.lower()

        if ("方案b" in q_lower) or ("风格b" in q_lower) or ("方案Ｂ" in q) or ("风格Ｂ" in q):
            return "B"
        if ("方案c" in q_lower) or ("风格c" in q_lower) or ("方案Ｃ" in q) or ("风格Ｃ" in q):
            return "C"
        if ("方案d" in q_lower) or ("风格d" in q_lower) or ("方案Ｄ" in q) or ("风格Ｄ" in q) \
                or ("交互式" in q) or ("interactive" in q_lower):
            return "D"
        if ("方案a" in q_lower) or ("风格a" in q_lower) or ("方案Ａ" in q) or ("风格Ａ" in q):
            return "A"

        if "style b" in q_lower or "option b" in q_lower:
            return "B"
        if "style c" in q_lower or "option c" in q_lower:
            return "C"
        if "style d" in q_lower or "option d" in q_lower:
            return "D"
        if "style a" in q_lower or "option a" in q_lower:
            return "A"

        return "A"

    def _wrap_html_in_iframe(self, raw_html: str) -> str:
        """
        把 pyvis 生成的整页 HTML 包装成一个可嵌入的 <iframe srcdoc="...">，
        这样在 Gradio 的 gr.HTML 中也能执行其中的 <script>。
        """
        if not raw_html:
            return "<div style='color:red;'>未生成交互式机理图谱 HTML。</div>"

        escaped = html.escape(raw_html, quote=True)
        iframe = f"""
        <iframe
            srcdoc="{escaped}"
            style="width:100%; height:680px; border:none;"
            sandbox="allow-scripts allow-same-origin"
        ></iframe>
        """
        return iframe

    def _query_mechanism_rules(self, query: str, top_k: int = 5) -> str:
        """
        从机理知识图谱（Neo4j）中检索与该问题相关的机理规则。
        """
        graph = self.mechanism_kg.graph
        lost_type = self._infer_lost_type_from_query(query)
        print(f"[HybridAgent] 推断 LostType: {lost_type}")

        if lost_type:
            cypher = """
            MATCH (r:Rule)-[:BELONGS_TO]->(lt:LostType {name:$lt})
            RETURN r
            LIMIT $top_k
            """
            res = graph.run(cypher, lt=lost_type, top_k=top_k).data()
        else:
            cypher = """
            MATCH (r:Rule)
            RETURN r
            LIMIT $top_k
            """
            res = graph.run(cypher, top_k=top_k).data()

        if not res:
            return ""

        lines = ["【机理规则摘录】"]
        for item in res:
            r = item["r"]
            rid = r.get("RuleID", "")
            name = r.get("name", "")
            lt = r.get("LostType", "")
            sev = r.get("SeverityLevel", "")
            mech_short = r.get("MechanismShort", "")
            mech_detail = r.get("MechanismDetail", "")

            lines.append(f"- 规则 {rid}（{name}，类型：{lt}，程度：{sev}）")
            if mech_short:
                lines.append(f"  · 概要：{mech_short}")
            if mech_detail:
                mech_detail_show = mech_detail[:300] + "……" if len(mech_detail) > 300 else mech_detail
                lines.append(f"  · 机理：{mech_detail_show}")

        return "\n".join(lines) + "\n"

    def _draw_mech_kg(self, query: str):
        """
        按问句自动选择 LostType + 风格（A/B/C/D），绘制机理知识图谱。
        返回：(mech_graph_image, info_text, mech_html)
        """
        lt = self._infer_lost_type_from_query(query)
        style = self._infer_kg_style(query)
        info = f"\n[机理图谱] 采用方案{style}，漏失类型：{lt}。\n"

        mech_html = ""
        try:
            buf = None

            if style == "A":
                buf = self.mechanism_kg.visualize_by_lost_type(
                    lost_type=lt,
                    save_path=None,
                    return_buffer=True
                )

            elif style == "B":
                if hasattr(self.mechanism_kg, "visualize_by_lost_type_compact"):
                    buf = self.mechanism_kg.visualize_by_lost_type_compact(
                        lost_type=lt,
                        save_path=None,
                        return_buffer=True
                    )
                else:
                    info += "[提示] 尚未实现方案B，已回退到方案A。\n"
                    buf = self.mechanism_kg.visualize_by_lost_type(
                        lost_type=lt,
                        save_path=None,
                        return_buffer=True
                    )

            elif style == "C":
                if hasattr(self.mechanism_kg, "visualize_layered"):
                    buf = self.mechanism_kg.visualize_layered(
                        lost_type=lt,
                        save_path=None,
                        return_buffer=True
                    )
                else:
                    info += "[提示] 尚未实现方案C，已回退到方案A。\n"
                    buf = self.mechanism_kg.visualize_by_lost_type(
                        lost_type=lt,
                        save_path=None,
                        return_buffer=True
                    )

            elif style == "D":
                if hasattr(self.mechanism_kg, "visualize_interactive"):
                    raw_html = self.mechanism_kg.visualize_interactive(
                        lost_type=lt,
                        view="full",
                        layout_mode="LAYER",
                        interactive=True,
                        return_html=True,
                        limit_rules=80
                    )
                    mech_html = self._wrap_html_in_iframe(raw_html)
                    info += "[机理图谱-方案D] 已嵌入交互式机理知识图谱。\n"

                    buf = self.mechanism_kg.visualize_by_lost_type(
                        lost_type=lt,
                        save_path=None,
                        return_buffer=True
                    )
                    info += "[机理图谱-方案D] 同时生成方案A的静态预览图。\n"
                else:
                    info += "[提示] 尚未实现方案D，已回退到方案A。\n"
                    buf = self.mechanism_kg.visualize_by_lost_type(
                        lost_type=lt,
                        save_path=None,
                        return_buffer=True
                    )

            else:
                info += f"[提示] 未识别的方案{style}，已回退方案A。\n"
                buf = self.mechanism_kg.visualize_by_lost_type(
                    lost_type=lt,
                    save_path=None,
                    return_buffer=True
                )

            if buf:
                img = Image.open(buf)
                img.load()
                info += "该图展示了规则节点、严重程度、风险等级、时间模式及关键参数之间的关联。\n"
                return img, info, mech_html

            info += "[机理图谱] 未生成可用图像缓冲区。\n"
            return None, info, mech_html

        except Exception as e:
            info += f"[机理图谱生成失败] {e}\n"
            return None, info, mech_html

    # ========= 通用 LM 生成函数 =========

    def _lm_generate(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=2048,          # 原来 2048
            temperature=0.7,             # 降低发散
            top_p=0.9,
            do_sample=True,             # 先改贪心/近贪心，速度更稳
            repetition_penalty=1.05,
            #eos_token_id=self.rag_chain.eos_id if self.rag_chain else self.tokenizer.eos_token_id,
            #pad_token_id=self.rag_chain.pad_id if self.rag_chain else self.tokenizer.pad_token_id,
            #eos_token_id=self.tokenizer.eos_token_id,
            #pad_token_id=self.tokenizer.pad_token_id,
            min_new_tokens=512,
        )

        gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        #text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        #if self.rag_chain:
        #    text = self.rag_chain.clean_output(text)
        return text.strip()

    def _tool_predict_loss_type(self, feature_dict: dict) -> str:
        """
        工具：根据 22 数值 + 4 类别特征，调用 CatBoost 漏失类型模型。
        feature_dict 的 key 必须是训练时的列名：
        ['TVD', 'LossSequence', ..., 'WaterContent',
         'WorkingCondition', 'LossFormation', 'Lithology', 'LossSeverity']
        """
        if not feature_dict:
            return "未收到用于预测漏失类型的参数，请检查前端是否正确传入特征。"

        result = self.loss_type_model.predict_single(feature_dict)
        pred_label = result["pred_label"]      # 中文：裂缝漏失 / 断层漏失 等
        proba = result["proba"]                # dict: {类别: 概率}

        # 概率排序后展示
        proba_sorted = sorted(proba.items(), key=lambda x: x[1], reverse=True)
        proba_text = "；".join(
            [f"{cls}: {p:.2%}" for cls, p in proba_sorted]
        )

        answer = (
            f"根据输入的井况、地层与钻井液参数，"
            f"漏失类型预测模型认为最可能的漏失类型为：**{pred_label}**。\n\n"
            f"各类型的概率为：{proba_text}。\n\n"
            f"该结果来自基于现场历史井漏数据训练的 CatBoost 多分类模型，"
            f"预测结果仅供参考，实际情况请结合地质、录井和随钻测井信息进行综合判断。"
        )
        return answer

    def set_loss_point_logs(self, df_logs):
        """由 main.py 上传曲线后调用，把数据缓存到 agent。"""
        self._loss_point_logs = df_logs

    def _sanitize_context_for_english(self, text: str) -> str:
            """
            Remove Chinese characters and keep only symbolic / structural information
            for English generation.
            """
            # 删除中文字符
            text = re.sub(r"[\u4e00-\u9fff]+", "", text)
            # 删除多余空行
            text = re.sub(r"\n{2,}", "\n", text)
            return text.strip()

    def _compact_supporting_info(self, ctx: str, max_chars: int = 3500) -> str:
        """压缩辅助信息：避免把长规则/长RAG片段整段塞给大模型，提升输出结构化程度。"""
        if not ctx:
            return ctx

        # 去掉常见报错/噪声行（但保留必要提示）
        noisy_patterns = [
            r"\[机理图谱生成失败\].*",
            r"draw_networkx_edges\(\) got an unexpected keyword argument.*",
        ]
        for pat in noisy_patterns:
            ctx = re.sub(pat, "[图谱提示] 图谱生成失败（绘图库版本不兼容或渲染异常）。", ctx)

        # 1) 机理规则：每条最多保留“编号+名称+概要”，机理细节严格截断
        def _shrink_rules(block: str) -> str:
            lines = block.splitlines()
            out = []
            keep_rules = 0
            for ln in lines:
                if ln.strip().startswith("- 规则"):
                    keep_rules += 1
                if keep_rules > 3:
                    continue
                # 把“· 机理”进一步截断
                if "· 机理：" in ln:
                    head, tail = ln.split("· 机理：", 1)
                    tail = tail.strip()
                    if len(tail) > 140:
                        tail = tail[:140] + "……"
                    ln = f"{head}· 机理：{tail}"
                out.append(ln)
            # 若规则很多，补一句提示
            if keep_rules > 3:
                out.append("（其余规则已省略）")
            return "\n".join(out)

        ctx = re.sub(r"\【机理规则摘录】[\s\S]*?(?=\n\[|\Z)", lambda mm: _shrink_rules(mm.group(0)), ctx)

        # 2) 文献RAG片段：只保留前2段，每段截断
        def _shrink_rag(block: str) -> str:
            text = block.replace("\r", "")
            parts = [p.strip() for p in text.split("\n\n") if p.strip()]
            parts = parts[:2]
            out = []
            for p in parts:
                if len(p) > 380:
                    p = p[:380] + "……"
                out.append(p)
            return "\n\n".join(out)

        ctx = re.sub(r"\[文献RAG片段\][\s\S]*?(?=\n\[|\Z)", lambda mm: "[文献RAG片段]\n" + _shrink_rag(mm.group(0).split("\n",1)[1] if "\n" in mm.group(0) else ""), ctx)

        # 3) 全局长度限制
        ctx = ctx.strip()
        if len(ctx) > max_chars:
            ctx = ctx[:max_chars] + "\n（辅助信息过长，已截断）"
        return ctx


    """"""
    def _build_cn_prompt(self, query: str, supporting: str) -> str:
        """中文结构化报告模板：强约束输出格式，减少“堆字”。"""
        return (
            "你是一个‘井漏智能诊断专家’，具备 MLR 计算、机理知识图谱推理以及文献 RAG 检索能力。\n"
            "请严格输出 **Markdown 报告**，必须包含以下一级标题（标题文字保持一致）：\n"
            "# 井漏智能诊断报告\n"
            "## 1 结论摘要\n"
            "## 2 风险评估（MLR）\n"
            "## 3 漏失类型与机理判断\n"
            "## 4 机理推理依据（规则/图谱）\n"
            "## 5 工程建议（监测/控制/堵漏/后续）\n"
            "## 6 证据摘录（可选）\n\n"
            "写作规则：\n"
            "1) 每个小节用要点列表（- 或 1. 2.），避免连续大段文字。\n"
            "2) 【机理规则】只写‘规则编号+一句话概要’，不要粘贴长段原文。\n"
            "3) 【文献RAG】只摘录关键结论，每条不超过两行。\n"
            "4) 若某项信息缺失，写“未提供/无法判断”，不要编造。\n\n"
            "重要：最终输出不要输出任何提示词、写作规则、【用户问题】或【辅助信息】原文。\n"
            "证据摘录只保留：规则编号+一句话结论；文献只列出关键结论/题名，不要粘贴长段原文。\n\n"
            f"【用户问题】\n{query}\n\n"
            f"【辅助信息】\n{supporting}\n"
        )

    def _clean_markdown_output(self, text: str) -> str:
  
        #清理模型输出中的提示词泄露、乱码字符、重复报告等，
        #并统一标题格式，保证前端 Markdown 渲染稳定。

        if not text:
            return text

        s = text.replace("\r", "\n")

        # 0) 去掉模型偶发加的“报告正文：”
        s = re.sub(r"^\s*报告正文[:：]\s*", "", s)

        # 1) 如果模型把提示词也输出了：保留最完整的一份报告
        key = "# 井漏智能诊断报告"
        if key in s:
            parts = s.split(key)
            candidates = []

            for part in parts[1:]:
                block = (key + part).strip()
                score = 0
                for h in [
                    "## 1 结论摘要",
                    "## 2 风险评估（MLR）",
                    "## 3 漏失类型与机理判断",
                    "## 4 机理推理依据（规则/图谱）",
                    "## 5 工程建议（监测/控制/堵漏/后续）",
                    "## 6 证据摘录（可选）",
                ]:
                    if h in block:
                        score += 1
                candidates.append((score, len(block), block))

            if candidates:
                candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
                s = candidates[0][2]

        # 2) 去掉常见提示词泄露段
        s = re.sub(r"^\s*\[gMASK\][\s\S]*?(?=\n# 井漏智能诊断报告|\Z)", "", s, flags=re.M)
        s = re.sub(r"^写作规则：[\s\S]*?(?=\n【用户问题】|\n# 井漏智能诊断报告|\Z)", "", s, flags=re.M)
        s = re.sub(r"^【用户问题】[\s\S]*?(?=\n【辅助信息】|\n# 井漏智能诊断报告|\Z)", "", s, flags=re.M)
        s = re.sub(r"^【辅助信息】[\s\S]*?$", "", s, flags=re.M)

        # 3) 删除私有区字符 + 压缩空格
        s = re.sub(r"[\uE000-\uF8FF]", "", s)
        s = s.replace("□", "")
        s = re.sub(r"[ \t]{2,}", " ", s)

        # 4) 修复“标题和正文粘在同一行”
        # 例如：正文。 ## 2 风险评估（MLR）
        s = re.sub(
            r'([。；：!?）\)])\s*(##\s*[2-6]\s+)',
            r'\1\n \2',
            s
        )

        # 5) 去掉 "===== ## 1 结论摘要" 这种前缀
        s = re.sub(
            r'^\s*[=]{3,}\s*(##\s*[1-6]\s+)',
            r'\1',
            s,
            flags=re.M
        )

        # 6) 如果标题前面还有普通文本，也强制断行
        s = re.sub(
            r'([^\n])\s+(##\s*[2-6]\s+)',
            r'\1\n\n \2',
            s
        )

        # 7) 修复
        # 1
        # 结论摘要
        # -> 1 结论摘要
        s = re.sub(
            r'\n([1-6])\s*\n\s*([^\n#]+)',
            r'\n\1 \2',
            s
        )
        # 8) 把“规则 MR_xxx ...”统一成 bullet
        lines = []
        for ln in s.split("\n"):
            raw = ln.strip()
            if not raw:
                lines.append("")
                continue

            raw = re.sub(r"^#{1,6}\s*(规则\s*MR_?\d+)", r"\1", raw)

            if re.match(r"^(规则\s*MR_?\d+)", raw) and not raw.startswith(("-", "1.", "2.", "3.")):
                raw = "- " + raw

            if raw.startswith("[文献RAG片段]"):
                lines.append(raw)
                continue

            lines.append(raw)

        s = "\n".join(lines)
        # 9) 控制证据摘录长度
        s = re.sub(
            r"(## 6 证据摘录（可选）\n)([\s\S]{1400,})",
            lambda m: m.group(1) + m.group(2)[:1400] + "\n\n（证据摘录过长，已截断）",
            s
        )
        # 10) 删除孤立 bullet（只剩 · 或 -）
        s = re.sub(r"^\s*[·\-•]\s*$", "·", s, flags=re.M)

        # 11) 标题后只允许一个换行
        s = re.sub(
            r"(##\s*[1-6][^\n]*)\n{2,}",
            r"\1\n",
            s
        )
        # 12) 全局空行收敛
        s = re.sub(r"\n{3,}", "\n\n", s).strip()

        return s


    def run_text(self, query: str, mode: str = "text_rag_agent", loss_features=None):
        """
        Gradio 单输出专用：只返回干净的 Markdown 文本
        """
        answer, _data_img, _mech_img, _mech_html = self.run(query=query, mode=mode, loss_features=loss_features)
        answer = self._clean_markdown_output(answer)
        return answer


    def _postprocess_cn_answer(self, ans: str) -> str:
        """兜底：若模型没有按模板输出，则自动修复为规范 Markdown 报告。"""
        if not ans:
            return ans

        def _as_bullets(text: str, limit: int = 8) -> str:
            if not text:
                return "- 未提供/无法判断"
            text = text.strip()
            lines = [ln.strip(" 	-•") for ln in text.splitlines() if ln.strip()]
            if not lines:
                return "- 未提供/无法判断"
            out = []
            for ln in lines[:limit]:
                if re.match(r"^(?:[-•]|\d+[\.、])\s*", ln):
                    out.append(ln)
                else:
                    out.append(f"- {ln}")
            return "\n".join(out)

        def _split_plain_report(text: str):
            text = text.replace("\r", "\n").strip()
            text = re.sub(r"^#+\s*井漏智能诊断报告\s*", "", text).strip()
            pattern = re.compile(
                r"(?:^|\n)\s*(?:##\s*)?([1-6])\s*(结论摘要|风险评估（MLR）|漏失类型与机理判断|机理推理依据（规则/图谱）|工程建议（监测/控制/堵漏/后续）|证据摘录（可选）)",
                re.M,
            )
            matches = list(pattern.finditer(text))
            if not matches:
                return None
            sections = {}
            for i, m in enumerate(matches):
                sec_no = m.group(1)
                start_pos = m.end()
                end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                body = text[start_pos:end_pos].strip()
                sections[sec_no] = body
            return sections

        def _repair_collapsed_sections(sections: dict):
            s1 = sections.get("1", "").strip()
            s2 = sections.get("2", "").strip()
            s3 = sections.get("3", "").strip()
            s4 = sections.get("4", "").strip()
            s5 = sections.get("5", "").strip()
            s6 = sections.get("6", "").strip()

            earlier_empty = sum(1 for x in [s1, s2, s3, s4, s5] if not x)
            if s6 and earlier_empty >= 3:
                raw = s6

                mlr_hits = re.findall(r"\[MLR计算结果\][\s\S]*?(?=(?:\n\s*\[机理规则摘录\]|\n\s*\[文献RAG片段\]|\n\s*\[数据知识图谱\]|\Z))", raw)
                rule_hits = re.findall(r"\[机理规则摘录\][\s\S]*?(?=(?:\n\s*\[机理图谱\]|\n\s*\[文献RAG片段\]|\n\s*\[数据知识图谱\]|\Z))", raw)

                if not s2 and mlr_hits:
                    s2 = "\n\n".join(x.strip() for x in mlr_hits if x.strip())
                if not s4 and rule_hits:
                    s4 = "\n\n".join(x.strip() for x in rule_hits if x.strip())

                pred_type = ""
                m_type = re.search(r"最可能的漏失类型为[:：]?\*\*(.+?)\*\*", raw)
                if m_type:
                    pred_type = m_type.group(1).strip()
                if not s3 and pred_type:
                    s3 = f"预测漏失类型：{pred_type}"

                if not s1:
                    summary_parts = []
                    if pred_type:
                        summary_parts.append(f"预测漏失类型为{pred_type}。")
                    m_level = re.search(r'"level"\s*:\s*"([^"]+)"', raw)
                    if m_level:
                        summary_parts.append(f"综合风险等级为{m_level.group(1)}。")
                    if summary_parts:
                        s1 = " ".join(summary_parts)

                if not s5:
                    sugg = []
                    for ln in raw.splitlines():
                        t = ln.strip()
                        if any(k in t for k in ["建议", "monitor", "堵漏", "控制", "后续", "监测"]):
                            sugg.append(t)
                    if sugg:
                        s5 = "\n".join(sugg[:6])

                s6 = raw

            return {
                "1": s1, "2": s2, "3": s3,
                "4": s4, "5": s5, "6": s6,
            }

        t = ans.lstrip()
        if t.startswith("# 井漏智能诊断报告"):
            return ans.strip()

        # 2) 只有纯文本标题：先补上 markdown，再判断是否需要把第6节内容回填到前面各节
        if t.startswith("井漏智能诊断报告"):
            lines = ans.strip().splitlines()
            fixed = []
            for i, ln in enumerate(lines):
                s = ln.strip()
                if i == 0 and s == "井漏智能诊断报告":
                    fixed.append("# 井漏智能诊断报告")
                    continue
                if re.match(r"^[1-6]\s+", s):
                    fixed.append("## " + s)
                else:
                    fixed.append(ln)

            s = "\n".join(fixed).strip()

            s = re.sub(r"\n=+\n", "\n", s)
            s = re.sub(r"\n-+\n", "\n", s)
            # ---- 检查是否出现“前5节空、第6节塞满” ----
            m = re.search(
                r"## 1 结论摘要\s*(.*?)\s*"
                r"## 2 风险评估（MLR）\s*(.*?)\s*"
                r"## 3 漏失类型与机理判断\s*(.*?)\s*"
                r"## 4 机理推理依据（规则/图谱）\s*(.*?)\s*"
                r"## 5 工程建议（监测/控制/堵漏/后续）\s*(.*?)\s*"
                r"## 6 证据摘录（可选）\s*([\s\S]*)$",
                s
            )

            if not m:
                return (
                        "# 井漏智能诊断报告\n"
                        f"\n{ans.strip()}\n"+ s
                )
                """
                return (
                        "# 井漏智能诊断报告\n"
                        "## 1 结论摘要\n 出错了，请联系工作人员。\n"
                        "## 2 风险评估（MLR）\n 未完整生成。\n"
                        "## 3 漏失类型与机理判断\n 未完整生成。\n"
                        "## 4 机理推理依据（规则/图谱）\n 未完整生成。\n"
                        "## 5 工程建议（监测/控制/堵漏/后续）\n 未完整生成。\n"
                        "## 6 证据摘录（可选）\n" + s
                )"""

                #return s

            sec1, sec2, sec3, sec4, sec5, sec6 = [x.strip() for x in m.groups()]

            def _empty(x: str) -> bool:
                return (not x) or x in ("- 未提供/无法判断", "未提供/无法判断")

            # 当前就是这种情况：前几节空，第6节很长
            if sum(_empty(x) for x in [sec1, sec2, sec3, sec4, sec5]) >= 3 and len(sec6) > 80:
                raw = sec6

                # 1 结论摘要
                summary = ""
                m_level = re.search(r'"level"\s*:\s*"([^"]+)"', raw)
                m_mlr = re.search(r'"MLR"\s*:\s*([0-9.]+)', raw)
                if m_level or m_mlr:
                    parts = []
                    if m_mlr:
                        parts.append(f"MLR={m_mlr.group(1)}")
                    if m_level:
                        parts.append(f"风险等级：{m_level.group(1)}")
                    summary = "；".join(parts)

                # 2 风险评估
                mlr_block = ""
                m_mlr_block = re.search(r"(\[MLR计算结果\][\s\S]*?)(?=\[机理规则摘录\]|\[文献RAG片段\]|\Z)", raw)
                if m_mlr_block:
                    mlr_block = m_mlr_block.group(1).strip()

                # 3 类型判断
                loss_type = ""
                m_type = re.search(r"(裂缝性漏失|渗透性漏失|多孔[-－]溶洞漏失|诱导裂缝漏失|完全失返)", raw)
                if m_type:
                    loss_type = f"初步判断为：{m_type.group(1)}"

                # 4 机理依据
                mech_block = ""
                m_mech = re.search(r"(\[机理规则摘录\][\s\S]*?)(?=\[文献RAG片段\]|\Z)", raw)
                if m_mech:
                    mech_block = m_mech.group(1).strip()
                else:
                    rule_lines = re.findall(r"(?:^|\n)\s*[-•]?\s*(规则\s*MR_?\d+[^\n]*)", raw)
                    if rule_lines:
                        mech_block = "\n".join(f"- {x.strip()}" for x in rule_lines[:5])

                # 5 工程建议
                suggest = ""
                suggest_lines = []
                for ln in raw.splitlines():
                    tln = ln.strip()
                    if any(k in tln for k in ["建议", "监测", "控制", "堵漏", "后续", "压井", "排量", "钻井液"]):
                        if len(tln) >= 8:
                            suggest_lines.append(f"- {tln.lstrip('-• ')}")
                if suggest_lines:
                    suggest = "\n".join(suggest_lines[:5])

                def _fill(x, default="- 未提供/无法判断"):
                    return x if x else default

                return (
                    "# 井漏智能诊断报告\n"
                    "## 1 结论摘要\n" + _fill(summary) + "\n"
                    "## 2 风险评估（MLR）\n" + _fill(mlr_block) + "\n"
                    "## 3 漏失类型与机理判断\n" + _fill(loss_type) + "\n"
                    "## 4 机理推理依据（规则/图谱）\n" + _fill(mech_block) + "\n"
                    "## 5 工程建议（监测/控制/堵漏/后续）\n" + _fill(suggest) + "\n"
                    "## 6 证据摘录（可选）\n" + raw
                ).strip()

            return s

    # ========= 主调度（支持消融模式） =========
    def run(self, query: str, mode: str = "text_rag_agent", loss_features = None):
        """
        ...
        新增:
        - loss_features: 当 mode == "loss_type" 时，用于漏失类型预测的特征字典
        """
        """
        智能体调度逻辑，mode 支持：
        - "lm_only"      : 纯大模型（无RAG、无KG、无MLR）
        - "text_rag"     : 大模型 + Text-RAG（仅文本检索，不用KG，不算MLR）
        - "text_rag_agent"  : 全量方案（MLR + 文本RAG + 机理KG + 数据KG）

        返回：
        - answer: 文本回答
        - data_graph_image: 基于现场数据的知识图谱（KGAgent）
        - mech_graph_image: 基于机理规则的知识图谱（MechanismKGAgent）
        - mech_html: 方案D的交互式HTML（否则为空字符串）
        """
        mode = (mode or "text_rag_agent").lower()
        if mode not in ("lm_only", "text_rag", "text_rag_agent","loss_type","loss_point"):
            mode = "text_rag_agent"

        lower_q = query.lower()

        # -------- 模式0：loss_type（漏失类型识别，只用 CatBoost 模型） --------
        if mode == "loss_type":
            # 不走 RAG / KG / MLR，直接调用结构化模型
            answer = self._tool_predict_loss_type(loss_features or {})
            # 该模式下不生成任何图谱，只返回文本
            return answer, None, None, ""
        # -----模式0：loss_point（漏点位置识别）-----
        if mode == "loss_point":
            if self._loss_point_logs is None:
                answer = "I don't have log curves loaded. Please upload the log CSV in the Loss Point tab first."
                return answer, None, None, ""

            # 这里从 query 里简单抽井名：如果你已有更好的解析函数，用你的
            # 最简单：默认取第一口井名（你也可以在 payload 里传 well）
            wells = sorted(self._loss_point_logs["WellName"].dropna().unique().tolist())
            target_well = wells[0] if wells else None

            # 如果用户在问题里写了井名，就用它
            for w in wells:
                if w in query:
                    target_well = w
                    break

            if not target_well:
                answer = "No WellName found in uploaded logs."
                return answer, None, None, ""

            seg_df, fig, summary, _debug = run_loss_point_from_logs(
                self._loss_point_logs,
                target_well=target_well,
                depth_step=0.5,
                baseline_mode="ge1",
                p_thr=0.6,
                min_seg_len=1.0,
                neighbor_k=3,
            )

            # 把 seg_df 简化成文本
            if seg_df.empty:
                answer = summary
            else:
                lines = [summary, "Segments:"]
                for i, r in seg_df.iterrows():
                    lines.append(f"- {i + 1}) {r['StartDepth']:.1f}–{r['EndDepth']:.1f} m (mean p={r['MeanProb']:.2f})")
                answer = "\n".join(lines)

            # 把图放到 data_graph_image 位置，保持你原来的 4 返回不变
            return answer, fig, None, ""

        # -------- 模式1：lm_only（纯LM） --------
        if mode == "lm_only":
            prompt = (
                "你是一名从事“井漏风险评估与防漏堵漏”的资深工程师。"
                "请仅基于你已有的通用与领域知识回答问题，不要虚构不存在的数据，也不要引用外部文献或图谱。\n\n"
                f"【用户问题】\n{query}\n\n"
                "【回答】\n"
            )
            answer = self._lm_generate(prompt)
            return answer, None, None, ""

        # -------- 模式2：text_rag（仅文本RAG） --------
        if mode == "text_rag":
            docs = self.retriever.search(query, top_k=6)
            ctx = self.rag_chain.build_context(docs) if self.rag_chain else ""
            prompt = (
                "你是一名“井漏风险评估与防漏堵漏”领域的专家，请优先基于下方给出的参考资料回答问题，"
                "当资料不足时再结合你已有的专业知识进行推理。\n\n"
                f"【用户问题】\n{query}\n\n"
                f"【参考资料】\n{ctx}\n\n"
                "【回答】\n"
            )
            answer = self._lm_generate(prompt)
            return answer, None, None, ""

        # -------- 模式3：text_rag_agent（全量方案） --------
        # 以下逻辑基本保持你原来的流程，只是在 Step4 里真正把文献片段作为上下文喂给模型
        calc_triggers = ["风险", "mlr", "risk", "ecd", "密度", "排量", "井深"]
        knowledge_triggers = ["机理", "原因", "原理", "why", "how", "mechanism", "reason"]
        graph_triggers = ["关系", "特征", "图谱", "网络", "关联", "mechanism", "relation", "graph","交互式","interactive"]
        loss_type_triggers = ["漏失类型", "类型判断", "类型识别", "类型预测", "loss type", "lost type"]
        loss_point_triggers = ["漏点", "漏失位置", "漏点识别", "漏点深度",
                               "loss point", "leak point", "leakage depth", "loss interval"]

        has_number = bool(re.search(r"[0-9]", query))

        final_context = ""
        data_graph_image = None
        mech_graph_image = None
        mech_html = ""

        # Step 1: MLR 计算
        if has_number and any(w in lower_q for w in calc_triggers):
            feats = self.extract_numeric_features(query)
            if feats:
                result = self.mlr.calc(feats)
                final_context += f"\n[MLR计算结果]\n{json.dumps(result, ensure_ascii=False, indent=2)}\n"
            else:
                final_context += "\n[未识别到可计算参数，跳过MLR计算]\n"

        # Step 2: 机理知识图谱（文本 + 可视化）
        force_mech = ("方案d" in lower_q) or ("风格d" in lower_q) or ("style d" in lower_q) or (
                    "interactive" in lower_q) or ("交互式" in query)
        if force_mech or any(w in lower_q for w in graph_triggers + knowledge_triggers):
            try:
                mech_text = self._query_mechanism_rules(query, top_k=5)
                if mech_text:
                    final_context += "\n" + mech_text
            except Exception as e:
                final_context += f"\n[机理知识图谱检索失败] {e}\n"

            mech_img, info_text, mech_html_part = self._draw_mech_kg(query)
            mech_graph_image = mech_img
            final_context += info_text
            if mech_html_part:
                mech_html = mech_html_part

        # Step 3: 数据知识图谱（现场数据）
        if any(w in lower_q for w in graph_triggers):
            try:
                buf = self.kg.visualize_examples_side_by_side(
                    mlr_threshold=0.55,
                    well_id=None,
                    save_path=None
                )
                if buf:
                    data_graph_image = Image.open(buf)
                    data_graph_image.load()
                    final_context += "\n[数据知识图谱] 已生成基于现场数据的典型井漏样本关系图。\n"
            except Exception as e:
                final_context += f"\n[数据知识图谱生成失败] {e}\n"

        # Step 4: 文献检索（RAG）—— 这次真正把文献片段放进上下文
        docs = []
        if any(w in lower_q for w in knowledge_triggers):
            docs = self.retriever.search(query, top_k=4)
            if docs:
                ctx = self.rag_chain.build_context(docs)
                final_context += f"\n[文献RAG片段]\n{ctx}\n"

        # Step 5: 漏失类型判断
        if any(t in lower_q for t in loss_type_triggers):
            # ① 如果前端传入了结构化特征字典，就优先用它
            lt_feats = loss_features if isinstance(loss_features, dict) and loss_features else {}

            # ② 否则尝试从 query 里解析字段=值（你需要有这个解析函数）
            if not lt_feats and hasattr(self, "extract_loss_type_features"):
                lt_feats = self.extract_loss_type_features(query)

            # ③ 字段够用才预测，不够就提示用户补充
            if lt_feats and len(lt_feats) >= 3:
                try:
                    lt_answer = self._tool_predict_loss_type(lt_feats)
                    final_context += f"\n[漏失类型预测-小模型]\n{lt_answer}\n"
                except Exception as e:
                    final_context += f"\n[漏失类型预测-小模型]\n预测失败：{e}\n"
            else:
                final_context += (
                    "\n[漏失类型预测-小模型]\n"
                    "检测到你在询问漏失类型，但可用特征不足。请补充关键字段（如：漏失速度、漏失量、漏点起止深度、工况、层位、岩性、漏失程度等），"
                    "或通过前端表单/上传数据传入完整特征。\n"
                )
        # Step 6 :漏点位置识别
        if any(t.lower() in query.lower() for t in loss_point_triggers):
            if self._loss_point_logs is None:
                tool_text = "[LossPointModel] No logs loaded. Ask user to upload log CSV in Loss Point tab."
            else:
                wells = sorted(self._loss_point_logs["WellName"].dropna().unique().tolist())
                target_well = wells[0] if wells else None
                for w in wells:
                    if w in query:
                        target_well = w
                        break

                if target_well:
                    seg_df, _fig, summary, _debug = run_loss_point_from_logs(self._loss_point_logs,
                                                                             target_well=target_well)
                    # 只注入摘要，避免把整df塞进上下文
                    tool_text = summary
                else:
                    tool_text = "[LossPointModel] Could not resolve WellName from query."

            # 把工具输出加进你最终 prompt 的上下文变量里（下面这行变量名你按你代码实际替换）
            final_context += "\n" + tool_text + "\n"

        # Step 7: 综合生成回答
        def _is_english(text: str) -> bool:
            return bool(re.search(r"[a-zA-Z]", text)) and not bool(re.search(r"[\u4e00-\u9fff]", text))
        is_english = _is_english(query)

        if is_english:
            clean_context = self._sanitize_context_for_english(final_context)
        else:
            clean_context = self._compact_supporting_info(final_context)

        if is_english:
            prompt = (
                "You are an expert in lost circulation diagnosis, with capabilities in "
                "MLR risk assessment, mechanism knowledge graph reasoning, and literature-based RAG.\n"
                "Please generate a structured answer STRICTLY in ENGLISH using the following format:\n"
                "[Conclusion]\n"
                "[Basis]\n"
                "[Suggestions]\n\n"
                "IMPORTANT RULES:\n"
                "1) The final answer MUST be written in ENGLISH ONLY.\n"
                "2) DO NOT include any Chinese characters.\n"
                "3) If a technical term appears in Chinese in the supporting information, "
                "translate it into standard technical English instead of copying it.\n\n"
                f"[User Question]\n{query}\n\n"
                f"[Supporting Information]\n{clean_context}\n"
            )

        else:
            prompt = self._build_cn_prompt(query, clean_context)

        print("[HybridAgent] Step 7 prompt length:", len(prompt))
        print("[HybridAgent] 开始大模型生成...")
        answer = self._lm_generate(prompt)
        print("[HybridAgent] 大模型生成完成")
        print("[HybridAgent] 大模型原始输出开始")
        print(repr(answer[:1200]))
        print("[HybridAgent] 大模型原始输出结束")
        if not is_english:
            answer = self._postprocess_cn_answer(answer)
            answer = self._clean_markdown_output(answer)  # ✅ 新增：强制清洗
        return answer, data_graph_image, mech_graph_image, mech_html
