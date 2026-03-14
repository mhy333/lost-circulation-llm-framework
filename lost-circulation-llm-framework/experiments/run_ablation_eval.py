# -*- coding: utf-8 -*-
"""
run_ablation_eval.py —— 井漏智能体三种知识注入模式的消融实验批量评测 + 可视化

使用方式（在项目根目录下）：
    python run_ablation_eval.py

前提：
    - 已准备好 questoins_ablation_1.csv
      必须包含列: QID(可选), QuestionText, RefAnswer(建议), 其它元信息可选
    - config.MODEL_DIR 指向 ChatGLM3-6B 本地权重
"""

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from evaluate import load as hf_load
from rouge_score.rouge_scorer import _score_lcs
import jieba

from config import MODEL_DIR
from retriever import HybridRetriever
from rag_chain import RAGChain
from hybrid_agent import HybridAgent

# ================= 指标函数 =================

def compute_distinct_n(text: str, n: int = 2) -> float:
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / max(len(ngrams), 1)


def compute_flags(text: str):
    """
    简单统计：回答中是否提到机理/图谱/MLR（0/1）
    """
    t = text.lower()
    has_mech = int(("机理" in text) or ("mechanism" in t) or ("原因" in text) or ("原理" in text))
    has_kg = int(("图谱" in text) or ("knowledge graph" in t) or ("kg" in t) or ("规则" in text) or ("rule " in t))
    has_mlr = int(("mlr" in t) or ("MLR" in text) or ("风险指数" in text))
    return has_mech, has_kg, has_mlr


def compute_structure_flags(text: str):
    """
    结构化输出指标：
    - 是否包含【结论】/Conclusion
    - 是否包含【依据】/Evidence
    - 是否包含【建议】/Recommendations
    - 项目符号数量（· 或 "- "）
    """
    t = text
    has_conclusion = int(("【结论】" in t) or ("Conclusion" in t))
    has_evidence = int(("【依据】" in t) or ("Evidence" in t))
    has_suggestion = int(("【建议】" in t) or ("Recommendations" in t))
    num_bullets = t.count("·") + t.count("- ")
    return has_conclusion, has_evidence, has_suggestion, num_bullets


def is_chinese(text: str) -> bool:
    if not text:
        return False
    zh_chars = [c for c in text if '\u4e00' <= c <= '\u9fff']
    return len(zh_chars) / max(len(text), 1) > 0.2


def compute_rouge_l(ref: str, hyp: str) -> float:
    """
    ROUGE-L F1, 中英文自适应：
    - 中文：jieba 分词 + LCS
    - 英文：空格分词 + LCS
    """
    ref = (ref or "").strip()
    hyp = (hyp or "").strip()
    if not ref or not hyp:
        return 0.0

    if is_chinese(ref + hyp):
        try:
            ref_tokens = list(jieba.cut(ref))
            hyp_tokens = list(jieba.cut(hyp))
        except Exception:
            ref_tokens = list(ref)
            hyp_tokens = list(hyp)
    else:
        ref_tokens = ref.lower().split()
        hyp_tokens = hyp.lower().split()

    score = _score_lcs(ref_tokens, hyp_tokens)
    if score.recall + score.precision == 0:
        return 0.0
    return 2 * score.recall * score.precision / (score.recall + score.precision)


# ================= 主流程 =================

def main():
    # 1. 读取问题集
    q_csv = r"E:\pycharm_project\lost-circ-rag\data\raw\data\questoins_ablation_with_ref.csv"
    print(f"[*] 期望的问题集路径：{q_csv}")
    print(f"[*] 文件是否存在：{os.path.exists(q_csv)}")
    if not os.path.exists(q_csv):
        raise FileNotFoundError(f"未找到问题集文件：{q_csv}，请先准备好。")

    df_q = pd.read_csv(q_csv, encoding="utf-8-sig")
    if "QuestionText" not in df_q.columns:
        raise ValueError("CSV 中必须包含列 'QuestionText'")
    if "RefAnswer" not in df_q.columns:
        print("[WARN] 未提供 RefAnswer，Accuracy/ROUGE/BERTScore 将全部置 0。")

    # 2. 加载大模型
    print("[*] 正在加载 ChatGLM3-6B 模型（4bit）用于消融实验...")
    qconfig = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        trust_remote_code=True,
        quantization_config=qconfig,
        device_map="auto"
    ).eval()
    print("[*] 模型加载完成 ✅")

    retriever = HybridRetriever()
    rag_chain = RAGChain(model=model, tokenizer=tokenizer)
    agent = HybridAgent(model=model, tokenizer=tokenizer,
                        retriever=retriever, rag_chain=rag_chain)

    # 三种模式：纯 LM / 文本 RAG / 全量 Agent（Text-RAG+KG+MLR）
    modes = ["lm_only", "text_rag", "text_rag_agent"]

    # SentenceTransformer 用于 ContextSim
    try:
        embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        embed_ok = True
        print("[*] 已加载 SentenceTransformer 用于 ContextSim 计算。")
    except Exception as e:
        print("[WARN] 无法加载 SentenceTransformer，ContextSim 将置 0：", e)
        embed_model = None
        embed_ok = False

    # BERTScore：用于 Accuracy / BERTScore
    try:
        bertscore = hf_load("bertscore")
        bert_ok = True
        print("[*] 已加载 BERTScore 评估器。")
    except Exception as e:
        print("[WARN] 无法加载 BERTScore，相关指标置 0：", e)
        bertscore = None
        bert_ok = False

    records = []

    # 3. 对每道题、每种模式依次生成回答并计算指标
    for idx, row in df_q.iterrows():
        qid = row.get("QID", f"Q{idx+1:03d}")
        question = str(row["QuestionText"])
        ref_answer = str(row.get("RefAnswer", "") or "")
        scenario = row.get("Scenario", "")
        qtype = row.get("Type", "")
        diff = row.get("Difficulty", "")
        need_kg = row.get("NeedKG", "")

        print(f"\n==================== Q{idx+1}/{len(df_q)}: {qid} ====================")
        print("问题：", question)

        for mode in modes:
            print(f"\n>>> 模式：{mode}")
            try:
                start = time.time()
                answer, _, _, _ = agent.run(question, mode=mode)
                elapsed = time.time() - start
            except Exception as e:
                answer = f"[ERROR] {e}"
                elapsed = -1.0

            len_chars = len(answer)
            distinct2 = compute_distinct_n(answer, n=2)
            has_mech, has_kg_flag, has_mlr = compute_flags(answer)
            has_conc, has_evid, has_sugg, num_bullets = compute_structure_flags(answer)

            # ---- ContextSim：回答与检索文献片段的一致性（仅对使用检索的模式计算）----
            context_sim = 0.0
            if embed_ok and mode != "lm_only":
                try:
                    docs = retriever.search(question, top_k=4)
                    ctx_texts = [d.get("text", "") for d in docs if d.get("text", "")]
                    if ctx_texts:
                        ans_emb = embed_model.encode([answer])
                        ctx_embs = embed_model.encode(ctx_texts)
                        context_sim = float(np.mean(cosine_similarity(ans_emb, ctx_embs)))
                except Exception as e:
                    print(f"[WARN] ContextSim 计算失败（{mode}, {qid}）：{e}")
                    context_sim = 0.0

            # ---- ROUGE-L & BERTScore & Accuracy ----
            rouge_l = 0.0
            bert_f1 = 0.0
            acc = 0.0

            if ref_answer.strip():
                # ROUGE-L
                rouge_l = compute_rouge_l(ref_answer, answer)

                # BERTScore
                if bert_ok:
                    try:
                        lang = "zh" if is_chinese(ref_answer + answer) else "en"
                        bs_res = bertscore.compute(
                            predictions=[answer],
                            references=[ref_answer],
                            lang=lang
                        )
                        bert_f1 = float(bs_res["f1"][0])
                    except Exception as e:
                        print(f"[WARN] BERTScore 计算失败（{mode}, {qid}）：{e}")
                        bert_f1 = 0.0

                # Accuracy: 这里用一个简单策略：BERTScore >= 0.8 认为“回答正确”
                # 你之后可以在 CSV 里加入人工标签列，比如 AccLabel，然后直接用人工标签覆盖
                if bert_f1 >= 0.8:
                    acc = 1.0
                else:
                    acc = 0.0

            # 如果 CSV 里有人工标签 AccLabel_Xxx，可以在这里覆盖 acc（可选）
            # 比如每道题你打一个“标准正确模式”，再用那一列替换 acc

            records.append({
                "QID": qid,
                "QuestionText": question,
                "RefAnswer": ref_answer,
                "Scenario": scenario,
                "Type": qtype,
                "Difficulty": diff,
                "NeedKG": need_kg,
                "Mode": mode,
                "Answer": answer,
                "Time_sec": elapsed,         # Latency
                "LenChars": len_chars,
                "Distinct2": distinct2,
                "HasMechanism": has_mech,
                "HasKG": has_kg_flag,
                "HasMLR": has_mlr,
                "HasConclusion": has_conc,
                "HasEvidence": has_evid,
                "HasSuggestion": has_sugg,
                "NumBullets": num_bullets,
                "ContextSim": context_sim,
                "ROUGE_L": rouge_l,
                "BERTScore": bert_f1,
                "Accuracy": acc
            })

    # 4. 保存总表
    out_csv = r"E:/pycharm_project/lost-circ-rag/data/raw/data/answers_ablation_all.csv"
    df_out = pd.DataFrame(records)
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n[*] 已保存消融实验结果到：{out_csv}")

    # 5. 可视化
    fig_dir = r"E:/pycharm_project/lost-circ-rag/outputs/ablation_figs"
    os.makedirs(fig_dir, exist_ok=True)

    # 5.1 各模式平均回答长度
    df_len = df_out.groupby("Mode")["LenChars"].mean().reset_index()
    plt.figure(figsize=(6, 4))
    plt.bar(df_len["Mode"], df_len["LenChars"])
    plt.ylabel("Average Answer Length (chars)")
    plt.title("Average Answer Length per Mode")
    for i, v in enumerate(df_len["LenChars"]):
        plt.text(i, v + 5, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    len_fig_path = os.path.join(fig_dir, "avg_answer_length_per_mode.png")
    plt.savefig(len_fig_path, dpi=200)
    plt.close()
    print(f"[*] 已保存图像：{len_fig_path}")

    # 5.2 机理 / KG / MLR 提及比例
    df_flag = df_out.groupby("Mode")[["HasMechanism", "HasKG", "HasMLR"]].mean().reset_index()
    x = np.arange(len(df_flag["Mode"]))
    width = 0.25

    plt.figure(figsize=(8, 4))
    plt.bar(x - width, df_flag["HasMechanism"], width, label="Mechanism-related")
    plt.bar(x, df_flag["HasKG"], width, label="KG-related")
    plt.bar(x + width, df_flag["HasMLR"], width, label="MLR-related")

    plt.xticks(x, df_flag["Mode"])
    plt.ylim(0, 1.05)
    plt.ylabel("Ratio")
    plt.title("Ratio of Answers Mentioning Mechanism / KG / MLR")
    plt.legend()
    for i in range(len(df_flag["Mode"])):
        plt.text(x[i] - width, df_flag["HasMechanism"][i] + 0.02,
                 f"{df_flag['HasMechanism'][i]:.2f}", ha="center", fontsize=8)
        plt.text(x[i], df_flag["HasKG"][i] + 0.02,
                 f"{df_flag['HasKG'][i]:.2f}", ha="center", fontsize=8)
        plt.text(x[i] + width, df_flag["HasMLR"][i] + 0.02,
                 f"{df_flag['HasMLR'][i]:.2f}", ha="center", fontsize=8)
    plt.tight_layout()
    flag_fig_path = os.path.join(fig_dir, "ratio_mech_kg_mlr_per_mode.png")
    plt.savefig(flag_fig_path, dpi=200)
    plt.close()
    print(f"[*] 已保存图像：{flag_fig_path}")

    # 5.3 平均 ContextSim（知识利用率）
    df_ctx = df_out.groupby("Mode")["ContextSim"].mean().reset_index()
    plt.figure(figsize=(6, 4))
    plt.bar(df_ctx["Mode"], df_ctx["ContextSim"])
    plt.ylabel("Average ContextSim")
    plt.ylim(0, 1.05)
    plt.title("Average Knowledge Utilization (ContextSim) per Mode")
    for i, v in enumerate(df_ctx["ContextSim"]):
        plt.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    ctx_fig_path = os.path.join(fig_dir, "avg_contextsim_per_mode.png")
    plt.savefig(ctx_fig_path, dpi=200)
    plt.close()
    print(f"[*] 已保存图像：{ctx_fig_path}")

    # 5.4 结构化输出比例（结论/依据/建议）
    df_struct = df_out.groupby("Mode")[["HasConclusion", "HasEvidence", "HasSuggestion"]].mean().reset_index()
    x = np.arange(len(df_struct["Mode"]))
    width = 0.25

    plt.figure(figsize=(8, 4))
    plt.bar(x - width, df_struct["HasConclusion"], width, label="Conclusion")
    plt.bar(x, df_struct["HasEvidence"], width, label="Evidence")
    plt.bar(x + width, df_struct["HasSuggestion"], width, label="Suggestion")

    plt.xticks(x, df_struct["Mode"])
    plt.ylim(0, 1.05)
    plt.ylabel("Ratio")
    plt.title("Ratio of Answers with Conclusion/Evidence/Suggestion")
    plt.legend()
    for i in range(len(df_struct["Mode"])):
        plt.text(x[i] - width, df_struct["HasConclusion"][i] + 0.02,
                 f"{df_struct['HasConclusion'][i]:.2f}", ha="center", fontsize=8)
        plt.text(x[i], df_struct["HasEvidence"][i] + 0.02,
                 f"{df_struct['HasEvidence'][i]:.2f}", ha="center", fontsize=8)
        plt.text(x[i] + width, df_struct["HasSuggestion"][i] + 0.02,
                 f"{df_struct['HasSuggestion'][i]:.2f}", ha="center", fontsize=8)
    plt.tight_layout()
    struct_fig_path = os.path.join(fig_dir, "ratio_structure_per_mode.png")
    plt.savefig(struct_fig_path, dpi=200)
    plt.close()
    print(f"[*] 已保存图像：{struct_fig_path}")

    # 5.5 质量指标综合对比（Accuracy / ROUGE-L / BERTScore / Distinct-2）
    df_quality = df_out.groupby("Mode")[["Accuracy", "ROUGE_L", "BERTScore", "Distinct2"]].mean().reset_index()
    metrics = ["Accuracy", "ROUGE_L", "BERTScore", "Distinct2"]
    x = np.arange(len(df_quality["Mode"]))
    width = 0.18

    plt.figure(figsize=(10, 4))
    for i, m in enumerate(metrics):
        plt.bar(x + (i - 1.5) * width, df_quality[m], width, label=m)
        for j in range(len(df_quality["Mode"])):
            plt.text(x[j] + (i - 1.5) * width, df_quality[m][j] + 0.01,
                     f"{df_quality[m][j]:.2f}", ha="center", fontsize=7)

    plt.xticks(x, df_quality["Mode"])
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Quality Metrics per Mode (Accuracy / ROUGE-L / BERTScore / Distinct-2)")
    plt.legend()
    plt.tight_layout()
    quality_fig_path = os.path.join(fig_dir, "quality_metrics_per_mode.png")
    plt.savefig(quality_fig_path, dpi=200)
    plt.close()
    print(f"[*] 已保存图像：{quality_fig_path}")

    # 5.6 Latency（响应时延）
    df_lat = df_out.groupby("Mode")["Time_sec"].mean().reset_index()
    plt.figure(figsize=(6, 4))
    plt.bar(df_lat["Mode"], df_lat["Time_sec"])
    plt.ylabel("Average Latency (s)")
    plt.title("Average Latency per Mode")
    for i, v in enumerate(df_lat["Time_sec"]):
        plt.text(i, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    lat_fig_path = os.path.join(fig_dir, "avg_latency_per_mode.png")
    plt.savefig(lat_fig_path, dpi=200)
    plt.close()
    print(f"[*] 已保存图像：{lat_fig_path}")

    print("\n[*] 消融实验批量运行与可视化已完成。你可以：")
    print(f"    - 用 Excel 打开 {out_csv} 做人工打分/检查单题效果；")
    print(f"    - 在 {fig_dir} 里查看")
    print(f"      - {len_fig_path}")
    print(f"      - {flag_fig_path}")
    print(f"      - {ctx_fig_path}")
    print(f"      - {struct_fig_path}")
    print(f"      - {quality_fig_path}")
    print(f"      - {lat_fig_path}")


if __name__ == "__main__":
    main()
