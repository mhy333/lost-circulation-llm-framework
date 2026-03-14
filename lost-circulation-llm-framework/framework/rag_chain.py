# -*- coding: utf-8 -*-
"""
rag_chain.py —— RAG 生成链（ChatGLM3 共享/独立双模式）
"""
from typing import List, Dict, Optional
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from config import MODEL_DIR, MAX_NEW_TOKENS, TEMPERATURE, TOP_P
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0


def _detect_lang(s: str) -> str:
    if not s:
        return "unknown"
    zh_ratio = len(re.findall(r'[\u4e00-\u9fff]', s)) / max(len(s), 1)
    if zh_ratio > 0.2:
        return "zh"
    try:
        lang = detect(s)
        if "zh" in lang:
            return "zh"
        if "en" in lang:
            return "en"
    except Exception:
        pass
    return "unknown"


def _looks_chinese(s: str) -> bool:
    if not s:
        return False
    zh = re.findall(r'[\u4e00-\u9fff]', s)
    return (len(zh) / max(len(s), 1)) > 0.2


class RAGChain:
    def __init__(self, model=None, tokenizer=None):
        """
        优先使用外部共享的 model/tokenizer；若为 None，则独立加载（用于 CLI/调试）。
        """
        if model is not None and tokenizer is not None:
            print("[*] RAGChain 使用共享 ChatGLM3-6B 实例 ✅")
            self.model = model
            self.tokenizer = tokenizer
        else:
            print("[*] 独立模式：加载 ChatGLM3-6B 本地模型（仅用于调试或 CLI）...")
            self.tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
            qconfig = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                str(MODEL_DIR),
                trust_remote_code=True,
                quantization_config=qconfig,
                device_map="auto"
            ).eval()

        # eos / pad 兜底
        self.eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if self.eos_id is None and hasattr(self.tokenizer, "eos_token"):
            self.eos_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)
        self.pad_id = getattr(self.tokenizer, "pad_token_id", None) or self.eos_id

        self.cn_header = (
            "你是“井漏风险评估与防漏堵漏”领域的科研专家助手。"
            "请参考已检索到的资料作答，并按照以下格式输出：\n"
            "【结论】\n"
            "【依据】详情见参考资料部分。\n"
            "【建议】\n"
        )
        self.en_header = (
            "You are a research assistant specialized in lost-circulation risk assessment and prevention. "
            "Please answer based on the retrieved references and output in the following format:\n"
            "Conclusion\n"
            "Evidence: See the References section.\n"
            "Recommendations\n"
        )

    def build_context(self, docs: List[Dict]) -> str:
        """
        把检索到的文档整理为一个连续的“参考资料”文本块。
        """
        if not docs:
            return ""
        lines = []
        for i, d in enumerate(docs, 1):
            src = d.get("src", "?")
            text = (d.get("text", "") or "").replace("\n", " ").strip()
            if len(text) > 300:
                text = text[:300] + "..."
            lines.append(f"[{i}] 来源: {src}\n{text}")
        return "\n\n".join(lines)

    def format_prompt(self, query: str, docs: List[Dict], lang: Optional[str] = None) -> str:
        """
        构造 RAG Prompt：
        - 若 docs 为空，则退化为纯 LM（但仍保留角色指令）；
        - 若 docs 非空，则在 Prompt 中追加【参考资料】段落。
        """
        if lang is None:
            lang = _detect_lang(query)

        context = self.build_context(docs)

        if lang == "en" or (lang == "unknown" and not _looks_chinese(query)):
            header = self.en_header
            if context:
                prompt = (
                    f"{header}\n"
                    f"User Question:\n{query}\n\n"
                    f"References:\n{context}\n\n"
                    f"Assistant:\n"
                )
            else:
                prompt = (
                    f"{header}\n"
                    f"User Question:\n{query}\n\n"
                    f"Assistant:\n"
                )
        else:
            header = self.cn_header
            if context:
                prompt = (
                    f"{header}\n"
                    f"【用户问题】\n{query}\n\n"
                    f"【参考资料】\n{context}\n\n"
                    f"【模型回答】\n"
                )
            else:
                prompt = (
                    f"{header}\n"
                    f"【用户问题】\n{query}\n\n"
                    f"【模型回答】\n"
                )

        return prompt.strip()

    def clean_output(self, output: str) -> str:
        output = re.sub(r"(参考文献|References?)[:：].*", "", output, flags=re.S | re.I)
        #output = re.sub(r"(参考文献|References?)[:：][^\n]*", "", output, flags=re.I)
        output = re.sub(r"(作者|Author)[:：].*", "", output)
        output = re.sub(r"\[[0-9]+\][^\n]*", "", output)
        output = re.sub(r"\n{2,}", "\n", output)
        output = re.sub(r"([。！？!?])\1+", r"\1", output)
        keep_labels = ["结论", "依据", "建议", "Conclusion", "Evidence", "Recommendations"]
        cleaned = []
        for line in output.splitlines():
            if not any(k in line for k in keep_labels) and len(line.strip()) == 0:
                continue
            cleaned.append(line)
        return "\n".join(cleaned).strip()

    @torch.no_grad()
    def generate(self, prompt: str) -> str:
        torch.cuda.empty_cache()
        try:
            model_device = next(self.model.parameters()).device
        except StopIteration:
            model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(model_device)
        input_len = inputs["input_ids"].shape[-1]

        out = self.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            repetition_penalty=1.05,
            eos_token_id=self.eos_id,
            pad_token_id=self.pad_id
        )

        gen_ids = out[0][input_len:]
        if gen_ids.numel() == 0:
            text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        else:
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        return self.clean_output(text)

    def answer(self, query: str, docs: List[Dict], lang: Optional[str] = None) -> str:
        prompt = self.format_prompt(query, docs, lang=lang)
        return self.generate(prompt)
