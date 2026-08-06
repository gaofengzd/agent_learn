import os
from openai import OpenAI
from loguru import logger
from pydantic import BaseModel, Field, ValidationError
# from dotenv import load_dotenv
from src.day1.config import AppConfig
from src.day1.models import UserProfile, AnswerFormat
import json
from typing import List, Dict

from loguru import logger

# 配置日志文件：每天零点生成新文件，保留7天日志，只记录INFO及以上级别
logger.add(
    sink="logs/llm_client_{time:YYYY-MM-DD}.log",  # 日志文件路径，{time}会自动替换成日期
    rotation="00:00",  # 轮转规则：每天0点生成新文件
    retention="7 days",  # 保留规则：只保留最近7天的日志
    compression="zip",  # 旧日志自动压缩成zip，节省空间
    level="INFO",  # 只记录INFO及以上级别的日志
    encoding="utf-8",  # 编码，避免中文乱码
    enqueue=True  # 异步写入，多线程/异步场景不阻塞
)

# 2. 初始化OpenAI客户端（兼容所有OpenAI格式的模型）
config = AppConfig()
client = OpenAI(
    api_key=config.api_key,
    base_url=config.api_base_url
)


class MyAgent:
    def __init__(self):
        # 1. 设定 System Prompt (系统角色)
        self.system_prompt = {
            "role": "system",
            "content": (
                "你是一个专业的 C909 飞机故障诊断辅助 Agent。你具备以下能力：\n"
                "1. 引导用户描述具体的故障现象。\n"
                "2. 当需要输出诊断报告时，必须严格返回 JSON 格式数据。\n"
                "请保持专业、简洁。"
            )
        }
        # 初始化上下文历史
        self.history: List[Dict[str, str]] = [self.system_prompt]
        # 设置触发总结的阈值（对话轮次）
        self.summarize_threshold = 6

    def chat_stream(self, user_input: str):
        """流式多轮对话"""
        # 2. 追加 User Prompt
        self.history.append({"role": "user", "content": user_input})

        print("\nAgent: ", end="", flush=True)

        # 3. 调用 API 并开启流式输出 (stream=True)
        response = client.chat.completions.create(
            model=config.llm_model,  # 或你使用的任意大模型
            messages=self.history,
            stream=True
        )

        full_response = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                word = chunk.choices[0].delta.content
                print(word, end="", flush=True)
                full_response += word

        print("\n")

        # 4. 追加 Assistant Message，形成完整的对话记忆闭环
        self.history.append({"role": "assistant", "content": full_response})

        # 检查是否需要压缩上下文
        self.check_and_summarize()

    def generate_json_report(self):
        """让模型输出固定的 JSON 格式"""
        print("\n[系统] 正在生成结构化诊断报告...")

        # 临时构造一条要求输出 JSON 的消息
        temp_messages = self.history.copy()
        temp_messages.append({
            "role": "user",
            "content": "请根据之前的对话，提取故障特征。必须严格输出JSON，包含字段：'fault_type'(字符串), 'severity'(字符串，如高/中/低), 'suggested_action'(字符串)。"
        })

        response = client.chat.completions.create(
            model=config.llm_model,
            messages=temp_messages,
            response_format={"type": "json_object"}  # 强制 JSON 模式
        )

        json_str = response.choices[0].message.content
        print(f"生成的 JSON 数据:\n{json_str}")

        # 可以直接被 Python 解析为字典
        data_dict = json.loads(json_str)
        return data_dict

    def check_and_summarize(self):
        """尝试让模型总结上下文，减少历史 Token"""
        # 除去 system prompt，如果历史对话超过阈值
        if len(self.history) > self.summarize_threshold:
            print("\n[系统] 对话历史过长，正在启动上下文压缩策略...")

            summary_prompt = [
                {"role": "system", "content": "你是一个内容总结助手。"},
                {"role": "user",
                 "content": f"请将以下对话历史总结为一段不超过100字的精简摘要，保留核心的故障信息和排查进度：\n{str(self.history[1:])}"}
            ]

            response = client.chat.completions.create(
                model=config.llm_model,
                messages=summary_prompt
            )
            summary = response.choices[0].message.content

            # 清空旧历史，用摘要替换，保留 system_prompt
            self.history = [
                self.system_prompt,
                {"role": "assistant", "content": f"之前的对话摘要：{summary}"}
            ]
            print(f"[系统] 压缩完成，当前上下文已重置为摘要状态。\n")


class ChatAagent:
    def __init__(self, system_prompt: str = "你是一个专业、简洁的AI助手，回答通俗易懂。"):
        # 维护全局对话上下文
        self.message_history = [
            {"role": "system", "content": system_prompt}
        ]

    def stream_chat(self, user_input: str):
        """流式对话，逐字打字机输出，返回完整回答文本"""
        # 追加用户提问到历史
        self.message_history.append({"role": "user", "content": user_input})
        full_response = ""

        try:
            logger.info(f"发起对话，历史消息总数：{len(self.message_history)}")
            # 开启流式输出 stream=True
            stream = client.chat.completions.create(
                model=config.llm_model,
                messages=self.message_history,
                stream=True,
                temperature=0.7
            )

            print("\n🤖 助手：", end="", flush=True)
            # 遍历流式数据块
            for chunk in stream:
                delta_content = chunk.choices[0].delta.content
                if delta_content:
                    full_response += delta_content
                    print(delta_content, end="", flush=True)
            print("\n")

            # 将完整回复存入上下文，实现多轮记忆
            self.message_history.append({"role": "assistant", "content": full_response})
            logger.success("本轮对话完成，已保存对话历史")
            return full_response

        except Exception as e:
            logger.exception("对话调用发生异常")
            print(f"\n❌ 对话失败：{str(e)}\n")
            return ""

    def structured_chat(self, user_input: str) -> AnswerFormat | None:
        """结构化输出模式：强制返回固定JSON格式"""
        messages = [
            {
                "role": "system",
                "content": "你是信息总结助手，严格输出纯JSON，不要任何多余文字。JSON字段要求：summary(一句话总结), key_points(要点数组), suggestion(建议)"
            },
            {"role": "user", "content": user_input}
        ]
        try:
            response = client.chat.completions.create(
                model=config.llm_model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.1
            )
            json_str = response.choices[0].message.content
            # Pydantic一步解析+校验JSON
            result = AnswerFormat.model_validate_json(json_str)
            logger.success("结构化提取成功")
            return result
        except ValidationError as e:
            logger.error(f"JSON格式校验失败：{e.errors()}")
            return None
        except Exception as e:
            logger.exception("结构化调用异常")
            return None

    def compress_history(self, max_msg_count: int = 8):
        """上下文压缩，减少token消耗（练习任务：总结历史缩短上下文）"""
        if len(self.message_history) <= max_msg_count:
            return
        # 保留system提示词 + 最近3轮对话，剩余历史做总结
        system_msg = self.message_history[0]
        latest_messages = self.message_history[-3:]
        old_messages = self.message_history[1:-3]

        summary_prompt = f"精简总结下面全部对话内容，控制在150字以内，只保留核心话题：{old_messages}"
        summary_res = self.stream_chat(summary_prompt)

        # 重构上下文：原system + 历史摘要 + 最新对话
        self.message_history = [
                                   system_msg,
                                   {"role": "system", "content": f"过往对话摘要：{summary_res}"}
                               ] + latest_messages
        logger.info(f"上下文已压缩，当前消息数量：{len(self.message_history)}")
