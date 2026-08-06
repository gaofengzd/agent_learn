import os
from datetime import datetime

from openai import OpenAI
from loguru import logger
import json

from src.day1.config import AppConfig

# 配置日志文件：每天零点生成新文件，保留7天日志，只记录INFO及以上级别
logger.add(
    sink="logs/agent_{time:YYYY-MM-DD}.log",  # 日志文件路径，{time}会自动替换成日期
    rotation="00:00",  # 轮转规则：每天0点生成新文件
    retention="7 days",  # 保留规则：只保留最近7天的日志
    compression="zip",  # 旧日志自动压缩成zip，节省空间
    level="INFO",  # 只记录INFO及以上级别的日志
    encoding="utf-8",  # 编码，避免中文乱码
    enqueue=True  # 异步写入，多线程/异步场景不阻塞
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "执行基本的数学运算，支持加减乘除",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，例如 '3 + 5' 或 '10 / 2'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前的日期和时间",
            "parameters": {}
        }
    }
]

config = AppConfig()
client = OpenAI(
    api_key=config.api_key,
    base_url=config.api_base_url
)
model = config.llm_model


def run_agent(user_message: str) -> str:
    messages = [{"role": "user", "content": user_message}]
    logger.info(f"用户的问题是：{user_message}")

    # 第一次调用模型，携带工具定义
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto"  # 让模型自动决定
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    # 情况1：模型直接回答（无需工具）
    if not tool_calls:
        logger.info("无需调用工具，直接回答问题")
        return response_message.content

    # 情况2：模型要求调用工具
    # 先把模型的响应（含 tool_calls）加入消息列表
    messages.append(response_message)

    # 遍历所有工具调用（一次可能调用多个）
    for tool_call in tool_calls:
        logger.info(f"tool_call的参数：{tool_call}")
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)  # 解析 JSON
        result = execute_tool(tool_name, tool_args)

        # 将工具执行结果以 role="tool" 发回
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result
        })

    # 第二次调用模型，传入工具结果
    second_response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    logger.info(f"agent回答如下：{second_response.choices[0].message.content}")
    return second_response.choices[0].message.content


def execute_tool(tool_name: str, tool_args: dict) -> str:
    """根据工具名和参数执行对应操作，返回结果字符串"""
    if tool_name == "calculator":
        logger.info("调用计算器工具")
        expr = tool_args.get("expression", "")
        try:
            # 注意：生产环境请用安全解析（如 ast.literal_eval 或计算库），此处为演示
            result = eval(expr)
            return str(result)
        except Exception as e:
            logger.exception(e)
            return f"计算错误: {str(e)}"

    elif tool_name == "get_current_time":
        logger.info("调用时间工具")
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    else:
        logger.error(f"未知工具: {tool_name}")
        return f"未知工具: {tool_name}"


# 测试
if __name__ == "__main__":
    print(run_agent("3 + 5 等于多少？"))
    print(run_agent("现在几点了？"))
    print(run_agent("今天天气如何？"))  # 无对应工具，模型会礼貌拒绝
