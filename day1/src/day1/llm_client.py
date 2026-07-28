import os
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from config import AppConfig

load_dotenv()


# 1. 定义Pydantic数据模型（和你之前学的写法完全一致）
class UserProfile(BaseModel):
    """用户个人信息结构"""
    name: str = Field(description="用户的真实姓名")
    age: int = Field(description="用户的年龄，整数类型")
    skills: list[str] = Field(description="用户掌握的技能列表，字符串数组")
    is_student: bool = Field(description="用户是否为在校学生")


# 2. 初始化OpenAI客户端（兼容所有OpenAI格式的模型）
config = AppConfig()
client = OpenAI(
    api_key=config.api_key,
    base_url=config.api_base_url
)


# 3. 调用结构化输出接口
def extract_user_info(text: str) -> UserProfile:
    response = client.beta.chat.completions.parse(
        model=config.llm_model,
        messages=[
            {"role": "system", "content": "你是信息提取专家，从用户输入的文本中提取结构化信息。"},
            {"role": "user", "content": f"从下面这段文本中提取用户信息：\n{text}"}
        ],
        # 核心：直接传入Pydantic类作为输出格式约束
        response_format=UserProfile,
        temperature=0.1
    )

    # 4. 直接拿到解析、校验完成的Pydantic对象
    return response.choices[0].message.parsed


if __name__ == "__main__":
    input_text = "我叫张三，今年25岁，会Python、数据分析和英语，已经工作了，不是学生。"
    user = extract_user_info(input_text)
    # 直接像普通对象一样访问字段，IDE有完整代码补全，类型安全
    print(user.name)  # 输出：张三