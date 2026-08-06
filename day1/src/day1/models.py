from pydantic import BaseModel, Field


# 1. 定义Pydantic数据模型（和你之前学的写法完全一致）
class UserProfile(BaseModel):
    """用户个人信息结构"""
    name: str = Field(description="用户的真实姓名")
    age: int = Field(description="用户的年龄，整数类型")
    skills: list[str] = Field(description="用户掌握的技能列表，字符串数组")
    is_student: bool = Field(description="用户是否为在校学生")
    

class AnswerFormat(BaseModel):
    summary: str
    key_points: list[str]
    suggestion: str