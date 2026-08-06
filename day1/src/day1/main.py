from src.day1.llm_agent.llm_client import MyAgent, ChatAagent

def main():
    bot = ChatAagent()
    print("===== 命令行流式聊天机器人 =====")
    print("指令说明：")
    print("  exit / 退出   ：关闭程序")
    print("  /json + 问题  ：结构化JSON输出模式")
    print("  /compress      ：手动压缩对话历史，减少token")
    print("====================================\n")

    while True:
        user_text = input("👤 你：").strip()
        if not user_text:
            continue
        # 退出指令
        if user_text.lower() in ["exit", "退出"]:
            print("👋 对话结束，再见！")
            break
        # 手动压缩上下文指令
        if user_text == "/compress":
            bot.compress_history()
            print("✅ 对话历史已完成压缩")
            continue
        # 结构化JSON输出指令
        if user_text.startswith("/json"):
            query = user_text.replace("/json", "").strip()
            if not query:
                print("请在/json后输入需要总结的内容")
                continue
            res_data = bot.structured_chat(query)
            if res_data:
                print("\n📋 结构化输出结果：")
                print(f"一句话总结：{res_data.summary}")
                print(f"核心要点：{res_data.key_points}")
                print(f"优化建议：{res_data.suggestion}\n")
            continue
        # 普通多轮流式聊天
        bot.stream_chat(user_text)

def main1():
    print("=== 故障诊断命令行 Agent 启动 (输入 'quit' 退出, 输入 'report' 生成报告) ===")
    agent = MyAgent()
    
    while True:
        user_input = input("User: ")
        if user_input.lower() in ['quit', 'exit']:
            break
        elif user_input.lower() == 'report':
            agent.generate_json_report()
            continue
            
        agent.chat_stream(user_input)

if __name__ == "__main__":
    main()
