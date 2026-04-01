from llm_client import call_openrouter

user_input = "اشرح لي مفهوم البرمجة الكائنية بطريقة بسيطة"
answer = call_openrouter(user_input)
print(answer)