from model import load_model_and_tokenizer  # 导入加载函数
import torch


def generate_answer(model, tokenizer, question, max_new_tokens=512, temperature=0.7):
    """使用加载好的模型和分词器生成回答"""
    # 输入格式必须与训练一致
    input_text = f"Instruction: {question}\nOutput: "
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(model.device)  # 自动匹配模型设备

    # 推理（禁用梯度计算节省显存）
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # 解码生成的内容
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 提取Output部分
    output_start = input_text.rfind("Output: ") + len("Output: ")
    answer = generated_text[output_start:].strip()

    return answer


# 主程序：加载模型并测试推理
if __name__ == "__main__":
    # 1. 加载模型和分词器（只需要加载一次）
    try:
        model, tokenizer = load_model_and_tokenizer()
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败：{e}")
        exit(1)

    # 2. 测试推理
    while True:
        question = input("请输入问题（输入'退出'结束）：")
        if question == "退出":
            break
        try:
            answer = generate_answer(model, tokenizer, question)
            print(f"\n回答：{answer}\n")
        except Exception as e:
            print(f"推理失败：{e}")    