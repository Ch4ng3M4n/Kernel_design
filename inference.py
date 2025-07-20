# inference_without_lora.py
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os

# 1. 配置路径
base_model_name = "deepseek-ai/deepseek-llm-7b-base"
cache_dir = "./models"
offload_folder = "./model_offload"
os.makedirs(offload_folder, exist_ok=True)

# 2. 加载分词器
tokenizer = AutoTokenizer.from_pretrained(
    base_model_name,
    cache_dir=cache_dir
)
tokenizer.pad_token = tokenizer.eos_token

# 3. 加载量化配置（与之前保持一致）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# 4. 加载基础模型（不应用任何LoRA适配器）
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
    cache_dir=cache_dir,
    offload_folder=offload_folder,
    trust_remote_code=True
)
model.eval()  # 推理模式


# 5. 推理函数（输入格式与LoRA版本保持一致）
def generate_answer(question, max_new_tokens=512, temperature=0.7):
    input_text = f"Instruction: {question}\nOutput:"
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="max_length"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# 6. 测试推理
if __name__ == "__main__":
    question = "操作系统的双重模式是啥"
    try:
        answer = generate_answer(question)
        print(f"问题：{question}")
        print(f"回答：{answer}")
    except Exception as e:
        print(f"推理失败：{e}")