# model.py
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
import os

def load_model_and_tokenizer():
    """加载基础模型、LoRA适配器和分词器，返回 (model, tokenizer)"""
    # 1. 配置路径（与训练代码一致）
    base_model_name = "deepseek-ai/deepseek-llm-7b-base"
    cache_dir = "./models"
    lora_dir = "./deepseek_lora_adapter"
    offload_folder = "./model_offload"
    os.makedirs(offload_folder, exist_ok=True)

    # 2. 验证LoRA目录完整性
    if not os.path.exists(lora_dir):
        raise FileNotFoundError(f"LoRA目录不存在：{lora_dir}，请先训练")
    required_files = ["adapter_config.json", "adapter_model.bin"]
    for file in required_files:
        if not os.path.exists(os.path.join(lora_dir, file)):
            raise FileNotFoundError(f"LoRA缺少文件：{file}，请重新训练")

    # 3. 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        cache_dir=cache_dir
    )
    tokenizer.pad_token = tokenizer.eos_token  # 设置pad_token

    # 4. 加载量化配置（与训练一致）
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # 5. 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir=cache_dir,
        offload_folder=offload_folder,
        trust_remote_code=True
    )

    # 6. 加载LoRA适配器
    model = PeftModel.from_pretrained(
        base_model,
        lora_dir,
        device_map="auto"
    )
    model.eval()  # 推理模式

    print("模型和分词器加载完成！")
    return model, tokenizer
