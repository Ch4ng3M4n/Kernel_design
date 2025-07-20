# 环境安装：pip install transformers peft datasets accelerate bitsandbytes
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch

# 1. 加载 DeepSeek 模型和分词器
model_name = "deepseek-ai/deepseek-llm-7b-base"  # 或 "deepseek-ai/deepseek-coder-6.7b"
cache_dir = "./models"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

# 量化配置（节省显存）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, # 4-bit 量化
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,  # 应用4-bit量化
    device_map="auto",              # 自动分配GPU/CPU
    trust_remote_code=True,    # DeepSeek 可能需要此参数
    cache_dir=cache_dir

)

# 2. 配置LoRA适配器
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    task_type="CAUSAL_LM",
    bias="none"
)
model = get_peft_model(model, peft_config)

# 3. 加载数据集
dataset = load_dataset("json", data_files="data/train.json")

# 4. 数据预处理（适配批量处理）
def tokenize_function(examples):
    # 修正：使用"question"和"answer"字段
    texts = [
        f"Instruction: {q}\nOutput: {a}"
        for q, a in zip(examples["question"], examples["answer"])
    ]

    # 批量处理文本
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=512,
        padding="max_length"  # 重要：确保所有序列长度一致
    )

    # 为了适配 Trainer API，可能需要调整标签
    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized


tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names  # 移除原始文本列，节省内存
)


# 5. 训练配置
training_args = TrainingArguments(
    output_dir="./deepseek_fine_tuned",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=10,
    fp16=True if torch.cuda.is_available() else False,
    logging_steps=1,
    save_strategy="epoch",
)

# 6. 训练模型
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)
trainer.train()

# 7. 保存模型
model.save_pretrained("./deepseek_lora_adapter")
