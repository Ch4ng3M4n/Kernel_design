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
# 关键修复：设置右侧填充，确保有效文本靠左
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    padding_side="right"  # 右侧填充，避免有效文本被挤压到右侧
)
# 补充pad_token（DeepSeek模型默认可能没有pad_token）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # 用EOS token作为填充标记

# 量化配置（节省显存）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 4-bit 量化
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,  # 应用4-bit量化
    device_map="auto",  # 自动分配GPU/CPU
    trust_remote_code=True,  # DeepSeek 需要此参数
    cache_dir=cache_dir
)

# 2. 配置LoRA适配器
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj"],
    lora_dropout=0.1,
    task_type="CAUSAL_LM",
    bias="none"
)
model = get_peft_model(model, peft_config)
# ...（前面的导入和模型加载代码保持不变）

# 3. 加载数据集并划分训练/验证集
dataset = load_dataset("json", data_files="data/train.json")
dataset = dataset["train"].train_test_split(test_size=0.1)  # 90%训练，10%验证


# 4. 优化后的数据预处理函数
def tokenize_function(examples):
    # 拼接输入文本
    texts = [
        f"Instruction: {q}\nOutput: {a}"
        for q, a in zip(examples["question"], examples["answer"])
    ]

    # 批量处理文本
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=512,
        padding="max_length",
        add_special_tokens=False
    )

    # 更简单的前缀匹配方式
    output_prefix = tokenizer.encode("\nOutput: ", add_special_tokens=False)
    prefix_len = len(output_prefix)

    labels = []
    for input_ids in tokenized["input_ids"]:
        # 查找"\nOutput: "的位置
        try:
            # 使用列表切片匹配
            prefix_pos = next(
                i for i in range(len(input_ids) - prefix_len + 1)
                if input_ids[i:i + prefix_len] == output_prefix
            )
            # 设置标签（忽略Instruction部分）
            label = [-100] * (prefix_pos + prefix_len) + input_ids[prefix_pos + prefix_len:]
        except StopIteration:
            # 如果没有找到前缀，使用全标签（不推荐，但比崩溃好）
            print("警告：未找到Output前缀，使用全标签")
            label = input_ids.copy()

        labels.append(label)

    tokenized["labels"] = labels
    return tokenized


# 应用预处理（同时处理训练和验证集）
print("开始预处理数据集...")
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
    desc="Tokenizing dataset"
)

# 5. 训练配置（添加验证相关参数）
training_args = TrainingArguments(
    output_dir="./deepseek_fine_tuned",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    num_train_epochs=6,
    evaluation_strategy="epoch",  # 每个epoch后评估
    fp16=True if torch.cuda.is_available() else False,
    logging_steps=1,
    warmup_ratio=0.1,              # 10%步数用于学习率热身
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,  # 训练结束时加载最佳模型
    metric_for_best_model="eval_loss",  # 根据验证损失选择最佳模型
)

# 6. 训练模型（添加验证集）
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],  # 添加验证集
)

# 训练并评估
print("开始训练...")
trainer.train()

# 7. 保存模型（保持不变）
model.save_pretrained("./deepseek_lora_adapter")
tokenizer.save_pretrained("./deepseek_lora_adapter")
print(f"模型已保存至 ./deepseek_lora_adapter")