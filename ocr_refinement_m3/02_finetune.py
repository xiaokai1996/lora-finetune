#!/Users/bytedance/codes/lora-finetune/.venv/bin/python3
"""
功能：加载Qwen-1.5-1.8B，用650条数据做LoRA微调，支持checkpoint续跑，保存模型到指定目录
执行方式：python3 02_finetune.py
依赖：需先执行01_generate_data.py生成数据，且安装依赖（pip3 install torch transformers datasets peft accelerate）
"""
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model

def main():
    # ===================== 1. 配置参数（可按需修改） =====================
    MODEL_NAME = "Qwen/Qwen1.5-1.8B"
    DATA_PATH = "/Users/bytedance/codes/lora-finetune/ocr_refinement_m3/data/650case.jsonl"
    OUTPUT_DIR = "/Users/bytedance/codes/lora-finetune/ocr_refinement_m3/model_finetuned_output"
    # 新增：指定checkpoint恢复路径（设为None则自动找最新的）
    RESUME_FROM_CHECKPOINT = None  # 也可以指定具体路径如：OUTPUT_DIR + "/checkpoint-50"

    # ===================== 2. 加载模型和Tokenizer =====================
    print("🚀 开始加载模型（首次运行会自动下载Qwen-1.8B，约1.8GB）")
    # Mac CPU专用配置：不量化、float32、device_map=cpu
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="right"  # 右填充，避免影响生成
    )
    # Qwen默认无pad_token，手动设置为eos_token
    tokenizer.pad_token = tokenizer.eos_token

    # ===================== 3. 配置LoRA（极简版，加快训练） =====================
    model = prepare_model_for_lora_training(model)  # 简化版prepare，适配CPU

    lora_config = LoraConfig(
        r=4,                
        lora_alpha=16,      # 缩放因子
        lora_dropout=0.05,  # Dropout防止过拟合
        target_modules=["q_proj", "v_proj"],  
        bias="none",
        task_type="CAUSAL_LM"  # 因果语言模型，适配Qwen
    )

    model = get_peft_model(model, lora_config)
    # 打印可训练参数（约50万，极快）
    model.print_trainable_parameters()

    # ===================== 4. 加载并格式化数据 =====================
    def format_example(example):
        """将数据格式化为Qwen的对话格式，并进行Masking和截断处理"""
        # 1. 分别处理Input和Output
        user_text = f"<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n"
        assistant_text = f"{example['output']}<|im_end|>"
        
        # 2. Tokenize
        user_tokens = tokenizer(user_text, add_special_tokens=False)["input_ids"]
        assistant_tokens = tokenizer(assistant_text, add_special_tokens=False)["input_ids"]
        
        # 3. 动态截断（保留完整的回复，截断过长的输入）
        # 预留给回复的长度
        max_seq_length = 2048  # 增加上下文长度到2048
        if len(assistant_tokens) > max_seq_length:
            # 如果回复本身就超长，强制截断回复（极端情况）
            assistant_tokens = assistant_tokens[:max_seq_length]
            user_tokens = []
        else:
            # 剩余空间给输入
            remaining_length = max_seq_length - len(assistant_tokens)
            if len(user_tokens) > remaining_length:
                # 截断输入（保留后面的部分通常更有用，或者保留前面的部分）
                # 这里选择保留前面的部分（OCR文档通常开头包含关键信息），如果需要保留末尾改为 user_tokens[-remaining_length:]
                user_tokens = user_tokens[:remaining_length]
        
        # 4. 拼接
        input_ids = user_tokens + assistant_tokens
        attention_mask = [1] * len(input_ids)
        
        # 5. 构建Labels（Mask掉User部分）
        # User部分设为-100（不计算Loss），Assistant部分保留原ID
        labels = [-100] * len(user_tokens) + assistant_tokens
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    # 加载JSONL数据
    dataset = load_dataset("json", data_files=DATA_PATH)["train"]
    # 格式化数据
    dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    # ===================== 5. 配置训练参数（Mac专用，支持Checkpoint） =====================
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,  # 梯度累积，模拟batch_size=8，稳定训练
        learning_rate=3e-4,        
        logging_steps=2,           
        # 核心修改1：开启checkpoint保存
        save_steps=50,            # 每50步保存一次checkpoint（可根据数据量调整）
        save_total_limit=3,       # 只保留最近3个checkpoint，避免占满磁盘
        save_strategy="steps",    # 按步数保存（替代原来的"no"）
        use_cpu=True,             
        fp16=False,                
        gradient_checkpointing=False,  
        report_to="none",          
        remove_unused_columns=False,
        # 核心修改2：开启断点续跑的关键参数
        load_best_model_at_end=False,  # 不需要加载最优模型（LoRA微调场景）
    )

    # 数据整理器（自动padding）
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt"
    )

    # ===================== 6. 启动训练（支持续跑） =====================
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # 自动检测最新的checkpoint
    if RESUME_FROM_CHECKPOINT is None and os.path.exists(OUTPUT_DIR):
        checkpoints = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("checkpoint-")]
        if checkpoints:
            latest_checkpoint_name = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            RESUME_FROM_CHECKPOINT = os.path.join(OUTPUT_DIR, latest_checkpoint_name)
            print(f"🔍 检测到最新checkpoint：{RESUME_FROM_CHECKPOINT}，将从该位置续跑")
    
    print("🔥 开始训练（支持断点续跑，650条数据，Mac CPU）")
    trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)

    # ===================== 7. 保存最终模型 =====================
    # 保存最终的LoRA模型（包含所有训练参数）
    final_model_dir = os.path.join(OUTPUT_DIR, "final_model")
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    print(f"✅ 训练完成！最终模型保存到：{final_model_dir}")
    print(f"📌 Checkpoint文件保存在：{OUTPUT_DIR}（以checkpoint-开头的文件夹）")

# 简化版prepare_model_for_kbit_training（适配CPU）
def prepare_model_for_lora_training(model):
    for param in model.parameters():
        param.requires_grad = False  # 冻结基座模型
        if param.ndim == 1:
            # 避免梯度溢出
            param.data = param.data.to(torch.float32)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    return model

if __name__ == "__main__":
    main()
    