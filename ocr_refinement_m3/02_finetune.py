#!/Users/bytedance/codes/lora-finetune/.venv/bin/python3
"""
功能：加载Qwen-1.5-1.8B，用100条数据做LoRA微调，保存模型到mac_demo/
执行方式：python3 02_finetune.py
依赖：需先执行01_generate_data.py生成数据，且安装依赖（pip3 install torch transformers datasets peft accelerate）
"""
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
    # lora_config = LoraConfig(
    #     r=4,                # 低秩，减少参数
    #     lora_alpha=16,      # 缩放因子
    #     lora_dropout=0.05,  # Dropout防止过拟合
    #     target_modules=["q_proj", "v_proj"],  # 训练q_proj和v_proj，适配Qwen1.5
    #     bias="none",
    #     task_type="CAUSAL_LM"  # 因果语言模型，适配Qwen
    # )

    # 适配小样本650条高质量数据训练参数
    lora_config = LoraConfig(
        r=2,                
        lora_alpha=8,      # 缩放因子
        lora_dropout=0.01,  # Dropout防止过拟合
        target_modules=["q_proj", "v_proj"],  
        bias="none",
        task_type="CAUSAL_LM"  # 因果语言模型，适配Qwen
    )

    model = get_peft_model(model, lora_config)
    # 打印可训练参数（约50万，极快）
    model.print_trainable_parameters()

    # ===================== 4. 加载并格式化数据 =====================
    def format_example(example):
        """将数据格式化为Qwen的对话格式"""
        prompt = f"<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n{example['output']}<|im_end|>"
        # Tokenize
        tokenized = tokenizer(
            prompt,
            truncation=True,
            max_length=512,  # 限制长度
            padding=False,   # 交给DataCollator处理padding
            return_tensors=None
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    # 加载JSONL数据
    dataset = load_dataset("json", data_files=DATA_PATH)["train"]
    # 格式化数据
    dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    # ===================== 5. 配置训练参数（Mac专用） =====================
    # training_args = TrainingArguments(
    #     output_dir=OUTPUT_DIR,
    #     num_train_epochs=EPOCHS,
    #     per_device_train_batch_size=BATCH_SIZE,
    #     learning_rate=3e-4,        # 较大的学习率，加快收敛
    #     logging_steps=2,           # 每2步打印日志，快速看进度
    #     save_steps=1000,           # 仅100条数据，无需频繁保存
    #     use_cpu=True,              # 强制使用CPU
    #     fp16=False,                # Mac CPU不支持FP16
    #     gradient_checkpointing=False,  # 关闭，节省内存
    #     report_to="none",          # 不使用wandb，避免依赖
    #     remove_unused_columns=False,
    #     save_strategy="no"         # 仅保存最终模型，加快速度
    # )

    # 650条高质量数据训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,                # 仅1轮
        per_device_train_batch_size=8,     # 最大批次，总步数仅82步
        learning_rate=5e-4,                # 大学习率，加快收敛
        warmup_ratio=0.0,                  # 关闭预热，减少计算
        lr_scheduler_type="constant",      # 固定学习率，最快

        # 日志/保存：极致减少IO耗时
        logging_steps=50,                  # 每50步打印1次（原2步太频繁）
        save_steps=1000,
        save_strategy="no",                # 仅训练结束保存

        # Mac CPU关键配置（支撑batch=8）
        use_cpu=True,
        fp16=False,
        gradient_checkpointing=True,       # 必须开！节省50%内存
        report_to="none",
        remove_unused_columns=False,
    )

    # 数据整理器（自动padding）
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt"
    )

    # ===================== 6. 启动训练 =====================
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )
    
    print("🔥 开始训练（650条数据，1轮，Mac CPU约2-5分钟）")
    trainer.train()

    # ===================== 7. 保存模型 =====================
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ 训练完成！模型保存到：{OUTPUT_DIR}")

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
