"""
功能：生成100条格式统一的测试数据，输出到demo_data.jsonl
执行方式：python3 01_generate_data.py
"""
import json
import random

def generate_demo_data(num_samples=100, output_path="/Users/bytedance/codes/lora-finetune/ocr_refinement_m3/data/demo_data.jsonl"):
    # 基础模板（固定格式，仅替换行号）
    base_template = {
        "input": "[doc]\n[{start_line}]3.1 双绞线\n[{end_line}]双绞线(Twisted Pair wire, TP)是建筑设备监控系统中最常用的一种传输介质。\n[/doc]",
        "output": "merge_line(start={start_line}, end={end_line}, content=\"3.1 双绞线\\n双绞线(Twisted Pair wire, TP)是建筑设备监控系统中最常用的一种传输介质。\")"
    }
    
    # 生成指定数量的样本
    demo_samples = []
    for i in range(num_samples):
        # 随机生成不重复的行号
        start_line = 10000 + i * 10
        end_line = start_line + 1
        
        # 替换行号
        input_text = base_template["input"].format(start_line=start_line, end_line=end_line)
        output_text = base_template["output"].format(start_line=start_line, end_line=end_line)
        
        demo_samples.append({
            "input": input_text,
            "output": output_text
        })
    
    # 保存为JSONL格式
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in demo_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    print(f"✅ 成功生成{num_samples}条测试数据，保存到：{output_path}")

if __name__ == "__main__":
    generate_demo_data(num_samples=100)
