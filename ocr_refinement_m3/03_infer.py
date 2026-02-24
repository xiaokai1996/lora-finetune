"""
功能：加载微调后的模型，输入测试文本，验证生成效果
执行方式：python3 03_infer.py
依赖：需先执行02_finetune.py完成训练
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

def main():
    # ===================== 1. 配置路径 =====================
    BASE_MODEL = "Qwen/Qwen1.5-1.8B"
    LORA_MODEL = "/Users/bytedance/codes/lora-finetune/ocr_refinement_m3/model_finetuned_output"  # 微调后的LoRA模型路径

    # ===================== 2. 加载模型和Tokenizer =====================
    print("🚀 加载模型和LoRA权重...")
    # 加载基座模型（Mac CPU专用）
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    # 加载LoRA权重（核心：将微调的适配器贴到基座模型上）
    model = PeftModel.from_pretrained(model, LORA_MODEL)
    
    # 加载微调时保存的Tokenizer（保证配置一致）
    tokenizer = AutoTokenizer.from_pretrained(
        LORA_MODEL,
        trust_remote_code=True,
        padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token

    # ===================== 3. 测试输入 =====================
    test_input = """[doc]
[doc]
[0000] 男人为什么要经常说谎啊？？？？当事业和爱情发生冲突时我迷茫，不知道如何处理！我是该放弃哪一头？
[0001] 女人可以什么都没有，但一定需要一份真挚的感情来呵护她；女人可以什么都没有，但一定要有一份自信来促进她；女人可以什么都没有，但一定更需要自尊自重.......一个爱她的男人可以给女人很多其他方面不能满足的需求；爱她会帮助她事业更有所成。但现在的男人会爱一个什么都没有只有感情需求的女人吗？哪怕她甚至没有工作？要是他是优秀的男人，更会对你自身要求更高。还是现实点吧，这个问题不该你来考虑，你应该先把自己的电充足了，如果他真爱你这么深，会给你答案的。
[0002] 现在的人不晓得几现实`好好工作有什么不对？不赚钱养家喝西北风去啊？不赚钱不赚钱不赚钱
[0003] 男人说谎这事怪不得男人，据说是男性基因释然，男人也没办法控制。 事业和爱情冲突时，以事业为主，没有事业就没有独立，没有独立的话，短期是可以的，长期势必失去尊重和任何感情。
[0004] 目前着个社会,男人在女人心里看来,是一个花言巧语,只是下贱的人的,几乎女人们都认为好男人绝种了,或许是被男人伤害的太深(女人看男人) 对于男人来说,风流是他们的本性,捻花惹草更不在话下,区区几句谎话只是随手拿来,或许只是习惯,更或者是一种自信过度`` 古人说,男人应该先立业再成家!其实不然`如果爱情对你的事业有帮助,那么有何不可爱情事业一起发展!一边是你深爱的人,一边是你的前程!就算你很爱她(他),你想过如果你放弃事业,为的是他,你将来会后悔吗?如果你决然的回答是不!或许应该放弃```假如放弃的不是事业`而是爱情```那么你以后飞黄腾达之时,你又会惋惜你的爱情么!不同的人有不同的想法,(只是随口说 说)))
[/doc]"""

    # ===================== 4. 推理生成 =====================
    print("🔍 开始生成结果...")
    # 文本转token（模型可识别的数字）
    inputs = tokenizer(
        test_input,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    # 生成修复函数（极简配置，加快速度）
    outputs = model.generate(
        **inputs,
        max_new_tokens=1000,  # 最多生成100个token
        temperature=0.1,     # 低温度，输出更稳定
        do_sample=False,     # 关闭采样，加快生成
        pad_token_id=tokenizer.pad_token_id
    )

    # ===================== 5. 输出结果 =====================
    # 解码token为文本，去掉特殊符号
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 提取模型生成的部分（去掉输入文本）
    generated_text = result.replace(test_input, "").strip()

    # 打印最终结果
    print("\n===== 推理结果 =====")
    print(f"输入文本：\n{test_input}")
    print(f"\n模型生成的修复函数：\n{generated_text}")
    print("\n🎉 推理验证完成！")

if __name__ == "__main__":
    main()
    