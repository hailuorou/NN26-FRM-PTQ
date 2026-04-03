from quantize.int_linear_real import load_quantized_model
from accelerate import infer_auto_device_map, dispatch_model
import torch

model_path = '/root/path'
wbits = 3
abits = 3
group_size = 128
use_act_quant = True
# qwen2.5-14b-instruct
sensitive_group= [27, 20, 23, 25, 29, 22, 24, 4, 47, 3, 2, 1]
robust_group= [41, 39, 38, 42, 15, 40, 11, 7, 12]
model, tokenizer = load_quantized_model(model_path=model_path, wbits=wbits, group_size=group_size, use_act_quant=use_act_quant, sensitive_group=sensitive_group, robust_group=robust_group)
print(f"memory footprint after loading quantized model: {torch.cuda.max_memory_allocated('cuda') / 1024**3:.2f}GiB")
model.eval()

model.cpu()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
prompt = "What are the logical circuits?"
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512,
    # min_new_tokens=512  # 强制生成固定长度
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)
