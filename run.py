import os
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

start_time = datetime.utcnow()
tokenizer = AutoTokenizer.from_pretrained(os.getcwd())
model = AutoModelForCausalLM.from_pretrained(
    os.getcwd(),
    low_cpu_mem_usage=True,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

prompt = input("Nhập 1 đoạn nói về chủ đề bạn muốn tạo: ")
while True:
    max_length = input('Nhập số ký tự tối đa của đoạn văn được tạo: ')
    if not str(max_length).isdigit():
        print('Số ký tự phải là số nguyên (VD: 1000)')
        continue
    max_length = int(max_length)
    break
input_ids = tokenizer(prompt, return_tensors="pt")['input_ids'].to(device)

gen_tokens = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=True,
        temperature=0.9,
        top_k=20,
    )

gen_text = tokenizer.batch_decode(gen_tokens)[0]
print('Đoạn văn của bạn là')
print('=' * 50)
print(gen_text)
print('=' * 50)
total_time = datetime.utcnow() - start_time
print(f'Thời gian xử lý: {total_time} giây')
