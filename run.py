import os
import requests
import sys
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

URL_MODEL = 'https://huggingface.co/VietAI/gpt-neo-1.3B-vietnamese-news/blob/' \
    'main/pytorch_model.bin'
start_time = datetime.utcnow()
trained_model_path = os.path.join(os.getcwd(), 'pytorch_model.bin')
if not os.path.exists(trained_model_path):
    with requests.get(URL_MODEL, stream=True) as trained_remote_file:
        trained_remote_file.raise_for_status()
        with open(trained_model_path, 'wb') as trained_local_file:
            for chunk in trained_remote_file.iter_content(chunk_size=8192):
                trained_local_file.write(chunk)

tokenizer = AutoTokenizer.from_pretrained(os.getcwd())
model = AutoModelForCausalLM.from_pretrained(
    os.getcwd(),
    low_cpu_mem_usage=True,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = sys.argv[1]
prompt = sys.argv[2]
while True:
    if not str(max_length).isdigit():
        print('Số ký tự phải là số nguyên (VD: 1000)')
        max_length = input('Nhập số ký tự tối đa của đoạn văn được tạo: ')
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
