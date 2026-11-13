import requests
import time
from tqdm import tqdm
import os

# windows+R输入cmd打开命令提示符，输入ollama run qwen2.5:14b加载相应模型，然后运行该文件即可。


# Ollama 默认 API 地址
OLLAMA_API = "http://localhost:11434/api/generate"

# 使用的模型名称
MODEL_NAME = "qwen2.5:14b" # qwen3:8b

# PROMPT_TEMPLATE = """
# You are a language assistant specialized in extracting geographic entities and related keywords from sentences describing remote sensing images or geospatial data. Your task is to assist in remote sensing image-text retrieval by identifying key geographic objects mentioned in the descriptions of these images.

# ### Task Description:
# From the given sentence, extract all noun phrases that meet the following criteria:
# 1. If a noun phrase is modified by a determiner or quantifier (like "a", "an", "the", "one", "two", "several") or a color word (like "green", "red", "blue"), return the full phrase including the modifier.
# 2. Only extract noun phrases that represent complete geographic entities or objects (e.g., "two large ships", "the gray port", "a green playground").
# 3. Do NOT extract prepositional phrases starting with "in", "on", "at", "around", "between", "among", "of", etc. (e.g., "on both sides of the gray port", "full of red ships")
# 4. Prefer phrases that are **subjects** or **direct objects** in the sentence — i.e., represent what the sentence is primarily about or describing.
# 5. Return the extracted phrases **in the order they appear in the original sentence**.
# 6. If the sentence does not contain any valid geographic entity or object that meets the above criteria (e.g., explanation-only sentences), return an **empty string**.

# ### Output Format:
# Return the results as a comma-separated list. If no valid geographic object is found, return an empty string. Do not add any extra text, explanation, or thinking process!

# ### Examples:

# Example 1:
# Input: "Two large ships are located on both sides of the gray port."

# Output: two large ships, the gray port

# Example 2:
# Input: "It was a field of uncut grass."

# Output: uncut grass

# Example 3:
# Input: "One side of the four table tennis courts is bare and the other side is green."

# Output: the four table tennis courts

# Example 4:
# Input: "The grass grows well on this field."

# Output: The grass

# Now process this sentence:
# "{sentence}"
# """



PROMPT_TEMPLATE = """
You are a language assistant in the field of remote sensing. Your task is to assist in remote sensing image-text retrieval by identifying salient  ground entities mentioned in the captions of remote sensing images.

Task Description:
From the given sentence, extract all noun phrases that meet the following criteria:
1. Only extract noun phrases that represent ground objects.
2. If a noun phrase is modified by a quantifier, color word, or descriptive adjective, return the full phrase including the modifier.
3. Return the extracted phrases in the order they appear in the original sentence.
4. Based on contextual understanding, ensure each key ground object is extracted only once, even if expressed differently.

Output Format:
Return the results as a comma-separated list. If no valid ground object is found, return an empty string.  Do not add any extra text, explanation, or thinking process!

Example 1:
Input: On both sides of the gray port were two large ships, full of red boats.
Output: the gray port, two large ships, red boats

Example 2:
Input: It's a satellite image of the scene.
Output: " "

Now process this sentence:
"{sentence}"
"""

def build_prompt(sentence):
    return PROMPT_TEMPLATE.format(sentence=sentence.strip())


def call_ollama(prompt, max_retries=3, retry_delay=2):
    data = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    for attempt in range(max_retries):
        try:
            response = requests.post(OLLAMA_API, json=data, timeout=60)
            if response.status_code == 200:
                return response.json()['response'].strip()
            else:
                print(f"Attempt {attempt + 1} - Error from Ollama: {response.text}")
                time.sleep(retry_delay)
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} - Network error: {e}")
            time.sleep(retry_delay)
    return ""


def process_file(input_path, output_path):
    print(f"🔄 正在处理：{os.path.basename(input_path)}")

    with open(input_path, 'r', encoding='utf-8') as infile:
        lines = [line.strip() for line in infile if line.strip()]

    results = []
    for idx, line in enumerate(tqdm(lines, desc="处理进度", leave=True)):
        prompt = build_prompt(line)
        extracted = call_ollama(prompt)

        if extracted:
            results.append(extracted)
        else:
            results.append("")  # 如果失败则留空

        # 控制请求频率（可调整或移除）
        if idx < len(lines) - 1:
            time.sleep(0.1)

    # 写入输出文件
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for result in results:
            outfile.write(result + '\n')

    print(f"✅ 已处理完成：{os.path.basename(input_path)}\n")


def get_output_path(input_path, output_dir="output_keywords"):
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_filename = f"{base_name}_keywords.txt"
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, output_filename)


def batch_process_directory(input_dir="key_word_test", output_dir="key_word_test"):
    if not os.path.exists(input_dir):
        print(f"❌ 输入目录不存在：{input_dir}")
        return

    file_list = [
        os.path.join(input_dir, f) for f in os.listdir(input_dir)
        if f.endswith('.txt')
    ]

    total_files = len(file_list)
    if total_files == 0:
        print(f"⚠️ 在目录 {input_dir} 中未找到 .txt 文件")
        return

    print(f"📁 找到 {total_files} 个 .txt 文件，开始批量处理...\n")

    for i, input_file in enumerate(file_list):
        if not os.path.exists(input_file):
            print(f"❌ 文件不存在：{input_file}")
            continue

        output_file = get_output_path(input_file, output_dir)
        process_file(input_file, output_file)

    print("🎉 所有文件处理完成！关键词已保存至 output_keywords 文件夹。")


if __name__ == "__main__":
    print("🚀 开始批量处理 key_word_extraction 文件夹中的文本文件...\n")
    batch_process_directory()