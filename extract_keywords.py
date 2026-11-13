import requests
import time
from tqdm import tqdm
import os


# Ollama default API
OLLAMA_API = "http://localhost:11434/api/generate"

# MODEL_NAME
MODEL_NAME = "qwen2.5:14b" # qwen3:8b


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
    print(f"processing：{os.path.basename(input_path)}")

    with open(input_path, 'r', encoding='utf-8') as infile:
        lines = [line.strip() for line in infile if line.strip()]

    results = []
    for idx, line in enumerate(tqdm(lines, desc="Processing Status", leave=True)):
        prompt = build_prompt(line)
        extracted = call_ollama(prompt)

        if extracted:
            results.append(extracted)
        else:
            results.append("")  

        
        if idx < len(lines) - 1:
            time.sleep(0.1)

    with open(output_path, 'w', encoding='utf-8') as outfile:
        for result in results:
            outfile.write(result + '\n')

    print(f"Finish：{os.path.basename(input_path)}\n")


def get_output_path(input_path, output_dir="output_keywords"):
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_filename = f"{base_name}_keywords.txt"
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, output_filename)


def batch_process_directory(input_dir="key_word_test", output_dir="key_word_test"):
    if not os.path.exists(input_dir):
        print(f"Input directory does not exist: {input_dir}")
        return

    file_list = [
        os.path.join(input_dir, f) for f in os.listdir(input_dir)
        if f.endswith('.txt')
    ]

    total_files = len(file_list)
    if total_files == 0:
        print(f"No .txt files found in directory {input_dir}")
        return

    print(f"Found {total_files} .txt file(s), starting batch processing...\n")

    for i, input_file in enumerate(file_list):
        if not os.path.exists(input_file):
            print(f"File does not exist: {input_file}")
            continue

        output_file = get_output_path(input_file, output_dir)
        process_file(input_file, output_file)

    print("All files done")


if __name__ == "__main__":
    print("Begin processing...\n")
    batch_process_directory()
