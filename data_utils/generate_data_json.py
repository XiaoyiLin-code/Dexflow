import os
import json

def generate_data_json(directory, json_file):
    data = {
        "train": [],
        "validate": []
    }

    # 获取目标文件夹下的第一层文件或子文件夹名称，不包含后缀
    for item in os.listdir(directory):
        full_path = os.path.join(directory, item)
        if os.path.isfile(full_path):  # 只处理文件
            file_name = os.path.splitext(item)[0]  # 去掉扩展名
            data["train"].append(file_name)  # 添加不含后缀的文件名

    # 保存为 JSON 文件
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"第一层文件名（不含后缀）已保存到 {json_file}")
    return data

if __name__ == "__main__":
    folder_path = "/home/lightcone/workspace/DRO-retarget/Noise-learn/data/dexgraspnet"  # 替换为目标文件夹路径
    output_json = "./mano_meshdata.json"  # 输出的JSON文件路径
    generate_data_json(folder_path, output_json)
