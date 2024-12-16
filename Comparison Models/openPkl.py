import pickle

# 指定 .pkl 文件路径和输出 .txt 文件路径
pkl_file_path = './data/yelp/tstMat.pkl'
txt_file_path = './data/yelp/tstMat.txt'

# 打开并加载 .pkl 文件内容
with open(pkl_file_path, 'rb') as file:
    data = pickle.load(file)

# 将数据保存到 .txt 文件
with open(txt_file_path, 'w', encoding='utf-8') as file:
    # 检查数据类型并格式化写入
    if isinstance(data, dict):
        for key, value in data.items():
            file.write(f"{key}: {value}\n")
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            file.write(f"Item {idx}: {item}\n")
    else:
        file.write(str(data))

print(f"内容已保存到 {txt_file_path}")
