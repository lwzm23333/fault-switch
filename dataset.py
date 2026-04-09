import os, json, random
from collections import defaultdict

def split_dataset(wav_files, test_ratio=0.3, seed=42, save_path="data/split.json"):
    """
    wav_files: [(filepath, label), ...]
    """
    random.seed(seed)

    # 按类别分组
    class_dict = defaultdict(list)
    for f, lbl in wav_files:
        class_dict[lbl].append(f)

    train_set = []
    test_set = []

    # 每个类别分别划分
    for lbl, files in class_dict.items():
        files = files.copy()
        random.shuffle(files)

        N = len(files)
        print(N)
        test_N = int(N * test_ratio)

        test_files = files[:test_N]
        train_files = files[test_N:]

        test_set += [(f, lbl) for f in test_files]
        train_set += [(f, lbl) for f in train_files]

    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf8") as f:
        json.dump(
            {"train": train_set, "test": test_set},
            f, indent=2, ensure_ascii=False
        )

    print(f"[Split] Train={len(train_set)}, Test={len(test_set)} saved to {save_path}")
    return train_set, test_set