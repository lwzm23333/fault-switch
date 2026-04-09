import pickle, os
import shutil
from main_train_experiment import extract_features_pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from dataset import split_dataset

def load_dataset(root):
    wav_files = []
    class_names = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    for lbl, cname in enumerate(class_names):
        folder = os.path.join(root, cname)
        for fn in os.listdir(folder):
            if fn.lower().endswith(".wav"):
                wav_files.append((os.path.join(folder, fn), lbl))
    return wav_files, class_names


def test_one_method(method, test_dir):
    print(f"\n===== Testing {method} + RF =====")

    feat_dir = f"outputs/different_features/{method}/test"
    model_path = f"outputs/models/{method}_rf.pkl"

    X_test, y_test = extract_features_pipeline(
        method, test_dir, feat_dir
    )

    with open(model_path, "rb") as f:
        clf = pickle.load(f)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {f1:.4f}")
    print("Confusion Matrix:\n", cm)

    return {
        "acc": acc,
        "f1": f1,
        "cm": cm
    }

def main_test_all():
    data_root = "D:/Work/pythonproject/Datasets/Data_Sets/Switch_machine_audio/data"
    test_dir = os.path.join(data_root, "test")

    # methods = ["stft", "mfcc", "mel", "vmd_fixed", "imf_token"]
    methods = ["stft"]

    results = {}
    for m in methods:
        results[m] = test_one_method(m, test_dir)

    print("\n===== Summary =====")
    for k, v in results.items():
        print(f"{k}: Acc={v['acc']:.4f}, F1={v['f1']:.4f}")

if __name__ == "__main__":
    main_test_all()

