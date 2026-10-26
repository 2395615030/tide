"""合并 tieba-01 + Neko-A 成字符级语料。"""
import json, glob, os
from collections import Counter

TIEBA = r"D:\ROOT\Datasets\tieba-01\train.jsonl"
TIEBA_VALID = r"D:\ROOT\Datasets\tieba-01\valid.jsonl"
NEKO = r"D:\ROOT\Datasets\Neko-A"


def load_tieba(path):
    out = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            t = d.get("text")
            if t:
                out.append(t.replace("<|im_start|>", "\n").replace("<|im_end|>", "\n"))
    return "".join(out)


def load_neko():
    out = []
    for f in sorted(glob.glob(os.path.join(NEKO, "*.json"))):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        sc = d.get("scenario", {})
        out.append(f"\n【场景】{sc.get('name','')}\n{sc.get('description','')}\n")
        for t in d.get("conversation", []):
            role = "用户" if t.get("role") == "user" else "猫娘"
            c = (t.get("content") or "").strip()
            if c:
                out.append(f"{role}：{c}\n")
    return "".join(out)


def build(chars_limit=3000):
    text = load_tieba(TIEBA) + load_neko()
    cnt = Counter(text)
    keep = [ch for ch, _ in cnt.most_common(chars_limit)]
    vocab = {ch: i + 1 for i, ch in enumerate(keep)}
    ids = [vocab.get(ch, 0) for ch in text]
    val_text = load_tieba(TIEBA_VALID)
    val_ids = [vocab.get(ch, 0) for ch in val_text]
    return text, ids, val_ids, keep


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    text, ids, vids, keep = build()
    print(f"train chars: {len(text)}  tokens: {len(ids)}")
    print(f"val   chars: {len(vids)}  tokens: {len(vids)}")
    print(f"vocab: {len(keep)+1}")
    cov = sum(1 for i in ids if i != 0) / len(ids)
    print(f"train coverage: {cov*100:.2f}%")
