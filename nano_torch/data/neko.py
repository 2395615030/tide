"""加载 Neko-A 多轮对话, 格式化成字符级语料。"""
import json, glob
from collections import Counter


def load_neko(chars_limit=2000):
    files = sorted(glob.glob(r"D:\ROOT\Datasets\Neko-A\*.json"))
    parts = []
    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        sc = d.get("scenario", {})
        parts.append(f"\n【场景】{sc.get('name','')}\n{sc.get('description','')}\n")
        for t in d.get("conversation", []):
            role = "用户" if t.get("role") == "user" else "猫娘"
            c = (t.get("content") or "").strip()
            if c:
                parts.append(f"{role}：{c}\n")
    text = "".join(parts)
    cnt = Counter(text)
    keep = [ch for ch, _ in cnt.most_common(chars_limit)]
    vocab = {ch: i + 1 for i, ch in enumerate(keep)}
    ids = [vocab.get(ch, 0) for ch in text]
    return text, ids, keep


if __name__ == "__main__":
    text, ids, keep = load_neko()
    print(f"chars: {len(text)}, tokens: {len(ids)}, vocab: {len(keep)+1}")
    print("sample:", repr(text[:160]))
