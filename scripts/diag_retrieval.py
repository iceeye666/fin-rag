import sys
sys.path.insert(0, ".")
from finrag.config import get_settings
import chromadb

cfg = get_settings()
client = chromadb.PersistentClient(path=str(cfg.chroma_dir))
col = client.get_collection(cfg.chroma_collection)
data = col.get(include=["documents", "metadatas"])
docs = data["documents"]
metas = data["metadatas"]

print("=== 全库摘要中含'营业收入'的块 ===")
cnt = 0
for t, m in zip(docs, metas):
    if "营业收入" in t:
        cnt += 1
        print(f"   p{m.get('page')} {m.get('category')} src={m.get('source','')[:20]}")
        print("     ", t[:160].replace("\n", " "))
print("含营业收入的摘要数:", cnt, "/", len(docs))
print("\n=== 含'营业收入'的原始表块是否存在（docstore）===")
import json
dp = cfg.docstore_path / "chunks.json"
if dp.exists():
    store = json.loads(dp.read_text())
    items = store.get("data", store)
    n = 0
    for cid, c in (items.items() if isinstance(items, dict) else []):
        txt = (c.get("text") if isinstance(c, dict) else str(c))
        if "营业收入" in txt:
            n += 1
            print("   chunk", cid, "page", c.get("page"), "cat", c.get("category"))
            print("     ", txt[:160].replace("\n", " "))
    print("docstore 含营业收入的块数:", n)
