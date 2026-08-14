# 查询动作并下载 GLB

这份文档覆盖从启动本地服务到查询动作、下载 GLB 的最短流程。

## 1. 启动服务

推荐使用项目自带启动脚本：

```bash
cd /data5/cy/multimotion/server_bundle_lazy
bash shell/start_api_server.sh
```

这个脚本默认监听：

```text
http://127.0.0.1:7091
```

如果要手工指定语义模型、索引和 GPU，可以用：

```bash
cd /data5/cy/multimotion/server_bundle_lazy
python3 local_motion_query_api.py \
  --host 0.0.0.0 \
  --port 7091 \
  --semantic-model /data5/cy/models/qwen3-embedding-0.6b \
  --semantic-index dataset/semantic_index_qwen3_06b \
  --semantic-device cuda:3 \
  --reranker-model /data5/cy/models/bge-reranker-v2-m3 \
  --reranker-device cuda:3 \
  --search-prewarm
```

后台启动并写日志：

```bash
cd /data5/cy/multimotion/server_bundle_lazy
mkdir -p runtime
nohup bash shell/start_api_server.sh > runtime/local_motion_query_api.log 2>&1 &
echo $! > runtime/local_motion_query_api.pid
```

检查服务是否可用：

```bash
curl http://127.0.0.1:7091/api/v1/health
```

能返回 JSON 就说明 API 已经启动。

## 2. 一条命令查询并下载 GLB

客户端脚本会自动完成两步：先搜索动作，再下载 top-k 个 GLB。

```bash
cd /data5/cy/multimotion/server_bundle_lazy
python3 query_api_client.py \
  --base-url http://127.0.0.1:7091 \
  --text "两个人握手" \
  --top-k 3 \
  --output-dir client_output/handshake
```

输出文件：

```text
client_output/handshake/query_status.json
client_output/handshake/models/rank_01_*.glb
client_output/handshake/models/rank_02_*.glb
client_output/handshake/models/rank_03_*.glb
```

`query_status.json` 保存完整搜索结果；`models/` 下是下载好的 GLB 文件。

更多例子：

```bash
python3 query_api_client.py --base-url http://127.0.0.1:7091 --text "拥抱" --top-k 3 --output-dir client_output/hug
python3 query_api_client.py --base-url http://127.0.0.1:7091 --text "一个人挥手" --top-k 3 --output-dir client_output/wave
python3 query_api_client.py --base-url http://127.0.0.1:7091 --text "两个人互相推搡" --top-k 3 --output-dir client_output/push
```

只查询、不下载 GLB：

```bash
python3 query_api_client.py \
  --base-url http://127.0.0.1:7091 \
  --text "两个人握手" \
  --top-k 3 \
  --skip-download-models
```

## 3. 直接调用搜索接口

搜索接口：

```text
POST /api/v1/searches
```

最小请求体只需要 `text` 和 `top_k`：

```bash
curl -s -X POST http://127.0.0.1:7091/api/v1/searches \
  -H 'Content-Type: application/json' \
  -d '{"text":"两个人握手","top_k":3}'
```

返回结果里每个候选动作都有 `object_id` 和 `glb.url`：

```json
{
  "status": "ok",
  "items": [
    {
      "rank": 1,
      "object_id": "84c0814294c64ce3321271ee3b9fdb13657ae394cdc70488f9dddc1bf768d30c",
      "description": "Two people shake hands.",
      "glb": {
        "url": "http://127.0.0.1:7091/api/v1/models/84c0814294c64ce3321271ee3b9fdb13657ae394cdc70488f9dddc1bf768d30c",
        "filename": "G022T000A001R005.glb",
        "content_type": "model/gltf-binary",
        "lazy": true
      }
    }
  ]
}
```

## 4. 用 object_id 下载 GLB

下载接口：

```text
GET /api/v1/models/{object_id}
```

如果已经知道 `object_id`：

```bash
curl -L \
  "http://127.0.0.1:7091/api/v1/models/84c0814294c64ce3321271ee3b9fdb13657ae394cdc70488f9dddc1bf768d30c" \
  -o result.glb
```

第一次下载某个动作时，服务端会按需把原始 motion 转成 GLB，所以首次请求可能较慢；之后再次下载同一个动作通常会更快。

## 5. Python 查询并下载第一个结果

```python
import json
from pathlib import Path
from urllib.request import Request, urlopen

base_url = "http://127.0.0.1:7091"
query_text = "两个人握手"
output_path = Path("result.glb")

payload = json.dumps(
    {"text": query_text, "top_k": 3},
    ensure_ascii=False,
).encode("utf-8")

request = Request(
    f"{base_url}/api/v1/searches",
    data=payload,
    headers={"Content-Type": "application/json; charset=utf-8"},
    method="POST",
)

with urlopen(request, timeout=3600) as response:
    search_result = json.loads(response.read().decode("utf-8"))

items = search_result.get("items") or []
if not items:
    raise RuntimeError(f"No motion result for query: {query_text}")

first = items[0]
glb_url = first["glb"]["url"]

with urlopen(glb_url, timeout=3600) as response:
    output_path.write_bytes(response.read())

print(json.dumps({
    "object_id": first["object_id"],
    "description": first.get("description"),
    "glb": str(output_path),
}, ensure_ascii=False, indent=2))
```

## 6. 公网访问

如果本地服务已经通过 FRP 暴露到公网，例如：

```text
http://42.193.117.211:7091
```

把 `--base-url` 换成公网地址：

```bash
python3 query_api_client.py \
  --base-url http://42.193.117.211:7091 \
  --text "两个人握手" \
  --top-k 3 \
  --output-dir client_output/remote_handshake
```

直接下载也同理：

```bash
curl -L "http://42.193.117.211:7091/api/v1/models/<object_id>" -o result.glb
```

## 7. 常见问题

- 服务连不上：先跑健康检查，确认 `local_motion_query_api.py` 还在监听 `7091`。
- `Address already in use`：说明已有旧服务占用 `7091`，用 `ss -ltnp 'sport = :7091'` 找 PID。
- 搜索结果为空：换更具体的中文动作描述，例如 `两个人握手`、`一个人挥手`、`两个人拥抱`。
- 下载很慢：首次下载会触发 lazy GLB 转换，属于正常行为。
- 找不到下载文件：客户端默认写到 `client_output/models/`；如果传了 `--output-dir`，就在该目录的 `models/` 子目录下。
