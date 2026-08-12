Multimotion Lazy Retrieval Bundle

这是一个“先检索、后按需转 GLB”的服务包。
它先用 caption 建索引，列表使用 NAS 上的静态 WebP 缩略图；只有播放或下载时才临时生成 GLB，响应结束后立即删除。
当前支持 InterHuman、InterX 和 Motion-X++。Motion-X++ 使用 `models/smplx/SMPLX_NEUTRAL.npz`，
首次下载时按需生成带 55 关节骨架和 50 个表情 morph targets 的 GLB。

动作库前端默认使用关键词、Qwen3-Embedding 与本地 BGE reranker 的混合检索。分类目录、列表和缩略图均使用固定索引结果，
重型模型只参与文本查询和按需转换，不参与普通标签跳转。

默认使用方式按中文查询来理解。
Qwen3 直接编码中英文查询；内置 alias 只用于补充中文词法召回，不再加载翻译模型。

## 你会用到的文件

- `local_motion_query_api.py`: 本地 API 服务。
- `query_api_client.py`: 查询并下载模型的客户端。
- `build_motion_index.py`: 生成动作元数据索引和可选的本地向量索引。
- `build_taxonomy_assignments.py`: 生成三级分类目录、分类模型、资产映射和覆盖报告。
- `search_index.py`: 关键词 FTS 与可选的本地多语言向量检索。
- `motion_taxonomy.py`: 三级分类规则、运行时数据结构和批量模型分类器。
- `generate_motion_previews.py`: 生成可断点续跑的全量静态 WebP 缩略图。
- `taxonomy/catalog.json`: 版本化的中英双语动作目录、别名和来源引用。
- `taxonomy/action_asset_source.json`: 动作资产三级分类源（一级大类、二级动作组、三级动作标签）。
- `taxonomy/sources.json`: WHO ICF、FIG、O*NET、ISCO 等分类来源元数据。
- `shell/start_api_server.sh`: 在固定端口 `7091` 启动本地 API。
- `config/frpc.ini` / `shell/start_api_frp.sh`: 把本地 API 暴露到公网。
- `requirements/base.txt`: 基础运行依赖说明。
- `requirements/semantic.txt`: Qwen3 向量检索与 BGE 重排依赖。
- `config/retrieval_eval.jsonl`: 106 条中英文检索质量基线，83 条开发集、23 条固定留出集。
- `runtime/`: 本地日志、PID 和运行时配置，不提交 Git。
- `taxonomy/`: 可编辑的分类定义、运行时目录和分类来源。
- `dataset/`: 动作资产索引及分类、审核、模型等可重建产物。
- `dataset/motion_index.jsonl`: 当前索引。
- `dataset/semantic_index_qwen3_06b/`: 本地生成的 Qwen3 语义索引，不提交 Git，可通过构建脚本重建。
- `/mnt/nas/cy/humanmotion/multimotion_previews/`: 长期保存的 WebP 缩略图。

## 先理解怎么用

1. 先启动本地 API。
2. 用中文文字描述去搜结果。
3. 中文查询由 Qwen3 直接编码，alias 同时补充词法候选。
4. 需要下载时，再请求 `/api/v1/models/{object_id}`。
5. 如果要给别的机器用，再起 FRP 公网入口。

## 安装依赖

纯词法搜索不需要额外三方依赖；Qwen3 语义检索与重排使用 `requirements/semantic.txt`。

## 启用语义检索

语义检索与三级分类解耦，模型和资产向量均在本机运行。默认使用 `Qwen/Qwen3-Embedding-0.6B` 和 `BAAI/bge-reranker-v2-m3`。先部署模型并构建一次独立索引：

```bash
huggingface-cli download Qwen/Qwen3-Embedding-0.6B \
  --local-dir /data5/cy/models/qwen3-embedding-0.6b
pip install -r requirements/semantic.txt
python3 build_motion_index.py --semantic-only \
  --semantic-model /data5/cy/models/qwen3-embedding-0.6b \
  --semantic-output dataset/semantic_index_qwen3_06b \
  --semantic-device cuda:3
```

启动时显式传入模型和索引目录：

```bash
python3 local_motion_query_api.py --host 0.0.0.0 --port 7091 \
  --semantic-model /data5/cy/models/qwen3-embedding-0.6b \
  --semantic-index dataset/semantic_index_qwen3_06b --semantic-device cuda:3 \
  --reranker-model /data5/cy/models/bge-reranker-v2-m3 \
  --reranker-device cuda:3 --search-prewarm
```

搜索接口支持 `lexical`、`semantic` 和 `hybrid` 三种 `retrieval_mode`，默认使用确定性的 `hybrid + rerank + diversity`。重排只处理前 20 个候选，Motion-X++ 的源文件动作名优先于冲突 caption；多样性只降低同系列重复项的排名，不删除结果。可传 `"rerank":false` 或 `"diversity":false` 做对比，也可显式使用 `lexical` 获得最低服务端成本。模型或索引不可用时会自动回退并在 `warnings` 中说明原因。

Qwen3 查询会添加动作检索 instruction，中文短动作名还会套用通用的活动语境；caption 不添加 instruction。编码使用 last-token pooling 和 L2 归一化。模型以 BF16 加载，caption 索引保持 FP16，并放到同一张 GPU 上计算。FTS 先截取宽松候选，再与向量结果融合。由于现有 reranker 和后续多样性排序会损害中文查询对英文 caption 的排序，CJK 查询保留 Qwen 混合排序，英文查询继续使用 reranker 和多样性。服务缓存向量查询、混合候选和重排分数，重复查询无需再次推理。索引 manifest 固定记录 Qwen 编码契约，旧 BGE/CLS 索引会被拒绝并提示重建；模型或索引缺失时服务仍可启动并回退到词法检索。

检索评测报告写入已忽略的 `runtime/`，配置集长期保留，便于迭代前后使用同一口径比较：

```bash
python3 query_api_client.py --eval-file config/retrieval_eval.jsonl \
  --split all --report runtime/retrieval_eval_report.json
```

报告包含 Hit@5、MRR@10、nDCG@10、Recall@50、前十重复率、禁止命中数和 p50/p95 延迟，并按语言和查询类型分别汇总。相关资产 ID 和系列规则可直接编辑，留出集不要参与参数选择。

## 本地启动

本项目的本地 API 和 FRP 端口统一固定为 `7091`。

```bash
cd /data5/cy/multimotion/server_bundle_lazy
bash shell/start_api_server.sh
```

或者手工启动：

```bash
cd /data5/cy/multimotion/server_bundle_lazy
python3 local_motion_query_api.py --port 7091
```

CPU 启动方式：

```bash
python3 local_motion_query_api.py --host 0.0.0.0 --port 7091 \
  --semantic-model /data5/cy/models/qwen3-embedding-0.6b \
  --semantic-index dataset/semantic_index_qwen3_06b --semantic-device cpu
```

检查健康状态：

```bash
curl http://127.0.0.1:7091/api/v1/health
```

动作库和三级分类树页面：

```text
http://127.0.0.1:7091/ui
http://127.0.0.1:7091/ui/action-taxonomy
```

前端交互使用以下轻量策略：

- 分类树和固定计数只初始化一次，标签跳转不再重建整棵树。
- 输入搜索有防抖，新的列表请求会取消仍在执行的旧请求。
- 分类页悬停或聚焦三级标签时预取对应列表，点击后复用同一缓存。
- Three.js、页面脚本和样式支持 gzip、ETag 与浏览器缓存。
- WebP 缩略图使用长期 immutable 缓存；3D 模型仍只在播放或下载时生成。

修改 `taxonomy/action_asset_source.json` 后，重新生成目录和资产分类：

```bash
python3 build_taxonomy_assignments.py --stage all \
  --classifier hybrid --device cuda:3
```

三级模型会为每条资产发布一个唯一的三级叶子标签，旧规则只作为轻量先验，不再锁定最终分类。低置信、规则冲突和分层抽样结果写入 `dataset/motion_taxonomy_review.jsonl`。“待复核”只是第一候选得分或与第二候选差距较低的提示，不表示资产未分类，也不影响文本检索。

## 搜索

最直接的方式是 POST 搜索接口。对外正式参数只需要 `text` 和 `top_k`：

```bash
curl -X POST http://127.0.0.1:7091/api/v1/searches   -H 'Content-Type: application/json'   -d '{"text":"握手","top_k":3}'
```

也可以直接写完整中文描述：

```bash
curl -X POST http://127.0.0.1:7091/api/v1/searches   -H 'Content-Type: application/json'   -d '{"text":"两个人握手","top_k":3}'
```

返回结果里最重要的是 `items[].glb.url`，它是可以直接请求的 GLB 下载地址：

```json
{
  "status": "ok",
  "query": {"text": "两个人握手", "top_k": 3, "result_count": 3},
  "items": [
    {
      "rank": 1,
      "object_id": "84c081...",
      "description": "Two people shake hands.",
      "glb": {
        "url": "http://127.0.0.1:7091/api/v1/models/84c081...",
        "filename": "G022T000A001R005.glb",
        "content_type": "model/gltf-binary",
        "lazy": true
      }
    }
  ]
}
```

## 下载 GLB

推荐直接使用搜索返回的 `items[0].glb.url` 下载模型：

```python
import json
import urllib.request

payload = json.dumps({"text": "两个人握手", "top_k": 3}, ensure_ascii=False).encode("utf-8")
request = urllib.request.Request(
    "http://127.0.0.1:7091/api/v1/searches",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(request) as response:
    result = json.loads(response.read().decode("utf-8"))

glb_url = result["items"][0]["glb"]["url"]
urllib.request.urlretrieve(glb_url, "result.glb")
```

也可以拿到 `object_id` 后，直接拼下载地址：

```bash
curl -L "http://127.0.0.1:7091/api/v1/models/<object_id>" -o result.glb
```

例如：

```bash
curl -L "http://127.0.0.1:7091/api/v1/models/84c0814294c64ce3321271ee3b9fdb13657ae394cdc70488f9dddc1bf768d30c" -o G022T000A001R005.glb
```

## 推荐用客户端

客户端会帮你把搜索结果和模型下载串起来。

只看搜索结果，不下载模型：

```bash
python3 query_api_client.py --base-url http://127.0.0.1:7091 --text "握手" --top-k 3 --skip-download-models
```

客户端默认使用服务端的确定性混合检索。需要探索性结果时加 `--randomness 0.25 --random-seed demo`。

搜索并下载 top-k 模型：

```bash
python3 query_api_client.py --base-url http://127.0.0.1:7091 --text "握手" --top-k 3
```

也可以直接写句子：

```bash
python3 query_api_client.py --base-url http://127.0.0.1:7091 --text "两个人握手" --top-k 3
```

默认输出目录是 `client_output/`。

## 中文查询说明

这个 bundle 的中文查询有两层处理：

1. Qwen3-Embedding 直接编码完整中文查询，负责语义召回。
2. 内置中文 alias 词表补充英文关键词，负责词法召回。

常见动作词可以直接搜，例如：

- `握手`
- `拥抱`
- `拍肩膀`
- `扇巴掌`
- `推`
- `拉`
- `挥手`

Qwen 模型或索引不可用时，API 会继续使用 alias 和 FTS 词法检索，并在响应 `warnings` 中说明回退原因。

## 公网访问

这套服务通过 FRP 暴露到公网。
当前公网地址是：

```text
http://42.193.117.211:7091
```

### 服务端怎么启动

在服务器上先起本地 API：

```bash
cd /data5/cy/multimotion/server_bundle_lazy
bash shell/start_api_server.sh
```

再起 FRP：

```bash
cd /data5/cy/multimotion/server_bundle_lazy
bash shell/start_api_frp.sh
```

本地 API、FRP 本地端口和公网端口统一固定为 `7091`。

### 外部机器怎么用

先测公网健康检查：

```bash
curl http://42.193.117.211:7091/api/v1/health
```

公网搜索：

```bash
curl -X POST http://42.193.117.211:7091/api/v1/searches   -H 'Content-Type: application/json'   -d '{"text":"握手","top_k":3}'
```

公网搜索结果同样从 `items[].glb.url` 读取 GLB 下载地址。

如果你想直接用客户端：

```bash
python3 query_api_client.py   --base-url http://42.193.117.211:7091   --text "握手"   --top-k 3
```

### 公网下载模型

先搜索，优先使用返回结果里的 `items[0].glb.url` 下载。也可以拿到 `object_id` 后直接拼下载地址：

```bash
curl -L "http://42.193.117.211:7091/api/v1/models/<object_id>" -o result.glb
```

例如：

```bash
curl -L "http://42.193.117.211:7091/api/v1/models/84c0814294c64ce3321271ee3b9fdb13657ae394cdc70488f9dddc1bf768d30c" -o G022T000A001R005.glb
```

第一次下载某个动作时，服务端会先做 lazy conversion，所以第一次可能会慢一点。

## 常见问题

- `Address already in use`：检查并停止占用 `7091` 的旧 API 进程，再重新启动。
- 公网连不上：查看 `runtime/frpc.log`，再确认 `shell/start_api_frp.sh` 是否在运行。
- 第一次下载慢：这是正常的，服务端会先把对应 motion 转成 GLB。
- 下载后找不到文件：去看 `client_output/`。
- Qwen 搜索回退：检查 Qwen 模型目录、`dataset/semantic_index_qwen3_06b/manifest.json` 和健康接口中的语义检索状态。

## 重新生成索引

原始动作索引更新后，需要同步重建确定性分类映射：

```bash
cd /data5/cy/multimotion/server_bundle_lazy
python3 build_motion_index.py --semantic-only \
  --semantic-model /data5/cy/models/qwen3-embedding-0.6b \
  --semantic-output dataset/semantic_index_qwen3_06b --semantic-device cuda:3
python3 build_taxonomy_assignments.py --stage model \
  --model /data5/cy/models/qwen3-embedding-0.6b --device cuda:3
python3 build_taxonomy_assignments.py --classifier hybrid --device cuda:3
```

输出文件：

- `dataset/motion_taxonomy_assignments.jsonl`: 按 `object_id` 保存预计算分类、证据和版本。
- `dataset/motion_taxonomy_report.json`: 分类数量、未分类率、节点分布和待审样例。
- `dataset/motion_taxonomy_review.jsonl`: 低置信及规则冲突候选审核队列。
- `dataset/taxonomy_model/`: 可重建的本地标签模型索引。

分类过程离线复用 Qwen3 caption 向量，直接编码中英双语分类原型，按三级动作、二级动作组和一级领域进行层级语义评分。所有资产固定归入一个三级叶子；置信度不足时仍发布该标签，同时标记为待复核。

全量生成静态缩略图：

```bash
/data1/cy/anaconda3/bin/python generate_motion_previews.py --dataset all --workers 8
```

脚本会跳过已有 WebP，失败项会记录并可重复执行；生成过程中使用临时 GLB，任务结束后不会留下模型缓存。
新生成的 WebP 会先写入同目录临时文件，完成解码、尺寸和空帧校验后再原子替换目标文件。

独立审计全部缩略图，不修改资产：

```bash
python3 generate_motion_previews.py --audit-existing --skip-migration --workers 24
```

审计并只修复缺失、损坏或空帧文件：

```bash
/data1/cy/anaconda3/bin/python generate_motion_previews.py --repair-invalid --skip-migration --workers 24
```

也可以重复传 `--object-id <id>`，只审计或重新生成指定动作；需要强制替换已有文件时同时传 `--overwrite`。

## 分类目录 API

`GET /api/v1/taxonomy` 返回分类版本、节点树、来源和素材计数。按节点检索时使用：

```bash
curl 'http://127.0.0.1:7091/api/v1/library?node_id=strength_squat&include_descendants=true'
```

父节点默认包含所有子节点；传 `include_descendants=false` 可只匹配直接挂载到该节点的素材。旧的 `action_category`、`action_tag` 和 `activity_domain` 参数继续兼容。

动作库响应中的 `summary.entry_count` 或精简响应中的 `global_total` 是固定的全库总量，`total` 是当前查询和筛选后的结果量。
完整响应的 `facets` 使用排除自身筛选的计数方式；Web 前端从 taxonomy 接口读取固定全局计数，切换标签时数字不会随请求重新计算。

Web 前端使用兼容的精简响应，外部客户端不受影响：

```text
GET /api/v1/taxonomy?view=compact
GET /api/v1/library?view=compact&include_facets=0&node_id=<node_id>
```

精简模式会省略重复的平铺节点、`data`、`hierarchy` 和 `summary`，并支持 gzip、ETag 与短期缓存。`include_facets=0` 会跳过动态 facet 计算和传输，适合前端搜索、分页和标签跳转；未传该参数时保持原有 API 行为。

如果你换了原始数据或 caption，重新生成索引：

```bash
cd /data5/cy/multimotion/server_bundle_lazy
python3 build_motion_index.py
```

索引会持久化三套数据集的帧数，列表请求不会再为补帧数逐条访问 NAS。全量重建会以当前 NAS 挂载内容为准，可能新增或移除资产；需要保持线上总数稳定时，应先输出到临时文件，对比 `object_id` 和数据集数量后再替换正式索引。

只重建 Motion-X++ 索引可以使用：

```bash
python3 build_motion_index.py --dataset motionxpp
```
