Multimotion Lazy Retrieval Bundle

这是一个“先检索、后按需转 GLB”的服务包。
它先用 caption 建索引，列表使用 NAS 上的静态 WebP 缩略图；只有播放或下载时才临时生成 GLB，响应结束后立即删除。
当前支持 InterHuman、InterX 和 Motion-X++。Motion-X++ 使用 `models/smplx/SMPLX_NEUTRAL.npz`，
首次下载时按需生成带 55 关节骨架和 50 个表情 morph targets 的 GLB。

默认使用方式按中文查询来理解。
API 现在会优先对短中文关键词直接走 alias，对较长中文描述再在进程内翻成英文，不需要再单独起一个翻译 HTTP 服务。

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
- `requirements/translation.txt`: 进程内中文翻译依赖。
- `requirements/semantic.txt`: 向量检索依赖。
- `runtime/`: 本地日志、PID 和运行时配置，不提交 Git。
- `taxonomy/`: 可编辑的分类定义、运行时目录和分类来源。
- `dataset/`: 动作资产索引及分类、审核、模型等可重建产物。
- `dataset/motion_index.jsonl`: 当前索引。
- `dataset/semantic_index/`: 本地生成的语义索引，不提交 Git，可通过构建脚本重建。
- `/mnt/nas/cy/humanmotion/multimotion_previews/`: 长期保存的 WebP 缩略图。

## 先理解怎么用

1. 先启动本地 API。
2. 用中文文字描述去搜结果。
3. 短中文关键词直接走 alias，较长中文描述才会尝试进程内翻译。
4. 需要下载时，再请求 `/api/v1/models/{object_id}`。
5. 如果要给别的机器用，再起 FRP 公网入口。

## 安装依赖

基础搜索不需要额外三方依赖。
如果你想启用“进程内中文翻译”，需要安装：

```bash
cd /data5/cy/multimotion/server_bundle_lazy
pip install -r requirements/translation.txt
```

## 启用语义检索

语义检索与三级分类解耦，模型和资产向量均在本机运行。首次准备好本地 `BAAI/bge-m3` 模型目录后，安装依赖并构建一次索引：

```bash
pip install -r requirements/semantic.txt
python3 build_motion_index.py --semantic-only \
  --semantic-model /data5/cy/models/bge-m3 --semantic-device cuda:0
```

启动时显式传入模型和索引目录：

```bash
python3 local_motion_query_api.py --host 0.0.0.0 --port 7091 \
  --semantic-model /data5/cy/models/bge-m3 \
  --semantic-index dataset/semantic_index --semantic-device cuda:0
```

搜索接口支持 `lexical`、`semantic` 和 `hybrid` 三种 `retrieval_mode`。旧请求默认仍是关键词检索；动作库页面会使用 `hybrid`。模型或索引不可用时会自动回退到关键词检索，并在 `warnings` 中说明原因。

默认使用的本地翻译模型路径是：

```text
/data5/cy/animodata/server_bundle_full/opus-mt-zh-en
```

## 本地启动

本项目的本地 API 和 FRP 端口统一固定为 `7091`。

```bash
cd /data5/cy/multimotion/server_bundle_lazy
bash shell/start_api_server.sh
```

或者手工启动：

```bash
cd /data5/cy/multimotion/server_bundle_lazy
python3 local_motion_query_api.py --host 0.0.0.0 --port 7091
```

如果你想显式指定翻译模型或设备：

```bash
python3 local_motion_query_api.py   --host 0.0.0.0   --port 7091   --translation-model /data5/cy/animodata/server_bundle_full/opus-mt-zh-en   --translation-device 0
```

CPU 启动方式：

```bash
python3 local_motion_query_api.py   --host 0.0.0.0   --port 7091   --translation-device -1
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

修改 `taxonomy/action_asset_source.json` 后，重新生成目录和资产分类：

```bash
python3 build_taxonomy_assignments.py --stage all \
  --classifier hybrid --device cuda:3 --translation-device cuda:0
```

三级模型会为每条资产发布一个唯一的三级叶子标签，旧规则只作为轻量先验，不再锁定最终分类。低置信、规则冲突和分层抽样结果写入 `dataset/motion_taxonomy_review.jsonl`。

## 搜索

最直接的方式是 POST 搜索接口。
默认先按中文使用：

```bash
curl -X POST http://127.0.0.1:7091/api/v1/searches   -H 'Content-Type: application/json'   -d '{"text":"握手","top_k":3}'
```

默认会启用轻微随机性，在高分候选里按分数加权抽样。泛化查询例如“走路”不会每次固定返回同几个结果。
如果需要完全固定排序，可以传 `"randomness":0`：

```bash
curl -X POST http://127.0.0.1:7091/api/v1/searches   -H 'Content-Type: application/json'   -d '{"text":"走路","top_k":3,"randomness":0}'
```

如果需要可复现的随机结果，可以同时传 `randomness` 和 `random_seed`：

```bash
curl -X POST http://127.0.0.1:7091/api/v1/searches   -H 'Content-Type: application/json'   -d '{"text":"走路","top_k":3,"randomness":0.25,"random_seed":"demo"}'
```

也可以直接写完整中文描述：

```bash
curl -X POST http://127.0.0.1:7091/api/v1/searches   -H 'Content-Type: application/json'   -d '{"text":"两个人握手","top_k":3}'
```

返回结果里最重要的是 `object_id` 和 `model.download_url`。

## 下载 GLB

拿到 `object_id` 后，直接下载模型：

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

客户端默认不传 `randomness`，使用服务端默认随机性。需要固定结果时加 `--randomness 0`；需要可复现随机结果时加 `--randomness 0.25 --random-seed demo`。

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

这个 bundle 现在有两层中文处理：

1. 内置中文 alias 词表（短关键词优先走这个，速度最快）
2. 进程内本地翻译模型（较长中文描述再走这个）

常见动作词可以直接搜，例如：

- `握手`
- `拥抱`
- `拍肩膀`
- `扇巴掌`
- `推`
- `拉`
- `挥手`

如果本地翻译模型没装依赖，或者模型不可用，API 会继续只用 alias 模式工作。

如果你已经有现成的外部中文翻译接口，也仍然可以在启动 API 时加上：

```bash
python3 local_motion_query_api.py   --host 0.0.0.0   --port 7091   --translation-url http://127.0.0.1:7099/translate
```

外部翻译现在只是可选回退，不再是默认路径。

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

公网接口同样默认启用轻微随机性；固定结果可在 JSON 里加 `"randomness":0`。

如果你想直接用客户端：

```bash
python3 query_api_client.py   --base-url http://42.193.117.211:7091   --text "握手"   --top-k 3
```

### 公网下载模型

先搜索，拿到返回结果里的 `object_id`。
再下载：

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
- 中文搜不到：先把描述缩短成关键词，例如把“其中一个人和另一个人轻轻握手”改成“握手”。
- 中文翻译不工作：先确认已经安装 `requirements/translation.txt` 里的依赖，并且本地模型路径存在。

## 重新生成索引

原始动作索引更新后，需要同步重建确定性分类映射：

```bash
cd /data5/cy/multimotion/server_bundle_lazy
python3 build_taxonomy_assignments.py --stage model \
  --device cuda:3 --translation-device cuda:0
python3 build_taxonomy_assignments.py --classifier hybrid --device cuda:3
```

输出文件：

- `dataset/motion_taxonomy_assignments.jsonl`: 按 `object_id` 保存预计算分类、证据和版本。
- `dataset/motion_taxonomy_report.json`: 分类数量、未分类率、节点分布和待审样例。
- `dataset/motion_taxonomy_review.jsonl`: 低置信及规则冲突候选审核队列。
- `dataset/taxonomy_model/`: 可重建的本地标签模型索引。

分类过程离线复用 BGE-M3 caption 向量，按三级动作、二级动作组和一级领域进行层级语义评分。所有资产固定归入一个三级叶子；置信度不足时仍发布该标签，同时标记为待复核。

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

动作库响应中的 `summary.entry_count` 是固定的全库总量，`total` 是当前查询和筛选后的结果量。
`facets` 使用排除自身筛选的计数方式：例如选中某个数据集后，`facets.dataset` 仍会列出其他数据集在其余条件下的可用数量。

Web 前端使用兼容的精简响应，外部客户端不受影响：

```text
GET /api/v1/taxonomy?view=compact
GET /api/v1/library?view=compact&node_id=<node_id>
```

精简模式会省略重复的平铺节点、`data`、`hierarchy` 和 `summary`，并支持 gzip、ETag 与短期缓存。

如果你换了原始数据或 caption，重新生成索引：

```bash
cd /data5/cy/multimotion/server_bundle_lazy
python3 build_motion_index.py
```

只重建 Motion-X++ 索引可以使用：

```bash
python3 build_motion_index.py --dataset motionxpp
```
