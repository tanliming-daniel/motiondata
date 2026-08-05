import * as THREE from "/ui/static/vendor/three.module.js";
import { GLTFLoader } from "/ui/static/vendor/GLTFLoader.js";

const state = { offset: 0, limit: 9, total: 0, taxonomy: [], taxonomyNodes: [], taxonomySources: [], taxonomyQuery: "", expandedNodes: new Set(), hierarchy: [], facets: {}, filters: {}, items: [] };
const $ = (selector) => document.querySelector(selector);
const escapeHtml = (value) => String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;");

let renderer, scene, camera, mixer, clock, animationFrame, currentObject, currentFixedFront = false;

function isMotionXppDataset(value) {
  const normalized = String(value || "").toLowerCase().replace(/[^a-z0-9]/g, "");
  return normalized === "motionx" || normalized === "motionxpp";
}

async function fetchJson(url) {
  const response = await fetch(url);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.message || payload.error || "请求失败");
  return payload;
}

function chip(label, count, selected, attrs) {
  return `<button class="chip${selected ? " selected" : ""}" ${attrs} type="button">${escapeHtml(label)} <small>${Number(count || 0)}</small></button>`;
}

function renderTaxonomyFilters() {
  const root = $("#taxonomyFilters");
  const auxiliary = state.taxonomy.filter((axis) => ["participants", "space"].includes(axis.key));
  const hierarchy = filterTaxonomyTree(state.hierarchy || [], state.taxonomyQuery);
  const selectedNode = state.filters.node_id || "";
  const renderNode = (node, depth = 0) => {
    const children = node.tags || [];
    const selected = selectedNode === node.id;
    const open = state.expandedNodes.has(node.id) || nodeContains(node, selectedNode) || Boolean(state.taxonomyQuery);
    const toggle = children.length
      ? `<button class="tree-toggle" type="button" data-toggle-node="${escapeHtml(node.id)}" aria-label="${open ? "收起" : "展开"}" aria-expanded="${open}"><span aria-hidden="true">${open ? "−" : "+"}</span></button>`
      : '<span class="tree-toggle-spacer" aria-hidden="true"></span>';
    const button = `<button class="tree-node${selected ? " selected" : ""}" type="button" data-node-id="${escapeHtml(node.id)}" style="--tree-depth:${depth}"><span>${escapeHtml(node.label)}</span><small>${Number(node.count || 0).toLocaleString()}</small></button>`;
    const nested = children.length && open ? `<div class="taxonomy-children">${children.map((child) => renderNode(child, depth + 1)).join("")}</div>` : "";
    return `<div class="taxonomy-branch${open ? " expanded" : ""}"><div class="tree-row">${toggle}${button}</div>${nested}</div>`;
  };
  const treeHtml = `<section class="taxonomy-section"><h3>动作分类</h3><input class="taxonomy-search" type="search" value="${escapeHtml(state.taxonomyQuery)}" placeholder="搜索分类名称"><div class="taxonomy-tree">${hierarchy.length ? hierarchy.map((node) => renderNode(node)).join("") : '<p class="tree-empty">没有匹配的分类。</p>'}</div></section>`;
  const auxiliaryHtml = auxiliary.map((axis) => {
    const counts = state.facets[axis.key] || {};
    return `<section><h3>${escapeHtml(axis.label)}</h3><div class="chip-row">${axis.values.map((value) => chip(value.label, counts[value.key], state.filters[axis.key] === value.key, `data-axis="${axis.key}" data-value="${value.key}"`)).join("")}</div></section>`;
  }).join("");
  root.innerHTML = treeHtml + auxiliaryHtml;
  const taxonomySearch = root.querySelector(".taxonomy-search");
  taxonomySearch.addEventListener("input", () => {
    state.taxonomyQuery = taxonomySearch.value;
    renderTaxonomyFilters();
    const nextInput = root.querySelector(".taxonomy-search");
    nextInput.focus();
    nextInput.setSelectionRange(nextInput.value.length, nextInput.value.length);
  });
  root.querySelectorAll("[data-toggle-node]").forEach((button) => button.addEventListener("click", () => {
    const nodeId = button.dataset.toggleNode;
    if (state.expandedNodes.has(nodeId)) state.expandedNodes.delete(nodeId);
    else state.expandedNodes.add(nodeId);
    renderTaxonomyFilters();
  }));
  root.querySelectorAll("[data-node-id]").forEach((button) => button.addEventListener("click", () => selectTaxonomyNode(button.dataset.nodeId)));
  root.querySelectorAll("[data-axis]").forEach((button) => button.addEventListener("click", () => {
    const { axis, value } = button.dataset;
    if (axis === "action_category" && state.filters.action_category === value) {
      state.filters.action_category = "";
      state.filters.action_tag = "";
    } else {
      state.filters[axis] = state.filters[axis] === value ? "" : value;
      if (axis === "action_category") state.filters.action_tag = "";
    }
    state.offset = 0;
    loadLibrary();
  }));
  renderTaxonomyDetail();
}

function filterTaxonomyTree(nodes, query) {
  const normalized = String(query || "").trim().toLowerCase();
  if (!normalized) return nodes;
  return nodes.map((node) => {
    const children = filterTaxonomyTree(node.tags || [], normalized);
    const matches = `${node.label || ""} ${node.en || ""}`.toLowerCase().includes(normalized);
    return matches || children.length ? { ...node, tags: children } : null;
  }).filter(Boolean);
}

function nodeContains(node, nodeId) {
  if (!nodeId) return false;
  if (node.id === nodeId) return true;
  return (node.tags || []).some((child) => nodeContains(child, nodeId));
}

function selectTaxonomyNode(nodeId) {
  state.filters.node_id = state.filters.node_id === nodeId ? "" : nodeId;
  state.filters.action_category = "";
  state.filters.action_tag = "";
  state.offset = 0;
  loadLibrary();
}

function renderTaxonomyDetail() {
  const root = $("#taxonomyDetail");
  const node = state.taxonomyNodes.find((item) => item.id === state.filters.node_id);
  if (!node) {
    root.classList.add("hidden");
    root.innerHTML = "";
    return;
  }
  const sourceMap = new Map(state.taxonomySources.map((source) => [source.id, source]));
  const sources = (node.source_refs || []).map((sourceId) => sourceMap.get(sourceId)).filter(Boolean);
  const sourceHtml = sources.map((source) => source.url
    ? `<a href="${escapeHtml(source.url)}" target="_blank" rel="noopener">${escapeHtml(source.publisher)}</a>`
    : `<span>${escapeHtml(source.publisher)}</span>`).join(" · ");
  root.classList.remove("hidden");
  root.innerHTML = `<div><p class="eyebrow">SELECTED TAXONOMY</p><h3>${escapeHtml(node.zh_cn)} <span>${escapeHtml(node.en)}</span></h3></div><div class="taxonomy-stats"><strong>${Number(node.count || 0).toLocaleString()}</strong> 条子树素材 · <strong>${Number(node.direct_count || 0).toLocaleString()}</strong> 条直接命中</div><p>${sourceHtml ? `来源：${sourceHtml}` : "来源信息未提供"}</p>`;
}

function renderDatasetFilters() {
  const counts = state.facets.dataset || {};
  const root = $("#datasetFilters");
  root.innerHTML = ["interhuman", "interx", "motionxpp"].map((dataset) => chip(dataset, counts[dataset], state.filters.dataset === dataset, `data-dataset="${dataset}"`)).join("");
  root.querySelectorAll("[data-dataset]").forEach((button) => button.addEventListener("click", () => {
    state.filters.dataset = state.filters.dataset === button.dataset.dataset ? "" : button.dataset.dataset;
    state.offset = 0;
    loadLibrary();
  }));
}

function tagsFor(item) {
  const labels = item.taxonomy_labels || {};
  const actionNodes = (item.action_nodes || []).slice(0, 3).map((node) => `<button class="tag action-tag" type="button" data-card-node-id="${escapeHtml(node.id)}">${escapeHtml(node.label)}</button>`).join("");
  return [
    actionNodes,
    `<span class="tag domain">${escapeHtml(labels.action_category || "其他动作")}</span>`,
    `<span class="tag">${escapeHtml(labels.action_tag || "其他动作")}</span>`,
    `<span class="tag">${escapeHtml(labels.participants || "未知")}</span>`,
    `<span class="tag">${escapeHtml(labels.space || "未说明")}</span>`,
  ].join("");
}

function renderCards(items) {
  const root = $("#resultGrid");
  if (!items.length) {
    root.innerHTML = '<article class="empty">没有符合当前条件的动作。可以清空筛选或换一个关键词。</article>';
    return;
  }
  root.innerHTML = items.map((item, index) => {
    const model = item.model || {};
    const frames = item.frame_count ? `${Number(item.frame_count).toLocaleString()} 帧` : "帧数未知";
    return `<article class="motion-card">
      <div class="card-preview" data-card-preview="${index}"><span>${item.preview?.available ? "读取预览" : "暂无缩略图"}</span></div>
      <div class="card-body">
        <div class="card-top"><span>${escapeHtml(item.dataset)} · ${escapeHtml(item.motion_id)}</span><span>WebP 缩略图</span></div>
        <h3>${escapeHtml(item.description || item.motion_id)}</h3>
        <p class="description">${escapeHtml((item.captions || []).join(" / "))}</p>
        <div class="tags">${tagsFor(item)}<span class="tag">${frames}</span></div>
        <div class="card-actions">
          <button type="button" data-preview-index="${index}">播放完整</button>
          <a href="${escapeHtml(model.download_url || "#")}" target="_blank" rel="noopener">下载 GLB</a>
        </div>
      </div>
    </article>`;
  }).join("");
  root.querySelectorAll("[data-preview-index]").forEach((button) => button.addEventListener("click", () => previewItem(items[Number(button.dataset.previewIndex)])));
  root.querySelectorAll("[data-card-node-id]").forEach((button) => button.addEventListener("click", () => selectTaxonomyNode(button.dataset.cardNodeId)));
  renderCardPreviews(items);
}

const cardPreviewQueue = { running: 0, jobs: [] };

function renderCardPreviews(items) {
  cardPreviewQueue.jobs = [];
  cardPreviewQueue.running = 0;
  items.forEach((item, index) => {
    const target = document.querySelector(`[data-card-preview="${index}"]`);
    if (!target || !item?.preview?.download_url) return;
    cardPreviewQueue.jobs.push(() => renderOneCardPreview(item, target));
  });
  pumpCardPreviewQueue();
}

function pumpCardPreviewQueue() {
  while (cardPreviewQueue.running < 2 && cardPreviewQueue.jobs.length) {
    const job = cardPreviewQueue.jobs.shift();
    cardPreviewQueue.running += 1;
    job().catch(() => {}).finally(() => {
      cardPreviewQueue.running -= 1;
      pumpCardPreviewQueue();
    });
  }
}

function renderOneCardPreview(item, target) {
  return new Promise((resolve) => {
    if (item.preview && item.preview.available && item.preview.download_url) {
      target.innerHTML = `<img src="${escapeHtml(item.preview.download_url)}" alt="${escapeHtml(item.motion_id)} 预览图" loading="lazy">`;
      return resolve();
    }
    target.innerHTML = "<span>暂无缩略图</span>";
    resolve();
  });
}

function fitObjectCamera(object, targetCamera, padding = 1.7, fixedFront = false) {
  if (fixedFront) {
    object.updateMatrixWorld(true);
    object.traverse((child) => { if (child.isSkinnedMesh) child.skeleton.update(); });
  }
  const box = new THREE.Box3().setFromObject(object, fixedFront);
  if (fixedFront) {
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const verticalFov = THREE.MathUtils.degToRad(targetCamera.fov);
    const horizontalFov = 2 * Math.atan(Math.tan(verticalFov / 2) * Math.max(targetCamera.aspect, 0.01));
    const heightDistance = size.y / Math.max(2 * Math.tan(verticalFov / 2), 0.01);
    const widthDistance = size.x / Math.max(2 * Math.tan(horizontalFov / 2), 0.01);
    const distance = Math.max(heightDistance, widthDistance, 1.0) * padding;
    const targetY = box.min.y + size.y * 0.5;
    targetCamera.position.set(center.x, targetY, box.max.z + distance);
    targetCamera.lookAt(center.x, targetY, center.z);
    targetCamera.near = Math.max(distance / 200, 0.01);
    targetCamera.far = Math.max(distance * 8, 80);
    targetCamera.updateProjectionMatrix();
    return;
  }
  const sphere = box.getBoundingSphere(new THREE.Sphere());
  const center = sphere.center;
  const radius = Math.max(sphere.radius, 1.0);
  const fov = THREE.MathUtils.degToRad(targetCamera.fov);
  const distance = (radius / Math.sin(fov / 2)) * padding;
  targetCamera.position.set(center.x + radius * 0.28, center.y + radius * 1.38, center.z + distance);
  targetCamera.lookAt(center.x, center.y + radius * 0.56, center.z);
  targetCamera.near = Math.max(distance / 200, 0.01);
  targetCamera.far = Math.max(distance * 8, 80);
  targetCamera.updateProjectionMatrix();
}

function renderPager() {
  const pager = $("#pager");
  const pages = Math.max(1, Math.ceil(state.total / state.limit));
  const current = Math.floor(state.offset / state.limit) + 1;
  pager.classList.toggle("hidden", state.total <= state.limit);
  $("#prevPage").disabled = state.offset === 0;
  $("#nextPage").disabled = state.offset + state.limit >= state.total;
  $("#pageText").textContent = `${current} / ${pages}`;
}

async function loadLibrary() {
  const params = new URLSearchParams({ limit: String(state.limit), offset: String(state.offset), sort: $("#sortSelect").value });
  Object.entries(state.filters).forEach(([key, value]) => { if (value) params.set(key, value); });
  const search = $("#searchInput").value.trim();
  if (search) params.set("q", search);
  const payload = await fetchJson(`/api/v1/library?${params}`);
  state.total = Number(payload.total || 0);
  state.facets = payload.facets || {};
  state.hierarchy = payload.hierarchy || state.hierarchy;
  state.items = payload.items || [];
  renderCards(state.items);
  renderDatasetFilters();
  renderTaxonomyFilters();
  renderPager();
  $("#resultTitle").textContent = `动作结果 · ${state.total}`;
  const summary = payload.summary || {};
  $("#statusBadge").textContent = `${Number(summary.entry_count || state.total).toLocaleString()} 条索引 · WebP 缩略图 · 按需 3D`;
}

function setViewerStatus(text) { $("#viewerStatus").textContent = text; }

function ensureViewer() {
  const root = $("#viewerCanvas");
  if (renderer) return;
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x101820);
  camera = new THREE.PerspectiveCamera(45, 1, 0.01, 200);
  camera.position.set(0, 1.5, 5);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x273241, 2.4));
  const light = new THREE.DirectionalLight(0xffffff, 2.2);
  light.position.set(3, 6, 4);
  scene.add(light);
  const grid = new THREE.GridHelper(6, 12, 0x506070, 0x2a3440);
  scene.add(grid);
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  root.appendChild(renderer.domElement);
  clock = new THREE.Clock();
  window.addEventListener("resize", resizeViewer);
  resizeViewer();
  animate();
}

function resizeViewer() {
  if (!renderer) return;
  const root = $("#viewerCanvas");
  const width = Math.max(root.clientWidth, 1);
  const height = Math.max(root.clientHeight, 1);
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}

function animate() {
  animationFrame = requestAnimationFrame(animate);
  const delta = clock ? clock.getDelta() : 0;
  if (mixer) mixer.update(delta);
  if (currentObject && !currentFixedFront) currentObject.rotation.y += delta * 0.08;
  renderer.render(scene, camera);
}

function fitCamera(object) {
  fitObjectCamera(object, camera, 1.65, currentFixedFront);
}

async function previewItem(item) {
  ensureViewer();
  $("#viewerTitle").textContent = `${item.dataset} / ${item.motion_id}`;
  $("#viewerMeta").innerHTML = `<p>${escapeHtml(item.description)}</p><p>${tagsFor(item)}</p>`;
  const url = item?.model?.download_url;
  if (!url) { setViewerStatus("这个动作没有可用的 GLB 下载地址。"); return; }
  $("#downloadLink").href = url;
  $("#downloadLink").classList.remove("disabled");
  setViewerStatus("正在临时生成 GLB，完成后立即清理文件...");
  const loader = new GLTFLoader();
  loader.load(url, (gltf) => {
    if (currentObject) scene.remove(currentObject);
    currentObject = gltf.scene;
    currentFixedFront = isMotionXppDataset(item.dataset);
    scene.add(currentObject);
    mixer = null;
    if (gltf.animations && gltf.animations.length) {
      mixer = new THREE.AnimationMixer(currentObject);
      gltf.animations.forEach((clip) => mixer.clipAction(clip).play());
    }
    fitCamera(currentObject);
    setViewerStatus(gltf.animations?.length ? "GLB 已加载，动画播放中。" : "GLB 已加载，但没有检测到动画轨道。");
  }, undefined, (error) => {
    setViewerStatus(`GLB 加载失败：${error.message || error}`);
  });
}

async function boot() {
  try {
    const taxonomy = await fetchJson("/api/v1/taxonomy");
    state.taxonomy = taxonomy.data?.axes || [];
    state.taxonomyNodes = taxonomy.data?.nodes || [];
    state.taxonomySources = taxonomy.data?.sources || [];
    state.hierarchy = taxonomy.data?.hierarchy || [];
    await loadLibrary();
  } catch (error) {
    $("#resultGrid").innerHTML = `<article class="empty">读取失败：${escapeHtml(error.message)}</article>`;
    $("#statusBadge").textContent = "读取失败";
  }
}

let searchTimer;
$("#searchInput").addEventListener("input", () => { clearTimeout(searchTimer); searchTimer = setTimeout(() => { state.offset = 0; loadLibrary(); }, 220); });
$("#sortSelect").addEventListener("change", () => { state.offset = 0; loadLibrary(); });
$("#clearFilters").addEventListener("click", () => { state.filters = {}; state.offset = 0; $("#searchInput").value = ""; loadLibrary(); });
$("#prevPage").addEventListener("click", () => { state.offset = Math.max(0, state.offset - state.limit); loadLibrary(); });
$("#nextPage").addEventListener("click", () => { state.offset += state.limit; loadLibrary(); });

boot();
