const state = { globalTotal: null, offset: 0, limit: 24, total: 0, taxonomy: [], taxonomyNodes: [], taxonomySources: [], taxonomyQuery: "", expandedNodes: new Set(), hierarchy: [], facets: {}, filters: {}, items: [], view: "library", assetTaxonomyQuery: "", taxonomyCounts: {} };
const $ = (selector) => document.querySelector(selector);
const escapeHtml = (value) => String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;");

let THREE, GLTFLoader, viewerModulesPromise;
let renderer, scene, camera, mixer, clock, animationFrame, currentObject, currentFixedFront = false;
let currentBlobUrl, currentItem, viewerAbortController, viewerRequestId = 0, lastViewerTrigger;
let libraryRequestId = 0, taxonomySearchTimer, taxonomyPrefetchTimer;
const requestCache = new Map();
const REQUEST_CACHE_LIMIT = 64;

function isMotionXppDataset(value) {
  const normalized = String(value || "").toLowerCase().replace(/[^a-z0-9]/g, "");
  return normalized === "motionx" || normalized === "motionxpp";
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.message || payload.error || "请求失败");
  return payload;
}

function cachedJson(url, { ttl = 30000 } = {}) {
  const now = Date.now();
  const existing = requestCache.get(url);
  if (existing && (existing.promise || existing.expires > now)) return existing.promise || Promise.resolve(existing.value);
  const promise = fetchJson(url).then((value) => {
    requestCache.set(url, { value, expires: Date.now() + ttl });
    while (requestCache.size > REQUEST_CACHE_LIMIT) requestCache.delete(requestCache.keys().next().value);
    return value;
  }).catch((error) => {
    requestCache.delete(url);
    throw error;
  });
  requestCache.set(url, { promise, expires: now + ttl });
  return promise;
}

function chip(label, count, selected, attrs) {
  return `<button class="chip${selected ? " selected" : ""}" ${attrs} type="button">${escapeHtml(label)} <small>${Number(count || 0)}</small></button>`;
}

function renderTaxonomyFilters() {
  const root = $("#taxonomyFilters");
  const auxiliary = state.taxonomy.filter((axis) => axis.key === "participants");
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
  const path = state.filters.node_id ? `/ui?node_id=${encodeURIComponent(state.filters.node_id)}` : "/ui";
  history.pushState({}, "", path);
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
  const currentNode = findHierarchyNode(state.hierarchy, node.id);
  const subtreeCount = currentNode?.count ?? node.count ?? 0;
  const directCount = currentNode?.direct_count ?? node.direct_count ?? 0;
  const sourceHtml = sources.map((source) => source.url
    ? `<a href="${escapeHtml(source.url)}" target="_blank" rel="noopener">${escapeHtml(source.publisher)}</a>`
    : `<span>${escapeHtml(source.publisher)}</span>`).join(" · ");
  root.classList.remove("hidden");
  root.innerHTML = `<div><p class="eyebrow">SELECTED TAXONOMY</p><h3>${escapeHtml(node.zh_cn)} <span>${escapeHtml(node.en)}</span></h3></div><div class="taxonomy-stats"><strong>${Number(subtreeCount).toLocaleString()}</strong> 条当前可用 · <strong>${Number(directCount).toLocaleString()}</strong> 条直接命中</div><p>${sourceHtml ? `来源：${sourceHtml}` : "来源信息未提供"}</p>`;
}

function findHierarchyNode(nodes, nodeId) {
  for (const node of nodes || []) {
    if (node.id === nodeId) return node;
    const child = findHierarchyNode(node.tags, nodeId);
    if (child) return child;
  }
  return null;
}

function flattenHierarchy(nodes, result = []) {
  (nodes || []).forEach((node) => {
    result.push({ ...node, zh_cn: node.label });
    flattenHierarchy(node.tags, result);
  });
  return result;
}

function applyFacetCounts(nodes, facets) {
  const aggregate = facets?.action_node || {};
  const direct = facets?.action_node_direct || {};
  (nodes || []).forEach((node) => {
    node.count = Number(aggregate[node.id] || 0);
    node.direct_count = Number(direct[node.id] || 0);
    applyFacetCounts(node.tags, facets);
  });
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

function filterLabel(key, value) {
  if (key === "dataset") return value;
  if (key === "node_id") return state.taxonomyNodes.find((node) => node.id === value)?.zh_cn || value;
  const axis = state.taxonomy.find((item) => item.key === key);
  return axis?.values?.find((item) => item.key === value)?.label || value;
}

function renderActiveFilters() {
  const root = $("#activeFilters");
  const filters = Object.entries(state.filters).filter(([, value]) => Boolean(value));
  root.classList.toggle("hidden", filters.length === 0);
  root.innerHTML = filters.map(([key, value]) => `<button class="active-filter" type="button" data-remove-filter="${escapeHtml(key)}">${escapeHtml(filterLabel(key, value))}</button>`).join("");
  root.querySelectorAll("[data-remove-filter]").forEach((button) => button.addEventListener("click", () => {
    state.filters[button.dataset.removeFilter] = "";
    state.offset = 0;
    loadLibrary();
  }));
}

function tagsFor(item, { interactive = true, limit = 3 } = {}) {
  const labels = item.taxonomy_labels || {};
  const allActionNodes = item.action_nodes || [];
  const visibleActionNodes = allActionNodes.filter((node) => state.taxonomyNodes.some((candidate) => candidate.id === node.id));
  const actionValues = (visibleActionNodes.length ? visibleActionNodes : allActionNodes)
    .slice(0, 1)
    .map((node) => ({ label: node.label, id: node.id, domain: true }));
  const values = [
    ...actionValues,
    ...(item.confidence_level === "low" ? [{ label: "待复核", confidence: true }] : []),
    { label: labels.participants || "未知" },
  ].slice(0, limit);
  return values.map((value) => interactive && value.id
    ? `<button class="tag action-tag${value.domain ? " domain" : ""}" type="button" data-card-node-id="${escapeHtml(value.id)}">${escapeHtml(value.label)}</button>`
    : `<span class="tag${value.domain ? " domain" : ""}${value.confidence ? " confidence-low" : ""}">${escapeHtml(value.label)}</span>`).join("");
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
    const previewUrl = item.preview?.download_path || item.preview?.download_url;
    const previewRotation = Number(item.preview?.rotation_degrees || 0);
    const previewTransform = Number.isFinite(previewRotation) && previewRotation
      ? ` style="--preview-rotation:${Math.max(-180, Math.min(180, previewRotation))}deg"`
      : "";
    const preview = item.preview?.available && previewUrl
      ? `<img src="${escapeHtml(previewUrl)}" alt="${escapeHtml(item.motion_id)} 动作缩略图" loading="lazy"${previewTransform}>`
      : "<span>暂无缩略图</span>";
    const modelUrl = model.rest_download_path || model.download_path || model.download_url || "#";
    return `<article class="motion-card">
      <button class="card-preview" type="button" data-preview-index="${index}" aria-label="播放 ${escapeHtml(item.motion_id)}">${preview}</button>
      <div class="card-body">
        <div class="card-top"><span class="dataset-mark">${escapeHtml(item.dataset)}</span><span>${escapeHtml(item.motion_id)}</span></div>
        <h3>${escapeHtml(item.description || item.motion_id)}</h3>
        <div class="tags">${tagsFor(item, { limit: 2 })}<span class="tag">${frames}</span></div>
        <div class="card-actions">
          <button class="button button-primary" type="button" data-preview-index="${index}">播放 3D</button>
          <a class="button button-secondary" href="${escapeHtml(modelUrl)}" target="_blank" rel="noopener">下载</a>
        </div>
      </div>
    </article>`;
  }).join("");
  root.querySelectorAll("[data-preview-index]").forEach((button) => button.addEventListener("click", () => previewItem(items[Number(button.dataset.previewIndex)])));
  root.querySelectorAll("[data-card-node-id]").forEach((button) => button.addEventListener("click", () => selectTaxonomyNode(button.dataset.cardNodeId)));
  root.querySelectorAll(".card-preview img").forEach((image) => {
    const showFallback = () => {
      if (!image.isConnected) return;
      const fallback = document.createElement("span");
      fallback.textContent = "缩略图暂不可用";
      image.replaceWith(fallback);
    };
    image.addEventListener("error", showFallback, { once: true });
    image.addEventListener("load", () => {
      try {
        const canvas = document.createElement("canvas");
        canvas.width = 32;
        canvas.height = 20;
        const context = canvas.getContext("2d", { willReadFrequently: true });
        context.drawImage(image, 0, 0, canvas.width, canvas.height);
        const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
        const minimum = [255, 255, 255];
        const maximum = [0, 0, 0];
        const cornerOffsets = [0, (canvas.width - 1) * 4, (canvas.height - 1) * canvas.width * 4, (canvas.width * canvas.height - 1) * 4];
        const background = [0, 1, 2].map((channel) => cornerOffsets
          .map((offset) => pixels[offset + channel])
          .sort((left, right) => left - right)[2]);
        let foregroundPixels = 0;
        for (let offset = 0; offset < pixels.length; offset += 4) {
          for (let channel = 0; channel < 3; channel += 1) {
            minimum[channel] = Math.min(minimum[channel], pixels[offset + channel]);
            maximum[channel] = Math.max(maximum[channel], pixels[offset + channel]);
          }
          if (Math.max(...background.map((value, channel) => Math.abs(pixels[offset + channel] - value))) >= 18) {
            foregroundPixels += 1;
          }
        }
        const channelRange = Math.max(...maximum.map((value, channel) => value - minimum[channel]));
        if (channelRange < 12 || foregroundPixels / (pixels.length / 4) < 0.02) showFallback();
      } catch (_) {
        // A preview that cannot be sampled can still be displayed normally.
      }
    }, { once: true });
  });
}

function subjectFacingDirection(object) {
  const nodes = [];
  object.traverse((node) => nodes.push(node));
  const left = nodes.find((node) => /leftshoulder$/i.test(node.name || ""));
  const right = nodes.find((node) => /rightshoulder$/i.test(node.name || ""));
  if (!left || !right) return new THREE.Vector3(0, 0, 1);
  const shoulderAxis = right.getWorldPosition(new THREE.Vector3())
    .sub(left.getWorldPosition(new THREE.Vector3()));
  shoulderAxis.y = 0;
  if (shoulderAxis.lengthSq() < 1e-6) return new THREE.Vector3(0, 0, 1);
  shoulderAxis.normalize();
  return new THREE.Vector3(0, 1, 0).cross(shoulderAxis).normalize();
}

function fitObjectCamera(object, targetCamera, padding = 1.7, fixedFront = false) {
  object.updateMatrixWorld(true);
  object.traverse((child) => { if (child.isSkinnedMesh) child.skeleton.update(); });
  const box = new THREE.Box3().setFromObject(object, true);
  if (fixedFront) {
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const verticalFov = THREE.MathUtils.degToRad(targetCamera.fov);
    const horizontalFov = 2 * Math.atan(Math.tan(verticalFov / 2) * Math.max(targetCamera.aspect, 0.01));
    const heightDistance = size.y / Math.max(2 * Math.tan(verticalFov / 2), 0.01);
    const horizontalSpan = Math.hypot(size.x, size.z);
    const widthDistance = horizontalSpan / Math.max(2 * Math.tan(horizontalFov / 2), 0.01);
    const distance = Math.max(heightDistance, widthDistance, 1.0) * padding;
    const targetY = box.min.y + size.y * 0.5;
    const viewDirection = subjectFacingDirection(object)
      .applyAxisAngle(new THREE.Vector3(0, 1, 0), THREE.MathUtils.degToRad(12));
    targetCamera.position.copy(center).addScaledVector(viewDirection, distance);
    targetCamera.position.y = targetY + size.y * 0.06;
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

function libraryUrl() {
  const params = new URLSearchParams({ view: "compact", limit: String(state.limit), offset: String(state.offset), sort: $("#sortSelect").value, retrieval_mode: "hybrid" });
  Object.entries(state.filters).forEach(([key, value]) => { if (value) params.set(key, value); });
  const search = $("#searchInput").value.trim();
  if (search) params.set("q", search);
  return `/api/v1/library?${params}`;
}

function applyLibraryPayload(payload) {
  const resultTotal = Number(payload.total);
  const globalTotal = Number(payload.global_total ?? payload.summary?.entry_count);
  if (!Number.isInteger(resultTotal) || resultTotal < 0) throw new Error("接口返回了无效的结果数量");
  if (!Number.isInteger(globalTotal) || globalTotal < 1) throw new Error("接口返回了无效的总数据量");
  state.total = resultTotal;
  state.globalTotal = globalTotal;
  state.facets = payload.facets || {};
  applyFacetCounts(state.hierarchy, state.facets);
  state.items = payload.items || [];
  renderCards(state.items);
  renderDatasetFilters();
  renderTaxonomyFilters();
  renderActiveFilters();
  renderPager();
  $("#resultTitle").textContent = `筛选结果 · ${state.total.toLocaleString()}`;
  $("#statusBadge").textContent = `总数据 ${state.globalTotal.toLocaleString()} · WebP 缩略图 · 按需 3D`;
  setLibraryNotice("");
}

function setLibraryNotice(message) {
  const notice = $("#libraryNotice");
  notice.textContent = message;
  notice.classList.toggle("hidden", !message);
}

async function loadLibrary() {
  const requestId = ++libraryRequestId;
  const url = libraryUrl();
  $("#resultGrid").setAttribute("aria-busy", "true");
  try {
    const payload = await cachedJson(url);
    if (requestId !== libraryRequestId) return false;
    applyLibraryPayload(payload);
    return true;
  } catch (error) {
    if (requestId !== libraryRequestId) return false;
    setLibraryNotice(`读取失败：${error.message || error}`);
    return false;
  } finally {
    if (requestId === libraryRequestId) {
      $("#resultGrid").removeAttribute("aria-busy");
    }
  }
}

function setViewerStatus(text) { $("#viewerStatus").textContent = text; }

async function ensureViewer() {
  const root = $("#viewerCanvas");
  if (renderer) return;
  if (!viewerModulesPromise) {
    viewerModulesPromise = Promise.all([
      import("/ui/static/vendor/three.module.js"),
      import("/ui/static/vendor/GLTFLoader.js"),
    ]);
  }
  const [threeModule, loaderModule] = await viewerModulesPromise;
  THREE = threeModule;
  GLTFLoader = loaderModule.GLTFLoader;
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
  if (!renderer || !scene || !camera) return;
  animationFrame = requestAnimationFrame(animate);
  const delta = clock ? clock.getDelta() : 0;
  if (mixer) mixer.update(delta);
  if (currentObject && !currentFixedFront) currentObject.rotation.y += delta * 0.08;
  renderer.render(scene, camera);
}

function fitCamera(object) {
  fitObjectCamera(object, camera, 1.65, currentFixedFront);
}

function disposeMaterial(material) {
  Object.values(material || {}).forEach((value) => { if (value?.isTexture) value.dispose(); });
  material?.dispose?.();
}

function clearCurrentModel() {
  if (!currentObject) return;
  scene?.remove(currentObject);
  currentObject.traverse((child) => {
    child.geometry?.dispose?.();
    if (Array.isArray(child.material)) child.material.forEach(disposeMaterial);
    else disposeMaterial(child.material);
  });
  currentObject = null;
  mixer = null;
}

function updateOverlayState() {
  document.body.classList.toggle("overlay-open", document.body.classList.contains("viewer-open") || document.body.classList.contains("filters-open"));
}

function openViewer() {
  if (!document.body.classList.contains("viewer-open")) lastViewerTrigger = document.activeElement;
  document.body.classList.add("viewer-open");
  $("#viewerDrawer").setAttribute("aria-hidden", "false");
  $("#viewerDrawer").inert = false;
  updateOverlayState();
  requestAnimationFrame(() => $("#closeViewer").focus());
}

function closeViewer() {
  viewerRequestId += 1;
  viewerAbortController?.abort();
  viewerAbortController = null;
  clearCurrentModel();
  if (currentBlobUrl) URL.revokeObjectURL(currentBlobUrl);
  currentBlobUrl = null;
  if (animationFrame) cancelAnimationFrame(animationFrame);
  animationFrame = null;
  window.removeEventListener("resize", resizeViewer);
  renderer?.dispose();
  renderer?.forceContextLoss?.();
  renderer = scene = camera = clock = null;
  $("#viewerCanvas").innerHTML = "";
  $("#downloadLink").classList.add("disabled");
  $("#downloadLink").removeAttribute("download");
  document.body.classList.remove("viewer-open");
  $("#viewerDrawer").setAttribute("aria-hidden", "true");
  $("#viewerDrawer").inert = true;
  updateOverlayState();
  lastViewerTrigger?.focus?.();
}

async function previewItem(item) {
  currentItem = item;
  openViewer();
  const viewerReady = ensureViewer();
  $("#viewerTitle").textContent = `${item.dataset} / ${item.motion_id}`;
  $("#viewerMeta").innerHTML = `<p class="detail-description">${escapeHtml(item.description || item.motion_id)}</p><p class="detail-caption">${escapeHtml((item.captions || []).join(" / "))}</p><div class="tags">${tagsFor(item, { interactive: false, limit: 8 })}</div>`;
  const url = item?.model?.rest_download_path || item?.model?.download_path || item?.model?.download_url;
  if (!url) { setViewerStatus("这个动作没有可用的 GLB 下载地址。"); return; }
  viewerAbortController?.abort();
  viewerAbortController = new AbortController();
  const requestId = ++viewerRequestId;
  clearCurrentModel();
  if (currentBlobUrl) URL.revokeObjectURL(currentBlobUrl);
  currentBlobUrl = null;
  $("#downloadLink").classList.add("disabled");
  $("#retryViewer").classList.add("hidden");
  setViewerStatus("正在临时生成 GLB，完成后立即清理文件...");
  try {
    const response = await fetch(url, { signal: viewerAbortController.signal });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const blob = await response.blob();
    if (requestId !== viewerRequestId) return;
    currentBlobUrl = URL.createObjectURL(blob);
    $("#downloadLink").href = currentBlobUrl;
    $("#downloadLink").download = `${item.motion_id.split("/").pop() || item.object_id}.glb`;
    $("#downloadLink").classList.remove("disabled");
    setViewerStatus("模型已生成，正在准备 3D 场景...");
    await viewerReady;
    if (requestId !== viewerRequestId) return;
    const loader = new GLTFLoader();
    loader.load(currentBlobUrl, (gltf) => {
      if (requestId !== viewerRequestId) return;
      currentObject = gltf.scene;
      currentFixedFront = isMotionXppDataset(item.dataset);
      scene.add(currentObject);
      mixer = null;
      if (gltf.animations?.length) {
        mixer = new THREE.AnimationMixer(currentObject);
        gltf.animations.forEach((clip) => mixer.clipAction(clip).play());
      }
      fitCamera(currentObject);
      setViewerStatus(gltf.animations?.length ? "动画播放中。下载将复用当前模型。" : "模型已加载，但没有检测到动画轨道。");
    }, undefined, (error) => {
      if (requestId !== viewerRequestId) return;
      setViewerStatus(`3D 场景加载失败：${error.message || error}`);
      $("#retryViewer").classList.remove("hidden");
    });
  } catch (error) {
    if (error.name === "AbortError" || requestId !== viewerRequestId) return;
    setViewerStatus(`GLB 生成失败：${error.message || error}`);
    $("#retryViewer").classList.remove("hidden");
  }
}

function compactLibraryUrlForNode(nodeId) {
  const params = new URLSearchParams({
    view: "compact",
    limit: String(state.limit),
    offset: "0",
    sort: $("#sortSelect").value,
    node_id: nodeId,
  });
  return `/api/v1/library?${params}`;
}

function showView(view, { updateHistory = false } = {}) {
  state.view = view;
  const taxonomyView = view === "taxonomy";
  $("#libraryView").classList.toggle("hidden", taxonomyView);
  $("#taxonomyView").classList.toggle("hidden", !taxonomyView);
  $("#libraryViewButton").setAttribute("aria-pressed", String(!taxonomyView));
  $("#taxonomyViewButton").setAttribute("aria-pressed", String(taxonomyView));
  document.body.classList.toggle("taxonomy-view", taxonomyView);
  document.title = taxonomyView ? "动作资产分类树" : "Multimotion 动作库";
  if (updateHistory) history.pushState({}, "", taxonomyView ? "/ui/action-taxonomy" : "/ui");
  if (taxonomyView) renderAssetTaxonomy();
}

function hydrateAssetGroup(details, group) {
  const root = details.querySelector(".asset-tags");
  if (!root || root.dataset.rendered === "true") return;
  root.dataset.rendered = "true";
  root.innerHTML = (group.tags || []).map((tag) => `<a class="asset-tag-link" href="/ui?node_id=${encodeURIComponent(tag.id)}" data-asset-node-id="${escapeHtml(tag.id)}"><span>${escapeHtml(tag.label)}</span><small>${Number(tag.count || 0).toLocaleString()}</small></a>`).join("") || '<span class="asset-node-count">暂无三级标签</span>';
}

function renderAssetTaxonomy() {
  if (!state.hierarchy.length) return;
  const root = $("#assetTaxonomyTree");
  const query = state.assetTaxonomyQuery;
  const majors = filterTaxonomyTree(state.hierarchy, query);
  if (!majors.length) {
    root.innerHTML = '<div class="empty">没有匹配的分类。</div>';
    return;
  }
  root.innerHTML = majors.map((major, majorIndex) => `<details class="asset-major" data-major-index="${majorIndex}"${query ? " open" : ""}>
    <summary><span class="asset-tree-toggle" aria-hidden="true"></span><span><span class="asset-node-name">${escapeHtml(major.label)}</span><span class="asset-node-description">${escapeHtml(major.summary || major.en || "")}</span></span><span class="asset-node-count">${Number(major.count || 0).toLocaleString()} 资产</span></summary>
    <div class="asset-groups">${(major.tags || []).map((group, groupIndex) => `<details class="asset-group" data-major-index="${majorIndex}" data-group-index="${groupIndex}"${query ? " open" : ""}>
      <summary><span class="asset-tree-toggle" aria-hidden="true"></span><span><span class="asset-node-name">${escapeHtml(group.label)}</span><span class="asset-node-description">${escapeHtml(group.focus || group.en || "")}</span></span><span class="asset-node-count">${Number(group.count || 0).toLocaleString()} 资产</span></summary>
      <div class="asset-tags"></div>
    </details>`).join("")}</div>
  </details>`).join("");
  root.querySelectorAll(".asset-group").forEach((details) => {
    const major = majors[Number(details.dataset.majorIndex)];
    const group = major?.tags?.[Number(details.dataset.groupIndex)];
    details.addEventListener("toggle", () => { if (details.open && group) hydrateAssetGroup(details, group); });
    if (query && group) hydrateAssetGroup(details, group);
  });
}

function prefetchTaxonomyNode(nodeId) {
  return cachedJson(compactLibraryUrlForNode(nodeId)).catch(() => null);
}

async function navigateToLibraryNode(nodeId, { replace = false } = {}) {
  state.filters = { node_id: nodeId };
  state.offset = 0;
  $("#searchInput").value = "";
  const url = `/ui?node_id=${encodeURIComponent(nodeId)}`;
  history[replace ? "replaceState" : "pushState"]({}, "", url);
  showView("library");
  await loadLibrary();
  $(".results").scrollIntoView({ behavior: "smooth", block: "start" });
}

function parseRoute() {
  if (window.location.pathname === "/ui/action-taxonomy" || decodeURIComponent(window.location.pathname) === "/动作资产分类树.html") {
    return { view: "taxonomy", nodeId: "" };
  }
  return { view: "library", nodeId: new URLSearchParams(window.location.search).get("node_id") || "" };
}

async function boot() {
  try {
    state.limit = window.matchMedia("(max-width: 620px)").matches ? 12 : 24;
    const route = parseRoute();
    state.view = route.view;
    if (route.nodeId) state.filters.node_id = route.nodeId;
    syncFilterAccessibility();
    const taxonomyPromise = cachedJson("/api/v1/taxonomy?view=compact", { ttl: 300000 });
    const libraryPromise = route.view === "library" ? cachedJson(libraryUrl()) : null;
    const taxonomy = await taxonomyPromise;
    state.taxonomy = taxonomy.data?.axes || [];
    state.taxonomySources = taxonomy.data?.sources || [];
    state.hierarchy = taxonomy.data?.hierarchy || [];
    state.taxonomyNodes = flattenHierarchy(state.hierarchy);
    state.taxonomyCounts = taxonomy.data?.asset_taxonomy_counts || {};
    $("#taxonomyMajorCount").textContent = Number(state.taxonomyCounts.major || state.hierarchy.length).toLocaleString();
    $("#taxonomyGroupCount").textContent = Number(state.taxonomyCounts.group || 0).toLocaleString();
    $("#taxonomyTagCount").textContent = Number(state.taxonomyCounts.tag || 0).toLocaleString();
    showView(route.view);
    if (libraryPromise) applyLibraryPayload(await libraryPromise);
    else $("#statusBadge").textContent = `分类 ${Number(state.taxonomyCounts.tag || 0).toLocaleString()} · 选择标签查看资产`;
  } catch (error) {
    setLibraryNotice(`初始化失败：${error.message || error}`);
    $("#statusBadge").textContent = "读取失败";
  }
}

function setFiltersOpen(open) {
  document.body.classList.toggle("filters-open", open);
  syncFilterAccessibility();
  updateOverlayState();
  if (open) requestAnimationFrame(() => $("#closeFilters").focus());
  else $("#filterToggle").focus();
}

function syncFilterAccessibility() {
  const mobile = window.matchMedia("(max-width: 900px)").matches;
  const open = document.body.classList.contains("filters-open");
  $("#filtersPanel").inert = mobile && !open;
  $("#filterToggle").setAttribute("aria-expanded", String(mobile ? open : !document.body.classList.contains("filters-collapsed")));
}

function toggleFilters() {
  if (window.matchMedia("(max-width: 900px)").matches) {
    setFiltersOpen(!document.body.classList.contains("filters-open"));
    return;
  }
  document.body.classList.toggle("filters-collapsed");
  $("#filterToggle").setAttribute("aria-expanded", String(!document.body.classList.contains("filters-collapsed")));
}

async function pageResults(direction) {
  state.offset = direction < 0 ? Math.max(0, state.offset - state.limit) : state.offset + state.limit;
  if (await loadLibrary()) $(".results").scrollIntoView({ behavior: "smooth", block: "start" });
}

let searchTimer;
$("#searchInput").addEventListener("input", () => { clearTimeout(searchTimer); searchTimer = setTimeout(() => { state.offset = 0; loadLibrary(); }, 220); });
$("#sortSelect").addEventListener("change", () => { state.offset = 0; loadLibrary(); });
$("#clearFilters").addEventListener("click", () => {
  state.filters = {};
  state.taxonomyQuery = "";
  state.offset = 0;
  $("#searchInput").value = "";
  history.pushState({}, "", "/ui");
  loadLibrary();
});
$("#prevPage").addEventListener("click", () => pageResults(-1));
$("#nextPage").addEventListener("click", () => pageResults(1));
$("#filterToggle").addEventListener("click", toggleFilters);
$("#closeFilters").addEventListener("click", () => setFiltersOpen(false));
$("#filterScrim").addEventListener("click", () => setFiltersOpen(false));
$("#closeViewer").addEventListener("click", closeViewer);
$("#viewerScrim").addEventListener("click", closeViewer);
$("#retryViewer").addEventListener("click", () => { if (currentItem) previewItem(currentItem); });
$("#libraryViewButton").addEventListener("click", () => {
  state.filters = {};
  state.offset = 0;
  $("#searchInput").value = "";
  showView("library", { updateHistory: true });
  loadLibrary();
});
$("#taxonomyViewButton").addEventListener("click", () => showView("taxonomy", { updateHistory: true }));
$("#assetTaxonomySearch").addEventListener("input", (event) => {
  clearTimeout(taxonomySearchTimer);
  taxonomySearchTimer = setTimeout(() => {
    state.assetTaxonomyQuery = event.target.value.trim();
    renderAssetTaxonomy();
  }, 120);
});
$("#assetTaxonomyExpand").addEventListener("click", () => {
  $("#assetTaxonomyTree").querySelectorAll("details").forEach((details) => { details.open = true; });
});
$("#assetTaxonomyCollapse").addEventListener("click", () => {
  $("#assetTaxonomyTree").querySelectorAll("details").forEach((details) => { details.open = false; });
});
$("#assetTaxonomyTree").addEventListener("click", (event) => {
  const link = event.target.closest("[data-asset-node-id]");
  if (!link) return;
  event.preventDefault();
  navigateToLibraryNode(link.dataset.assetNodeId);
});
$("#assetTaxonomyTree").addEventListener("pointerover", (event) => {
  const link = event.target.closest("[data-asset-node-id]");
  if (!link) return;
  clearTimeout(taxonomyPrefetchTimer);
  taxonomyPrefetchTimer = setTimeout(() => prefetchTaxonomyNode(link.dataset.assetNodeId), 80);
});
$("#assetTaxonomyTree").addEventListener("pointerout", () => clearTimeout(taxonomyPrefetchTimer));
$("#assetTaxonomyTree").addEventListener("focusin", (event) => {
  const link = event.target.closest("[data-asset-node-id]");
  if (link) prefetchTaxonomyNode(link.dataset.assetNodeId);
});
window.addEventListener("popstate", () => {
  const route = parseRoute();
  if (route.view === "taxonomy") {
    showView("taxonomy");
    return;
  }
  state.filters = route.nodeId ? { node_id: route.nodeId } : {};
  state.offset = 0;
  showView("library");
  loadLibrary();
});
window.addEventListener("resize", syncFilterAccessibility);
document.addEventListener("keydown", (event) => {
  if (event.key === "Tab") {
    const container = document.body.classList.contains("viewer-open")
      ? $("#viewerDrawer")
      : document.body.classList.contains("filters-open") ? $("#filtersPanel") : null;
    if (container) {
      const focusable = [...container.querySelectorAll('button:not([disabled]), a[href]:not(.disabled), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])')];
      const first = focusable[0];
      const last = focusable.at(-1);
      if (first && (!container.contains(document.activeElement) || (event.shiftKey && document.activeElement === first) || (!event.shiftKey && document.activeElement === last))) {
        event.preventDefault();
        (event.shiftKey ? last : first).focus();
      }
    }
  }
  if (event.key !== "Escape") return;
  if (document.body.classList.contains("viewer-open")) closeViewer();
  else if (document.body.classList.contains("filters-open")) setFiltersOpen(false);
});

boot();
