// TruthLens AI — Frontend Logic
const API = 'http://localhost:8000';

// ── Particle canvas ──────────────────────────────────────
const canvas = document.getElementById('particle-canvas');
const ctx = canvas.getContext('2d');
let particles = [];

function resizeCanvas() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
}
resizeCanvas();
window.addEventListener('resize', resizeCanvas);

for (let i = 0; i < 60; i++) {
  particles.push({
    x: Math.random() * window.innerWidth,
    y: Math.random() * window.innerHeight,
    r: Math.random() * 1.5 + 0.5,
    dx: (Math.random() - 0.5) * 0.4,
    dy: (Math.random() - 0.5) * 0.4,
    alpha: Math.random() * 0.5 + 0.1,
  });
}

function animateParticles() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  particles.forEach(p => {
    p.x += p.dx; p.y += p.dy;
    if (p.x < 0) p.x = canvas.width;
    if (p.x > canvas.width) p.x = 0;
    if (p.y < 0) p.y = canvas.height;
    if (p.y > canvas.height) p.y = 0;
    ctx.beginPath();
    ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(124,58,237,${p.alpha})`;
    ctx.fill();
  });
  requestAnimationFrame(animateParticles);
}
animateParticles();

// ── Tab switching ────────────────────────────────────────
function switchTab(tab) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.input-panel').forEach(p => p.classList.remove('active'));
  document.getElementById('tab-' + tab).classList.add('active');
  document.getElementById('panel-' + tab).classList.add('active');
  hideResults();
}

// ── Char counter ─────────────────────────────────────────
const textInput = document.getElementById('text-input');
if (textInput) {
  textInput.addEventListener('input', () => {
    document.getElementById('char-count').textContent = textInput.value.length;
  });
}

// ── Samples ──────────────────────────────────────────────
const SAMPLES = {
  fake: `BREAKING: Scientists EXPOSED for lying about vaccines — share before it gets deleted! The DEEP STATE has been hiding the REAL data from the public for years. You won't believe what they found! This miracle cure is being suppressed by Big Pharma. URGENT: Share this NOW before they remove it!`,
  real: `According to data released by the Reserve Bank of India on Wednesday, the country's foreign exchange reserves increased by $2.3 billion to reach $645 billion. RBI Governor stated that the central bank would continue to monitor inflation trends, which stood at 4.8% in the previous month, according to official Ministry of Statistics figures.`,
  hinglish: `Viral ho rahi hai yeh khabar ke sarkar ne EVM mein gaddari ki hai. Yeh sach hai, share karo sabko! Nakli khabar phailai ja rahi hai — desh ke logon ko jagruk karo. Tukde tukde gang ka naya षड्यंत्र saamne aaya hai. Turant share karo!`,
};

function loadSample(type) {
  document.getElementById('text-input').value = SAMPLES[type];
  document.getElementById('char-count').textContent = SAMPLES[type].length;
}

// ── URL helpers ──────────────────────────────────────────
function setURL(url) {
  document.getElementById('url-input').value = url;
}

// ── Image upload ─────────────────────────────────────────
let selectedImageFile = null;

function handleImageSelect(event) {
  const file = event.target.files[0];
  if (file) setImageFile(file);
}

function handleDrop(event) {
  event.preventDefault();
  document.getElementById('drop-zone').classList.remove('drag-over');
  const file = event.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) setImageFile(file);
}

function setImageFile(file) {
  selectedImageFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    const preview = document.getElementById('img-preview');
    preview.src = e.target.result;
    preview.style.display = 'block';
  };
  reader.readAsDataURL(file);
  document.getElementById('btn-analyze-image').disabled = false;
  document.querySelector('#panel-image .drop-text').textContent = `✅ ${file.name} (${(file.size/1024).toFixed(0)} KB)`;
}

// ── Video upload ─────────────────────────────────────────
let selectedVideoFile = null;

function handleVideoSelect(event) {
  const file = event.target.files[0];
  if (file) setVideoFile(file);
}

function handleVideoDrop(event) {
  event.preventDefault();
  document.getElementById('drop-zone-video').classList.remove('drag-over');
  const file = event.dataTransfer.files[0];
  if (file && file.type.startsWith('video/')) setVideoFile(file);
}

function setVideoFile(file) {
  selectedVideoFile = file;
  const url = URL.createObjectURL(file);
  const preview = document.getElementById('video-preview');
  preview.src = url;
  preview.style.display = 'block';
  preview.load();
  document.getElementById('btn-analyze-video').disabled = false;
  document.getElementById('drop-text-video').textContent = `✅ ${file.name} (${(file.size/1024/1024).toFixed(1)} MB)`;
}

// ── Loading helpers ──────────────────────────────────────
const LOADING_STEPS = {
  text: ['🔤 Parsing linguistic features...', '🧠 Running NLP classifier...', '🔍 Cross-validating sources...', '💡 Generating XAI explanation...'],
  image: ['🖼 Loading image...', '📊 FFT frequency analysis...', '🔬 Noise pattern scan...', '⚡ Deepfake scoring...'],
  url: ['🌐 Extracting domain info...', '🔍 Checking credibility database...', '📋 Fetching page metadata...'],
  video: ['🎥 Extracting frames...', '📊 Frame-by-frame analysis...', '🔬 Checking temporal consistency...', '⚡ Deepfake video scoring...'],
};

function showLoading(type) {
  const overlay = document.getElementById('loading-overlay');
  const steps = document.getElementById('loading-steps');
  overlay.style.display = 'flex';
  steps.innerHTML = '';
  let i = 0;
  const stepList = LOADING_STEPS[type];
  const interval = setInterval(() => {
    if (i < stepList.length) {
      document.getElementById('loading-text').textContent = stepList[i];
      steps.innerHTML += `<div style="opacity:.6">${stepList[i]}</div>`;
      i++;
    } else {
      clearInterval(interval);
    }
  }, 400);
}

function hideLoading() {
  document.getElementById('loading-overlay').style.display = 'none';
}

function hideResults() {
  document.getElementById('results-panel').style.display = 'none';
}

// ── Render results ────────────────────────────────────────
function renderVerdict(verdict, confidence, sub) {
  const banner = document.getElementById('verdict-banner');
  const label = document.getElementById('verdict-label');
  const subEl = document.getElementById('verdict-sub');
  const badge = document.getElementById('verdict-badge');
  const confBar = document.getElementById('confidence-bar');
  const confVal = document.getElementById('confidence-value');

  banner.className = 'verdict-banner ' + verdict.toLowerCase();

  const MAP = {
    FAKE:       { icon: '🔴', badge: '🔴 FAKE',       color: 'var(--red)'    },
    REAL:       { icon: '🟢', badge: '🟢 RELIABLE',   color: 'var(--green)'  },
    SUSPICIOUS: { icon: '🟡', badge: '🟡 SUSPICIOUS', color: 'var(--yellow)' },
  };
  const m = MAP[verdict] || MAP.SUSPICIOUS;
  document.getElementById('verdict-icon').textContent = m.icon;
  label.textContent = verdict === 'REAL' ? 'LIKELY REAL' : verdict;
  label.style.color = m.color;
  subEl.textContent = sub || '';
  badge.textContent = m.badge;

  const pct = Math.round(confidence * 100);
  setTimeout(() => { confBar.style.width = pct + '%'; }, 100);
  confVal.textContent = pct + '%';
  confVal.style.color = m.color;
}

function renderSignals(signals) {
  const section = document.getElementById('signals-section');
  const list = document.getElementById('signals-list');
  if (!signals || !signals.length) { section.style.display = 'none'; return; }
  section.style.display = 'block';
  list.innerHTML = signals.map(s => {
    const cls = s.startsWith('🔴') ? 'red' : s.startsWith('🟢') ? 'green' : 'yellow';
    return `<div class="signal-item ${cls}">${s}</div>`;
  }).join('');
}

function renderCredibility(cred) {
  if (!cred) { document.getElementById('credibility-card').style.display = 'none'; return; }
  document.getElementById('credibility-card').style.display = 'block';
  const score = cred.credibility_score ?? cred.score ?? 0;
  document.getElementById('cred-score').textContent = score;
  document.getElementById('cred-label').textContent = cred.tier ? `Tier: ${cred.tier}` : 'Credibility';
  const circle = document.getElementById('cred-circle');
  const circumference = 213.6;
  setTimeout(() => {
    circle.style.strokeDashoffset = circumference - (score / 100) * circumference;
  }, 200);
}

function renderVerification(v) {
  if (!v) { document.getElementById('verification-card').style.display = 'none'; return; }
  document.getElementById('verification-card').style.display = 'block';
  const el = document.getElementById('verify-status');
  el.textContent = v.message;
  el.style.background = v.status === 'verified' ? 'rgba(16,185,129,.12)' : v.status === 'disputed' ? 'rgba(239,68,68,.1)' : 'rgba(245,158,11,.1)';
  const src = document.getElementById('verify-sources');
  src.textContent = v.sources?.length ? 'Sources: ' + v.sources.slice(0,3).join(', ') : '';
}

function renderPhrases(phrases) {
  const section = document.getElementById('phrases-section');
  if (!phrases?.length) { section.style.display = 'none'; return; }
  section.style.display = 'block';
  document.getElementById('phrases-list').innerHTML = phrases.map(p =>
    `<span class="phrase-tag">"${p}"</span>`
  ).join('');
}

function renderRecommendation(text) {
  const section = document.getElementById('recommendation-section');
  if (!text) { section.style.display = 'none'; return; }
  section.style.display = 'block';
  document.getElementById('recommendation-text').textContent = text;
}

function renderFlags(flags) {
  const section = document.getElementById('flags-section');
  if (!flags?.length) { section.style.display = 'none'; return; }
  section.style.display = 'block';
  document.getElementById('flags-list').innerHTML = flags.map(f =>
    `<div class="flag-item">🚩 ${f}</div>`
  ).join('');
}

function renderImageScores(scores, artifacts) {
  const section = document.getElementById('image-detail-section');
  if (!scores) { section.style.display = 'none'; return; }
  section.style.display = 'block';
  const scoreHTML = Object.entries(scores).map(([k,v]) => {
    const label = k.replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase());
    const pct = Math.round(v * 100);
    return `<div class="score-item">
      <div class="score-label">${label}</div>
      <div class="score-bar-wrap"><div class="score-bar-fill" style="width:${pct}%"></div></div>
      <div class="score-val">${pct}%</div>
    </div>`;
  }).join('');
  document.getElementById('analysis-scores').innerHTML = scoreHTML;
  document.getElementById('artifacts-list').innerHTML = (artifacts||[]).map(a =>
    `<span class="artifact-tag">⚠ ${a}</span>`
  ).join('');
}

// ── Analyze Text ─────────────────────────────────────────
async function analyzeText() {
  const text = document.getElementById('text-input').value.trim();
  if (text.length < 10) { alert('Please enter at least 10 characters.'); return; }
  showLoading('text');

  try {
    const body = { text, url: document.getElementById('url-hint').value.trim() || null };
    const res = await fetch(`${API}/analyze-text`, {
      method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body)
    });
    const data = await res.json();
    hideLoading();

    renderVerdict(data.verdict, data.confidence, `Processed in ${data.processing_time_ms}ms`);
    renderSignals(data.signals);
    renderPhrases(data.highlighted_phrases);
    renderCredibility(data.credibility);
    renderVerification(data.verification);
    renderRecommendation(data.explanation?.recommendation);
    document.getElementById('credibility-card').style.display = data.credibility ? 'block' : 'none';
    document.getElementById('image-detail-section').style.display = 'none';
    document.getElementById('flags-section').style.display = 'none';
    document.getElementById('processing-tag').textContent = `⚡ Analyzed in ${data.processing_time_ms}ms`;
    document.getElementById('results-panel').style.display = 'block';
    document.getElementById('results-panel').scrollIntoView({behavior:'smooth', block:'start'});
  } catch(e) {
    hideLoading();
    alert('Backend not reachable. Make sure uvicorn is running on port 8000.\n\n' + e.message);
  }
}

// ── Analyze Image ────────────────────────────────────────
async function analyzeImage() {
  if (!selectedImageFile) { alert('Please select an image first.'); return; }
  showLoading('image');

  try {
    const form = new FormData();
    form.append('file', selectedImageFile);
    const res = await fetch(`${API}/analyze-image`, { method:'POST', body: form });
    const data = await res.json();
    hideLoading();

    renderVerdict(data.verdict, data.confidence, `Deepfake analysis complete`);
    renderSignals(data.signals);
    renderImageScores(data.analysis_scores, data.artifacts_detected);
    document.getElementById('credibility-card').style.display = 'none';
    document.getElementById('verification-card').style.display = 'none';
    document.getElementById('phrases-section').style.display = 'none';
    document.getElementById('recommendation-section').style.display = 'none';
    document.getElementById('flags-section').style.display = 'none';
    document.getElementById('processing-tag').textContent = `⚡ Analyzed in ${data.processing_time_ms}ms`;
    document.getElementById('results-panel').style.display = 'block';
    document.getElementById('results-panel').scrollIntoView({behavior:'smooth', block:'start'});
  } catch(e) {
    hideLoading();
    alert('Backend error: ' + e.message);
  }
}

// ── Analyze Video ────────────────────────────────────────
async function analyzeVideo() {
  if (!selectedVideoFile) { alert('Please select a video first.'); return; }
  showLoading('video');

  try {
    const form = new FormData();
    form.append('file', selectedVideoFile);
    const res = await fetch(`${API}/analyze-video`, { method:'POST', body: form });
    const data = await res.json();
    hideLoading();

    if (res.status !== 200) {
        throw new Error(data.detail || 'Analysis failed');
    }

    renderVerdict(data.verdict, data.confidence, `Deepfake video analysis complete (${data.frames_analyzed} frames)`);
    renderSignals(data.signals);
    renderImageScores(data.analysis_scores, data.artifacts_detected);
    document.getElementById('credibility-card').style.display = 'none';
    document.getElementById('verification-card').style.display = 'none';
    document.getElementById('phrases-section').style.display = 'none';
    document.getElementById('recommendation-section').style.display = 'none';
    document.getElementById('flags-section').style.display = 'none';
    document.getElementById('processing-tag').textContent = `⚡ Analyzed in ${data.processing_time_ms}ms`;
    document.getElementById('results-panel').style.display = 'block';
    document.getElementById('results-panel').scrollIntoView({behavior:'smooth', block:'start'});
  } catch(e) {
    hideLoading();
    alert('Backend error: ' + e.message);
  }
}

// ── Analyze URL ──────────────────────────────────────────
async function analyzeURL() {
  const url = document.getElementById('url-input').value.trim();
  if (!url) { alert('Please enter a URL.'); return; }
  showLoading('url');

  try {
    const res = await fetch(`${API}/analyze-url`, {
      method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({url})
    });
    const data = await res.json();
    hideLoading();

    const sub = data.page_title ? `"${data.page_title.slice(0,60)}"` : `Domain: ${data.domain}`;
    renderVerdict(data.verdict, data.credibility_score / 100, sub);
    renderSignals(data.signals);
    renderFlags(data.flags);
    renderCredibility({ credibility_score: data.credibility_score, tier: data.tier });
    document.getElementById('verification-card').style.display = 'none';
    document.getElementById('phrases-section').style.display = 'none';
    document.getElementById('image-detail-section').style.display = 'none';
    document.getElementById('recommendation-section').style.display = 'none';
    document.getElementById('processing-tag').textContent = `⚡ Analyzed in ${data.processing_time_ms}ms`;
    document.getElementById('results-panel').style.display = 'block';
    document.getElementById('results-panel').scrollIntoView({behavior:'smooth', block:'start'});
  } catch(e) {
    hideLoading();
    alert('Backend error: ' + e.message);
  }
}

// ── Enter key shortcuts ───────────────────────────────────
document.getElementById('url-input')?.addEventListener('keydown', e => {
  if (e.key === 'Enter') analyzeURL();
});
