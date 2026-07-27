// ────────────────────────────────────────────────────────────────
// LlamaMed-3.1-8B-Reasoner-Agent — local UI
// All network traffic goes here. Change this if the backend moves.
// ────────────────────────────────────────────────────────────────
const API_BASE = 'http://localhost:8000';

const els = {
  sidebar:        document.getElementById('sidebar'),
  chatList:       document.getElementById('chat-list'),
  newChatBtn:     document.getElementById('new-chat-btn'),
  chatColumn:     document.getElementById('chat-column'),
  chatScroll:     document.getElementById('chat-scroll'),
  composerInput:  document.getElementById('composer-input'),
  sendBtn:        document.getElementById('send-btn'),
  fileInput:      document.getElementById('file-input'),
  attachments:    document.getElementById('attachments'),
  sessionTitle:   document.getElementById('session-title'),
  collapseBtn:    document.getElementById('collapse-sidebar'),
  openSidebarBtn: document.getElementById('open-sidebar'),
};

const state = {
  currentSessionId: null,
  sessions: [],
  attachments: [],  // { id, filename, status, chunks, error }
  sending: false,
};

// ─── utilities ─────────────────────────────────────────────────
function relTime(iso) {
  if (!iso) return '';
  const t = new Date(iso).getTime();
  if (isNaN(t)) return '';
  const diff = Date.now() - t;
  const m = Math.floor(diff / 60000);
  if (m < 1) return 'just now';
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.floor(h / 24);
  if (d < 7) return `${d}d ago`;
  return new Date(iso).toLocaleDateString();
}

function escapeHtml(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    els.chatScroll.scrollTop = els.chatScroll.scrollHeight;
  });
}

function autosize() {
  const ta = els.composerInput;
  ta.style.height = 'auto';
  ta.style.height = Math.min(ta.scrollHeight, 200) + 'px';
}

// ─── sessions list ─────────────────────────────────────────────
async function loadSessions() {
  try {
    const r = await fetch(`${API_BASE}/api/sessions`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    state.sessions = data.sessions || [];
    renderSidebar();
  } catch (e) {
    console.error('Failed to load sessions', e);
  }
}

function renderSidebar() {
  els.chatList.innerHTML = '';
  for (const s of state.sessions) {
    const item = document.createElement('div');
    item.className = 'chat-item' + (s.id === state.currentSessionId ? ' active' : '');
    item.dataset.id = s.id;
    item.innerHTML = `
      <div class="chat-item-row">
        <div class="chat-title">${escapeHtml(s.title || 'Untitled')}</div>
        <button class="delete-btn" title="Delete chat" aria-label="Delete chat">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
        </button>
      </div>
      <div class="chat-time">${escapeHtml(relTime(s.updated_at))}</div>
    `;
    item.addEventListener('click', (e) => {
      if (e.target.closest('.delete-btn')) return;
      openSession(s.id);
    });
    item.querySelector('.delete-btn').addEventListener('click', (e) => {
      e.stopPropagation();
      deleteSession(s.id);
    });
    els.chatList.appendChild(item);
  }
}

async function createNewSession() {
  try {
    const r = await fetch(`${API_BASE}/api/sessions`, { method: 'POST' });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    state.currentSessionId = data.id;
    state.attachments = [];
    renderAttachments();
    els.chatColumn.innerHTML = '';
    renderEmptyState();
    els.sessionTitle.textContent = 'New chat';
    await loadSessions();
    renderSidebar();
    els.composerInput.focus();
    if (window.innerWidth < 768) els.sidebar.classList.add('collapsed');
  } catch (e) {
    showInlineError(`Couldn't create new chat: ${e.message}`);
  }
}

async function openSession(id) {
  state.currentSessionId = id;
  state.attachments = [];
  renderAttachments();
  try {
    const r = await fetch(`${API_BASE}/api/sessions/${id}/messages`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    els.chatColumn.innerHTML = '';
    const msgs = data.messages || [];
    if (msgs.length === 0) renderEmptyState();
    else for (const m of msgs) renderMessage(m.role, m.text, m.trace);
    const session = state.sessions.find(s => s.id === id);
    els.sessionTitle.textContent = session ? (session.title || 'New chat') : 'Chat';
    renderSidebar();
    scrollToBottom();
    if (window.innerWidth < 768) els.sidebar.classList.add('collapsed');
  } catch (e) {
    showInlineError(`Couldn't load chat: ${e.message}`);
  }
}

async function deleteSession(id) {
  try {
    const r = await fetch(`${API_BASE}/api/sessions/${id}`, { method: 'DELETE' });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    if (state.currentSessionId === id) {
      state.currentSessionId = null;
      state.attachments = [];
      renderAttachments();
      els.chatColumn.innerHTML = '';
      renderEmptyState();
      els.sessionTitle.textContent = 'New chat';
    }
    await loadSessions();
  } catch (e) {
    showInlineError(`Couldn't delete chat: ${e.message}`);
  }
}

// ─── messages ──────────────────────────────────────────────────
function renderEmptyState() {
  const div = document.createElement('div');
  div.className = 'empty-state';
  div.innerHTML = `
    <h2>Ask about your documents</h2>
    <p>Attach a PDF using the paperclip, then ask a question to get started.</p>
  `;
  els.chatColumn.appendChild(div);
}

function renderMessage(role, text, trace) {
  const empty = els.chatColumn.querySelector('.empty-state');
  if (empty) empty.remove();

  const msg = document.createElement('div');
  msg.className = `msg ${role}`;

  if (role === 'user') {
    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';
    bubble.textContent = text;
    msg.appendChild(bubble);
  } else {
    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = 'L';
    msg.appendChild(avatar);

    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';
    const content = document.createElement('div');
    content.className = 'msg-content';

    const textDiv = document.createElement('div');
    textDiv.textContent = text;
    content.appendChild(textDiv);

    if (Array.isArray(trace) && trace.length > 0) {
      content.appendChild(buildReasoning(trace));
    }

    bubble.appendChild(content);
    msg.appendChild(bubble);
  }

  els.chatColumn.appendChild(msg);
  scrollToBottom();
  return msg;
}

function buildReasoning(trace) {
  const wrap = document.createElement('div');
  wrap.className = 'reasoning';

  const toggle = document.createElement('button');
  toggle.className = 'reasoning-toggle';
  toggle.innerHTML = `
    <svg class="chevron" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18l6-6-6-6"/></svg>
    Show reasoning
  `;

  const content = document.createElement('div');
  content.className = 'reasoning-content';

  for (const step of trace) {
    const stepDiv = document.createElement('div');
    stepDiv.className = 'trace-step';
    let html = '';
    if (step.thought)     html += `<div><span class="trace-label">Thought:</span> ${escapeHtml(step.thought)}</div>`;
    if (step.action)      html += `<div><span class="trace-label">Action:</span> ${escapeHtml(step.action)}</div>`;
    if (step.action_input != null)
                          html += `<div><span class="trace-label">Input:</span> ${escapeHtml(typeof step.action_input === 'string' ? step.action_input : JSON.stringify(step.action_input))}</div>`;
    if (step.observation) html += `<div><span class="trace-label">Observation:</span> ${escapeHtml(step.observation)}</div>`;
    stepDiv.innerHTML = html;
    content.appendChild(stepDiv);
  }

  toggle.addEventListener('click', () => {
    toggle.classList.toggle('open');
    content.classList.toggle('open');
    toggle.lastChild.textContent = toggle.classList.contains('open') ? ' Hide reasoning' : ' Show reasoning';
  });

  wrap.appendChild(toggle);
  wrap.appendChild(content);
  return wrap;
}

function renderThinking() {
  const empty = els.chatColumn.querySelector('.empty-state');
  if (empty) empty.remove();

  const msg = document.createElement('div');
  msg.className = 'msg assistant';
  msg.id = 'thinking-msg';
  msg.innerHTML = `
    <div class="avatar">L</div>
    <div class="msg-bubble">
      <div class="msg-content">
        <div class="thinking">
          <span class="dot"></span><span class="dot"></span><span class="dot"></span>
        </div>
      </div>
    </div>
  `;
  els.chatColumn.appendChild(msg);
  scrollToBottom();
}

function removeThinking() {
  const t = document.getElementById('thinking-msg');
  if (t) t.remove();
}

async function sendMessage() {
  if (state.sending) return;
  const text = els.composerInput.value.trim();
  if (!text) return;

  if (!state.currentSessionId) {
    await createNewSession();
    if (!state.currentSessionId) return;
  }

  state.sending = true;
  els.sendBtn.disabled = true;
  els.composerInput.value = '';
  autosize();

  renderMessage('user', text);
  renderThinking();

  try {
    const r = await fetch(`${API_BASE}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: state.currentSessionId, message: text }),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    removeThinking();
    renderMessage('assistant', data.reply, data.trace);
    loadSessions(); // refresh timestamps/titles in sidebar
  } catch (e) {
    removeThinking();
    showInlineError(`Couldn't send message: ${e.message}`);
  } finally {
    state.sending = false;
    els.sendBtn.disabled = false;
    els.composerInput.focus();
  }
}

// ─── attachments ───────────────────────────────────────────────
function renderAttachments() {
  els.attachments.innerHTML = '';
  for (const a of state.attachments) {
    const chip = document.createElement('div');
    chip.className = 'chip';
    chip.dataset.id = a.id;

    let statusHtml = '';
    if (a.status === 'indexing') statusHtml = `<span class="chip-status">Indexing…</span>`;
    else if (a.status === 'ok')  statusHtml = `<span class="chip-status ok">Indexed (${a.chunks} chunks)</span>`;
    else if (a.status === 'error') statusHtml = `<span class="chip-status error">${escapeHtml(a.error || 'Error')}</span>`;

    chip.innerHTML = `
      <span class="chip-icon">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/></svg>
      </span>
      <span class="chip-name">${escapeHtml(a.filename)}</span>
      ${statusHtml}
      <button class="chip-remove" title="Remove" aria-label="Remove attachment">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6L6 18M6 6l12 12"/></svg>
      </button>
    `;
    chip.querySelector('.chip-remove').addEventListener('click', () => {
      state.attachments = state.attachments.filter(x => x.id !== a.id);
      renderAttachments();
    });
    els.attachments.appendChild(chip);
  }
}

async function attachFile(file) {
  if (!state.currentSessionId) {
    await createNewSession();
    if (!state.currentSessionId) return;
  }

  const id = 'att_' + Date.now() + '_' + Math.random().toString(36).slice(2, 8);
  const att = { id, filename: file.name, status: 'indexing', chunks: 0, error: null };
  state.attachments.push(att);
  renderAttachments();

  const fd = new FormData();
  fd.append('file', file);
  fd.append('session_id', state.currentSessionId);

  try {
    const r = await fetch(`${API_BASE}/api/attach`, { method: 'POST', body: fd });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    const target = state.attachments.find(x => x.id === id);
    if (!target) return;
    if (data.status === 'ok') {
      target.status = 'ok';
      target.chunks = data.chunks_indexed;
    } else {
      target.status = 'error';
      target.error = data.error || 'Failed to index';
    }
    renderAttachments();
  } catch (e) {
    const target = state.attachments.find(x => x.id === id);
    if (target) {
      target.status = 'error';
      target.error = e.message;
      renderAttachments();
    }
  }
}

// ─── inline errors ─────────────────────────────────────────────
function showInlineError(message) {
  const div = document.createElement('div');
  div.className = 'inline-error';
  div.textContent = message;
  els.chatColumn.appendChild(div);
  scrollToBottom();
  setTimeout(() => { if (div.parentNode) div.parentNode.removeChild(div); }, 6000);
}

// ─── event bindings ────────────────────────────────────────────
els.newChatBtn.addEventListener('click', createNewSession);
els.sendBtn.addEventListener('click', sendMessage);
els.composerInput.addEventListener('input', autosize);
els.composerInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});
els.fileInput.addEventListener('change', (e) => {
  const file = e.target.files && e.target.files[0];
  if (file) attachFile(file);
  e.target.value = '';
});
els.collapseBtn.addEventListener('click', () => els.sidebar.classList.add('collapsed'));
els.openSidebarBtn.addEventListener('click', () => els.sidebar.classList.remove('collapsed'));

// ─── init ──────────────────────────────────────────────────────
(async function init() {
  if (window.innerWidth < 768) els.sidebar.classList.add('collapsed');
  await loadSessions();
  renderEmptyState();
  els.composerInput.focus();
})();
