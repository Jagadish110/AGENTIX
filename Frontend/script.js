const BACKEND_URL = "http://localhost:8000";  // ← Change to Render URL later

const openBtn   = document.getElementById('open-chat');
const modal     = document.getElementById('chat-modal');
const closeBtn  = document.getElementById('close-modal');
const chatBox   = document.getElementById('chat');
const input     = document.getElementById('question');
const sendBtn   = document.getElementById('send-btn');
const status    = document.getElementById('status');

let userId = localStorage.getItem('agentix_uid');
if (!userId) {
  userId = 'u_' + Date.now();
  localStorage.setItem('agentix_uid', userId);
}

openBtn.addEventListener('click', () => modal.classList.remove('hidden'));
closeBtn.addEventListener('click', () => modal.classList.add('hidden'));
modal.addEventListener('click', e => { if (e.target === modal) modal.classList.add('hidden'); });

async function send() {
  const q = input.value.trim();
  if (!q) return;

  addMsg(q, 'user');
  input.value = '';
  sendBtn.disabled = true;
  showStatus('Thinking…');

  try {
    const res = await fetch(`${BACKEND_URL}/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q, user_id: userId })
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    addMsg(data.response || 'No response.', 'bot', data.source_tool);
  } catch (err) {
    addMsg(`Error: ${err.message}`, 'bot');
  } finally {
    sendBtn.disabled = false;
    clearStatus();
  }
}

function addMsg(text, sender, source = null) {
  const el = document.createElement('div');
  el.className = `message ${sender}`;
  el.innerHTML = text.replace(/\n/g, '<br>');
  if (source) {
    const src = document.createElement('div');
    src.className = 'source';
    src.textContent = `Source: ${source}`;
    el.appendChild(src);
  }
  chatBox.appendChild(el);
  chatBox.scrollTop = chatBox.scrollHeight;
}
function showStatus(txt) { status.innerHTML = `${txt}<span class="loading"></span>`; }
function clearStatus() { status.textContent = ''; }

sendBtn.addEventListener('click', send);
input.addEventListener('keypress', e => { if (e.key === 'Enter') send(); });

window.addEventListener('load', () => {
  addMsg('Hi! I’m Agentix. Ask me anything.', 'bot');
});
