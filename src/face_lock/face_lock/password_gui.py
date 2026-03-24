"""Web GUI lifecycle node for FaceLock password setup."""

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Optional

import cv2
import numpy as np
import rclpy
from rclpy.client import Client
from rclpy.lifecycle import LifecycleNode, State, TransitionCallbackReturn
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger

from face_lock.constants import (
    BLENDSHAPE_THRESHOLD,
    IDENTITIES_DIR,
    IGNORED_BLENDSHAPES,
    PASSWORDS_DIR,
)
from robot_interfaces.msg import FaceBlendshapes
from robot_interfaces.srv import SetName

GUI_PORT = 8080

_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>FaceLock Setup</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: monospace; background: #0d0d0d; color: #ddd; height: 100vh; display: flex; overflow: hidden; }

#left { width: 55%; display: flex; flex-direction: column; padding: 12px; gap: 8px; border-right: 1px solid #1a1a1a; min-width: 0; }
#stream-img { flex: 1; object-fit: contain; background: #000; border: 1px solid #2a2a2a; min-height: 0; }
.stream-row { display: flex; gap: 8px; flex-shrink: 0; }
.stream-row button { flex: 1; padding: 8px; background: #141414; color: #666; border: 1px solid #252525; cursor: pointer; font-family: monospace; font-size: 12px; }
.stream-row button:hover { background: #1c1c1c; color: #999; }
.stream-row button.active-stream { border-color: #3a7abf; color: #6ab4ff; background: #0e1a28; }

#right { flex: 1; display: flex; flex-direction: column; overflow-y: auto; }
.panel { padding: 16px 20px; display: flex; flex-direction: column; gap: 10px; border-bottom: 1px solid #1a1a1a; }
.panel:last-child { border-bottom: none; }
h2 { font-size: 10px; color: #444; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 2px; }

input[type=text] { width: 100%; padding: 9px 12px; background: #111; color: #ddd; border: 1px solid #2a2a2a; font-family: monospace; font-size: 13px; outline: none; }
input[type=text]:focus { border-color: #3a7abf; }

.big-btn { padding: 12px 16px; background: #141414; color: #bbb; border: 1px solid #2a2a2a; cursor: pointer; font-family: monospace; font-size: 13px; width: 100%; text-align: center; transition: background 0.15s, border-color 0.15s; }
.big-btn:hover:not(:disabled) { background: #1c1c1c; border-color: #444; color: #eee; }
.big-btn:disabled { opacity: 0.35; cursor: default; }
.big-btn.ok  { border-color: #2d6a3f !important; color: #5db87a !important; background: #0a1c10 !important; }
.big-btn.err { border-color: #6a2d2d !important; color: #b85d5d !important; background: #1c0a0a !important; }
.big-btn.primary { border-color: #3a7abf; color: #6ab4ff; background: #0e1a28; }
.big-btn.primary:hover:not(:disabled) { background: #132434; border-color: #5a9adf; }
.big-btn.done { border-color: #7a3abf; color: #c46aff; background: #1a0e28; font-size: 14px; padding: 14px; }
.big-btn.done:hover:not(:disabled) { background: #241332; border-color: #a45adf; }
.btn-row { display: flex; gap: 8px; }
.btn-row .big-btn { flex: 1; }

.counter { font-size: 48px; font-weight: bold; text-align: center; color: #4af; line-height: 1; margin: 2px 0; }
.counter-label { font-size: 11px; color: #444; text-align: center; margin-bottom: 4px; }
.active-label { font-size: 11px; color: #555; text-align: center; min-height: 14px; }
.active-label span { color: #6ab4ff; }

.msg { font-size: 11px; color: #555; min-height: 16px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

.expression-box { background: #0a0a0a; border: 1px solid #1e1e1e; padding: 10px 12px; min-height: 52px; font-size: 12px; color: #7ae; white-space: pre-wrap; line-height: 1.7; }
.expression-empty { color: #333; }

.lib { display: flex; flex-direction: column; gap: 3px; }
.lib-item { display: flex; align-items: center; gap: 6px; padding: 6px 10px; background: #0d0d0d; border: 1px solid #1e1e1e; font-size: 12px; }
.lib-item.active { border-color: #3a7abf; background: #0e1a28; color: #6ab4ff; }
.lib-item span { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.lib-item button { padding: 3px 8px; font-size: 11px; background: #141414; color: #888; border: 1px solid #2a2a2a; cursor: pointer; font-family: monospace; flex-shrink: 0; }
.lib-item button:hover { color: #ccc; border-color: #555; }
.lib-item button.del { color: #7a3535; border-color: #3a1a1a; }
.lib-item button.del:hover { color: #c05555; border-color: #6a2a2a; background: #1a0a0a; }
.lib-empty { font-size: 11px; color: #2a2a2a; padding: 4px 0; }

#steps { display: flex; flex-direction: column; gap: 4px; }
.step { padding: 7px 12px; background: #0a130a; border: 1px solid #1a2e1a; font-size: 12px; color: #6b9; display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px; }
.step b { color: #4c7; }
.step .del-step { margin-left: auto; padding: 1px 6px; font-size: 10px; background: #141414; color: #7a3535; border: 1px solid #3a1a1a; cursor: pointer; font-family: monospace; }
.step .del-step:hover { color: #c05555; border-color: #6a2a2a; background: #1a0a0a; }
.step-empty { font-size: 12px; color: #2a2a2a; padding: 4px 0; }
</style>
</head>
<body>
<div id="left">
  <img id="stream-img" src="/stream/normal" alt="camera stream">
  <div class="stream-row">
    <button id="btn-normal" class="active-stream" onclick="switchStream('normal')">Normal</button>
    <button id="btn-debug" onclick="switchStream('debug')">Landmarks</button>
  </div>
</div>

<div id="right">

  <!-- Identity -->
  <div class="panel">
    <h2>Identity</h2>
    <div class="counter" id="sample-count">0</div>
    <div class="counter-label">samples recorded</div>
    <div class="active-label">Active: <span id="active-identity">none</span></div>
    <input type="text" id="identity-name" placeholder="Identity name…" oninput="syncName('identity')">
    <div class="btn-row">
      <button class="big-btn" onclick="doAction('record_person', this, 'msg-id')">+ Record Sample</button>
      <button class="big-btn primary" onclick="doNamedAction('save_identity','identity-name',this,'msg-id')">&#10003; Save</button>
    </div>
    <div class="msg" id="msg-id"></div>
    <h2 style="margin-top:4px">Saved Identities</h2>
    <div class="lib" id="lib-identities"><div class="lib-empty">None saved yet.</div></div>
  </div>

  <!-- Expression -->
  <div class="panel">
    <h2>Current Expression</h2>
    <div class="expression-box" id="cur-action"><span class="expression-empty">Waiting for face\u2026</span></div>
  </div>

  <!-- Password -->
  <div class="panel">
    <h2>Password</h2>
    <div class="active-label">Active: <span id="active-password">none</span></div>
    <input type="text" id="password-name" placeholder="Password name\u2026" oninput="syncName('password')">
    <div class="btn-row">
      <button class="big-btn" onclick="doAction('record_action', this, 'msg-pw')">+ Record Step</button>
      <button class="big-btn primary" onclick="doNamedAction('save_password','password-name',this,'msg-pw')">&#10003; Save</button>
    </div>
    <div class="msg" id="msg-pw"></div>
    <div id="steps"><div class="step-empty">No steps recorded yet.</div></div>
    <h2 style="margin-top:4px">Saved Passwords</h2>
    <div class="lib" id="lib-passwords"><div class="lib-empty">None saved yet.</div></div>
  </div>

  <!-- Done -->
  <div class="panel">
    <button class="big-btn done" id="btn-done" onclick="doDone(this)">&#10003;&#160; Done — Complete Setup</button>
    <div class="msg" id="msg-done"></div>
  </div>

</div>

<script>
function switchStream(type) {
  document.getElementById('stream-img').src = '/stream/' + type + '?t=' + Date.now();
  document.getElementById('btn-normal').classList.toggle('active-stream', type === 'normal');
  document.getElementById('btn-debug').classList.toggle('active-stream', type === 'debug');
}

function syncName(kind) {
  // live-sync name input so the server knows it before save
  const val = document.getElementById(kind + '-name').value.trim();
  fetch('/set_name/' + kind, { method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({name: val}) });
}

async function post(url, body) {
  const r = await fetch(url, { method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body || {}) });
  return r.json();
}

async function doAction(action, btn, msgId) {
  btn.disabled = true;
  try {
    const d = await post('/action/' + action);
    flash(btn, d.success);
    document.getElementById(msgId).textContent = d.message;
  } catch(e) {
    flash(btn, false);
    document.getElementById(msgId).textContent = 'Request failed';
  } finally { btn.disabled = false; }
}

async function doNamedAction(action, nameInputId, btn, msgId) {
  const name = document.getElementById(nameInputId).value.trim();
  if (!name) { document.getElementById(msgId).textContent = 'Enter a name first.'; return; }
  btn.disabled = true;
  try {
    const d = await post('/action/' + action, {name});
    flash(btn, d.success);
    document.getElementById(msgId).textContent = d.message;
    if (d.success) refreshLibs();
  } catch(e) {
    flash(btn, false);
    document.getElementById(msgId).textContent = 'Request failed';
  } finally { btn.disabled = false; }
}

async function loadItem(kind, name) {
  const d = await post('/action/load_' + kind, {name});
  document.getElementById('msg-' + (kind === 'identity' ? 'id' : 'pw')).textContent = d.message;
  if (d.success) refreshLibs();
}

async function deleteItem(kind, name) {
  if (!confirm('Delete ' + kind + ' "' + name + '"?')) return;
  const d = await post('/action/delete_' + kind, {name});
  document.getElementById('msg-' + (kind === 'identity' ? 'id' : 'pw')).textContent = d.message;
  refreshLibs();
}

async function deleteStep(index) {
  const d = await post('/action/delete_step', {index});
  document.getElementById('msg-pw').textContent = d.message;
}

async function doDone(btn) {
  btn.disabled = true;
  try {
    const d = await post('/action/complete_setup');
    flash(btn, d.success);
    document.getElementById('msg-done').textContent = d.message;
  } catch(e) {
    flash(btn, false);
    document.getElementById('msg-done').textContent = 'Request failed';
  } finally { btn.disabled = false; }
}

function flash(btn, ok) {
  btn.classList.add(ok ? 'ok' : 'err');
  setTimeout(() => btn.classList.remove('ok', 'err'), 2500);
}

function renderLib(elId, kind, items, activeName) {
  const el = document.getElementById(elId);
  if (!items.length) { el.innerHTML = '<div class="lib-empty">None saved yet.</div>'; return; }
  el.innerHTML = items.map(name => `
    <div class="lib-item${name === activeName ? ' active' : ''}">
      <span>${name}</span>
      <button onclick="loadItem('${kind}','${name}')">Load</button>
      <button class="del" onclick="deleteItem('${kind}','${name}')">&#10005;</button>
    </div>`).join('');
}

async function refreshLibs() {
  try {
    const r = await fetch('/status');
    const d = await r.json();
    renderLib('lib-identities', 'identity', d.identities, d.active_identity);
    renderLib('lib-passwords',  'password', d.passwords,  d.active_password);
  } catch(_) {}
}

async function poll() {
  try {
    const r = await fetch('/status');
    const d = await r.json();
    document.getElementById('sample-count').textContent = d.sample_count;
    document.getElementById('active-identity').textContent = d.active_identity || 'none';
    document.getElementById('active-password').textContent = d.active_password || 'none';

    const box = document.getElementById('cur-action');
    box.innerHTML = d.current_blendshapes.length
      ? d.current_blendshapes.map(b => `<span>${b}</span>`).join('\\n')
      : '<span class="expression-empty">Waiting for face\u2026</span>';

    const stepsEl = document.getElementById('steps');
    stepsEl.innerHTML = d.recorded_actions.length
      ? d.recorded_actions.map((s, i) =>
          `<div class="step"><b>Step ${i+1}</b>
           <button class="del-step" onclick="deleteStep(${i})" title="Remove step">&#10005;</button>
           <br>${s.join(', ') || '(none)'}</div>`
        ).join('')
      : '<div class="step-empty">No steps recorded yet.</div>';

    renderLib('lib-identities', 'identity', d.identities, d.active_identity);
    renderLib('lib-passwords',  'password', d.passwords,  d.active_password);
  } catch(_) {}
  setTimeout(poll, 500);
}

poll();
</script>
</body>
</html>"""


def _list_npy(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        return []
    return sorted(f[:-4] for f in os.listdir(directory) if f.endswith(".npy"))


class PasswordGuiNode(LifecycleNode):
    def __init__(self) -> None:
        super().__init__("password_gui")
        self._lock = threading.Lock()
        self._normal_frame: Optional[bytes] = None
        self._debug_frame: Optional[bytes] = None
        self._current_blendshapes: List[str] = []
        self._recorded_actions: List[List[str]] = []
        self._sample_count: int = 0
        self._server: Optional[ThreadingHTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._active_identity_name: str = ""
        self._active_password_name: str = ""

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.create_subscription(Image, "/camera/image_raw", self._image_cb, 1)
        self.create_subscription(
            Image, "/face_recognition/debug_landmarks", self._debug_image_cb, 1
        )
        self.create_subscription(
            FaceBlendshapes, "/face_recognition/blendshapes", self._blendshapes_cb, 1
        )
        self._svc_record_person = self.create_client(
            Trigger, "/face_recognition/record_person"
        )
        self._svc_save_identity = self.create_client(
            Trigger, "/face_recognition/save_identity"
        )
        self._svc_load_identity = self.create_client(
            Trigger, "/face_recognition/load_identity"
        )
        self._svc_delete_identity = self.create_client(
            Trigger, "/face_recognition/delete_identity"
        )
        self._svc_record_action = self.create_client(
            Trigger, "/face_recognition/record_action"
        )
        self._svc_save_password = self.create_client(
            Trigger, "/face_recognition/save_password"
        )
        self._svc_load_password = self.create_client(
            Trigger, "/face_recognition/load_password"
        )
        self._svc_delete_password = self.create_client(
            Trigger, "/face_recognition/delete_password"
        )
        self._svc_password_state = self.create_client(
            Trigger, "/face_recognition/password_state"
        )
        self._svc_complete_setup = self.create_client(
            Trigger, "/face_lock_manager/complete_setup"
        )
        self._svc_set_identity_name = self.create_client(
            SetName, "/face_recognition/set_identity_name"
        )
        self._svc_set_password_name = self.create_client(
            SetName, "/face_recognition/set_password_name"
        )
        self._svc_remove_password_step = self.create_client(
            SetName, "/face_recognition/remove_password_step"
        )
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self._start_server()
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self._stop_server()
        with self._lock:
            self._normal_frame = None
            self._debug_frame = None
            self._current_blendshapes.clear()
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        with self._lock:
            self._recorded_actions.clear()
            self._sample_count = 0
            self._active_identity_name = ""
            self._active_password_name = ""
        return TransitionCallbackReturn.SUCCESS

    def _serving(self) -> bool:
        return self._server is not None

    def _image_cb(self, msg: Image) -> None:
        if not self._serving():
            return
        frame = _encode_jpeg(msg)
        if frame:
            with self._lock:
                self._normal_frame = frame

    def _debug_image_cb(self, msg: Image) -> None:
        if not self._serving():
            return
        frame = _encode_jpeg(msg)
        if frame:
            with self._lock:
                self._debug_frame = frame

    def _blendshapes_cb(self, msg: FaceBlendshapes) -> None:
        if not self._serving():
            return
        active = sorted(
            name
            for name, score in zip(msg.shape_names, msg.coefficients)
            if score > BLENDSHAPE_THRESHOLD and name not in IGNORED_BLENDSHAPES
        )
        with self._lock:
            self._current_blendshapes = active

    def _call_sync(
        self, client: Client, timeout: float = 5.0
    ) -> Optional[Trigger.Response]:
        if not client.service_is_ready():
            return None
        future = client.call_async(Trigger.Request())
        deadline = time.time() + timeout
        while not future.done() and time.time() < deadline:
            time.sleep(0.05)
        return future.result() if future.done() else None

    def _call_set_name(
        self, client: Client, name: str, timeout: float = 2.0, ret: bool = False
    ):
        if not client.service_is_ready():
            return None if ret else None
        req = SetName.Request()
        req.name = name
        future = client.call_async(req)
        deadline = time.time() + timeout
        while not future.done() and time.time() < deadline:
            time.sleep(0.02)
        return future.result() if ret and future.done() else None

    def _set_face_rec_names(
        self, identity: Optional[str] = None, password: Optional[str] = None
    ) -> None:
        """Push active name(s) to the face_recognition node via SetName services."""
        if identity is not None:
            self._active_identity_name = identity
            self._call_set_name(self._svc_set_identity_name, identity)
        if password is not None:
            self._active_password_name = password
            self._call_set_name(self._svc_set_password_name, password)

    def _get_status(self) -> dict:
        with self._lock:
            return {
                "sample_count": self._sample_count,
                "current_blendshapes": list(self._current_blendshapes),
                "recorded_actions": [list(a) for a in self._recorded_actions],
                "active_identity": self._active_identity_name,
                "active_password": self._active_password_name,
                "identities": _list_npy(IDENTITIES_DIR),
                "passwords": _list_npy(PASSWORDS_DIR),
            }

    # --- action handlers (called from HTTP thread) ---

    def _action_record_person(self, _body: dict) -> dict:
        result = self._call_sync(self._svc_record_person)
        if result and result.success:
            with self._lock:
                self._sample_count += 1
        return _fmt(result)

    def _action_save_identity(self, body: dict) -> dict:
        name = body.get("name", "").strip()
        if not name:
            return {"success": False, "message": "Name is required."}
        self._set_face_rec_names(identity=name)
        result = self._call_sync(self._svc_save_identity)
        if result and result.success:
            with self._lock:
                self._sample_count = 0
        return _fmt(result)

    def _action_load_identity(self, body: dict) -> dict:
        name = body.get("name", "").strip()
        if not name:
            return {"success": False, "message": "Name is required."}
        self._set_face_rec_names(identity=name)
        return _fmt(self._call_sync(self._svc_load_identity))

    def _action_delete_identity(self, body: dict) -> dict:
        name = body.get("name", "").strip()
        if not name:
            return {"success": False, "message": "Name is required."}
        self._set_face_rec_names(identity=name)
        result = _fmt(self._call_sync(self._svc_delete_identity))
        if result.get("success") and self._active_identity_name == name:
            with self._lock:
                self._active_identity_name = ""
        return result

    def _action_record_action(self, _body: dict) -> dict:
        result = self._call_sync(self._svc_record_action)
        if result and result.success:
            state_res = self._call_sync(self._svc_password_state)
            if state_res and state_res.success:
                with self._lock:
                    self._recorded_actions = json.loads(state_res.message)
        return _fmt(result)

    def _action_save_password(self, body: dict) -> dict:
        name = body.get("name", "").strip()
        if not name:
            return {"success": False, "message": "Name is required."}
        self._set_face_rec_names(password=name)
        result = self._call_sync(self._svc_save_password)
        if result and result.success:
            with self._lock:
                self._recorded_actions.clear()
        return _fmt(result)

    def _action_load_password(self, body: dict) -> dict:
        name = body.get("name", "").strip()
        if not name:
            return {"success": False, "message": "Name is required."}
        self._set_face_rec_names(password=name)
        result = self._call_sync(self._svc_load_password)
        if result and result.success:
            state_res = self._call_sync(self._svc_password_state)
            if state_res and state_res.success:
                with self._lock:
                    self._recorded_actions = json.loads(state_res.message)
        return _fmt(result)

    def _action_delete_password(self, body: dict) -> dict:
        name = body.get("name", "").strip()
        if not name:
            return {"success": False, "message": "Name is required."}
        self._set_face_rec_names(password=name)
        result = _fmt(self._call_sync(self._svc_delete_password))
        if result.get("success") and self._active_password_name == name:
            with self._lock:
                self._active_password_name = ""
                self._recorded_actions.clear()
        return result

    def _action_delete_step(self, body: dict) -> dict:
        idx = body.get("index")
        if idx is None:
            return {"success": False, "message": "No index provided."}
        res = self._call_set_name(self._svc_remove_password_step, str(idx), ret=True)
        if res and res.success:
            state_res = self._call_sync(self._svc_password_state)
            if state_res and state_res.success:
                with self._lock:
                    self._recorded_actions = json.loads(state_res.message)
        return {
            "success": res.success if res else False,
            "message": res.message if res else "Service not ready or timed out",
        }

    def _action_complete_setup(self, _body: dict) -> dict:
        return _fmt(self._call_sync(self._svc_complete_setup, timeout=10.0))

    def _start_server(self) -> None:
        node = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                pass

            def _read_body(self) -> dict:
                length = int(self.headers.get("Content-Length", 0))
                if not length:
                    return {}
                try:
                    return json.loads(self.rfile.read(length))
                except Exception:
                    return {}

            def do_GET(self) -> None:
                path = self.path.split("?")[0]
                if path == "/":
                    self._send_html(_HTML.encode())
                elif path.startswith("/stream/"):
                    self._mjpeg(path[8:])
                elif path == "/status":
                    self._send_json(node._get_status())
                else:
                    self.send_error(404)

            def do_POST(self) -> None:
                path = self.path.split("?")[0]
                # name sync endpoint (fire-and-forget from JS oninput)
                if path.startswith("/set_name/"):
                    kind = path[10:]
                    body = self._read_body()
                    name = body.get("name", "").strip()
                    # Respond immediately, push to face_recognition in background
                    self._send_json({"ok": True})

                    def _push(k=kind, n=name):
                        if k == "identity":
                            node._set_face_rec_names(identity=n)
                        elif k == "password":
                            node._set_face_rec_names(password=n)

                    threading.Thread(target=_push, daemon=True).start()
                    return

                actions = {
                    "/action/record_person": node._action_record_person,
                    "/action/save_identity": node._action_save_identity,
                    "/action/load_identity": node._action_load_identity,
                    "/action/delete_identity": node._action_delete_identity,
                    "/action/record_action": node._action_record_action,
                    "/action/save_password": node._action_save_password,
                    "/action/load_password": node._action_load_password,
                    "/action/delete_password": node._action_delete_password,
                    "/action/delete_step": node._action_delete_step,
                    "/action/complete_setup": node._action_complete_setup,
                }
                fn = actions.get(path)
                if fn:
                    self._send_json(fn(self._read_body()))
                else:
                    self.send_error(404)

            def _send_html(self, body: bytes) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_json(self, data: dict) -> None:
                body = json.dumps(data).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _mjpeg(self, stream_type: str) -> None:
                self.send_response(200)
                self.send_header(
                    "Content-Type", "multipart/x-mixed-replace; boundary=frame"
                )
                self.end_headers()
                try:
                    while True:
                        with node._lock:
                            frame = (
                                node._normal_frame
                                if stream_type == "normal"
                                else node._debug_frame
                            )
                        if frame:
                            self.wfile.write(
                                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                                + frame
                                + b"\r\n"
                            )
                        time.sleep(0.033)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self._server = ThreadingHTTPServer(("0.0.0.0", GUI_PORT), _Handler)
        self._server_thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )
        self._server_thread.start()
        self.get_logger().info(f"Password GUI started at http://localhost:{GUI_PORT}")

    def _stop_server(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server = None
        if self._server_thread:
            self._server_thread.join(timeout=2.0)
            self._server_thread = None


def _encode_jpeg(msg: Image) -> Optional[bytes]:
    try:
        raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            (msg.height, msg.width, -1)
        )
        bgr = raw if "bgr" in msg.encoding.lower() else raw[:, :, ::-1].copy()
        _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return bytes(buf)
    except Exception:
        return None


def _fmt(result: Optional[Trigger.Response]) -> dict:
    if result is None:
        return {"success": False, "message": "Service not ready or timed out"}
    return {"success": bool(result.success), "message": result.message}


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = PasswordGuiNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
