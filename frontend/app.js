// RockfallGuard Client-Side Application Engine (Crash-Proof Telemetry & Graph Stream)
// -----------------------------------------------------------------------------
// Authentication + tenant-aware UI (Step 9).
// -----------------------------------------------------------------------------
// The frontend is a THIN role-aware layer on top of the JWT contract:
//   * Every fetch carries the Bearer token via the wrapper below.
//   * On login the API returns `{ access_token, user: {role, mine_id, ...} }`;
//     we cache the user block in localStorage so subsequent page loads can
//     tailor the UI without another round-trip.
//   * Non-admin operators get a stripped-down variant: the mine dropdown is
//     hidden, activeMineId is pinned to their assigned mine, admin actions
//     (register-mine) are hidden. The BACKEND enforces the real access on
//     every request; this is UX only.

const API_BASE = "/api";
let AUTH_TOKEN = localStorage.getItem("rockfallguard_token") || null;
let CURRENT_USER = _readStoredUser();  // { id, username, role, mine_id, company_name }
let LOGIN_IN_PROGRESS = false;

function _readStoredUser() {
    try {
        const raw = localStorage.getItem("rockfallguard_user");
        return raw ? JSON.parse(raw) : null;
    } catch (_e) {
        return null;
    }
}

function isAdminUser() {
    return CURRENT_USER && CURRENT_USER.role === "admin";
}

function scopedMineId() {
    // Admins have no fixed mine; operators are pinned to theirs.
    // Falls back to 1 as a safe default so a broken/empty token doesn't
    // stall the telemetry loop with `null`.
    if (isAdminUser()) return null;
    return (CURRENT_USER && CURRENT_USER.mine_id) || 1;
}

// Wrap every fetch so it carries the Bearer token when we have one.
// If a call comes back 401 we drop the stored token and prompt again.
const _originalFetch = window.fetch.bind(window);
window.fetch = async (input, init = {}) => {
    init.headers = init.headers || {};
    if (AUTH_TOKEN && !init.headers["Authorization"]) {
        init.headers["Authorization"] = `Bearer ${AUTH_TOKEN}`;
    }
    const res = await _originalFetch(input, init);
    if (res.status === 401) {
        localStorage.removeItem("rockfallguard_token");
        localStorage.removeItem("rockfallguard_user");
        AUTH_TOKEN = null;
        CURRENT_USER = null;
        showLoginPrompt();
    }
    return res;
};

async function showLoginPrompt() {
    // Re-entry guard: page load fires many fetches at once, so a burst of
    // 401s would otherwise stack one overlay per fetch and the topmost would
    // swallow clicks meant for the one beneath. The flag stays set for the
    // lifetime of the overlay -- a successful sign-in reloads the page, which
    // clears it.
    if (LOGIN_IN_PROGRESS || document.getElementById("loginOverlay")) return;
    LOGIN_IN_PROGRESS = true;

    // An in-page overlay rather than window.prompt(): prompt() blocks the
    // whole page behind a browser-chrome dialog, is unavailable in embedded
    // webviews (where it throws outright), and cannot show a validation
    // error without a second dialog. Access tokens expire after an hour,
    // so this surfaces mid-session too, not only on first load.
    const overlay = document.createElement("div");
    overlay.id = "loginOverlay";
    overlay.style.cssText = [
        "position:fixed", "inset:0", "z-index:9999",
        "display:flex", "align-items:center", "justify-content:center",
        "background:rgba(2,6,23,0.82)", "backdrop-filter:blur(4px)"
    ].join(";");
    overlay.innerHTML = `
        <form id="loginCard" style="width:min(92vw,360px);padding:1.5rem;border-radius:14px;
             background:#0f172a;border:1px solid rgba(148,163,184,0.22);
             box-shadow:0 24px 60px rgba(0,0,0,0.55);font-family:inherit;">
            <div style="font-size:1.05rem;font-weight:700;color:#e2e8f0;">Sign in</div>
            <div style="margin-top:0.2rem;font-size:0.82rem;color:#94a3b8;">RockfallGuard operations console</div>
            <input id="loginUser" autocomplete="username" placeholder="Username" style="
                 width:100%;margin-top:1.1rem;padding:0.6rem 0.7rem;border-radius:8px;
                 border:1px solid rgba(148,163,184,0.28);background:#1e293b;color:#e2e8f0;font-size:0.9rem;">
            <input id="loginPass" type="password" autocomplete="current-password" placeholder="Password" style="
                 width:100%;margin-top:0.55rem;padding:0.6rem 0.7rem;border-radius:8px;
                 border:1px solid rgba(148,163,184,0.28);background:#1e293b;color:#e2e8f0;font-size:0.9rem;">
            <div id="loginError" style="display:none;margin-top:0.6rem;font-size:0.82rem;color:#f87171;"></div>
            <button id="loginSubmit" type="submit" style="
                 width:100%;margin-top:0.9rem;padding:0.6rem;border:0;border-radius:8px;
                 background:#3b82f6;color:#fff;font-weight:600;font-size:0.9rem;cursor:pointer;">Sign in</button>
            <div style="margin-top:0.85rem;font-size:0.76rem;color:#64748b;line-height:1.5;">
                Demo accounts: <code style="color:#94a3b8;">admin / admin123</code> (all mines),
                <code style="color:#94a3b8;">grasberg_user / user123</code> (one tenant).
            </div>
        </form>`;
    document.body.appendChild(overlay);

    const card = overlay.querySelector("#loginCard");
    const userEl = overlay.querySelector("#loginUser");
    const passEl = overlay.querySelector("#loginPass");
    const errEl = overlay.querySelector("#loginError");
    const btn = overlay.querySelector("#loginSubmit");
    userEl.focus();

    const fail = (msg) => {
        errEl.textContent = msg;
        errEl.style.display = "block";
        btn.disabled = false;
        btn.textContent = "Sign in";
    };

    card.addEventListener("submit", async (ev) => {
        ev.preventDefault();
        errEl.style.display = "none";
        const username = userEl.value.trim();
        const password = passEl.value;
        if (!username || !password) return fail("Enter a username and password.");
        btn.disabled = true;
        btn.textContent = "Signing in...";
        try {
            // _originalFetch, not the wrapped one: the wrapper would see the
            // 401 from a bad password and recurse straight back into here.
            const res = await _originalFetch(`${API_BASE}/auth/login`, {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({username, password}),
            });
            if (!res.ok) {
                return fail(res.status === 401
                    ? "That username and password did not match."
                    : `Login failed (${res.status}).`);
            }
            const data = await res.json();
            AUTH_TOKEN = data.access_token;
            CURRENT_USER = data.user || null;
            localStorage.setItem("rockfallguard_token", AUTH_TOKEN);
            if (CURRENT_USER) {
                localStorage.setItem("rockfallguard_user", JSON.stringify(CURRENT_USER));
            }
            overlay.remove();
            location.reload();
        } catch (err) {
            fail(`Network error: ${err.message}`);
        }
    });
}

if (!AUTH_TOKEN) showLoginPrompt();

// Apply the role-driven UI variant. Called from DOMContentLoaded so the
// header + toolbar reflect the identity before any data fetches start.
function applyRoleUI() {
    // Header user badge (present in the new UI, absent in older HTML).
    const nameElem = document.getElementById("userNameLabel");
    const roleElem = document.getElementById("userRoleLabel");
    if (nameElem && CURRENT_USER) nameElem.textContent = CURRENT_USER.username;
    if (roleElem && CURRENT_USER) roleElem.textContent = CURRENT_USER.role;

    // Non-admin: hide fleet-wide affordances so they don't confuse a
    // single-mine operator (the backend refuses them anyway; this is
    // to avoid dead UI).
    if (!isAdminUser()) {
        const mineSelectWrap = document.querySelector(".toolbar-select");
        if (mineSelectWrap) mineSelectWrap.classList.add("hidden");
        const regBtn = document.getElementById("openRegisterModalBtn");
        if (regBtn) regBtn.classList.add("hidden");
        document.querySelectorAll(".admin-only").forEach(el => el.classList.add("hidden"));
    }
}

// Turn any FastAPI error body into one readable line. `detail` is a string
// for HTTPException but a list of {loc, msg} objects for 422 validation
// errors, which used to render as "Error: [object Object]".
function describeError(body, status) {
    const d = body && body.detail;
    if (typeof d === "string") return d;
    if (Array.isArray(d)) {
        return d.map(e => {
            const field = Array.isArray(e.loc) ? e.loc.filter(x => x !== "body").join(".") : "";
            return field ? `${field}: ${e.msg}` : e.msg;
        }).join("; ");
    }
    return `Request failed (${status})`;
}

async function readJsonSafe(res) {
    try { return await res.json(); } catch (_e) { return null; }
}

async function signOut() {
    try {
        // Revokes the token server-side (Redis blocklist). A 503 here means
        // Redis is down and the token was NOT revoked; we still clear the
        // local session, which is the part the user is asking for.
        await _originalFetch(`${API_BASE}/auth/logout`, {
            method: "POST",
            headers: AUTH_TOKEN ? {"Authorization": `Bearer ${AUTH_TOKEN}`} : {},
        });
    } catch (_e) { /* offline: fall through */ }
    localStorage.removeItem("rockfallguard_token");
    localStorage.removeItem("rockfallguard_user");
    location.reload();
}

let activeMineId = scopedMineId() || 1;
let currentScenario = "normal";
let isSoundMuted = false;
let telemetryTimer = null;
let leafletMap = null;
let leafletMarker = null;

// Real-Time Chart Instances
let displacementChart = null;
let porePressureChart = null;
let seismicChart = null;

// Telemetry History Buffers for Live Charts
const MAX_DATA_POINTS = 10;
let chartTimeLabels = [];
let velData = [];
let poreData = [];
let seismicData = [];

// Initialize Web Audio API Alert Beep Synthesizer
const audioCtx = new (window.AudioContext || window.webkitAudioContext)();

function playEmergencySiren() {
    if (isSoundMuted || !audioCtx) return;
    try {
        if (audioCtx.state === 'suspended') {
            audioCtx.resume();
        }
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(880, audioCtx.currentTime); // A5 note
        osc.frequency.exponentialRampToValueAtTime(440, audioCtx.currentTime + 0.4);
        gain.gain.setValueAtTime(0.2, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.4);
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start();
        osc.stop(audioCtx.currentTime + 0.4);
    } catch (e) {
        console.log("Audio play error:", e);
    }
}

// Resume Web Audio Context on user gesture
document.addEventListener("click", () => {
    if (audioCtx && audioCtx.state === 'suspended') {
        audioCtx.resume().catch(() => {});
    }
});

// Startup Initialization
document.addEventListener("DOMContentLoaded", () => {
    // Event wiring first: if a chart library failed to load, initCharts
    // used to throw here and take every button on the page down with it.
    setupEventListeners();
    applyRoleUI();
    try { initCharts(); } catch (e) { console.error("Chart init failed:", e); }
    // No token yet: the sign-in overlay is up and every call would 401.
    // A successful sign-in reloads the page and lands back here.
    if (!AUTH_TOKEN) return;
    loadMinesList();
    loadAlertHistory();
    startTelemetryPolling();
    setInterval(loadAlertHistory, 15000);
});

// Seed Initial Chart Points for Continuous Scrolling Line Display
function seedInitialChartData() {
    const now = new Date();
    chartTimeLabels = [];
    velData = [];
    poreData = [];
    seismicData = [];

    for (let i = 7; i >= 0; i--) {
        const t = new Date(now.getTime() - i * 1000);
        chartTimeLabels.push(t.toTimeString().split(' ')[0]);
        velData.push(parseFloat((0.03 + Math.random() * 0.01).toFixed(4)));
        poreData.push(parseFloat((38.0 + Math.random() * 2.0).toFixed(2)));
        seismicData.push(parseFloat((0.012 + Math.random() * 0.003).toFixed(4)));
    }
}

// Setup Chart.js Graphs
function initCharts() {
    seedInitialChartData();
    if (typeof Chart === "undefined") {
        console.warn("Chart.js not loaded; live charts disabled");
        return;
    }

    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        scales: {
            x: {
                grid: { color: 'rgba(255,255,255,0.05)' },
                ticks: { color: '#9ca3af', font: { size: 9 }, maxRotation: 0, autoSkip: false }
            },
            y: {
                grid: { color: 'rgba(255,255,255,0.08)' },
                ticks: { color: '#9ca3af', font: { size: 9 } }
            }
        },
        plugins: { legend: { display: false } }
    };

    // 1. Extensometer Chart
    const ctxDisp = document.getElementById("displacementChart")?.getContext("2d");
    if (ctxDisp) {
        if (displacementChart) displacementChart.destroy();
        displacementChart = new Chart(ctxDisp, {
            type: 'line',
            data: {
                labels: [...chartTimeLabels],
                datasets: [
                    { label: 'Velocity (mm/h)', data: [...velData], borderColor: '#6f9bb2', backgroundColor: 'rgba(111, 155, 178, 0.15)', fill: true, tension: 0.3, borderWidth: 2, pointRadius: 3 }
                ]
            },
            options: chartOptions
        });
    }

    // 2. Pore Pressure Chart
    const ctxPore = document.getElementById("porePressureChart")?.getContext("2d");
    if (ctxPore) {
        if (porePressureChart) porePressureChart.destroy();
        porePressureChart = new Chart(ctxPore, {
            type: 'line',
            data: {
                labels: [...chartTimeLabels],
                datasets: [
                    { label: 'Pore Pressure (kPa)', data: [...poreData], borderColor: '#c19a4d', backgroundColor: 'rgba(193, 154, 77, 0.15)', fill: true, tension: 0.3, borderWidth: 2, pointRadius: 3 }
                ]
            },
            options: chartOptions
        });
    }

    // 3. Microseismic Filtered Chart
    const ctxSeismic = document.getElementById("seismicChart")?.getContext("2d");
    if (ctxSeismic) {
        if (seismicChart) seismicChart.destroy();
        seismicChart = new Chart(ctxSeismic, {
            type: 'line',
            data: {
                labels: [...chartTimeLabels],
                datasets: [
                    { label: 'Acoustic RMS (g)', data: [...seismicData], borderColor: '#b8625b', backgroundColor: 'rgba(184, 98, 91, 0.15)', fill: true, tension: 0.3, borderWidth: 2, pointRadius: 3 }
                ]
            },
            options: chartOptions
        });
    }
}

// Load Mines List
let MINES = [];

async function loadMinesList(preferId = null) {
    try {
        const res = await fetch(`${API_BASE}/mines`);
        if (!res.ok) return;
        const mines = await res.json();
        MINES = mines;
        const select = document.getElementById("mineSelect");

        // For operators the backend returns exactly one mine (their own);
        // pin activeMineId to it and skip the dropdown population.
        if (!isAdminUser()) {
            if (mines.length > 0) {
                activeMineId = mines[0].id;
                updateMineDetails(mines[0]);
            }
            return;
        }

        if (select) {
            select.innerHTML = "";
            mines.forEach(mine => {
                const opt = document.createElement("option");
                opt.value = mine.id;
                opt.textContent = `${mine.name} (${mine.company})`;
                select.appendChild(opt);
            });

            if (mines.length > 0) {
                // Keep the current mine selected across reloads of the list
                // (register / delete); only fall back to the first one when
                // the current mine is gone.
                const wanted = preferId ?? activeMineId;
                const current = mines.find(m => m.id === wanted) || mines[0];
                activeMineId = current.id;
                select.value = String(activeMineId);
                updateMineDetails(current);
            } else {
                updateMineDetails(null);
            }
        }
    } catch (err) {
        console.error("Error loading mines list:", err);
    }
}

function updateMineDetails(mine) {
    if (!mine) {
        ["infoCompany", "infoCoords", "infoDepth", "infoThreshold"].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.textContent = "—";
        });
        return;
    }
    const companyElem = document.getElementById("infoCompany");
    if (companyElem) companyElem.textContent = `${mine.name} (${mine.company})`;

    const coordsElem = document.getElementById("infoCoords");
    if (coordsElem) coordsElem.textContent = `${mine.latitude}°, ${mine.longitude}°`;

    const depthElem = document.getElementById("infoDepth");
    if (depthElem) depthElem.textContent = `${mine.pit_depth_m}m | ${mine.slope_angle_deg}° Slope`;

    const threshElem = document.getElementById("infoThreshold");
    if (threshElem) threshElem.textContent = `${mine.alert_threshold_pct ?? 70.0}%`;

    updateMineMap(mine);
}

function minePopup(mine, lat, lon) {
    // Built with textContent: mine names come from the register form, so
    // interpolating them into bindPopup's HTML string was an XSS sink.
    const box = document.createElement("div");
    const b = document.createElement("b");
    b.textContent = mine.name;
    box.appendChild(b);
    box.appendChild(document.createElement("br"));
    box.appendChild(document.createTextNode(`GPS: ${lat}°, ${lon}°`));
    return box;
}

function updateMineMap(mine) {
    if (typeof L === 'undefined') return;
    const mapDiv = document.getElementById("interactiveMineMap");
    if (!mapDiv) return;

    const lat = mine.latitude ?? -4.05;
    const lon = mine.longitude ?? 137.11;

    try {
        if (!leafletMap) {
            leafletMap = L.map('interactiveMineMap', { zoomControl: false }).setView([lat, lon], 9);
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                maxZoom: 18,
                attribution: '© OpenStreetMap'
            }).addTo(leafletMap);
            leafletMarker = L.marker([lat, lon]).addTo(leafletMap)
                .bindPopup(minePopup(mine, lat, lon))
                .openPopup();
        } else {
            leafletMap.setView([lat, lon], 9);
            leafletMarker.setLatLng([lat, lon])
                .bindPopup(minePopup(mine, lat, lon))
                .openPopup();
        }
        setTimeout(() => {
            if (leafletMap) leafletMap.invalidateSize();
        }, 300);
    } catch (e) {
        console.error("Map update error:", e);
    }
}

// Setup Event Listeners
function setupEventListeners() {
    // Mine Select Change
    const mineSelect = document.getElementById("mineSelect");
    if (mineSelect) {
        mineSelect.addEventListener("change", async (e) => {
            activeMineId = parseInt(e.target.value);
            currentScenario = "normal";
            document.querySelectorAll(".sim-btn").forEach(b => b.classList.remove("active"));
            const normalBtn = document.querySelector('.sim-btn[data-scenario="normal"]');
            if (normalBtn) normalBtn.classList.add("active");

            const m = MINES.find(x => x.id === activeMineId);
            if (m) updateMineDetails(m);

            seedInitialChartData();
            fetchTelemetryAndPredict();
        });
    }

    // Scenario Toggle Buttons
    document.querySelectorAll(".sim-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".sim-btn").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            currentScenario = btn.getAttribute("data-scenario") || "normal";
            fetchTelemetryAndPredict();
        });
    });

    // Mute/Unmute Audio
    const soundBtn = document.getElementById("soundToggleBtn");
    if (soundBtn) {
        soundBtn.addEventListener("click", () => {
            isSoundMuted = !isSoundMuted;
            const icon = document.getElementById("soundIcon");
            if (icon) icon.textContent = isSoundMuted ? "🔇" : "🔊";
        });
    }

    // Dismiss Alert Banner
    const dismissBtn = document.getElementById("dismissAlertBtn");
    if (dismissBtn) {
        dismissBtn.addEventListener("click", () => {
            const banner = document.getElementById("emergencyAlertBanner");
            if (banner) banner.classList.add("hidden");
            alertDismissed = true;
        });
    }

    document.getElementById("logoutBtn")?.addEventListener("click", signOut);

    // Remove the selected mine (admin only; the API enforces it as well).
    document.getElementById("deleteMineBtn")?.addEventListener("click", async () => {
        const mine = MINES.find(m => m.id === activeMineId);
        if (!mine) return;
        if (!confirm(`Remove "${mine.name}" and stop monitoring it?`)) return;
        const res = await fetch(`${API_BASE}/mines/${mine.id}`, { method: "DELETE" });
        const body = await readJsonSafe(res);
        if (!res.ok) {
            alert(`Could not remove the mine: ${describeError(body, res.status)}`);
            return;
        }
        await loadMinesList();
        seedInitialChartData();
        fetchTelemetryAndPredict();
        loadAlertHistory();
    });

    // Modals
    const regModal = document.getElementById("registerModal");
    const uploadModal = document.getElementById("uploadModal");
    const droneModal = document.getElementById("droneModal");

    document.getElementById("openRegisterModalBtn")?.addEventListener("click", () => regModal?.classList.remove("hidden"));
    document.getElementById("closeRegisterModalBtn")?.addEventListener("click", () => regModal?.classList.add("hidden"));
    document.getElementById("cancelRegisterBtn")?.addEventListener("click", () => regModal?.classList.add("hidden"));

    document.getElementById("openUploadModalBtn")?.addEventListener("click", () => uploadModal?.classList.remove("hidden"));
    document.getElementById("closeUploadModalBtn")?.addEventListener("click", () => uploadModal?.classList.add("hidden"));
    document.getElementById("cancelUploadBtn")?.addEventListener("click", () => uploadModal?.classList.add("hidden"));

    document.getElementById("openDroneModalBtn")?.addEventListener("click", () => droneModal?.classList.remove("hidden"));
    document.getElementById("closeDroneModalBtn")?.addEventListener("click", () => droneModal?.classList.add("hidden"));
    document.getElementById("cancelDroneBtn")?.addEventListener("click", () => droneModal?.classList.add("hidden"));

    // Register-mine form. The modal and its Register button existed but no
    // handler was attached, so submitting did a native GET form submit that
    // reloaded the page and dropped everything that was typed.
    document.getElementById("registerForm")?.addEventListener("submit", async (e) => {
        e.preventDefault();
        const form = e.target;
        const out = document.getElementById("registerResult");
        const submitBtn = document.querySelector('button[form="registerForm"]');
        const fd = new FormData(form);
        const num = (k) => {
            const v = String(fd.get(k) ?? "").trim();
            return v === "" ? undefined : Number(v);
        };
        const str = (k) => {
            const v = String(fd.get(k) ?? "").trim();
            return v === "" ? undefined : v;
        };
        const payload = {
            name: str("name"),
            company: str("company"),
            location_name: str("location_name"),
            latitude: num("latitude"),
            longitude: num("longitude"),
            pit_depth_m: num("pit_depth_m"),
            slope_angle_deg: num("slope_angle_deg"),
            contact_email: str("contact_email"),
            contact_phone: str("contact_phone"),
            alert_threshold_pct: num("alert_threshold_pct"),
        };
        // Leave unset optionals out so the API defaults apply.
        Object.keys(payload).forEach(k => payload[k] === undefined && delete payload[k]);

        if (submitBtn) submitBtn.disabled = true;
        try {
            const res = await fetch(`${API_BASE}/mines`, {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify(payload),
            });
            const body = await readJsonSafe(res);
            if (!res.ok) {
                if (out) {
                    out.textContent = `Error: ${describeError(body, res.status)}`;
                    out.style.color = "var(--crit)";
                    out.classList.remove("hidden");
                }
                return;
            }
            form.reset();
            if (out) out.classList.add("hidden");
            regModal?.classList.add("hidden");
            // Switch the dashboard to the mine that was just registered.
            await loadMinesList(body && body.mine_id);
            seedInitialChartData();
            fetchTelemetryAndPredict();
        } catch (err) {
            if (out) {
                out.textContent = `Network error: ${err.message}`;
                out.style.color = "var(--crit)";
                out.classList.remove("hidden");
            }
        } finally {
            if (submitBtn) submitBtn.disabled = false;
        }
    });

    // Drag-and-drop onto the upload zones. The copy says "click or drop",
    // but dropping a file used to make the browser navigate away to it.
    function wireDropzone(inputId) {
        const input = document.getElementById(inputId);
        const zone = input?.closest(".dropzone");
        if (!input || !zone) return;
        ["dragenter", "dragover"].forEach(ev => zone.addEventListener(ev, (e) => {
            e.preventDefault();
            zone.classList.add("dragover");
        }));
        ["dragleave", "drop"].forEach(ev => zone.addEventListener(ev, (e) => {
            e.preventDefault();
            zone.classList.remove("dragover");
        }));
        zone.addEventListener("drop", (e) => {
            if (!e.dataTransfer || !e.dataTransfer.files.length) return;
            input.files = e.dataTransfer.files;
            input.dispatchEvent(new Event("change"));
        });
    }
    wireDropzone("csvFileInput");
    wireDropzone("droneImageInput");
    // A file dropped anywhere else should not replace the dashboard.
    window.addEventListener("dragover", (e) => e.preventDefault());
    window.addEventListener("drop", (e) => e.preventDefault());

    // Render a "Selected File: <name> (<size> KB)" label without
    // interpolating the (attacker-controlled) file name into an
    // innerHTML string. The previous version was a DOM-XSS sink:
    // uploading a file literally named `<img src=x onerror=alert(1)>`
    // executed the payload as soon as the input changed. Building the
    // nodes and setting textContent keeps the file name inert.
    function renderSelectedFileLabel(textElem, file) {
        textElem.textContent = "";
        const label = document.createElement("strong");
        label.textContent = "Selected File: ";
        textElem.appendChild(label);
        textElem.appendChild(document.createTextNode(
            `${file.name} (${(file.size / 1024).toFixed(1)} KB)`
        ));
        textElem.style.color = "#3b82f6";
    }

    // Display selected drone image filename on file input change
    const droneInputElem = document.getElementById("droneImageInput") || document.getElementById("droneFileInput");
    droneInputElem?.addEventListener("change", (e) => {
        const file = e.target.files[0];
        const textElem = document.querySelector("#droneModal .dropzone-text span:last-child");
        if (file && textElem) {
            renderSelectedFileLabel(textElem, file);
        }
    });

    // Display selected CSV filename on file input change
    document.getElementById("csvFileInput")?.addEventListener("change", (e) => {
        const file = e.target.files[0];
        const textElem = document.querySelector("#uploadModal .dropzone-text span:last-child");
        if (file && textElem) {
            renderSelectedFileLabel(textElem, file);
        }
    });

    // --- Async job helper ------------------------------------------------
    // /api/upload_csv and /api/analyze_drone_image no longer do the work
    // inline: they persist the upload, push an arq job and return 202 with
    // {job_id, poll_url}. Poll that until the worker reports "complete",
    // then hand back the job's result. If a deployment ever answers with the
    // finished payload directly (no job_id), pass it straight through.
    async function awaitJobResult(accepted, onStatus) {
        if (!accepted || !accepted.job_id) {
            // Processed inline (no queue): the body is the finished result.
            if (accepted && accepted.status === "error") {
                throw new Error(accepted.message || "processing failed");
            }
            return accepted;
        }
        const pollUrl = accepted.poll_url || `/api/jobs/${accepted.job_id}`;
        const deadline = Date.now() + 90000;
        let ticks = 0;
        while (Date.now() < deadline) {
            await new Promise((r) => setTimeout(r, 800));
            const res = await fetch(window.location.origin + pollUrl);
            if (!res.ok) throw new Error(`job status check failed (${res.status})`);
            const snap = await res.json();
            if (snap.status === "complete") {
                if (snap.success === false) {
                    throw new Error("the worker raised while running this job");
                }
                const result = snap.result || {};
                if (result.status === "error") {
                    throw new Error(result.message || "the worker returned an error");
                }
                return result;
            }
            if (snap.status === "not_found") {
                throw new Error("job expired before it was collected");
            }
            if (snap.status === "unavailable") {
                throw new Error("queue unreachable - is Redis up and the arq worker running?");
            }
            if (onStatus) onStatus(snap.status, ++ticks);
        }
        throw new Error("timed out after 90s waiting for the worker");
    }

    // CSV Upload Form Submit
    document.getElementById("csvUploadForm")?.addEventListener("submit", async (e) => {
        e.preventDefault();
        const fileInput = document.getElementById("csvFileInput");
        if (!fileInput || !fileInput.files[0]) {
            alert("Please select a CSV file first.");
            return;
        }

        const formData = new FormData();
        formData.append("file", fileInput.files[0]);

        const resDiv = document.getElementById("uploadResult");
        if (resDiv) {
            resDiv.innerHTML = "⏳ Processing & Evaluating Telemetry CSV Data...";
            resDiv.classList.remove("hidden");
        }

        try {
            const url = window.location.origin + "/api/upload_csv";
            const res = await fetch(url, {
                method: "POST",
                body: formData
            });
            const accepted = (await readJsonSafe(res)) || {};
            if (res.ok) {
                const data = await awaitJobResult(accepted, (state, ticks) => {
                    resDiv.innerHTML = `\u23f3 Scoring rows on the worker (${state}, ${ticks}s)...`;
                });
                resDiv.innerHTML = `
                    <div style="padding: 0.75rem; border-radius: 8px; background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.4); line-height: 1.6;">
                        <strong style="color: #34d399; font-size: 0.95rem;">✓ Geotechnical Evaluation Complete!</strong><br>
                        <strong>Dataset:</strong> ${escapeHtml(fileInput.files[0].name)} (${data.total_records} Telemetry Rows Evaluated)<br>
                        <strong>Mean Failure Risk:</strong> <span style="color: #60a5fa; font-weight: 700;">${data.avg_risk_percentage}%</span> | <strong>Peak Risk:</strong> <span style="color: #f87171; font-weight: 700;">${data.max_risk_percentage}%</span><br>
                        <strong>Risk Tiers:</strong> 🟢 Safe: ${data.safe_records} | 🟡 Warning: ${data.warning_records} | 🔴 Critical: ${data.critical_records}<br>
                        <strong>Primary Hazard Factor:</strong> ${escapeHtml(data.top_risk_driver)}${
                            Array.isArray(data.columns_used)
                                ? `<br><strong>Columns used:</strong> ${escapeHtml(data.columns_used.join(", "))}`
                                : ""}
                    </div>
                `;
            } else {
                showError(resDiv, describeError(accepted, res.status));
            }
        } catch (err) {
            showError(resDiv, err.message);
        }
    });

    // Drone Upload Form Submit
    const droneFormElem = document.getElementById("droneImageForm") || document.getElementById("droneUploadForm");
    droneFormElem?.addEventListener("submit", async (e) => {
        e.preventDefault();
        const fileInput = document.getElementById("droneImageInput") || document.getElementById("droneFileInput");
        if (!fileInput || !fileInput.files[0]) {
            alert("Please select an aerial drone image first.");
            return;
        }

        const formData = new FormData();
        formData.append("file", fileInput.files[0]);

        const resDiv = document.getElementById("droneResult");
        if (resDiv) {
            resDiv.innerHTML = "⏳ Running PyTorch CNN Vision Analysis on Pit Wall Imagery...";
            resDiv.classList.remove("hidden");
        }

        try {
            const url = window.location.origin + "/api/analyze_drone_image";
            const res = await fetch(url, {
                method: "POST",
                body: formData
            });
            const accepted = (await readJsonSafe(res)) || {};
            if (res.ok) {
                const data = await awaitJobResult(accepted, (state, ticks) => {
                    resDiv.innerHTML = `\u23f3 Running the CNN on the worker (${state}, ${ticks}s)...`;
                });
                const status = data.anomaly_status ?? "unknown";
                // The note below is not decoration. The model is a terrain
                // classifier trained on open-pit bounding boxes; without this
                // line the panel reads like a crack-severity finding, which
                // is exactly the overclaim MODEL_CARD.md exists to prevent.
                const note = data.cnn_trained === false
                    ? `<div style="margin-top:0.5rem;padding:0.4rem 0.6rem;border-radius:6px;background:rgba(245,158,11,0.12);border:1px solid rgba(245,158,11,0.4);color:#fbbf24;font-size:0.85rem;">
                           No trained weights loaded - ${data.cnn_architecture}. Treat this as plumbing, not a measurement.
                       </div>`
                    : `<div style="margin-top:0.5rem;padding:0.4rem 0.6rem;border-radius:6px;background:rgba(148,163,184,0.1);border:1px solid rgba(148,163,184,0.3);color:#94a3b8;font-size:0.8rem;line-height:1.5;">
                           ${data.caveat ?? ""}
                       </div>`;
                const conf = data.mine_region_confidence_pct;
                resDiv.innerHTML = `
                    <div style="padding: 0.5rem; border-radius: 6px; background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.3);">
                        <strong style="color: #60a5fa;">Classification:</strong> ${status}<br>
                        <strong>Mine-terrain confidence:</strong> ${conf}% <span style="color:#64748b;">(threshold ${data.decision_threshold})</span><br>
                        <strong>Edge gradient intensity:</strong> ${data.edge_gradient_intensity} <span style="color:#64748b;">(texture statistic, not a model output)</span><br>
                        <strong>Next step:</strong> ${data.recommendation}
                        ${note}
                    </div>
                `;
            } else {
                showError(resDiv, describeError(accepted, res.status));
            }
        } catch (err) {
            showError(resDiv, err.message);
        }
    });
}

function showError(el, message) {
    if (!el) return;
    el.textContent = "";
    const span = document.createElement("span");
    span.style.color = "var(--crit)";
    span.textContent = `Error: ${message}`;
    el.appendChild(span);
    el.classList.remove("hidden");
}

// Real-Time Telemetry Polling Loop
// setTimeout chain rather than setInterval: when a poll takes longer than a
// second (cold model cache, slow weather API) setInterval stacked requests
// on top of each other and frames arrived out of order on the charts.
const POLL_INTERVAL_MS = 1000;
let telemetryInFlight = false;
// Dismissing the banner used to last exactly one poll: the next frame
// (a second later) put it straight back. It now stays dismissed until the
// risk drops below the threshold, i.e. until this alert episode is over.
let alertDismissed = false;

function startTelemetryPolling() {
    if (telemetryTimer) clearTimeout(telemetryTimer);
    const tick = async () => {
        if (!document.hidden) await fetchTelemetryAndPredict();
        telemetryTimer = setTimeout(tick, POLL_INTERVAL_MS);
    };
    tick();
}

async function fetchTelemetryAndPredict() {
    if (telemetryInFlight || !AUTH_TOKEN) return;
    telemetryInFlight = true;
    const mineId = activeMineId;
    try {
        const res = await fetch(`${API_BASE}/telemetry/${mineId}?scenario=${encodeURIComponent(currentScenario)}`);
        if (!res.ok) return;
        const data = await res.json();
        // Drop a frame that belongs to the mine we just switched away from.
        if (mineId !== activeMineId) return;
        updateDashboardUI(data);
        if (data.alert_recorded) loadAlertHistory();
    } catch (e) {
        console.error("Telemetry fetch error:", e);
    } finally {
        telemetryInFlight = false;
    }
}

function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, c => (
        {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]
    ));
}

async function loadAlertHistory() {
    const body = document.getElementById("alertHistoryBody");
    if (!body || !AUTH_TOKEN) return;
    try {
        const res = await fetch(`${API_BASE}/alerts?limit=25`);
        if (!res.ok) return;
        const rows = await res.json();
        body.textContent = "";
        if (!rows.length) {
            const tr = document.createElement("tr");
            const td = document.createElement("td");
            td.colSpan = 5;
            td.className = "table-empty";
            td.textContent = "No alerts recorded yet.";
            tr.appendChild(td);
            body.appendChild(tr);
            return;
        }
        rows.forEach(a => {
            const tr = document.createElement("tr");
            // triggered_at is stored as naive UTC; mark it so the browser
            // converts to local time instead of reading it as local.
            const ts = a.triggered_at
                ? new Date(/[zZ]|[+-]\d\d:\d\d$/.test(a.triggered_at) ? a.triggered_at : a.triggered_at + "Z")
                : null;
            const level = String(a.risk_level || "").toLowerCase();
            const cells = [
                [ts ? ts.toLocaleString() : "—", ""],
                [a.mine_name || `Mine ${a.mine_id}`, ""],
                [a.risk_percentage != null ? `${Number(a.risk_percentage).toFixed(1)}%` : "—", "num"],
                [null, ""],
                [a.top_shap_reason || "—", "reason"],
            ];
            cells.forEach(([text, cls], i) => {
                const td = document.createElement("td");
                if (cls) td.className = cls;
                if (i === 3) {
                    const badge = document.createElement("span");
                    badge.className = "badge " + (level === "critical" ? "badge-critical"
                        : level === "warning" ? "badge-warning" : "badge-safe");
                    badge.textContent = (a.risk_level || "—").toUpperCase();
                    td.appendChild(badge);
                } else {
                    td.textContent = text;
                }
                tr.appendChild(td);
            });
            body.appendChild(tr);
        });
    } catch (e) {
        console.error("Alert history fetch error:", e);
    }
}

// Safe Dashboard UI Renderer
function updateDashboardUI(data) {
    try {
        if (!data || !data.telemetry || !data.prediction) return;

        const pred = data.prediction;
        const weather = data.weather || {};
        const tel = data.telemetry;
        const threshold = data.alert_threshold_pct ?? 70.0;

        // 1. Update Risk Percentage & Gauge
        const riskPct = Number(pred.risk_percentage ?? 0);
        const riskLevel = pred.risk_level || "Safe";

        const riskPctElem = document.getElementById("riskPctVal");
        if (riskPctElem) riskPctElem.textContent = `${riskPct.toFixed(1)}%`;

        // Gauge SVG stroke calculation (perimeter 264)
        const strokeDash = 264 - (264 * (riskPct / 100.0));
        const gaugeFill = document.getElementById("gaugeFill");
        if (gaugeFill) {
            gaugeFill.style.strokeDashoffset = strokeDash;
            if (riskLevel === "Critical" || riskPct >= 65.0) {
                gaugeFill.style.stroke = "#b8625b";
            } else if (riskLevel === "Warning" || riskPct >= 35.0) {
                gaugeFill.style.stroke = "#c19a4d";
            } else {
                gaugeFill.style.stroke = "#4f8f74";
            }
        }

        // Category Badge
        const badge = document.getElementById("riskCategoryBadge");
        if (badge) {
            badge.className = "badge";
            if (riskLevel === "Critical" || riskPct >= 65.0) {
                badge.textContent = "CRITICAL";
                badge.classList.add("badge-critical");
            } else if (riskLevel === "Warning" || riskPct >= 35.0) {
                badge.textContent = "WARNING";
                badge.classList.add("badge-warning");
            } else {
                badge.textContent = "SAFE";
                badge.classList.add("badge-safe");
            }
        }

        // Probability Bars
        const probs = pred.probabilities || { safe: 85, warning: 10, critical: 5 };
        const pSafeBar = document.getElementById("probSafeBar");
        const pSafeVal = document.getElementById("probSafeVal");
        const pWarnBar = document.getElementById("probWarnBar");
        const pWarnVal = document.getElementById("probWarnVal");
        const pCritBar = document.getElementById("probCritBar");
        const pCritVal = document.getElementById("probCritVal");

        if (pSafeBar) pSafeBar.style.width = `${probs.safe}%`;
        if (pSafeVal) pSafeVal.textContent = `${probs.safe}%`;
        if (pWarnBar) pWarnBar.style.width = `${probs.warning}%`;
        if (pWarnVal) pWarnVal.textContent = `${probs.warning}%`;
        if (pCritBar) pCritBar.style.width = `${probs.critical}%`;
        if (pCritVal) pCritVal.textContent = `${probs.critical}%`;

        // 2. Weather Cards
        const wRainElem = document.getElementById("wRainVal");
        // Telemetry rainfall, not the raw weather feed: the heavy-rain and
        // failure scenarios add simulated rain on top of the live reading,
        // and the risk model scores the telemetry value.
        if (wRainElem) wRainElem.textContent = `${Number(tel.rainfall_mm ?? weather.rainfall_mm ?? 0).toFixed(1)} mm/h`;

        const wHumElem = document.getElementById("wHumidityVal");
        if (wHumElem) wHumElem.textContent = `${Number(tel.humidity_pct ?? weather.humidity_pct ?? 55).toFixed(1)}%`;

        const wTempElem = document.getElementById("wTempVal");
        if (wTempElem && weather.temperature_c !== undefined) wTempElem.textContent = `${weather.temperature_c.toFixed(1)} °C`;

        const wPressElem = document.getElementById("wPressureVal");
        if (wPressElem && weather.pressure_hpa !== undefined) wPressElem.textContent = `${weather.pressure_hpa.toFixed(0)} hPa`;

        const wSrcElem = document.getElementById("weatherSourceBadge");
        if (wSrcElem) wSrcElem.textContent = weather.source || "Open-Meteo API";

        // 3. SHAP Reasons List
        const shapContainer = document.getElementById("shapReasonsContainer");
        if (shapContainer && pred.shap_explanations && pred.shap_explanations.length > 0) {
            shapContainer.innerHTML = "";
            pred.shap_explanations.forEach(reason => {
                const div = document.createElement("div");
                div.className = "shap-item";
                const impactScore = Number(reason.impact_score ?? reason.impact_value ?? 0);
                const info = document.createElement("div");
                info.className = "shap-info";
                const h4 = document.createElement("h4");
                h4.textContent = reason.readable_name || reason.feature;
                const p = document.createElement("p");
                p.textContent = reason.explanation || "";
                info.append(h4, p);
                const score = document.createElement("div");
                score.className = `shap-score ${impactScore > 0 ? "positive" : "negative"}`;
                score.textContent = `${impactScore > 0 ? "+" : ""}${impactScore.toFixed(2)}`;
                div.append(info, score);
                shapContainer.appendChild(div);
            });
        }

        // 4. Update Charts Safely (Crash-Proof Property Guards)
        const timeLabel = tel.timestamp || new Date().toTimeString().split(' ')[0];
        const velVal = tel.velocity_mm_h !== undefined ? tel.velocity_mm_h : 0.04;
        const poreVal = tel.pore_pressure_kpa !== undefined ? tel.pore_pressure_kpa : 38.0;
        const seismicVal = (tel.raw_seismic_rms_g !== undefined) ? tel.raw_seismic_rms_g : 0.015;

        chartTimeLabels.push(timeLabel);
        velData.push(velVal);
        poreData.push(poreVal);
        seismicData.push(seismicVal);

        if (chartTimeLabels.length > MAX_DATA_POINTS) {
            chartTimeLabels.shift();
            velData.shift();
            poreData.shift();
            seismicData.shift();
        }

        if (typeof Chart !== "undefined" && (!displacementChart || !porePressureChart || !seismicChart)) {
            initCharts();
        }

        if (displacementChart) {
            displacementChart.data.labels = [...chartTimeLabels];
            displacementChart.data.datasets[0].data = [...velData];
            displacementChart.update();
        }
        if (porePressureChart) {
            porePressureChart.data.labels = [...chartTimeLabels];
            porePressureChart.data.datasets[0].data = [...poreData];
            porePressureChart.update();
        }
        if (seismicChart) {
            seismicChart.data.labels = [...chartTimeLabels];
            seismicChart.data.datasets[0].data = [...seismicData];
            seismicChart.update();
        }

        // 5. Emergency Alert Notification Banner (>60-80% Risk)
        const alertBanner = document.getElementById("emergencyAlertBanner");
        if (alertBanner) {
            if (riskPct >= threshold && alertDismissed) {
                // Episode still running, operator has acknowledged it.
            } else if (riskPct >= threshold) {
                alertBanner.classList.remove("hidden");
                const titleElem = document.getElementById("alertTitle");
                if (titleElem) titleElem.textContent = `CRITICAL SLOPE COLLAPSE ALERT (${riskPct.toFixed(1)}%)`;

                const bodyElem = document.getElementById("alertBody");
                if (bodyElem) bodyElem.textContent = `Geological Hazard Risk for ${data.mine ? data.mine.name : 'Active Mine'} exceeded safety threshold (${threshold}%). Evacuate bench sectors!`;

                const topShap = pred.shap_explanations && pred.shap_explanations[0] ? pred.shap_explanations[0].explanation : "High Pore Pressure & Acceleration";
                const shapSumElem = document.getElementById("alertShapSummary");
                if (shapSumElem) shapSumElem.textContent = `Top Trigger: ${topShap}`;

                playEmergencySiren();
            } else {
                alertBanner.classList.add("hidden");
                alertDismissed = false;
            }
        }
    } catch (err) {
        console.error("UI update exception:", err);
    }
}
