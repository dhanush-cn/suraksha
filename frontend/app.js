// RockfallGuard Client-Side Application Engine (Crash-Proof Telemetry & Graph Stream)
const API_BASE = "/api";

let activeMineId = 1;
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
    initCharts();
    loadMinesList();
    setupEventListeners();
    startTelemetryPolling();
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
                    { label: 'Velocity (mm/h)', data: [...velData], borderColor: '#3b82f6', backgroundColor: 'rgba(59, 130, 246, 0.15)', fill: true, tension: 0.3, borderWidth: 2, pointRadius: 3 }
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
                    { label: 'Pore Pressure (kPa)', data: [...poreData], borderColor: '#f59e0b', backgroundColor: 'rgba(245, 158, 11, 0.15)', fill: true, tension: 0.3, borderWidth: 2, pointRadius: 3 }
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
                    { label: 'Acoustic RMS (g)', data: [...seismicData], borderColor: '#ef4444', backgroundColor: 'rgba(239, 68, 68, 0.15)', fill: true, tension: 0.3, borderWidth: 2, pointRadius: 3 }
                ]
            },
            options: chartOptions
        });
    }
}

// Load Mines List
async function loadMinesList() {
    try {
        const res = await fetch(`${API_BASE}/mines`);
        if (!res.ok) return;
        const mines = await res.json();
        const select = document.getElementById("mineSelect");
        if (select) {
            select.innerHTML = "";
            mines.forEach(mine => {
                const opt = document.createElement("option");
                opt.value = mine.id;
                opt.textContent = `${mine.name} (${mine.company})`;
                select.appendChild(opt);
            });

            if (mines.length > 0) {
                activeMineId = mines[0].id;
                select.value = activeMineId;
                updateMineDetails(mines[0]);
            }
        }
    } catch (err) {
        console.error("Error loading mines list:", err);
    }
}

function updateMineDetails(mine) {
    if (!mine) return;
    const companyElem = document.getElementById("infoCompany");
    if (companyElem) companyElem.textContent = `${mine.name} (${mine.company})`;

    const coordsElem = document.getElementById("infoCoords");
    if (coordsElem) coordsElem.textContent = `${mine.latitude}°, ${mine.longitude}°`;

    const depthElem = document.getElementById("infoDepth");
    if (depthElem) depthElem.textContent = `${mine.pit_depth_m}m | ${mine.slope_angle_deg}° Slope`;

    const threshElem = document.getElementById("infoThreshold");
    if (threshElem) threshElem.textContent = `${mine.alert_threshold_pct || 70.0}%`;

    updateMineMap(mine);
}

function updateMineMap(mine) {
    if (typeof L === 'undefined') return;
    const mapDiv = document.getElementById("interactiveMineMap");
    if (!mapDiv) return;

    const lat = mine.latitude || -4.05;
    const lon = mine.longitude || 137.11;

    try {
        if (!leafletMap) {
            leafletMap = L.map('interactiveMineMap', { zoomControl: false }).setView([lat, lon], 9);
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                maxZoom: 18,
                attribution: '© OpenStreetMap'
            }).addTo(leafletMap);
            leafletMarker = L.marker([lat, lon]).addTo(leafletMap)
                .bindPopup(`<b>${mine.name}</b><br>GPS: ${lat}°, ${lon}°`)
                .openPopup();
        } else {
            leafletMap.setView([lat, lon], 9);
            leafletMarker.setLatLng([lat, lon])
                .bindPopup(`<b>${mine.name}</b><br>GPS: ${lat}°, ${lon}°`)
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

            const res = await fetch(`${API_BASE}/mines`);
            if (res.ok) {
                const mines = await res.json();
                const m = mines.find(x => x.id === activeMineId);
                if (m) updateMineDetails(m);
            }

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
        });
    }

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

    // Display selected drone image filename on file input change
    const droneInputElem = document.getElementById("droneImageInput") || document.getElementById("droneFileInput");
    droneInputElem?.addEventListener("change", (e) => {
        const file = e.target.files[0];
        const textElem = document.querySelector("#droneModal .dropzone-text span:last-child");
        if (file && textElem) {
            textElem.innerHTML = `<strong>Selected File:</strong> ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
            textElem.style.color = "#3b82f6";
        }
    });

    // Display selected CSV filename on file input change
    document.getElementById("csvFileInput")?.addEventListener("change", (e) => {
        const file = e.target.files[0];
        const textElem = document.querySelector("#uploadModal .dropzone-text span:last-child");
        if (file && textElem) {
            textElem.innerHTML = `<strong>Selected File:</strong> ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
            textElem.style.color = "#3b82f6";
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
            const data = await res.json();
            if (res.ok) {
                resDiv.innerHTML = `
                    <div style="padding: 0.5rem; border-radius: 6px; background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.3);">
                        <strong style="color: #60a5fa;">CNN Visual Classification:</strong> ${data.visual_status}<br>
                        <strong>Visual Surface Risk:</strong> ${data.visual_risk_percentage}%<br>
                        <strong>Crack Density / Anomaly:</strong> ${data.crack_severity}<br>
                        <strong>Recommendation:</strong> ${data.recommendation}
                    </div>
                `;
            } else {
                resDiv.innerHTML = `<span style="color: #ef4444;">Error: ${data.detail}</span>`;
            }
        } catch (err) {
            if (resDiv) resDiv.innerHTML = `<span style="color: #ef4444;">Network Error: ${err.message}</span>`;
        }
    });
}

// Real-Time Telemetry Polling Loop
function startTelemetryPolling() {
    if (telemetryTimer) clearInterval(telemetryTimer);
    fetchTelemetryAndPredict();
    telemetryTimer = setInterval(fetchTelemetryAndPredict, 1000);
}

async function fetchTelemetryAndPredict() {
    try {
        const res = await fetch(`${API_BASE}/telemetry/${activeMineId}?scenario=${currentScenario}`);
        if (!res.ok) return;
        const data = await res.json();
        updateDashboardUI(data);
    } catch (e) {
        console.error("Telemetry fetch error:", e);
    }
}

// Safe Dashboard UI Renderer
function updateDashboardUI(data) {
    try {
        if (!data || !data.telemetry || !data.prediction) return;

        const pred = data.prediction;
        const weather = data.weather || {};
        const tel = data.telemetry;
        const threshold = data.alert_threshold_pct || 70.0;

        // 1. Update Risk Percentage & Gauge
        const riskPct = pred.risk_percentage || 0.0;
        const riskLevel = pred.risk_level || "Safe";
        
        const riskPctElem = document.getElementById("riskPctVal");
        if (riskPctElem) riskPctElem.textContent = `${riskPct.toFixed(1)}%`;
        
        // Gauge SVG stroke calculation (perimeter 264)
        const strokeDash = 264 - (264 * (riskPct / 100.0));
        const gaugeFill = document.getElementById("gaugeFill");
        if (gaugeFill) {
            gaugeFill.style.strokeDashoffset = strokeDash;
            if (riskLevel === "Critical" || riskPct >= 65.0) {
                gaugeFill.style.stroke = "#ef4444";
            } else if (riskLevel === "Warning" || riskPct >= 35.0) {
                gaugeFill.style.stroke = "#f59e0b";
            } else {
                gaugeFill.style.stroke = "#10b981";
            }
        }

        // Category Badge
        const badge = document.getElementById("riskCategoryBadge");
        if (badge) {
            badge.className = "badge";
            if (riskLevel === "Critical" || riskPct >= 65.0) {
                badge.textContent = "CRITICAL / DANGER";
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
        if (wRainElem) wRainElem.textContent = `${(weather.rainfall_mm || tel.rainfall_mm || 0.0).toFixed(1)} mm/h`;

        const wHumElem = document.getElementById("wHumidityVal");
        if (wHumElem) wHumElem.textContent = `${(weather.humidity_pct || tel.humidity_pct || 55.0).toFixed(1)}%`;

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
                const impactScore = reason.impact_score !== undefined ? reason.impact_score : (reason.impact_value || 0.5);
                const scoreClass = impactScore > 0 ? "positive" : "negative";
                div.innerHTML = `
                    <div class="shap-info">
                        <h4>${reason.readable_name || reason.feature}</h4>
                        <p>${reason.explanation}</p>
                    </div>
                    <div class="shap-score ${scoreClass}">${impactScore > 0 ? '+' : ''}${impactScore.toFixed(2)}</div>
                `;
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

        if (!displacementChart || !porePressureChart || !seismicChart) {
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
            if (riskPct >= threshold) {
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
            }
        }
    } catch (err) {
        console.error("UI update exception:", err);
    }
}
