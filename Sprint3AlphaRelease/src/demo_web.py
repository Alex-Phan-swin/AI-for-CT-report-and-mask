import json
import mimetypes
import shutil
import subprocess
import sys
import time
import threading
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "web_demo"
UPLOAD_ROOT = OUTPUT_ROOT / "uploads"
CHECKPOINT = ROOT / "models" / "unet_brain_mri.pth"
DEFAULT_PORT = 8765


def open_browser_later(url):
    time.sleep(0.4)
    try:
        webbrowser.open(url)
    except Exception:
        pass


APP_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MediScan Sprint 3 Demo</title>
  <style>
    :root {
      --cream: #fff8b8;
      --cream-2: #fff3a3;
      --red: #e50914;
      --red-2: #c60710;
      --red-soft: #ff3b45;
      --ink: #241516;
      --muted: #735e5f;
      --panel: #e50914;
      --white: #fffdf7;
      --line: rgba(255,255,255,.34);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background:
        radial-gradient(circle at 28% 24%, rgba(255,255,255,.48), transparent 28%),
        linear-gradient(135deg, var(--cream), var(--cream-2));
    }

    .shell {
      width: min(1180px, calc(100vw - 34px));
      margin: 0 auto;
      padding: 30px 0 42px;
    }

    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 24px;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .mark {
      width: 44px;
      height: 44px;
      display: grid;
      place-items: center;
      color: #fff;
      background: linear-gradient(145deg, var(--red), #b8050d);
      border-radius: 11px;
      box-shadow: 0 12px 24px rgba(229,9,20,.22);
      font-weight: 900;
      font-size: 22px;
    }

    h1 {
      margin: 0;
      font-size: 25px;
      letter-spacing: 0;
    }

    .subtitle {
      margin-top: 3px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 650;
    }

    .status-pill {
      padding: 9px 13px;
      border: 1px solid rgba(36,21,22,.12);
      border-radius: 999px;
      background: rgba(255,255,255,.5);
      color: #5f4748;
      font-size: 13px;
      font-weight: 750;
    }

    .layout {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 24px;
    }

    .input-workflow {
      width: min(920px, 100%);
    }

    .panel {
      background: var(--panel);
      color: var(--white);
      border-radius: 14px;
      box-shadow: 0 22px 44px rgba(173, 29, 29, .22);
      border: 1px solid rgba(94, 0, 0, .12);
    }

    .card-pad { padding: 24px; }

    h2 {
      margin: 0 0 18px;
      font-size: 18px;
      letter-spacing: 0;
    }

    .scan-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }

    .scan-option {
      min-height: 80px;
      border: 1px solid transparent;
      border-radius: 10px;
      background: rgba(255,255,255,.16);
      color: #fff;
      padding: 14px;
      display: grid;
      grid-template-columns: 34px 1fr;
      align-items: center;
      gap: 11px;
      cursor: pointer;
    }

    .scan-option.active {
      border-color: rgba(255,255,255,.95);
      background: rgba(0,0,0,.16);
      box-shadow: inset 0 0 0 1px rgba(255,255,255,.6);
    }

    .scan-icon {
      width: 34px;
      height: 34px;
      border-radius: 50%;
      display: grid;
      place-items: center;
      background: rgba(123,0,8,.36);
      font-weight: 900;
    }

    .scan-option.active .scan-icon {
      background: #fff;
      color: var(--red);
    }

    .scan-title { font-size: 15px; font-weight: 850; }
    .scan-copy { font-size: 12px; opacity: .82; margin-top: 3px; font-weight: 650; }

    .upload {
      margin-top: 20px;
      min-height: 280px;
      border: 2px dashed rgba(229,9,20,.32);
      border-radius: 14px;
      background: rgba(255,253,247,.74);
      color: var(--ink);
      display: grid;
      place-items: center;
      text-align: center;
      padding: 28px;
      transition: transform .18s ease, background .18s ease, border-color .18s ease;
    }

    .upload.dragover {
      transform: translateY(-2px);
      background: rgba(255,253,247,.92);
      border-color: var(--red);
    }

    .upload-icon {
      width: 74px;
      height: 74px;
      margin: 0 auto 18px;
      border-radius: 50%;
      display: grid;
      place-items: center;
      background: #fff;
      color: var(--red);
      font-size: 32px;
      font-weight: 900;
    }

    .upload h3 {
      margin: 0 0 8px;
      font-size: 19px;
      color: var(--ink);
    }

    .upload p {
      margin: 0;
      color: #503d3e;
      font-size: 14px;
      font-weight: 650;
    }

    .small {
      margin-top: 16px;
      font-size: 12px;
      color: #735e5f;
      opacity: 1;
    }

    .actions {
      display: flex;
      justify-content: center;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 20px;
    }

    button, .file-button {
      appearance: none;
      border: 0;
      border-radius: 8px;
      background: #111827;
      color: #fff;
      padding: 11px 15px;
      font-weight: 850;
      font-size: 13px;
      cursor: pointer;
      box-shadow: 0 10px 22px rgba(0,0,0,.18);
    }

    button.secondary {
      background: #fffdf7;
      color: var(--red-2);
      border: 1px solid rgba(229,9,20,.24);
    }

    button:disabled {
      opacity: .56;
      cursor: wait;
    }

    input[type="file"] { display: none; }

    .report-panel {
      width: 100%;
      min-height: 360px;
      display: grid;
      grid-template-rows: auto 1fr;
    }

    .empty-state {
      min-height: 260px;
      display: grid;
      place-items: center;
      text-align: center;
      color: rgba(255,255,255,.9);
      padding: 32px;
    }

    .empty-state .mark {
      margin: 0 auto 16px;
      background: rgba(123,0,8,.4);
      box-shadow: none;
    }

    .results {
      display: none;
      background: #fffdf7;
      color: var(--ink);
      border-radius: 0 0 14px 14px;
      padding: 18px;
    }

    .result-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 14px;
    }

    .image-tile {
      border: 1px solid #eadfe0;
      border-radius: 10px;
      background: #211516;
      overflow: hidden;
    }

    .image-tile img {
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: contain;
      display: block;
    }

    .image-tile span {
      display: block;
      padding: 8px 10px;
      color: #fffdf7;
      font-size: 12px;
      font-weight: 800;
      background: #241516;
    }

    .metrics {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 8px;
      margin: 12px 0;
    }

    .metric {
      background: #fff2e8;
      border: 1px solid #f0d6cd;
      border-radius: 9px;
      padding: 10px;
      min-height: 66px;
    }

    .metric span {
      display: block;
      color: #7c6664;
      font-size: 11px;
      font-weight: 800;
    }

    .metric strong {
      display: block;
      margin-top: 5px;
      font-size: 18px;
    }

    .report-meta {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
      margin-top: 8px;
    }

    .report-badge {
      display: inline-flex;
      align-items: center;
      min-height: 30px;
      padding: 7px 10px;
      border-radius: 999px;
      border: 1px solid #d8c8c9;
      background: #fff5d6;
      color: #5f4748;
      font-size: 12px;
      font-weight: 850;
    }

    .report-badge.qwen {
      border-color: rgba(10,132,255,.28);
      background: #eef6ff;
      color: #0f4c81;
    }

    .report-badge.fallback {
      border-color: rgba(229,9,20,.24);
      background: #fff1f2;
      color: #9f1239;
    }

    .report-box, .validation-box {
      margin-top: 12px;
      padding: 14px;
      border-radius: 10px;
      border: 1px solid #eadfe0;
      background: #fffaf0;
      white-space: pre-wrap;
      line-height: 1.45;
    }

    .report-box {
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
      color: #2f2223;
    }

    .validation-box {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
    }

    .region-table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
      font-size: 12px;
    }

    .region-table th,
    .region-table td {
      border-top: 1px solid #eadfe0;
      padding: 8px 6px;
      text-align: left;
      vertical-align: top;
    }

    .region-table th {
      color: #7c6664;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0;
    }

    .swatch {
      width: 16px;
      height: 16px;
      display: inline-block;
      border-radius: 50%;
      border: 1px solid rgba(36,21,22,.18);
      vertical-align: middle;
      margin-right: 6px;
    }

    .toast {
      width: fit-content;
      max-width: 100%;
      margin: 16px auto 0;
      min-height: 22px;
      padding: 8px 12px;
      border-radius: 999px;
      border: 1px solid rgba(229,9,20,.18);
      background: rgba(255,255,255,.72);
      color: #4f383a;
      font-size: 14px;
      font-weight: 850;
      line-height: 1.25;
      box-shadow: 0 8px 18px rgba(36,21,22,.08);
    }

    .toast:empty {
      display: none;
    }

    .toast.success {
      border-color: rgba(22, 163, 74, .32);
      background: #edfdf3;
      color: #14532d;
    }

    .toast.error {
      border-color: rgba(229,9,20,.34);
      background: #fff1f2;
      color: #9f1239;
    }

    @media (max-width: 980px) {
      .input-workflow { width: 100%; }
      .result-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .metrics { grid-template-columns: repeat(2, 1fr); }
    }

    @media (max-width: 560px) {
      .shell { width: min(100vw - 22px, 1180px); padding-top: 16px; }
      header { align-items: flex-start; flex-direction: column; }
      .scan-grid, .result-grid { grid-template-columns: 1fr; }
      .card-pad { padding: 18px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div class="brand">
        <div class="mark">~</div>
        <div>
          <h1>MediScan AI</h1>
          <div class="subtitle">Sprint 3 grounded medical imaging demonstration</div>
        </div>
      </div>
      <div class="status-pill" id="statusPill">Ready for MRI upload</div>
    </header>

    <main class="layout">
      <section class="input-workflow">
        <div class="panel card-pad">
          <h2>Choose Type of Scan</h2>
          <div class="scan-grid">
            <div class="scan-option disabled"><div class="scan-icon">~</div><div><div class="scan-title">X-Ray</div><div class="scan-copy">Coming later</div></div></div>
            <div class="scan-option disabled"><div class="scan-icon">~</div><div><div class="scan-title">CT Scan</div><div class="scan-copy">Coming later</div></div></div>
            <div class="scan-option active"><div class="scan-icon">~</div><div><div class="scan-title">MRI</div><div class="scan-copy">Brain tumour segmentation</div></div></div>
            <div class="scan-option disabled"><div class="scan-icon">~</div><div><div class="scan-title">Ultrasound</div><div class="scan-copy">Coming later</div></div></div>
          </div>
        </div>

        <div class="panel upload" id="dropZone">
          <div>
            <div class="upload-icon">↑</div>
            <h3>Upload Medical Image</h3>
            <p id="fileText">Drag and drop an MRI image here, or click to browse</p>
            <div class="small">Supports PNG, JPG, JPEG, TIF, TIFF, BMP</div>
            <div class="actions">
              <label class="file-button" for="fileInput">Select Image</label>
              <button class="secondary" type="button" id="sampleButton">Use demo MRI</button>
            </div>
            <input id="fileInput" type="file" accept=".png,.jpg,.jpeg,.tif,.tiff,.bmp,image/*">
            <div class="toast" id="toast"></div>
          </div>
        </div>
      </section>

      <section class="panel report-panel">
        <div class="card-pad">
          <h2>Grounded Report Output</h2>
          <div class="subtitle" style="color:rgba(255,255,255,.82)">Image -> segmentation -> evidence regions -> report validation</div>
        </div>
        <div class="empty-state" id="emptyState">
          <div>
            <div class="mark">~</div>
            <h3>No Report Yet</h3>
            <p>Upload a medical image to generate a grounded analysis report.</p>
          </div>
        </div>
        <div class="results" id="results">
          <div class="result-grid">
            <div class="image-tile"><img id="originalImg" alt="Input scan"><span>Input scan</span></div>
            <div class="image-tile"><img id="maskImg" alt="Predicted mask"><span>Predicted mask</span></div>
            <div class="image-tile"><img id="overlayImg" alt="Evidence overlay"><span>Evidence overlay</span></div>
            <div class="image-tile"><img id="heatmapImg" alt="Probability heatmap"><span>Probability heatmap</span></div>
          </div>
          <div class="metrics">
            <div class="metric"><span>Finding</span><strong id="findingMetric">-</strong></div>
            <div class="metric"><span>Mask area</span><strong id="areaMetric">-</strong></div>
            <div class="metric"><span>Mean prob.</span><strong id="probMetric">-</strong></div>
            <div class="metric"><span>Regions</span><strong id="regionMetric">-</strong></div>
          </div>
          <h2 style="color:#241516;margin-top:14px">Generated Report</h2>
          <div class="report-meta">
            <span class="report-badge" id="writerBadge">Report writer: checking...</span>
            <span class="report-badge" id="modelBadge">Model: -</span>
          </div>
          <div class="report-box" id="reportBox"></div>
          <h2 style="color:#241516;margin-top:14px">Colour-Coded Evidence</h2>
          <table class="region-table">
            <thead>
              <tr><th>Region</th><th>Colour</th><th>Area</th><th>Mean prob.</th><th>Location</th></tr>
            </thead>
            <tbody id="regionRows"></tbody>
          </table>
          <h2 style="color:#241516;margin-top:14px">Validation</h2>
          <div class="validation-box" id="validationBox"></div>
        </div>
      </section>
    </main>
  </div>

  <script>
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const fileText = document.getElementById('fileText');
    const toast = document.getElementById('toast');
    const statusPill = document.getElementById('statusPill');
    const sampleButton = document.getElementById('sampleButton');

    function setBusy(isBusy, message) {
      statusPill.textContent = message;
      toast.className = 'toast';
      toast.textContent = message;
      sampleButton.disabled = isBusy;
    }

    function cacheBust(url) {
      return url + '?t=' + Date.now();
    }

    function showResults(data) {
      document.getElementById('emptyState').style.display = 'none';
      document.getElementById('results').style.display = 'block';
      document.getElementById('originalImg').src = cacheBust(data.images.original);
      document.getElementById('maskImg').src = cacheBust(data.images.mask);
      document.getElementById('overlayImg').src = cacheBust(data.images.overlay);
      document.getElementById('heatmapImg').src = cacheBust(data.images.heatmap);
      document.getElementById('findingMetric').textContent = data.metrics.finding ? 'Yes' : 'No';
      document.getElementById('areaMetric').textContent = data.metrics.area_percent.toFixed(2) + '%';
      document.getElementById('probMetric').textContent = data.metrics.mean_probability.toFixed(2);
      document.getElementById('regionMetric').textContent = data.metrics.region_count;
      const writerBadge = document.getElementById('writerBadge');
      const modelBadge = document.getElementById('modelBadge');
      const reportMeta = data.report_meta || {};
      if (reportMeta.qwen_enabled && !reportMeta.qwen_error) {
        writerBadge.className = 'report-badge qwen';
        writerBadge.textContent = 'Report writer: Qwen';
      } else if (reportMeta.qwen_enabled && reportMeta.qwen_error) {
        writerBadge.className = 'report-badge fallback';
        writerBadge.textContent = 'Report writer: deterministic fallback';
      } else {
        writerBadge.className = 'report-badge';
        writerBadge.textContent = 'Report writer: deterministic generator';
      }
      modelBadge.textContent = reportMeta.qwen_model ? `Model: ${reportMeta.qwen_model}` : 'Model: local rules';
      document.getElementById('reportBox').textContent = data.report;
      document.getElementById('validationBox').textContent = data.validation;
      const rows = document.getElementById('regionRows');
      if (!data.regions.length) {
        rows.innerHTML = '<tr><td colspan="5">No evidence region exceeded the configured thresholds.</td></tr>';
      } else {
        rows.innerHTML = data.regions.map(region => `
          <tr>
            <td>${region.id}</td>
            <td><span class="swatch" style="background:${region.color.hex}"></span>${region.color.name}</td>
            <td>${region.area_percent.toFixed(2)}%</td>
            <td>${region.mean_probability.toFixed(2)}</td>
            <td>${region.region_label}</td>
          </tr>
        `).join('');
      }
      statusPill.textContent = 'Analysis complete';
      toast.className = 'toast success';
      toast.textContent = 'Report generated and validated.';
    }

    async function analyseFile(file) {
      if (!file) return;
      fileText.textContent = file.name;
      const form = new FormData();
      form.append('image', file);
      setBusy(true, 'Analysing image...');
      try {
        const response = await fetch('/api/analyse', { method: 'POST', body: form });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Analysis failed.');
        showResults(data);
      } catch (error) {
        statusPill.textContent = 'Analysis failed';
        toast.className = 'toast error';
        toast.textContent = error.message;
      } finally {
        sampleButton.disabled = false;
      }
    }

    async function analyseSample() {
      setBusy(true, 'Analysing demo MRI...');
      try {
        const response = await fetch('/api/sample', { method: 'POST' });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Analysis failed.');
        fileText.textContent = 'Demo MRI sample';
        showResults(data);
      } catch (error) {
        statusPill.textContent = 'Analysis failed';
        toast.className = 'toast error';
        toast.textContent = error.message;
      } finally {
        sampleButton.disabled = false;
      }
    }

    dropZone.addEventListener('click', (event) => {
      if (event.target.tagName !== 'BUTTON' && event.target.tagName !== 'LABEL') fileInput.click();
    });
    fileInput.addEventListener('change', () => analyseFile(fileInput.files[0]));
    sampleButton.addEventListener('click', analyseSample);
    ['dragenter', 'dragover'].forEach(name => dropZone.addEventListener(name, event => {
      event.preventDefault();
      dropZone.classList.add('dragover');
    }));
    ['dragleave', 'drop'].forEach(name => dropZone.addEventListener(name, event => {
      event.preventDefault();
      dropZone.classList.remove('dragover');
    }));
    dropZone.addEventListener('drop', event => analyseFile(event.dataTransfer.files[0]));
  </script>
</body>
</html>
"""


def parse_multipart(body, content_type):
    marker = "boundary="
    if marker not in content_type:
        raise ValueError("Missing multipart boundary")
    boundary = content_type.split(marker, 1)[1].strip().strip('"').encode()
    delimiter = b"--" + boundary
    for part in body.split(delimiter):
        if b"\r\n\r\n" not in part:
            continue
        headers, payload = part.split(b"\r\n\r\n", 1)
        if b'name="image"' not in headers:
            continue
        filename = "upload.png"
        for header_part in headers.decode(errors="ignore").split(";"):
            header_part = header_part.strip()
            if header_part.startswith("filename="):
                filename = header_part.split("=", 1)[1].strip().strip('"') or filename
        payload = payload.rstrip(b"\r\n-")
        return filename, payload
    raise ValueError("No image file was included")


def safe_upload_name(filename):
    suffix = Path(filename).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        suffix = ".png"
    return f"{int(time.time())}_{uuid.uuid4().hex[:8]}{suffix}"


def run_pipeline(image_path):
    run_id = f"run_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    output_dir = OUTPUT_ROOT / run_id
    command = [
        sys.executable,
        str(ROOT / "src" / "predict_report.py"),
        "--checkpoint",
        str(CHECKPOINT),
        "--image",
        str(image_path),
        "--output-dir",
        str(output_dir),
        "--area-threshold",
        "0.2",
        "--mean-probability-threshold",
        "0.55",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout or "Pipeline failed")

    evidence = json.loads((output_dir / "evidence.json").read_text(encoding="utf-8"))
    findings = evidence["mask_derived_findings"]
    report = (output_dir / "report.txt").read_text(encoding="utf-8")
    validation = (output_dir / "report_validation.txt").read_text(encoding="utf-8")
    validation_json = json.loads((output_dir / "report_validation.json").read_text(encoding="utf-8"))
    qwen_meta = validation_json.get("qwen", {})
    relative = f"/outputs/web_demo/{run_id}"
    return {
        "run_id": run_id,
        "images": {
            "original": f"{relative}/original.png",
            "mask": f"{relative}/predicted_mask.png",
            "overlay": f"{relative}/overlay.png",
            "heatmap": f"{relative}/probability_heatmap.png",
        },
        "metrics": {
            "finding": findings["finding_present"],
            "area_percent": findings["mask_area_percent"],
            "mean_probability": findings["mean_mask_probability"],
            "region_count": len(evidence.get("evidence_regions", [])),
        },
        "regions": evidence.get("evidence_regions", []),
        "colour_key": evidence["visual_evidence"].get("colour_key", {}),
        "report_meta": {
            "qwen_enabled": bool(qwen_meta.get("enabled")),
            "qwen_model": qwen_meta.get("model"),
            "qwen_error": qwen_meta.get("error"),
        },
        "report": report,
        "validation": validation,
    }


def json_response(handler, status, payload):
    data = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


class DemoHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"{self.address_string()} - {format % args}")

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            data = APP_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if parsed.path.startswith("/outputs/"):
            requested = (ROOT / unquote(parsed.path.lstrip("/"))).resolve()
            try:
                requested.relative_to(ROOT.resolve())
            except ValueError:
                self.send_error(403)
                return
            if not requested.is_file():
                self.send_error(404)
                return
            content_type = mimetypes.guess_type(requested.name)[0] or "application/octet-stream"
            data = requested.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        self.send_error(404)

    def do_POST(self):
        try:
            if self.path == "/api/sample":
                sample = ROOT / "demo_input/TCGA_CS_4944_20010208_1.tif"
                json_response(self, 200, run_pipeline(sample))
                return

            if self.path == "/api/analyse":
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                filename, payload = parse_multipart(body, self.headers.get("Content-Type", ""))
                UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
                upload_path = UPLOAD_ROOT / safe_upload_name(filename)
                upload_path.write_bytes(payload)
                json_response(self, 200, run_pipeline(upload_path))
                return

            self.send_error(404)
        except Exception as error:
            json_response(self, 500, {"error": str(error)})


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not CHECKPOINT.exists():
        raise RuntimeError(f"Missing model checkpoint: {CHECKPOINT}")
    server = ThreadingHTTPServer(("127.0.0.1", port), DemoHandler)
    url = f"http://127.0.0.1:{port}"
    print(f"MediScan Sprint 3 demo running at {url}")
    print("Press Ctrl+C to stop.")
    threading.Thread(target=open_browser_later, args=(url,), daemon=True).start()
    server.serve_forever()


if __name__ == "__main__":
    main()
