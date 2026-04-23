# Building BEET 2.0

Three shells are supported and share one frontend (`beet/gui/static/index.html`):

| Shell | Backend | Frontend transport | Dist format |
|-------|---------|-------------------|-------------|
| `beet gui` | local HTTP server (`beet/gui/server.py`) | `fetch()` | Python package |
| `beet serve` | FastAPI (requires `beet[api]` extras) | `fetch()` | Python package |
| Tauri desktop | Python sidecar subprocess over stdio JSON-RPC | `invoke()` | `.dmg` / `.msi` / `.AppImage` |

The `index.html` auto-detects the host (`window.__TAURI_INTERNALS__`) and routes calls through `beetApi`. The same file works in all three.

---

## Prerequisites

- **Python** 3.11+ (project already requires this)
- **Rust** (stable toolchain) — https://rustup.rs
- **Tauri CLI** — `cargo install tauri-cli --version "^2.0"`
- **Node.js** 18+ — Tauri requires it even for vanilla-HTML frontends
- **PyInstaller** — `pip install pyinstaller` (only needed for bundled builds, not dev)
- Platform-native build toolchains:
  - macOS: Xcode Command Line Tools (`xcode-select --install`)
  - Windows: Microsoft C++ Build Tools + WebView2 runtime (pre-installed on Win11)
  - Linux: `webkit2gtk-4.1`, `libayatana-appindicator3-dev`, `librsvg2-dev`, `build-essential`

Verify:
```bash
rustc --version
cargo tauri --version
node --version
python --version
```

---

## Repo layout (Tauri-relevant)

```
.
├── beet/                    # Python package (unchanged)
│   ├── gui/
│   │   ├── server.py        # HTTP shell
│   │   └── static/index.html # single-file SPA, dual-mode (fetch + invoke)
│   └── sidecar.py           # JSON-RPC loop over stdio (Tauri backend)
├── src-tauri/
│   ├── Cargo.toml
│   ├── tauri.conf.json      # frontendDist: ../beet/gui/static
│   ├── build.rs
│   ├── src/
│   │   ├── main.rs          # Tauri app entry + sidecar lifecycle
│   │   ├── commands.rs      # #[tauri::command] handlers
│   │   └── sidecar.rs       # spawn/manage Python sidecar subprocess
│   ├── icons/               # populated via `cargo tauri icon`
│   └── binaries/            # PyInstaller output lives here for bundling
└── docs/building.md         # this file
```

---

## Development loop (no bundling)

You want the fastest iteration cycle. Run the Python sidecar from source, launch Tauri in dev mode, and skip PyInstaller entirely.

**1. One-time icon setup** (Tauri won't build without them):
```bash
cd src-tauri
cargo tauri icon path/to/any-square-png.png
cd ..
```
This generates `src-tauri/icons/*.{png,icns,ico}` in all required sizes.

**2. Point the Rust side at your local Python sidecar**:

macOS / Linux:
```bash
export BEET_SIDECAR_CMD="python -m beet.sidecar --profile default"
```

Windows (PowerShell):
```powershell
$env:BEET_SIDECAR_CMD = "python -m beet.sidecar --profile default"
```

The Rust sidecar manager checks this env var first; if set, it uses it verbatim instead of looking for a bundled binary. (Whitespace-split — don't put paths with spaces in the command.)

**3. Launch**:
```bash
cd src-tauri
cargo tauri dev
```

First run compiles the Rust dependencies (~2–5 min). Subsequent runs are incremental. The window opens with the BEET frontend; paste text, analyze, submit feedback, all backed by your local Python.

**Debugging tips:**
- Sidecar stderr (Python warnings, logs) is inherited into the Tauri process stderr. Watch the terminal.
- Add `--log-level DEBUG` to `BEET_SIDECAR_CMD` for verbose logs.
- Frontend DevTools: right-click in the window (dev builds only) → Inspect Element.
- Rust logs: `RUST_LOG=beet=debug cargo tauri dev`.

---

## Production build (bundled sidecar + Tauri binary)

Builds are **per-platform** — you build on macOS to get `.dmg`, on Windows to get `.msi`, on Linux to get `.AppImage`. Cross-compilation is painful; use CI runners or separate machines.

**Step 1 — build the Python sidecar binary:**
```bash
pyinstaller \
  --onefile \
  --name beet-sidecar \
  --distpath src-tauri/binaries \
  --paths . \
  beet/sidecar.py
```

On Windows, PyInstaller emits `beet-sidecar.exe`; on macOS / Linux, `beet-sidecar`. Tauri's bundler rewrites the name to `beet-sidecar-<target-triple>` for platform-specific lookup — see [Tauri's sidecar docs](https://tauri.app/v1/guides/building/sidecar) for the exact convention your Tauri version expects. (If the bundler errors on the name, rename the PyInstaller output to match the target triple, e.g. `beet-sidecar-x86_64-apple-darwin`.)

Sanity check the binary works:
```bash
echo '{"id":"1","method":"health"}' | src-tauri/binaries/beet-sidecar
# Expect: ready event, then a result line
```

**Step 2 — unset the dev override** so Rust uses the bundled binary:
```bash
unset BEET_SIDECAR_CMD   # or Remove-Item Env:BEET_SIDECAR_CMD on PowerShell
```

**Step 3 — build the Tauri app:**
```bash
cd src-tauri
cargo tauri build
```

Output:
- macOS: `src-tauri/target/release/bundle/dmg/BEET_2.0.0_aarch64.dmg` (or `x86_64`)
- Windows: `src-tauri/target/release/bundle/msi/BEET_2.0.0_x64_en-US.msi`
- Linux: `src-tauri/target/release/bundle/appimage/beet_2.0.0_amd64.AppImage`

**Step 4 — smoke test the bundle:**
1. Launch the installed app.
2. Paste text, click Analyze. Verdict should render within ~1–2 seconds.
3. Submit Human/LLM feedback. Check that `data/reviewer_feedback.jsonl` gets written — the path is relative to the app's working directory, which Tauri sets per-OS.
4. Close the window. Verify the Python subprocess exits (no orphaned `beet-sidecar` process in Task Manager / `ps`).

---

## Unsigned-build caveats

Shipping without code signing is fine for internal use. End users will see:

- **macOS**: *"BEET can't be opened because it is from an unidentified developer."* Right-click → Open → Open again dismisses it. Gatekeeper may also require `xattr -d com.apple.quarantine /Applications/BEET.app` the first time.
- **Windows**: SmartScreen warning. Click *More info* → *Run anyway*.
- **Linux**: no warning. Users just `chmod +x` the AppImage.

To sign later:
- macOS: Apple Developer ID ($99/yr) + `tauri.conf.json` > `bundle.macOS.signingIdentity`
- Windows: Azure Trusted Signing or an EV code signing certificate + `bundle.windows.certificateThumbprint`

---

## CI-friendly build

A three-OS GitHub Actions matrix looks like:

```yaml
jobs:
  build:
    strategy:
      matrix:
        os: [macos-latest, windows-latest, ubuntu-latest]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - uses: dtolnay/rust-toolchain@stable
      - uses: actions/setup-node@v4
      - name: Install Linux deps
        if: matrix.os == 'ubuntu-latest'
        run: |
          sudo apt-get update
          sudo apt-get install -y libwebkit2gtk-4.1-dev libayatana-appindicator3-dev librsvg2-dev
      - run: pip install -e . pyinstaller
      - name: Build sidecar
        run: pyinstaller --onefile --name beet-sidecar --distpath src-tauri/binaries beet/sidecar.py
      - run: cargo install tauri-cli --version "^2.0"
      - run: cd src-tauri && cargo tauri build
      - uses: actions/upload-artifact@v4
        with:
          name: beet-${{ matrix.os }}
          path: src-tauri/target/release/bundle/**/*
```

---

## Known limitations (v0)

- **No auto-restart** of the sidecar if it crashes. In-flight requests fail with `ERR_SIDECAR_GONE`; the user must relaunch the app. Add a restart loop in `src-tauri/src/sidecar.rs::reader_loop` when desired.
- **Single-window only.** Tauri supports multi-window; adding a second window requires passing the `SidecarManager` to it via Tauri state (already `Arc<…>` — trivial).
- **`data/` path is process-relative.** On packaged builds, feedback JSONL may land somewhere unexpected. Resolve an app-data dir via `app.path().app_data_dir()` and pass it to the sidecar via a flag.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `sidecar binary not found at ... and BEET_SIDECAR_CMD not set` | Dev without env var, or production without PyInstaller output | Set `BEET_SIDECAR_CMD` or build the sidecar binary into `src-tauri/binaries/` |
| White window, no content | `frontendDist` path wrong | Check `tauri.conf.json`: should be `../beet/gui/static` (relative to `src-tauri/`) |
| `invoke is not defined` in browser mode | Someone opened `index.html` directly as `file://` | Always serve via `beet gui` (HTTP) or Tauri (invoke) |
| `ERR_SIDECAR_GONE` on every call | Python subprocess died at startup | Run `BEET_SIDECAR_CMD` manually in a shell; check the error |
| `cargo tauri icon` fails | Source image is not square | Crop to square before running |
