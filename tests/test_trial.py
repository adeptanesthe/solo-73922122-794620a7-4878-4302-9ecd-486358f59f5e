"""试剪峰值路由的集成测试：对照正确性、空片、512 上限、过期重算、不改剪法。

运行：PYTHONPATH=. python tests/test_trial.py
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.engine import CHUNK  # noqa: E402
from app.main import app  # noqa: E402
from app.wavio import build_mono_pcm16_wav  # noqa: E402

FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  ok  " if cond else " FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def wav_bytes(pcm16: np.ndarray, sr: int = 16000) -> bytes:
    return build_mono_pcm16_wav(pcm16.astype("<i2").tobytes(), sr)


def reference_peak(pcm: np.ndarray, active: list[bool], n: int, a: float):
    s = 0.0
    cursor = 0
    best_abs = -1.0
    best_k = -1
    for b in range(n):
        if not active[b]:
            continue
        seg = pcm[b * CHUNK:(b + 1) * CHUNK].astype(np.float64)
        for v in seg:
            s = a * s + (1.0 - a) * v
            if abs(s) > best_abs:
                best_abs = abs(s)
                best_k = cursor
            cursor += 1
    return (best_abs, best_k) if best_k >= 0 else None


def main() -> None:
    app.config["CLIP_DATA_DIR"] = tempfile.mkdtemp(prefix="cliplab-trial-web-")
    import app.main as m
    m.DATA_DIR = app.config["CLIP_DATA_DIR"]
    client = app.test_client()
    rng = np.random.default_rng(31)

    print("[1] 上传并试算，两种剪法峰值对照逐样点参考实现")
    n = 6
    total = n * CHUNK + 37
    pcm = rng.integers(-30000, 30000, size=total).astype(np.int16)
    a = 0.85
    r = client.post("/upload",
                    data={"a": str(a), "wav": (io.BytesIO(wav_bytes(pcm)), "d.wav")},
                    content_type="multipart/form-data")
    check(r.status_code == 302, f"上传 302（实得 {r.status_code}）")
    pid = r.headers["Location"].rstrip("/").split("/")[-1]

    # 先撤两块
    for off in (1, 4):
        client.post(f"/api/p/{pid}/chunks/{off}/toggle", json={"active": False})

    target = 2
    r = client.post(f"/api/p/{pid}/chunks/{target}/peak-trial")
    d = r.get_json()
    check(r.status_code == 200, f"试算 200（实得 {r.status_code}）")
    st = client.get(f"/api/p/{pid}/state").get_json()
    active = [c["active"] for c in st["chunks"]]

    ref_cur = reference_peak(pcm, active, st["n_chunks"], a)
    sw_active = active[:]
    sw_active[target] = not sw_active[target]
    ref_sw = reference_peak(pcm, sw_active, st["n_chunks"], a)

    cur, sw = d["results"]["current"], d["results"]["switched"]
    check(not cur["empty"] and abs(cur["peak"] - ref_cur[0]) <= 0.01,
          f"当前剪法峰值 {cur['peak']:.4f} vs 参考 {ref_cur[0]:.4f}")
    check(cur["sample_index"] == ref_cur[1], f"当前剪法标记样点号 {cur['sample_index']} == {ref_cur[1]}")
    check(abs(sw["peak"] - ref_sw[0]) <= 0.01 and sw["sample_index"] == ref_sw[1],
          "切换剪法峰值与标记样点号正确")
    check(cur["max_per_chunk_comparisons"] <= 32 and sw["max_per_chunk_comparisons"] <= 32,
          "单块（上壳+下壳）比较总数 ≤ 32")

    print("[2] 试算不改剪法")
    st2 = client.get(f"/api/p/{pid}/state").get_json()
    check([c["active"] for c in st2["chunks"]] == active, "试算前后剪法完全一致")
    check(st2["kept_samples"] == cur["kept_samples"], "保留样点数未被试算改变")

    print("[3] 空片：撤掉全部块后试算，当前剪法显示空片")
    for i in range(st["n_chunks"]):
        client.post(f"/api/p/{pid}/chunks/{i}/toggle", json={"active": False})
    r = client.post(f"/api/p/{pid}/chunks/0/peak-trial")
    d = r.get_json()
    check(r.status_code == 200, "空片试算 200")
    check(d["results"]["current"]["empty"] and d["results"]["current"]["peak"] is None,
          "当前剪法为空片（peak=null）")
    check(not d["results"]["switched"]["empty"], "对照剪法恢复块 0 后非空")

    print("[4] 越界块号 400")
    r = client.post(f"/api/p/{pid}/chunks/999/peak-trial")
    check(r.status_code == 400, f"越界 -> 400（实得 {r.status_code}）")

    print("[4b] 超过 512 块 -> 400 提示")
    big = rng.integers(-1000, 1000, size=513 * CHUNK).astype(np.int16)
    r = client.post("/upload",
                    data={"a": "0.8", "wav": (io.BytesIO(wav_bytes(big)), "big.wav")},
                    content_type="multipart/form-data")
    pid_big = r.headers["Location"].rstrip("/").split("/")[-1]
    r = client.post(f"/api/p/{pid_big}/chunks/0/peak-trial")
    d = r.get_json()
    check(r.status_code == 400 and "512" in d["error"], f"513 块试算 -> 400 提示上限（实得 {r.status_code}）")
    # 512 块恰好允许
    edge = rng.integers(-1000, 1000, size=512 * CHUNK).astype(np.int16)
    r = client.post("/upload",
                    data={"a": "0.8", "wav": (io.BytesIO(wav_bytes(edge)), "edge.wav")},
                    content_type="multipart/form-data")
    pid_edge = r.headers["Location"].rstrip("/").split("/")[-1]
    r = client.post(f"/api/p/{pid_edge}/chunks/10/peak-trial")
    check(r.status_code == 200, f"512 块试算 200（实得 {r.status_code}）")

    print("[5] 试算期间另一窗口改剪法 -> 409 提示重算（即使改回）")
    # 重新上传一个干净工程以便控制
    pcm2 = rng.integers(-20000, 20000, size=4 * CHUNK).astype(np.int16)
    r = client.post("/upload",
                    data={"a": "0.8", "wav": (io.BytesIO(wav_bytes(pcm2)), "e.wav")},
                    content_type="multipart/form-data")
    pid2 = r.headers["Location"].rstrip("/").split("/")[-1]
    import app.main as m2
    proj = m2.PROJECTS[pid2]
    lock = m2.LOCKS[pid2]
    errors = []

    def worker():
        try:
            with lock:
                v0 = proj.edit_version
            # 锁外计算阶段，主线程会在此时改剪法
            time.sleep(0.3)
            proj.peak_trial(0)
            with lock:
                stale = proj.edit_version != v0
            errors.append(("stale_flag", stale))
        except Exception as exc:  # noqa: BLE001
            errors.append(("exc", repr(exc)))

    t = threading.Thread(target=worker)
    t.start()
    time.sleep(0.05)
    # 撤下再恢复，版本号 +2，剪法外观回到原样
    client.post(f"/api/p/{pid2}/chunks/1/toggle", json={"active": False})
    client.post(f"/api/p/{pid2}/chunks/1/toggle", json={"active": True})
    t.join()
    check(errors and errors[0][1] is True, "试算期间剪法变过（含改回），版本号识别为 stale")

    # 路由层：用 monkeypatch 让试算在锁外计算期间发生另一窗口的撤+恢复（版本号 +2）
    import app.engine as engine
    orig_peak_trial = engine.Project.peak_trial

    def slow_peak_trial(self, idx):
        time.sleep(0.3)
        return orig_peak_trial(self, idx)

    engine.Project.peak_trial = slow_peak_trial
    stale_result = {}

    def concurrent_edit():
        time.sleep(0.05)
        c2 = app.test_client()
        c2.post(f"/api/p/{pid2}/chunks/1/toggle", json={"active": False})
        c2.post(f"/api/p/{pid2}/chunks/1/toggle", json={"active": True})

    th = threading.Thread(target=concurrent_edit)
    th.start()
    r = client.post(f"/api/p/{pid2}/chunks/0/peak-trial")
    th.join()
    engine.Project.peak_trial = orig_peak_trial
    d = r.get_json()
    check(r.status_code == 409 and d.get("stale") is True and "重新试算" in d["error"],
          "试算期间剪法变化（含改回）-> 409 且提示重新试算")

    # 无并发时正常返回 200
    r = client.post(f"/api/p/{pid2}/chunks/0/peak-trial")
    check(r.status_code == 200 and r.get_json()["stale"] is False, "无并发改动时试算 200")

    print()
    if FAILS:
        print(f"{len(FAILS)} 项失败")
        sys.exit(1)
    print("试算路由集成测试全部通过")


if __name__ == "__main__":
    main()
