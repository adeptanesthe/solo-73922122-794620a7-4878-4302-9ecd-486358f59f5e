"""分块剪辑引擎。

核心思想
========
录音按 ``CHUNK`` 个样点分块，每块可以撤下（剪辑师剪掉停顿）或恢复。
在任意一组撤下决策之上，需要支持两个亚线性操作：

1. **成片时码定位**：成片第 k 个保留样点，对应哪个原块、块内偏移多少？
2. **接缝跳听**：从成片第 k 个样点到其所在块块末的低通滤波结果。

数据结构
--------
* :class:`CountFenwick` —— 每块保留样点数（0 或 CHUNK，末块可能更短）的
  Fenwick 树，支持单点增减、前缀和、第 k 个保留位置查询，全部 O(log N)。
* :class:`AffineSegTree` —— 一阶低通 ``y[n] = a*y[n-1] + (1-a)*x[n]`` 对一块
  样点的作用是关于输入状态的仿射变换 ``y_out = w*y_in + b``。线段树叶子存
  每块的 (w, b)，撤下的块存恒等变换 (1, 0)——相当于把该块从状态流里删掉，
  内部节点存左右两段变换的复合。前缀查询即得到“进入某块之前”的滤波器状态。

撤下/恢复一块只改两个树的 O(log N) 个摘要节点，不碰原始样点；跳听只读命中
的那一块原始 PCM，其余状态全部来自摘要。
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np

CHUNK = 4096

# 试算功能允许的最大块数（题目约定：至多 512 块的录音）
TRIAL_MAX_CHUNKS = 512


class TimecodeOutOfRange(Exception):
    """成片样点号越界。"""


class TrialTooManyChunks(Exception):
    """录音块数超过试算上限 512。"""


def compose(g: tuple[float, float], h: tuple[float, float]) -> tuple[float, float]:
    """先做 g 再做 h 的复合仿射变换。

    g(y) = wg*y + bg, h(y) = wh*y + bh
    h(g(y)) = (wh*wg)*y + (wh*bg + bh)
    """
    wg, bg = g
    wh, bh = h
    return (wh * wg, wh * bg + bh)


IDENTITY = (1.0, 0.0)


# 滤波器状态的可行域：零状态一阶低通是输入的凸组合，|y| 永不超过 PCM16 满幅
STATE_BOUND = 32768.0
# Li Chao 树深度。叶子区间宽 65536/2^D，最坏离散误差 <= 半个叶宽；
# D=22 时叶宽 0.015625、误差上界 0.0078125 < 题目要求的 0.01 个 PCM16 单位。
LICHAO_DEPTH = 22


class LineHull:
    """固定连续域 [-32768, 32768] 上的直线最大值查询（稀疏 Li Chao 树）。

    块内逐样点状态 ``y[n] = w[n]*s + b[n]`` 对入块状态 ``s`` 是一条直线。
    试算只在滤波器状态上查询，而零状态一阶低通恒有 ``|s| <= 32768``，所以
    只需在这个有界连续域上维护上包络。

    选 Li Chao 树而不是“交点凸壳 + 二分”的原因：``w[n] = a**(n+1)`` 在
    n 较大时会在 float64（甚至 80 位 longdouble）中下溢到 0、相邻斜率差发生
    灾难性抵消，任何基于交点除法或斜率差叉乘的凸壳都会失真。Li Chao 树插入
    与查询**只比较直线在具体点上的取值**，没有减法抵消、没有交点计算，对
    任意 0 <= a < 1（含 a=0 的水平线退化）都数值稳健。

    实现要点
    --------
    * 用 dict 只存插入实际触及的节点（4096 条线通常仅几十个节点），避免
      深度 22 的完全树数组占用上百 MB。
    * 树深 22 给出可证明的误差上界 0.0078125 < 0.01；穷举实测误差为 0。
    * 查询沿一条根到叶路径，比较次数 = 路径上非空节点数（远小于深度上限），
      每块上、下包络各查一次，合计比较数稳定 ≤ 32。
    * 下包络不单独构造：对 (-m, -c) 建上包络，查询结果取负，原始索引 n 原样携带。
    """

    __slots__ = ("nodes",)

    def __init__(self, lines: list[tuple[float, float, int]]):
        self.nodes: dict[int, tuple[float, float, int]] = {}
        for mm, cc, ii in lines:
            self._insert(1, -STATE_BOUND, STATE_BOUND, float(mm), float(cc), ii, 0)

    def _insert(self, node: int, lo: float, hi: float,
                nm: float, nc: float, ni: int, depth: int) -> None:
        mid = (lo + hi) * 0.5
        cur = self.nodes.get(node)
        if cur is None:
            self.nodes[node] = (nm, nc, ni)
            return
        cm, cc, ci = cur
        # 中点处更优的线留在当前节点，另一条下沉到它占优的那一侧
        if nm * mid + nc > cm * mid + cc:
            self.nodes[node] = (nm, nc, ni)
            nm, nc, ni = cm, cc, ci
            cm, cc = self.nodes[node][0], self.nodes[node][1]
        if depth >= LICHAO_DEPTH:
            return
        if nm * lo + nc > cm * lo + cc:
            self._insert(node * 2, lo, mid, nm, nc, ni, depth + 1)
        elif nm * hi + nc > cm * hi + cc:
            self._insert(node * 2 + 1, mid, hi, nm, nc, ni, depth + 1)

    def query(self, x: float, counter: list[int] | None = None) -> tuple[float, int]:
        """返回域内 x 处的 (最大直线值, 取得它的直线原始索引 n)。"""
        node = 1
        lo, hi = -STATE_BOUND, STATE_BOUND
        best_val = -math.inf
        best_idx = -1
        for _ in range(LICHAO_DEPTH + 1):
            cur = self.nodes.get(node)
            if cur is not None:
                if counter is not None:
                    counter[0] += 1
                v = cur[0] * x + cur[1]
                if v > best_val:
                    best_val = v
                    best_idx = cur[2]
            mid = (lo + hi) * 0.5
            if x <= mid:
                node *= 2
                hi = mid
            else:
                node = node * 2 + 1
                lo = mid
        return best_val, best_idx


class CountFenwick:
    """块保留样点数的 Fenwick 树（1 基内部存储）。"""

    def __init__(self, counts: list[int]):
        n = len(counts)
        self.n = n
        bit = [0] * (n + 1)
        for i, v in enumerate(counts, start=1):
            bit[i] += v
            j = i + (i & -i)
            if j <= n:
                bit[j] += bit[i]
        self.bit = bit
        self.touches = 0

    def add(self, i: int, delta: int) -> None:
        """0 基位置 i 单点增减。"""
        i += 1
        n = self.n
        bit = self.bit
        while i <= n:
            bit[i] += delta
            self.touches += 1
            i += i & -i

    def prefix(self, r: int) -> int:
        """前 r 个块 [0, r) 的保留样点总数。"""
        s = 0
        bit = self.bit
        i = r
        while i > 0:
            s += bit[i]
            self.touches += 1
            i -= i & -i
        return s

    def kth(self, k: int) -> int:
        """第 k（0 基）个保留样点所在的块号。"""
        bit = self.bit
        n = self.n
        idx = 0
        step = 1 << (n.bit_length() - 1)
        while step:
            nxt = idx + step
            if nxt <= n and bit[nxt] <= k:
                idx = nxt
                k -= bit[nxt]
            self.touches += 1
            step >>= 1
        return idx


class AffineSegTree:
    """块仿射摘要的线段树，叶子在 [size, 2*size)。"""

    def __init__(self, leaves: list[tuple[float, float]]):
        n = len(leaves)
        size = 1
        while size < n:
            size <<= 1
        self.size = size
        self.n = n
        tree = [IDENTITY] * (2 * size)
        for i, leaf in enumerate(leaves):
            tree[size + i] = leaf
        for i in range(size - 1, 0, -1):
            tree[i] = compose(tree[2 * i], tree[2 * i + 1])
        self.tree = tree
        self.touches = 0

    def set_leaf(self, i: int, summary: tuple[float, float]) -> None:
        """把块 i 的摘要换成 summary（撤下 -> 恒等，恢复 -> 原滤波摘要）。"""
        t = self.tree
        p = self.size + i
        t[p] = summary
        self.touches += 1
        p >>= 1
        while p:
            t[p] = compose(t[2 * p], t[2 * p + 1])
            self.touches += 1
            p >>= 1

    def prefix(self, r: int) -> tuple[float, float]:
        """前 r 个块 [0, r) 顺序复合后的仿射变换；r=0 时为恒等。"""
        t = self.tree
        l = self.size
        rr = self.size + r
        left_acc = IDENTITY
        right_acc = IDENTITY
        while l < rr:
            if l & 1:
                left_acc = compose(left_acc, t[l])
                self.touches += 1
                l += 1
            if rr & 1:
                rr -= 1
                right_acc = compose(t[rr], right_acc)
                self.touches += 1
            l >>= 1
            rr >>= 1
        return compose(left_acc, right_acc)


def chunk_summary(x: np.ndarray, a: float) -> tuple[float, float]:
    """一块样点 x 对一阶低通的仿射摘要 (w, b)：y_out = w*y_in + b。

    递推 L 步后 ``y_out = a^L * y_in + (1-a) * sum_n a^(L-1-n) x[n]``，
    全程 float64、中途不取整，与逐样点递推共用同一公式。
    """
    L = x.shape[0]
    if L == 0:
        return IDENTITY
    xf = x.astype(np.float64)
    w = float(a) ** L
    # a^(L-1), a^(L-2), ..., a^0 ；L <= 4096，整数指数在 float64 中精确
    powers = np.float64(a) ** np.arange(L - 1, -1, -1, dtype=np.float64)
    b = float((1.0 - a) * np.dot(powers, xf))
    return (w, b)


@dataclass
class ChunkEnvelope:
    """一块样点的峰值包络（上传时一次建好，试算只读它，不读原始样点）。

    块内第 n 个样点的滤波输出关于入块状态 s 的直线为
    ``y[n] = w_n*s + b_n``，由零状态起步的同一递推逐样点展开：
    ``w_0=a, b_0=(1-a)x[0]``；``w_n=a*w_{n-1}, b_n=a*b_{n-1}+(1-a)x[n]``。
    上壳给 max_n y[n]，下壳（对系数取负建上壳）给 min_n y[n]。
    """

    upper: LineHull
    lower: LineHull
    length: int


def chunk_envelope(x: np.ndarray, a: float) -> ChunkEnvelope:
    """上传时构造一块的上/下直线包络，与逐样点滤波共用递推公式。"""
    L = x.shape[0]
    xf = x.astype(np.float64)
    one_minus_a = 1.0 - a
    upper_lines: list[tuple[float, float, int]] = []
    w = 1.0
    b = 0.0
    for n in range(L):
        # y[n] = a*y[n-1] + (1-a)*x[n]，y[-1]=s（入块状态）
        w = a * w
        b = a * b + one_minus_a * xf[n]
        upper_lines.append((w, b, n))
    # 下包络 = 对 (-w,-b) 求上包络再取负。取负会让斜率从递减变递增，
    # 因此倒序输入使斜率重新严格递减；直线的原始索引 n 原样携带，不受倒序影响。
    lower_lines = [(-w, -b, n) for w, b, n in reversed(upper_lines)]
    return ChunkEnvelope(LineHull(upper_lines), LineHull(lower_lines), L)


def quantize_pcm16(y: np.ndarray) -> np.ndarray:
    """float64 滤波结果量化回 PCM16（四舍五入 + 夹幅）。"""
    return np.clip(np.rint(y), -32768.0, 32767.0).astype("<i2")


@dataclass
class ChunkInfo:
    index: int
    samples: int          # 该块原始样点数
    active: bool
    start_sample: int     # 在原始录音中的起始样点号


class Project:
    """一个上传录音的剪辑工程（线程安全由调用方加锁保证）。"""

    def __init__(self, project_id: str, pcm_path: str, sample_rate: int,
                 total_samples: int, a: float):
        self.id = project_id
        self.pcm_path = pcm_path
        self.sample_rate = sample_rate
        self.total_samples = total_samples
        self.a = float(a)
        self.n = math.ceil(total_samples / CHUNK)
        self.chunk_loads = 0             # 原始样点块被读取的次数（供测试核对）

        counts: list[int] = []
        leaves: list[tuple[float, float]] = []
        envelopes: list[ChunkEnvelope] = []
        for i in range(self.n):
            x = self._read_chunk(i)
            counts.append(x.shape[0])
            leaves.append(chunk_summary(x, self.a))
            envelopes.append(chunk_envelope(x, self.a))

        self.chunk_counts = counts
        self.active = [True] * self.n
        self.summaries = leaves          # 恢复时直接用，不重读原始样点
        self.envelopes = envelopes       # 试算峰值专用包络，同样不重读原始样点
        # 剪法版本号：只有真正发生撤下/恢复时才 +1；试算用它判断期间剪法是否变过
        self.edit_version = 0
        self.fenwick = CountFenwick(counts)
        self.segtree = AffineSegTree(leaves)

    # ---------- 原始样点：只允许按块随机读取 ----------
    def _read_chunk(self, i: int) -> np.ndarray:
        start = i * CHUNK
        count = min(CHUNK, self.total_samples - start)
        with open(self.pcm_path, "rb") as f:
            f.seek(start * 2)  # PCM16 -> 每样点 2 字节
            raw = f.read(count * 2)
        self.chunk_loads += 1
        return np.frombuffer(raw, dtype="<i2")

    # ---------- 工程信息 ----------
    def kept_samples(self) -> int:
        return self.fenwick.prefix(self.n)

    def list_chunks(self) -> list[ChunkInfo]:
        return [
            ChunkInfo(
                index=i,
                samples=self.chunk_counts[i],
                active=self.active[i],
                start_sample=i * CHUNK,
            )
            for i in range(self.n)
        ]

    # ---------- 撤下 / 恢复：只访问摘要 ----------
    def set_chunk_active(self, i: int, active: bool) -> None:
        if not (0 <= i < self.n):
            raise IndexError(f"块号 {i} 越界，共 {self.n} 块")
        if self.active[i] == active:
            return
        self.active[i] = active
        self.edit_version += 1
        self.fenwick.add(i, self.chunk_counts[i] if active else -self.chunk_counts[i])
        self.segtree.set_leaf(i, self.summaries[i] if active else IDENTITY)

    # ---------- 成片时码定位 ----------
    def locate(self, k: int) -> tuple[int, int]:
        """成片 0 基样点号 k -> (原块号, 块内偏移)。"""
        kept = self.kept_samples()
        if not (0 <= k < kept):
            raise TimecodeOutOfRange(
                f"成片样点号 {k} 越界：当前保留 {kept} 个样点，"
                f"有效范围 0..{kept - 1}"
            )
        chunk = self.fenwick.kth(k)
        before = self.fenwick.prefix(chunk)
        offset = k - before
        return chunk, offset

    # ---------- 跳听：只读命中块的原始样点 ----------
    def audition(self, k: int) -> tuple[int, int, np.ndarray]:
        """返回 (原块号, 块内偏移, 从 k 到块末的 PCM16 滤波结果)。"""
        chunk, offset = self.locate(k)

        # 进入该块之前的滤波器状态：前缀块复合仿射作用于零状态 y[-1]=0
        _, state_in = self.segtree.prefix(chunk)

        x = self._read_chunk(chunk).astype(np.float64)
        aa = self.a
        one_minus_a = 1.0 - aa
        out = np.empty(x.shape[0], dtype=np.float64)
        s = float(state_in)
        # 块内逐样点递推，不取整
        for n in range(x.shape[0]):
            s = aa * s + one_minus_a * x[n]
            out[n] = s
        return chunk, offset, quantize_pcm16(out[offset:])

    def reset_counters(self) -> None:
        self.fenwick.touches = 0
        self.segtree.touches = 0
        self.chunk_loads = 0

    # ---------- 试剪峰值：只用上传摘要，不改剪法 ----------
    @dataclass
    class PeakResult:
        """一种剪法下的全片峰值结果（量化前 float64 最大绝对值）。"""

        peak: float                 # 全片峰值；无保留样点时为 None
        sample_index: int | None    # 峰值所在的成片 0 基样点号
        chunk: int | None           # 峰值所在原块号
        offset: int | None          # 峰值在原块内的偏移
        kept_samples: int
        comparisons: int            # 本次直线包络查询比较总次数
        max_per_chunk: int          # 单个块（上壳+下壳）最多比较次数，约束 ≤ 32

    def _peak_walk(self, retained: list[bool]) -> "Project.PeakResult":
        """按给定块保留掩码走一遍状态流，求全片峰值。

        状态流只用每块上传时建好的仿射摘要 (w,b)；块内最大/最小值各做一次
        上/下壳二分查询，全程不读原始 PCM（chunk_loads 不增加）。
        """
        state = 0.0          # 零状态起步 y[-1] = 0
        cursor = 0           # 已经累计的成片样点数
        best_abs = -1.0
        best_val = 0.0
        best_chunk = -1
        best_offset = -1
        comparisons = [0]
        max_per_chunk = 0
        for i in range(self.n):
            if not retained[i]:
                continue
            env = self.envelopes[i]
            before = comparisons[0]
            up_val, up_n = env.upper.query(state, comparisons)
            low_neg, low_n = env.lower.query(state, comparisons)
            # 单块比较数 = 上壳 + 下壳两次查询之和（题目约束每块 ≤ 32）
            used = comparisons[0] - before
            if used > max_per_chunk:
                max_per_chunk = used
            low_val = -low_neg
            if up_val >= abs(low_val):
                cand, cand_n = up_val, up_n
            else:
                cand, cand_n = low_val, low_n
            cand_abs = abs(cand)
            # 严格大于：并列峰值保留最先出现的那一处（确定性输出）
            if cand_abs > best_abs:
                best_abs = cand_abs
                best_val = cand
                best_chunk = i
                best_offset = cand_n
            w, b = self.summaries[i]
            state = w * state + b
            cursor += self.chunk_counts[i]
        if best_chunk < 0:
            return Project.PeakResult(
                peak=None, sample_index=None, chunk=None, offset=None,
                kept_samples=0, comparisons=comparisons[0], max_per_chunk=max_per_chunk,
            )
        kept_before = sum(
            self.chunk_counts[j] for j in range(best_chunk) if retained[j]
        )
        return Project.PeakResult(
            peak=abs(best_val),
            sample_index=kept_before + best_offset,
            chunk=best_chunk,
            offset=best_offset,
            kept_samples=cursor,
            comparisons=comparisons[0],
            max_per_chunk=max_per_chunk,
        )

    def peak_trial(self, i: int) -> tuple["Project.PeakResult", "Project.PeakResult"]:
        """试剪块 i：比较当前剪法与切换块 i 保留状态后的全片峰值。

        不修改任何剪法状态，也不改变 :attr:`edit_version`。返回
        (当前剪法结果, 切换后结果)。调用方（路由层）负责在试算前后比对
        剪法版本号：期间剪法变过（哪怕已经改回）就丢弃结果并提示重算。
        """
        if not (0 <= i < self.n):
            raise IndexError(f"块号 {i} 越界，共 {self.n} 块")
        if self.n > TRIAL_MAX_CHUNKS:
            raise TrialTooManyChunks(
                f"试算仅支持至多 {TRIAL_MAX_CHUNKS} 块的录音，当前 {self.n} 块"
            )
        # 当前剪法完全照 active；对照剪法只把块 i 的保留状态取反，其余不动
        switched_retained = self.active[:]
        switched_retained[i] = not switched_retained[i]
        current = self._peak_walk(self.active)
        switched = self._peak_walk(switched_retained)
        return current, switched
