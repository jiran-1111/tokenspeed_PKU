#!/usr/bin/env python3
# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Render the current evidence into ten Chinese defense pages and source charts."""

import argparse
import hashlib
import json
import math
import os
import re
import statistics
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--repo", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--font", type=Path, required=True)
parser.add_argument("--bold-font", type=Path, required=True)
a = parser.parse_args()
repo = a.repo.resolve()
data_path = repo / "docs/evidence/project_agent_20261031/final_verification.json"
v = json.loads(data_path.read_text(encoding="utf-8"))
models = ("0.8b", "4b", "9b", "27b")
assert v["all_quality_passed"] and v["all_performance_passed"]
assert set(v["quality"]) == set(v["performance"]) == set(models)
assert all(
    v["quality"][m]["output_pairs_matched"]
    == v["quality"][m]["context_pairs_validated"]
    == 32
    for m in models
)
assert all(
    v["performance"][m]["formal_requests"] == 192
    and v["performance"][m]["warmup_requests"] == 64
    for m in models
)
assert a.font.is_file() and a.bold_font.is_file()
out = a.output_dir.resolve()
assert out.is_relative_to(repo)
out.mkdir(parents=True, exist_ok=False)
figures = out / "figures"
figures.mkdir()
os.environ["MPLCONFIGDIR"] = str(repo / ".cache/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties

fp = FontProperties(fname=str(a.font), size=14)
plt.rcParams["axes.unicode_minus"] = False


def formal(model, mode):
    rows = [
        x
        for x in v["performance"][model]["arms"]
        if x["phase"] == "formal" and x["mode"] == mode
    ]
    assert len(rows) == 3
    return rows


def paired_chart(filename, field, label, *, logarithmic):
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=160)
    x = np.arange(4)
    for offset, mode, title, color in (
        (-0.18, "session", "Session复用", "#087F8C"),
        (0.18, "stateless", "完整历史重算", "#DE8F05"),
    ):
        values = [[row[field] for row in formal(m, mode)] for m in models]
        mid = [statistics.median(z) for z in values]
        err = [
            [mid[i] - min(z) for i, z in enumerate(values)],
            [max(z) - mid[i] for i, z in enumerate(values)],
        ]
        ax.bar(x + offset, mid, 0.32, yerr=err, capsize=5, label=title, color=color)
    if logarithmic:
        ax.set_yscale("log")
    ax.set_xticks(x, [m.upper() for m in models], fontproperties=fp)
    ax.set_ylabel(label, fontproperties=fp)
    ax.legend(prop=fp, frameon=False)
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(figures / filename)
    plt.close(fig)


paired_chart(
    "ttft.png", "ttft_ms_p50", "客户端首字延迟（ms，对数轴）", logarithmic=True
)
paired_chart(
    "e2e_p95.png", "e2e_ms_p95", "客户端请求完成延迟P95（ms）", logarithmic=False
)
fig, ax = plt.subplots(figsize=(10, 5.5), dpi=160)
ratios = [
    [
        p["throughput_ratio_session_over_stateless"]
        for p in v["performance"][m]["paired_repetitions"]
    ]
    for m in models
]
mid = [statistics.median(z) for z in ratios]
ax.bar(
    np.arange(4),
    mid,
    0.5,
    color="#087F8C",
    yerr=[
        [mid[i] - min(z) for i, z in enumerate(ratios)],
        [max(z) - mid[i] for i, z in enumerate(ratios)],
    ],
    capsize=6,
)
ax.axhline(1, linestyle="--", color="#666666", linewidth=1)
ax.set_xticks(np.arange(4), [m.upper() for m in models], fontproperties=fp)
ax.set_ylabel("配对输出吞吐比（Session / 重算）", fontproperties=fp)
ax.set_ylim(bottom=0)
ax.grid(axis="y", alpha=0.18)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(figures / "throughput_ratio.png")
plt.close(fig)
fig, ax = plt.subplots(figsize=(10, 5.5), dpi=160)
labels, before, peak = [], [], []
for m in models:
    perf = v["performance"][m]
    for chip, value in sorted(perf["hbm_observed_peak_all_phases_mb_per_chip"].items()):
        labels.append(f"{m.upper()} / chip{chip}")
        before.append(perf["resource_observations"]["before_engine_mb"][chip])
        peak.append(value)
x = np.arange(len(labels))
ax.bar(x - 0.18, before, 0.32, label="启动前共享总用量", color="#A8AFB5")
ax.bar(x + 0.18, peak, 0.32, label="性能采样区间共享总峰", color="#087F8C")
ax.set_xticks(x, labels, fontproperties=fp, rotation=10)
ax.set_ylabel("目标chip已用HBM（npu-smi MB）", fontproperties=fp)
ax.legend(prop=fp, frameon=False)
ax.grid(axis="y", alpha=0.18)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(figures / "observed_hbm.png")
plt.close(fig)

BG, PANEL, WHITE, MUTED, TEAL, ORANGE = (
    "#0B132B",
    "#17243B",
    "#F4F7FB",
    "#B3C1D1",
    "#2EC4B6",
    "#FF9F1C",
)
slides, images = [], []
font = lambda size, bold=False: ImageFont.truetype(
    str(a.bold_font if bold else a.font), size
)


def wrap(draw, text, width, size, bold=False):
    lines, current = [], ""
    tokens = re.findall(
        r"[A-Za-z0-9][A-Za-z0-9_./%+=:–-]*[，。；：、！？）]*|\n|.[，。；：、！？）]*",
        text,
    )
    for token in tokens:
        if token == "\n":
            lines.append(current.rstrip())
            current = ""
            continue
        if draw.textlength(token, font=font(size, bold)) > width:
            pieces = list(token)
        else:
            pieces = [token]
        for piece in pieces:
            if (
                current
                and draw.textlength(current + piece, font=font(size, bold)) > width
            ):
                lines.append(current.rstrip())
                current = piece.lstrip()
            else:
                current += piece
    if current:
        lines.append(current.rstrip())
    return lines


def text(draw, value, x, y, width, size=28, color=WHITE, bold=False, max_y=628):
    for line in wrap(draw, value, width, size, bold):
        assert y + size + 8 <= max_y, f"Text exceeds page: {value}"
        draw.text((x, y), line, font=font(size, bold), fill=color)
        y += size + 10
    return y


def page(title, subtitle, notes):
    image = Image.new("RGB", (1280, 720), BG)
    draw = ImageDraw.Draw(image)
    draw.rectangle((60, 58, 69, 118), fill=TEAL)
    text(draw, title, 92, 58, 1110, 43, bold=True, max_y=180)
    text(draw, subtitle, 94, 126, 1090, 21, MUTED, max_y=190)
    draw.line((64, 654, 1216, 654), fill=PANEL, width=2)
    draw.text(
        (64, 677),
        "智能计算系统课程项目  |  张晟祺 · 李书砚 · 计然",
        font=font(16),
        fill=MUTED,
    )
    draw.text((1150, 677), f"{len(slides) + 1:02d} / 10", font=font(16), fill=MUTED)
    slides.append({"title": title, "subtitle": subtitle, "notes": notes})
    images.append(image)
    return draw


def cards(draw, entries):
    for i, (title, body) in enumerate(entries):
        x, y = 64 + (i % 2) * 592, 205 + (i // 2) * 207
        draw.rounded_rectangle((x, y, x + 560, y + 184), radius=12, fill=PANEL)
        text(draw, title, x + 23, y + 19, 514, 27, TEAL, True, max_y=y + 65)
        text(draw, body, x + 23, y + 66, 514, 25, max_y=y + 180)


def chart_page(title, subtitle, filename, note, footer):
    draw = page(title, subtitle, note)
    picture = Image.open(figures / filename).convert("RGB")
    picture.thumbnail((1120, 418))
    images[-1].paste(picture, (640 - picture.width // 2, 196))
    text(draw, footer, 72, 618, 1130, 18, MUTED, max_y=650)


d = page(
    "固定轨迹下验证原生Session复用",
    "四模型统一路径、共享条件同源实验、完整边界披露",
    "本轮实现与材料由AI工具辅助完成。原生128配对和性能768正式请求分别统计，不混用历史时长。",
)
for x, value, label in (
    (80, "4", "模型，同一套系统"),
    (480, "128", "原生输出与重算配对"),
    (880, "768", "正式性能请求"),
):
    d.text((x, 263), value, font=font(88, True), fill=TEAL)
    text(d, label, x, 384, 340, 26)
text(
    d,
    "源码固定1d6099da；1535项源清单。统一评测及主机预算尚未取得。",
    80,
    525,
    1120,
    27,
    MUTED,
)

d = page(
    "四模型部署，明确并行和精度",
    "每次只运行一个本组NPU实验；至多使用同一物理卡的两个chip",
    "最终配置保持原已验证策略，不扩展全部TP候选；配置4096不等于本轮覆盖4096边界。",
)
cards(
    d,
    [
        ("0.8B与4B", "BF16主体，TP1。分别部署于同一张卡的一个chip。"),
        ("9B", "BF16主体，显式FP32输出头，TP1。保留精度与资源取舍。"),
        ("27B", "BF16主体，TP2，使用同卡两个64GB chip。"),
        ("实际负载范围", "两会话、16轮、每轮8输出；上下文增长按实际轨迹记录。"),
    ],
)

d = page(
    "沿用统一Engine与缓存路径",
    "复用实际已生成历史，不新增第二条推理执行路径",
    "Engine原生open_session/generate/close_session接口保持；ON提交增量suffix，OFF提交完整实际历史。",
)
for i, (title, body) in enumerate(
    (
        ("Engine会话", "保存实际生成IDs"),
        ("调度与混合缓存", "MHA / GDN状态"),
        ("NPU执行", "复用后继续生成"),
    )
):
    x = 64 + i * 392
    d.rounded_rectangle((x, 244, x + 364, 420), radius=12, fill=PANEL)
    text(d, title, x + 20, 267, 326, 29, TEAL, True)
    text(d, body, x + 20, 332, 326, 26)
    if i < 2:
        d.polygon(((x + 369, 316), (x + 384, 328), (x + 369, 340)), fill=ORANGE)
text(
    d,
    "工具由脚本执行求和465并等待2秒；另一会话继续推进。不是模型自主选择工具。",
    80,
    496,
    1100,
    28,
    MUTED,
)

d = page(
    "同源、同权重、同任务配置配对",
    "先验证组合，再测独立Session ON/OFF性能",
    "每模型64预热、192正式。三轮ON/OFF、OFF/ON、ON/OFF，全部真实生成并匹配原固定IDs。",
)
cards(
    d,
    [
        (
            "质量组合",
            "原生阶段结束并关闭后，串行重算32条完整实际历史；逐ID和context分别检查。",
        ),
        ("性能两臂", "Session只提交增量；重算臂提交完整历史。两臂均关闭prefix cache。"),
        ("计时口径", "每臂wall包含工具等待；排除模型启动、创建和关闭Session。"),
        (
            "重复边界",
            "三轮共享同一Engine，不是三个独立进程样本；不报告显著性或置信区间。",
        ),
    ],
)

d = page(
    "四模型原生组合均按实际结果审计",
    "输出、历史、工具窗口、持页及关闭分别检查",
    "具体完成轮数和正活动块样本来自各模型本轮记录，不能把旧单chip证据代替新测组合。",
)
cards(
    d,
    [
        (
            m.upper(),
            f"输出 {v['quality'][m]['output_pairs_matched']}/32；context {v['quality'][m]['context_pairs_validated']}/32。工具窗口另会话完成{len(v['quality'][m]['tool_observations'][0]['other_session_requests_completed_inside_window'])}条；close后active=0。",
        )
        for m in models
    ],
)

chart_page(
    "首字延迟：按三轮各臂P50展示",
    "柱高为三轮各臂P50的中位数；逐ID匹配后统计",
    "ttft.png",
    "条高是三轮各臂P50的中位数；误差线为min/max。原始P95和全部请求另见CSV。",
    "对数轴；误差线为三轮min/max，不是置信区间。包含客户端与调度开销。",
)
chart_page(
    "吞吐：每轮配对计算再汇总",
    "每臂256个实际输出token除以该臂wall，包含工具等待",
    "throughput_ratio.png",
    "吞吐比是本系统Session与本系统完整历史重算的配对比，不与其他引擎比较。",
    "三轮共享同一Engine；共享设备条件下的描述性结果，不保证独占收益。",
)
chart_page(
    "资源：共享目标chip的观察HBM",
    "保留启动前共享用量与性能采样区间总峰，不归因为纯模型占用",
    "observed_hbm.png",
    "背景与总峰均按本轮实际设备身份解析。观察峰不是瞬时峰，也不是单Session KV量。",
    "查询不是连续瞬时采样；总用量包含背景服务，不能用两根柱的差值证明模型净开销。",
)

d = page(
    "负结果与验证边界保持可见",
    "固定轨迹一致不等于任意输入的数值或语义质量",
    "保留HW1旧差异、HW2短输入TP差异和27B旧TP1容量拒绝；不恢复额外全层诊断，不改变门槛或参考。",
)
cards(
    d,
    [
        (
            "历史差异",
            "4B/9B短输入TP1/TP2差异继续保留；0.8B旧greedy和1%失败不被Session结果覆盖。",
        ),
        (
            "27B容量",
            "旧TP1默认预算是缓存/调度容量拒绝，不是权重放不下；95%候选也不等于本轮Agent结果。",
        ),
        (
            "共享条件",
            "实时PID身份、显存余量与退出检查保护原服务；轮询不是全局设备预留。",
        ),
        ("未知课程评测", "统一负载、规定Kunpeng主机预算及细则尚未取得，未宣称通过。"),
    ],
)

d = page(
    "固定源码、复现入口与成员责任",
    "代码、实验、分析和材料均如实披露AI辅助",
    "张晟祺承担实现/部署责任域，李书砚承担正确性/协议，计然承担性能/资源。责任域不是亲自操作或工时记录。教学网用户提交。",
)
cards(
    d,
    [
        (
            "实现与复现",
            "源码1d6099da，1535项。复现见docs/project/final_delivery_20261101.md。",
        ),
        (
            "软件验证",
            f"本地{v['software']['local_cpu']['passed']}通过/{v['software']['local_cpu']['skipped']}跳过；服务器{v['software']['server_cpu']['passed']}通过/{v['software']['server_cpu']['skipped']}跳过，范围分别说明。",
        ),
        (
            "责任边界",
            "张晟祺：实现与部署；李书砚：正确性与协议；计然：性能与资源分析。",
        ),
        ("AI与提交", "不虚构成员亲自执行、个人工时或理解考核。教学网由用户本人提交。"),
    ],
)

assert len(slides) == len(images) == 10
for i, image in enumerate(images, 1):
    image.quantize(colors=128, method=Image.Quantize.FASTOCTREE).save(
        out / f"slide-{i:02d}.png", optimize=True
    )
(out / "slides.json").write_text(
    json.dumps(slides, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
pages = [Image.open(out / f"slide-{i:02d}.png").convert("RGB") for i in range(1, 11)]
with PdfPages(out / "course_defense.pdf") as pdf:
    for image in pages:
        figure = plt.figure(figsize=(1280 / 96, 720 / 96), dpi=96)
        figure.figimage(image, origin="upper")
        pdf.savefig(figure, dpi=96)
        plt.close(figure)
montage = Image.new("RGB", (1280, 1800), "white")
for i, image in enumerate(pages):
    montage.paste(image.resize((640, 360)), ((i % 2) * 640, (i // 2) * 360))
montage.quantize(colors=128, method=Image.Quantize.FASTOCTREE).save(
    out / "montage.png", optimize=True
)
manifest = {
    "slides": 10,
    "size_pixels": [1280, 720],
    "raster_pages": True,
    "font": a.font.name,
    "bold_font": a.bold_font.name,
    "font_sha256": hashlib.sha256(a.font.read_bytes()).hexdigest(),
    "bold_font_sha256": hashlib.sha256(a.bold_font.read_bytes()).hexdigest(),
    "source_manifest_sha256": v["source_manifest_sha256"],
    "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
    "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "files": {
        p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in out.rglob("*")
        if p.is_file()
    },
}
(out / "render_manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
)
print(
    json.dumps(
        {
            "slides": 10,
            "source_sha256": v["source_manifest_sha256"],
            "all_data_audited": True,
        }
    )
)
