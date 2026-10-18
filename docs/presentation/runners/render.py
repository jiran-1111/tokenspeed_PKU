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

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

p = argparse.ArgumentParser()
p.add_argument("--repo", type=Path, required=True)
p.add_argument("--output-dir", type=Path, required=True)
p.add_argument("--font", type=Path, required=True)
p.add_argument("--bold-font", type=Path, required=True)
a = p.parse_args()
repo = a.repo.resolve()
v = json.loads(
    (repo / "docs/evidence/course_final_20261018/verification.json").read_text()
)
assert v["final_quality_models"] == 4 and v["final_quality_pairs"] == 128
assert a.font.is_file() and a.bold_font.is_file()
audits = {
    m: json.loads(
        (repo / f"docs/evidence/native_performance_20261018/{m}/audit.json").read_text()
    )
    for m in ("0.8b", "4b", "9b", "27b")
}
out = a.output_dir.resolve()
out.mkdir(parents=True, exist_ok=False)
BG, PANEL, WHITE, MUTED, TEAL, ORANGE = (
    "#0B132B",
    "#17243B",
    "#F4F7FB",
    "#B3C1D1",
    "#2EC4B6",
    "#FF9F1C",
)
slides = []
images = []
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


d = page(
    "保留会话状态，减少历史重算",
    "Ascend统一四模型部署、原生Agent负载与有限正确性验证",
    "最终报告：docs/report/course_project_final_20261018.md。实际代码、实验与材料由AI工具辅助完成。最终配置同源1415项实现清单与报告分别核验。",
)
for x, value, label in (
    (80, "4", "模型，同一套系统"),
    (480, "768", "主性能请求逐ID匹配"),
    (880, "128", "最终原生/重算质量配对"),
):
    d.text((x, 266), value, font=font(92, True), fill=TEAL)
    text(d, label, x, 392, 330, 27)
text(
    d,
    "Session复用在固定轨迹上降低首字延迟；旧失败和精度代价全部保留。",
    80,
    531,
    1120,
    27,
    MUTED,
)

d = page(
    "统一执行路径，状态随会话保留",
    "复用上游Engine、C++调度和混合缓存；课程实现集成于原fork",
    "上游起点cf4a4f89；最终报告第1、2节。工具是脚本受控注入，不是模型自主选择工具。",
)
for i, label in enumerate(
    ("两条Agent会话", "统一Engine", "C++调度/缓存", "NPU MHA/GDN", "实际历史/前沿")
):
    x = 64 + i * 234
    d.rounded_rectangle((x, 224, x + 215, 318), radius=9, fill=PANEL)
    text(d, label, x + 12, 251, 195, 23, TEAL, True, max_y=316)
    if i < 4:
        d.text((x + 216, 251), "→", font=font(24), fill=WHITE)
text(
    d, "增量suffix → 实际生成 → 保留完整历史 → 下一轮复用", 92, 380, 1100, 33, bold=True
)
text(
    d,
    "0.8B/4B：BF16 TP1    9B：FP32输出头 TP1    27B：BF16 TP2\n同一物理NPU，最多两个64GiB chip；每次检查映射/空闲并串行运行。",
    92,
    468,
    1080,
    26,
    MUTED,
)

d = page(
    "数值、greedy与Session分别验收",
    "通过哪一层，就只对哪一层作结论",
    "HW1独立提交1d113e54，最终报告第3节；R5头精度分析见docs/project/9b_lm_head_precision_20261015.md。",
)
cards(
    d,
    [
        (
            "HW1完整logits",
            "原16输入、272向量通过预先冻结的双门槛；主参考CPU FP32/eager。",
        ),
        ("旧失败保留", "旧1%门槛68向量失败；条件argmax 13处差异，不改称greedy一致。"),
        (
            "9B输出头精度",
            "默认BF16八例7通过/1失败；显式FP32头正式图模式8/8，代价单独测量。",
        ),
        (
            "最终原生Session",
            "四模型各32组实际历史重算匹配；context、工具留存和关闭分别检查。",
        ),
    ],
)

d = page(
    "独立性能臂，不混入质量控制竞争",
    "每模型2会话 × 16轮 × 8输出；第8轮前求和465并等待2秒",
    "R9r2协议SHA c9bdb5b7e6f5d766a1f862b499410c97ba6019e51a5d46139b0186525d0259af；完整报告docs/project/native_agent_performance_20261018.md。",
)
cards(
    d,
    [
        ("Session ON", "只提交增量suffix；保留实际历史，逐次真实生成并核对固定IDs。"),
        ("完整历史OFF", "每轮重算完整实际历史；prefix cache关闭，无强制参考token。"),
        ("预热与三轮次序", "双方各32请求预热；ON/OFF、OFF/ON、ON/OFF，共192正式请求。"),
        (
            "计时与边界",
            "wall含工具等待；TTFT/E2E是客户端观测，同Engine重复不是独立统计样本。",
        ),
    ],
)

d = page(
    "首字延迟：四模型均观察到下降",
    "横轴为对数；每个数是三轮各32请求P50的中位数，非全请求合并P50",
    "R9r2 audit.json paired_repetitions；三轮描述性观测，不是置信区间或通用最优结论。",
)
for value in (10, 100, 1000, 10000):
    xx = 240 + 680 * (math.log10(value) - 1) / 3
    d.line((xx, 203, xx, 582), fill=PANEL, width=2)
    d.text((xx - 18, 588), str(value), font=font(16), fill=MUTED)
for i, (model, audit) in enumerate(audits.items()):
    pairs = audit["paired_repetitions"]
    y = 210 + i * 101
    d.text((65, y + 12), model.upper(), font=font(26, True), fill=WHITE)
    for j, (key, label, color) in enumerate(
        (("session_ttft_p50_ms", "ON", TEAL), ("stateless_ttft_p50_ms", "OFF", ORANGE))
    ):
        value = statistics.median(p[key] for p in pairs)
        x_end = 240 + 680 * (math.log10(value) - 1) / 3
        yy = y + j * 35
        d.rectangle((240, yy, x_end, yy + 23), fill=color)
        d.text((175, yy - 3), label, font=font(20), fill=color)
        d.text((x_end + 12, yy - 5), f"{value:.1f} ms", font=font(21), fill=WHITE)
text(
    d,
    "ON：增量复用     OFF：完整历史重算     详细P95、范围、逐请求CSV见报告",
    65,
    618,
    1150,
    19,
    MUTED,
    max_y=652,
)

d = page(
    "吞吐收益与HBM占用同时报告",
    "吞吐wall包含工具等待；HBM为全运行观察峰值，非瞬时峰值或仅KV",
    "主性能为R9r2的768正式请求；每模型192。最终1415源中原1409项实现字节一致。HBM包括预热/臂间，未证明Session减少预分配HBM。",
)
headers = ("模型/TP", "配对首字P50下降", "配对吞吐比", "观察峰 MB / chip")
xs = (75, 310, 660, 915)
for x, h in zip(xs, headers):
    text(d, h, x, 218, 330, 23, TEAL, True, max_y=270)
for i, (model, audit) in enumerate(audits.items()):
    pairs = audit["paired_repetitions"]
    red = [p["ttft_p50_reduction_percent"] for p in pairs]
    ratio = [p["throughput_ratio_session_over_stateless"] for p in pairs]
    peak = "/".join(
        str(n)
        for _, n in sorted(audit["hbm_observed_peak_all_phases_mb_per_chip"].items())
    )
    values = (
        f"{model.upper()} / {2 if model == '27b' else 1}",
        f"{min(red):.2f}–{max(red):.2f}%",
        f"{min(ratio):.2f}–{max(ratio):.2f}×",
        peak,
    )
    for x, value in zip(xs, values):
        text(d, value, x, 285 + i * 66, 330, 24, max_y=620)
text(
    d,
    "9B为显式FP32输出头。三轮先配对计算比值，再列min/max；不作显著性推断。",
    75,
    585,
    1110,
    22,
    MUTED,
)

d = page(
    "最终配置在同一源快照再次验证",
    "1415项清单；原1409项实现不变，仅新增4个final与2个TP候选配置",
    "最终源SHA d05a47d932b519b6e50ed0c668725935d93b744fffffa68e95f37a0227f6c6ea；docs/evidence/course_final_20261018/verification.json。",
)
lines = []
for model in ("0.8b", "4b", "9b", "27b"):
    r = v["results"][f"final_{model}"]
    lines.append(
        f"{model.upper()}：32/32输出与context匹配；工具窗口其他会话完成{r['tool_other_completions']}轮，close后active=0。"
    )
text(d, "\n".join(lines), 80, 216, 1120, 28)
for i, label in enumerate(("tp4b_tp2", "tp27b_tp1")):
    status = v["results"][label]["summary"]["status"]
    if status != "passed":
        status = (
            "启动缓存容量拒绝，无有效性能数据"
            if v["results"][label]["classification"]
            == "configured_cache_capacity_rejected"
            else "实验未有效完成，无性能结论"
        )
    text(
        d,
        f"TP候选 {label}：{status}；完整结果及采用取舍见总报告。",
        80,
        461 + i * 48,
        1110,
        24,
        TEAL,
    )
text(
    d,
    "保守交付：4B TP1 / 27B TP2。不同TP独立Engine，未作随机跨进程显著性推断。",
    80,
    581,
    1110,
    22,
    MUTED,
)

d = page(
    "精度修复有代价，Profiler也有边界",
    "不给失败改名，不把任务累计时长当端到端耗时",
    "9B头精度成本为R5源；HW1 Profiler为固定1396源、127实际Decode步；均不冒称最终1415源新采。",
)
cards(
    d,
    [
        (
            "9B FP32头成本",
            "逻辑多1.8945GiB；旧同源配对观察Decode +7.5%–9.3%，质量八例8/8。",
        ),
        (
            "HW1 Decode Profiler",
            "conv/state 36.944%、MatMul 21.107%、ViewCopy 13.334%，按任务累计时长。",
        ),
        (
            "不能互相替代",
            "AICore子集MatMul 92.15%不是整个Decode；时间隙不能直接归因Host或设备idle。",
        ),
        (
            "保留负结果",
            "旧1%、greedy、工具窗口和采样失败均留档；chunk/prefix旧结果不外推一般Agent。",
        ),
    ],
)

d = page(
    "责任均衡，实际执行如实披露",
    "责任边界不是成员个人工时或亲自执行记录",
    "最终报告第7节。实际代码、服务器实验、原始同步、分析及报告均由AI工具辅助完成；不宣称已完成学生个人理解考核。",
)
for i, (name, scope) in enumerate(
    (
        ("张晟祺", "计算路径、0.8B/4B数值与HW1 Profiler解释"),
        ("李书砚", "Session生命周期、9B精度与多轮正确性"),
        ("计然", "四模型/27B与TP评测、资源及集成交付"),
    )
):
    y = 222 + i * 102
    d.rounded_rectangle((65, y, 1215, y + 82), radius=10, fill=PANEL)
    text(d, name, 88, y + 23, 180, 29, TEAL, True, max_y=y + 80)
    text(d, scope, 300, y + 25, 880, 27, max_y=y + 80)
text(
    d,
    "AI完成实际操作；用回归、NPU实测、独立复算和SHA核验支持结论。",
    80,
    573,
    1110,
    25,
    MUTED,
)

d = page(
    "可复现交付与答辩关注点",
    "固定分支、明确配置、原始哈希、完整失败记录",
    "分支course/final-20261018；HW1固定1d113e54；R9固定82788a85。教学网由用户本人提交。演示命令见docs/project/final_delivery_20261018.md。",
)
cards(
    d,
    [
        (
            "复现顺序",
            "项目环境 → 完整软件检查 → 设备空闲 → 单个原生验证/性能 → 独立分析。",
        ),
        (
            "演示重点",
            "解释suffix/current-frontier、工具等待、关闭引用，以及ON/OFF如何保持相同质量。",
        ),
        (
            "结论范围",
            "固定轨迹、有限样例、同Engine重复；不宣称任意输入一致或未知统一负载验收。",
        ),
        (
            "提交与理解",
            "代码/报告/证据使用同源allowlist上传；教学网本人提交，成员理解不由AI代替。",
        ),
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
pages[0].save(
    out / "course_defense.pdf", save_all=True, append_images=pages[1:], resolution=96
)
thumb = Image.new("RGB", (1280, 5 * 360), "#FFFFFF")
for i, image in enumerate(pages):
    thumb.paste(image.resize((640, 360)), ((i % 2) * 640, (i // 2) * 360))
thumb.save(out / "montage.png")
hashes = {
    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
    for path in out.iterdir()
    if path.is_file()
}
(out / "render_manifest.json").write_text(
    json.dumps(
        {
            "slides": 10,
            "size_pixels": [1280, 720],
            "raster_pages": True,
            "font": a.font.name,
            "bold_font": a.bold_font.name,
            "font_sha256": hashlib.sha256(a.font.read_bytes()).hexdigest(),
            "bold_font_sha256": hashlib.sha256(a.bold_font.read_bytes()).hexdigest(),
            "source_manifest_sha256": v["source_manifest_sha256"],
            "files": hashes,
        },
        indent=2,
    )
    + "\n"
)
print(
    json.dumps(
        {
            "slides": 10,
            "slide_png_bytes": sum(
                (out / f"slide-{i:02d}.png").stat().st_size for i in range(1, 11)
            ),
        }
    )
)
