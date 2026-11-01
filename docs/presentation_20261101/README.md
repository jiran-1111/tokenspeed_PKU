# 20261101当期演示与重生成

- [10页PPTX](course_defense.pptx)、[10页PDF](rendered/course_defense.pdf)、[总览](rendered/montage.png)。
- [完整报告](../report/course_project_final_20261101.md)、[讲稿与核验问题](../project/defense_outline_20261101.md)、[当前复现入口](../project/final_delivery_20261101.md)。
- [四张独立实验图](rendered/figures/)、[页面元数据与讲者备注](rendered/slides.json)、[渲染来源与文件SHA](rendered/render_manifest.json)、[交付与视觉核查](artifact_verification.json)。

只使用当前`project_agent_20261031/final_verification.json`的审计结果生成图表和数据：0.8B TP1、4B TP1、9B FP32头TP1、27B TP2；128输出/128context配对，768正式/256预热请求。旧20261018演示目录保持冻结。共享条件、固定轨迹、旧负结果、未知课程负载/规定预算及AI使用均在页面可见披露。

柱高为三轮各臂P50的中位数，误差线为min/max；吞吐先每轮配对再展示，不是独立样本置信区间。共享观察HBM包括背景，非模型净占用/瞬时峰。全部详细P50/P95、E2E、近似TPOT和逐请求数据仍以报告/CSV为准。

PPTX每页嵌入一张1280×720 PNG，讲者备注为文本；PDF同样为图像页面。图表有独立PNG，正文和布局可在生成脚本中编辑，`slides.json`提供标题、说明与备注；它们不等同PPTX原生可编辑文本或无障碍标记文档。未声称经过PowerPoint/LibreOffice二次渲染；本次核验PPTX页面/备注数量和内嵌PNG SHA，并对PDF用Poppler实际渲染检查。

生成使用工作区内Python的Pillow13.0.0、matplotlib3.11.2，以及工作区Node依赖中的PptxGenJS。字体显式选择本机`msyh.ttc`/`msyhbd.ttc`，字体文件不分发，哈希在渲染清单中。其他机器更换字体会改变像素，应另立输出目录记录，不把视觉差异隐去。Node和Python依赖放自己的工作区环境，不全局安装。

在固定源码检出、完整审计数据齐备后执行（输出目录和PPTX必须不存在）：

```bash
python docs/presentation_20261101/runners/render.py \
  --repo "$PWD" --output-dir "$PWD/.tmp/defense-reproduce-unique" \
  --font /absolute/path/to/msyh.ttc --bold-font /absolute/path/to/msyhbd.ttc
NODE_PATH=/absolute/workspace/node_modules \
node docs/presentation_20261101/runners/package_deck.cjs \
  .tmp/defense-reproduce-unique .tmp/defense-reproduce-unique.pptx
pdfinfo .tmp/defense-reproduce-unique/course_defense.pdf
pdftoppm -scale-to 1280 -png .tmp/defense-reproduce-unique/course_defense.pdf \
  .tmp/defense-pdf-page
```

程序先核对四模型审计、质量/性能数量，再生成四图、10页PNG/PDF、总览和清单；拒绝覆盖已有目录。MIT头与历史布局/打包工具来源保留。修改后应重新查看总览、密集页面及PDF实际渲染；不能只凭生成命令退出0声称排版合格。
