// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

const fs = require('node:fs');
const path = require('node:path');
const pptxgen = require('pptxgenjs');
async function main() {
  if (process.argv.length !== 4) throw new Error('Usage: node package_deck.cjs INPUT_DIR OUTPUT_PPTX');
  const input = path.resolve(process.argv[2]);
  const output = path.resolve(process.argv[3]);
  if (fs.existsSync(output)) throw new Error('Refuse to overwrite existing deck');
  const slides = JSON.parse(fs.readFileSync(path.join(input, 'slides.json'), 'utf8'));
  const pres = new pptxgen();
  pres.layout = 'LAYOUT_WIDE';
  pres.author = '张晟祺、李书砚、计然（AI工具辅助制作）';
  pres.title = '面向Agent负载的Ascend大模型推理系统优化';
  pres.subject = '固定轨迹、同源验证、独立Session ON/OFF性能';
  pres.company = '智能计算系统课程项目';
  pres.lang = 'zh-CN';
  slides.forEach((meta, i) => {
    const slide = pres.addSlide();
    slide.addImage({path:path.join(input, `slide-${String(i+1).padStart(2,'0')}.png`), x:0, y:0, w:13.333333, h:7.5});
    slide.addNotes(`${meta.title}\n${meta.subtitle}\n${meta.notes}\n图像页面；可编辑文本与绘制来源见slides.json及生成脚本。`);
  });
  await pres.writeFile({fileName:output});
  process.stdout.write(JSON.stringify({slides:slides.length,bytes:fs.statSync(output).size})+'\n');
}
main().catch(e => {process.stderr.write(e.message+'\n');process.exit(1);});
