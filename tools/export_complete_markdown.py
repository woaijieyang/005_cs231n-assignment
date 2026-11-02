"""把三份作业的完整 notebook 内容、输出图像及核心源码合并为 Markdown。"""
from pathlib import Path
import base64, hashlib, json, re
from urllib.parse import quote
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'002_三份作业完整学习记录.md'
ASSETS=ROOT/'assets'/'complete_record'

def fence(text,language=''):
    longest=max([len(x) for x in re.findall(r'`+',text)]+[2])
    marker='`'*max(3,longest+1)
    return f'{marker}{language}\n{text.rstrip()}\n{marker}\n'

def clean_console(text):
    return re.sub(r'\x1b\[[0-9;]*[A-Za-z]','',text)

def rebase_links(text,directory):
    def replace(match):
        target=match.group(2)
        if re.match(r'^(https?://|mailto:|#|data:)',target):return match.group(0)
        raw=target.strip('<>')
        if ' ' in raw and not (directory/raw).exists():return match.group(0)
        path=(directory/raw).resolve()
        if not path.is_relative_to(ROOT):return match.group(0)
        return match.group(1)+quote(path.relative_to(ROOT).as_posix(),safe='/#')+')'
    return re.sub(r'(!?\[[^\]]*\]\()([^\)]+)\)',replace,text)

def notebook_markdown(path):
    notebook=json.loads(path.read_text(encoding='utf-8'));lines=[f'## {path.stem}\n',f'[可执行 notebook]({quote(path.relative_to(ROOT).as_posix())})\n']
    folder=ASSETS/path.stem;folder.mkdir(parents=True,exist_ok=True)
    counts={'markdown_cells':0,'code_cells':0,'images':0,'error_outputs':0}
    for index,cell in enumerate(notebook['cells']):
        source=''.join(cell.get('source',[]))
        if cell['cell_type']=='markdown':
            counts['markdown_cells']+=1
            lines.append(rebase_links(source,path.parent)+'\n')
        elif cell['cell_type']=='code':
            if not source.strip():continue
            counts['code_cells']+=1;lines.append(fence(source,'python'))
            pending_stream=[]
            def flush_stream():
                if pending_stream:
                    text=clean_console(''.join(pending_stream))
                    if text.strip():lines.append(fence(text,'text'))
                    pending_stream.clear()
            for j,output in enumerate(cell.get('outputs',[])):
                kind=output.get('output_type')
                if kind=='stream':
                    pending_stream.append(''.join(output.get('text',[])))
                    continue
                elif kind=='error':
                    flush_stream()
                    counts['error_outputs']+=1
                    lines.append(fence(clean_console('\n'.join(output.get('traceback',[]))),'text'))
                elif kind in ('display_data','execute_result'):
                    flush_stream()
                    data=output.get('data',{})
                    if 'image/png' in data:
                        image=folder/f"{cell.get('id',str(index))}-{j:02d}.png"
                        image.write_bytes(base64.b64decode(''.join(data['image/png'])))
                        lines.append(f'![{path.stem}，单元 {cell.get("id",index)} 的实际输出]({image.relative_to(ROOT).as_posix()})\n');counts['images']+=1
                    elif 'image/svg+xml' in data:
                        image=folder/f"{cell.get('id',str(index))}-{j:02d}.svg"
                        image.write_text(''.join(data['image/svg+xml']),encoding='utf-8')
                        lines.append(f'![实际输出]({image.relative_to(ROOT).as_posix()})\n');counts['images']+=1
                    elif 'text/plain' in data:lines.append(fence(''.join(data['text/plain']),'text'))
            flush_stream()
    return '\n'.join(lines),counts

def main():
    ASSETS.mkdir(parents=True,exist_ok=True)
    lines=['# CS231n：三份作业完整学习记录\n',
           '本文件按作业顺序汇总题目要求、原笔记、逐块中文解析、完整代码、本机运行输出与改进实验。图像以相对路径保存在 `assets/complete_record`；阅读或移动本文件时需同时保留资源目录。底层课程源码列在文末附录，便于把 notebook 调用和实际实现对应起来。\n',
           '[总实现流程](000_总实现流程.md) · [文件索引](001_文件索引.md) · [问题与解决记录](003_问题与解决记录.md)\n']
    manifest={}
    for group in sorted(ROOT.glob('*_Assignment*')):
        lines.append('\n---\n')
        lines.append(rebase_links((group/'000_作业要求.md').read_text(encoding='utf-8'),group))
        for notebook in sorted(group.glob('[0-9][0-9][0-9]_*.ipynb')):
            text,counts=notebook_markdown(notebook);lines.append(text);manifest[notebook.stem]=counts
    lines.append('\n# 附录：底层实现与运行工具\n\n以下按作业保持独立的 `cs231n` 包；同名模块不代表可以跨作业混用。TensorFlow 参考和历史二进制不属于本次执行源码。\n')
    sources=[]
    for group in sorted(ROOT.glob('*_Assignment*')):
        sources.extend(sorted((group/'cs231n').rglob('*.py')))
        sources.extend(sorted((group/'cs231n').rglob('*.pyx')))
    sources.extend([ROOT/'study_runtime.py'])
    sources.extend(sorted((ROOT/'tools').glob('*.py')))
    for source in sources:
        if source.name in ('squeezenet.py','archive_auxiliary.py','full_captioning.py','train_full_coco.py') or 'archive' in source.parts:continue
        relative=source.relative_to(ROOT).as_posix()
        lines.append(f'## {relative}\n\n[源文件]({quote(relative)})\n')
        lines.append(fence(source.read_text(encoding='utf-8'),'cython' if source.suffix=='.pyx' else 'python'))
    # 汇总文档清理行尾空格，不修改源 notebook 或 Python 文件。
    rendered='\n'.join(line.rstrip() for line in '\n'.join(lines).splitlines())+'\n'
    DEST.write_text(rendered,encoding='utf-8')
    manifest['_source_files']=len(sources)
    manifest['_markdown_bytes']=DEST.stat().st_size
    (ROOT/'outputs/markdown_export_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'notebooks':len(manifest)-2,'images':sum(v.get('images',0) for v in manifest.values() if isinstance(v,dict)),'bytes':DEST.stat().st_size},ensure_ascii=False))

if __name__=='__main__':main()
