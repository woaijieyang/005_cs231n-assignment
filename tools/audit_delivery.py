"""交付一致性检查：notebook、执行证据、题目、解析、链接、图像与训练预算。"""
from pathlib import Path
import hashlib,json,re,subprocess,sys
from urllib.parse import unquote
import nbformat
import mistune
from PIL import Image
from jupyter_client.kernelspec import KernelSpecManager
ROOT=Path(__file__).resolve().parents[1]

def code_hash(notebook):
    blocks=[(c.get('id'),''.join(c['source'])) for c in notebook['cells'] if c['cell_type']=='code' and ''.join(c['source']).strip()]
    return hashlib.sha256(json.dumps(blocks,ensure_ascii=False).encode()).hexdigest()

def markdown_targets(text):
    # 使用真正的 Markdown 解析器；三反引号行内代码不能误当成未闭合围栏。
    tree=mistune.create_markdown(renderer='ast')(text)
    targets=[]
    def walk(items):
        for item in items:
            if item['type'] in ('link','image'):
                targets.append(item.get('attrs',{}).get('url',''))
            elif item['type'] in ('inline_html','block_html'):
                targets.extend(re.findall(r'<img[^>]+src=["\']([^"\']+)',item.get('raw','')))
            walk(item.get('children',[]))
    walk(tree)
    return targets

def main():
    failures=[];notebooks=[];execution={}
    old_manifest_path=ROOT/'outputs/execution_manifest.json'
    old_manifest=json.loads(old_manifest_path.read_text(encoding='utf-8')) if old_manifest_path.exists() else {}
    paths=sorted(ROOT.glob('*_Assignment*/[0-9][0-9][0-9]_*.ipynb'))
    if len(paths)!=15:failures.append(f'Expected 15 notebooks, found {len(paths)}')
    for path in paths:
        notebook=nbformat.read(path,as_version=4);nbformat.validate(notebook)
        code_cells=[c for c in notebook.cells if c.cell_type=='code' and c.source.strip()]
        errors=[o for c in code_cells for o in c.outputs if o.output_type=='error']
        if errors:failures.append(f'{path.name}: error outputs')
        if notebook.cells[0].get('id')!='assignment-requirements':failures.append(f'{path.name}: missing opening requirements')
        if notebook.metadata.kernelspec.name!='cs231n_zx_test' or notebook.metadata.kernelspec.display_name!='cs231n_zx_test':failures.append(f'{path.name}: incorrect kernel')
        missing=[];unexecuted=[]
        for index,cell in enumerate(notebook.cells):
            if cell.cell_type!='code' or not cell.source.strip():continue
            if index==0 or notebook.cells[index-1].cell_type!='markdown':missing.append(cell.id)
            if cell.execution_count is None:unexecuted.append(cell.id)
        if missing:failures.append(f'{path.name}: missing preceding explanation {missing}')
        if unexecuted:failures.append(f'{path.name}: unexecuted {unexecuted}')
        for cell in notebook.cells:
            if cell.cell_type=='markdown' and re.search(r'TODO:\s*(?:YOUR ANSWER|Describe)|fill this in|fill in this',cell.source,re.I):failures.append(f'{path.name}: unanswered prompt')
            if '\ufffd' in cell.source:failures.append(f'{path.name}: replacement character in source')
        actual_hash=code_hash(notebook)
        runs=sorted((ROOT/'outputs'/path.stem).glob('run-*.ipynb'))
        successful=[]
        for run in runs:
            log=run.with_suffix('.jsonl')
            if not log.exists():continue
            lines=log.read_text(encoding='utf-8').splitlines()
            if lines and json.loads(lines[-1]).get('event')=='complete':successful.append(run)
        if successful:
            latest=successful[-1];executed=nbformat.read(latest,as_version=4)
            if code_hash(executed)!=actual_hash:failures.append(f'{path.name}: code differs from latest complete run')
            execution[path.stem]={'code_sha256':actual_hash,'completed_run':latest.stem,'code_cells':len(code_cells),'error_outputs':len(errors)}
        elif path.stem in old_manifest:
            execution[path.stem]=old_manifest[path.stem]
            if execution[path.stem]['code_sha256']!=actual_hash:failures.append(f'{path.name}: code differs from recorded execution')
        else:failures.append(f'{path.name}: no complete execution evidence')
        notebooks.append({'notebook':path.relative_to(ROOT).as_posix(),'code_cells':len(code_cells),'preceding_explanation_missing':missing,'unexecuted':unexecuted,'error_outputs':len(errors)})
    old_manifest_path.write_text(json.dumps(execution,ensure_ascii=False,indent=2),encoding='utf-8')
    mapping=json.loads((ROOT/'file_mapping.json').read_text(encoding='utf-8')) if (ROOT/'file_mapping.json').exists() else []
    if mapping and len(mapping)!=150:failures.append('Incomplete local file inventory')
    for record in mapping:
        if record.get('status')=='active' and not (ROOT/record['destination']).is_file():
            failures.append('Missing active mapped file: '+record['destination'])
        # archived_local 的条目不随 Git 发布；在线通过固定初始版本提交追溯。
    # 只检查交付文本，源代码围栏中的示例链接不当作实际文档依赖。
    missing_links=[]
    documents=list(ROOT.glob('*.md'))+list(ROOT.glob('*_Assignment*/*.md'))
    for path in documents:
        for target in markdown_targets(path.read_text(encoding='utf-8')):
            if re.match(r'^(https?://|mailto:|#|data:)',target):continue
            target=unquote(target.strip('<>').split('#')[0])
            if target and not (path.parent/target).exists():missing_links.append({'document':path.name,'target':target})
    if missing_links:failures.append(f'{len(missing_links)} broken local Markdown links')
    image_count=0
    for image in (ROOT/'assets/complete_record').rglob('*.png'):
        with Image.open(image) as img:img.verify()
        image_count+=1
    budgets={}
    for kind in ('rnn','lstm'):
        path=ROOT/'outputs'/f'full_coco_{kind}_gpu_equivalent/metrics.json'
        result=json.loads(path.read_text())
        passed=result['status']=='complete' and result['epochs_completed']==20 and result['steps_completed']==80040 and result['train_captions_per_epoch']==400135
        if not passed:failures.append(f'{kind}: incomplete full-data training')
        budgets[kind]={'epochs':20,'updates':80040,'passed':passed}
    for name,epochs,steps in [('vanilla',10,3900),('lsgan',10,3900),('dcgan',5,1950)]:
        result=json.loads((ROOT/f'outputs/015_GANs_PyTorch/{name}_report.json').read_text())
        if result['epochs_completed']!=epochs or result['steps_completed']!=steps:failures.append(name+': incomplete training')
        budgets[name]={'epochs':epochs,'updates':steps}
    extension=json.loads((ROOT/'outputs/010_PyTorch/augmentation_extension/metrics.json').read_text())
    if extension['status']!='complete' or extension['additional_epochs']!=30:failures.append('CIFAR extension incomplete')
    kernel=KernelSpecManager().get_kernel_spec('cs231n_zx_test')
    kernel_ok=Path(kernel.argv[0]).resolve()==Path(sys.executable).resolve() and kernel.display_name=='cs231n_zx_test'
    if not kernel_ok:failures.append('Jupyter kernel executable mismatch')
    baseline=ROOT.parent/'base'
    baseline_clean=None
    if (baseline/'.git').exists():
        baseline_clean=not subprocess.check_output(['git','status','--porcelain'],cwd=baseline).strip()
        if not baseline_clean:failures.append('Reference base has modifications')
    report={'status':'passed' if not failures else 'failed','notebooks':notebooks,'total_nonempty_code_cells':sum(x['code_cells'] for x in notebooks),'images_verified':image_count,'kernel_executable_matches_environment':kernel_ok,'training_budgets':budgets,'extended_cifar_epochs':extension['total_epoch_budget'],'missing_links':missing_links,'failures':failures,'evidence_boundary':'This audit checks delivered executed artifacts and recorded complete runs; it does not rerun training.'}
    (ROOT/'outputs/delivery_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('notebooks','training_budgets')},ensure_ascii=False,indent=2))
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
