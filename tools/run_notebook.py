"""干净内核运行，逐单元保存输出；失败输出与成功结果明确区分。"""
import argparse, datetime, json, os, pathlib, sys, time, traceback
import nbformat
from nbclient import NotebookClient

ROOT=pathlib.Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('number',help='Three-digit notebook number')
    parser.add_argument('--original-only',action='store_true',help='跳过方法二，并保留完整源 notebook 不变')
    args=parser.parse_args()
    matches=list(ROOT.glob(f'*_Assignment*/{args.number}_*.ipynb'))
    if len(matches)!=1: raise ValueError(matches)
    source=matches[0]
    out=ROOT/'outputs'/source.stem
    out.mkdir(parents=True,exist_ok=True)
    stamp=datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    prefix='original-only-run' if args.original_only else 'run'
    result=out/f'{prefix}-{stamp}.ipynb'
    logpath=out/f'{prefix}-{stamp}.jsonl'
    nb=nbformat.read(source,as_version=4)
    if args.original_only:
        nb.cells=[cell for cell in nb.cells if 'method2' not in cell.get('metadata',{}).get('tags',[])]
    for cell in nb.cells:
        if cell.cell_type=='code': cell.outputs=[];cell.execution_count=None
    started=time.time()
    def log(event,**fields):
        record={'event':event,'time':datetime.datetime.now().isoformat(),'elapsed_s':time.time()-started,**fields}
        with logpath.open('a',encoding='utf-8') as stream: stream.write(json.dumps(record,ensure_ascii=False)+'\n')
        print(json.dumps(record,ensure_ascii=False),flush=True)
    def on_start(cell,cell_index,**kwargs): log('cell_start',index=cell_index,id=cell.get('id'))
    def on_executed(cell,cell_index,**kwargs):
        nbformat.write(nb,result)
        log('cell_executed',index=cell_index,id=cell.get('id'))
    client=NotebookClient(nb,kernel_name='cs231n_zx_test',timeout=None,allow_errors=False,resources={'metadata':{'path':str(source.parent)}},on_cell_start=on_start,on_cell_executed=on_executed)
    log('start',source=str(source.relative_to(ROOT)),python=sys.executable,scope='original_only' if args.original_only else 'full')
    try:
        client.execute()
        nbformat.write(nb,result)
        if not args.original_only:
            nbformat.write(nb,source)
        log('complete',artifact=str(result.relative_to(ROOT)))
    except Exception as exc:
        nbformat.write(nb,result)
        log('failed',error=str(exc),traceback=traceback.format_exc())
        raise

if __name__=='__main__': main()
