"""先验证等价性，再执行或恢复完整 COCO GPU 训练。"""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'03_Assignment3'))
from cs231n.coco_utils import load_coco_data
from cs231n.torch_captioning import verify_equivalence,train_gpu
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('cell_type',choices=['rnn','lstm']);args=parser.parse_args()
    report={device:verify_equivalence(device) for device in ('cpu','cuda')}
    (ROOT/'outputs/caption_backend_equivalence.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Forward, gradient, decoding and Adam equivalence passed on CPU and CUDA.',flush=True)
    train_gpu(load_coco_data(pca_features=True),args.cell_type)
