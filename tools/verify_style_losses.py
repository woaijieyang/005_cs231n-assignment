"""直接提取当前 notebook 的四个实际损失函数，用双精度数值梯度检查。"""
from pathlib import Path
import json
import torch
ROOT=Path(__file__).resolve().parents[1]

def main():
    path=ROOT/'03_Assignment3/014_StyleTransfer_PyTorch.ipynb'
    notebook=json.loads(path.read_text(encoding='utf-8'))
    namespace={'torch':torch}
    for cell in notebook['cells']:
        if cell.get('metadata',{}).get('original_cell_index') in (10,14,18,22):
            exec(''.join(cell['source']),namespace)
    torch.manual_seed(231)
    x=torch.randn(2,3,4,5,dtype=torch.float64,requires_grad=True)
    target=torch.randn_like(x)
    checks={}
    checks['content']=torch.autograd.gradcheck(lambda z:namespace['content_loss'](.1,z,target),(x,))
    checks['gram']=torch.autograd.gradcheck(namespace['gram_matrix'],(x,))
    checks['tv']=torch.autograd.gradcheck(lambda z:namespace['tv_loss'](z,.2),(x,))
    gram_target=namespace['gram_matrix'](target).detach()
    checks['style']=torch.autograd.gradcheck(lambda z:namespace['style_loss']([z],[0],[gram_target],[.3]),(x,))
    (ROOT/'outputs/style_loss_gradcheck.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(checks)

if __name__=='__main__':main()
