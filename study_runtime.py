"""课程共享路径、MNIST 只读加载和旧图像接口的显式兼容实现。"""
from pathlib import Path
import gzip, json, os, struct
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent

def data_path(key):
    """读取本机路径覆盖；未设置时使用项目旁共享目录，不触发网络下载。"""
    config=ROOT/'data_paths.local.json'
    paths=json.loads(config.read_text(encoding='utf-8')) if config.exists() else {}
    defaults={'cifar10':ROOT.parent/'datasets/cifar-10-batches-py','coco':ROOT.parent/'datasets/coco_captioning','imagenet_val':ROOT.parent/'datasets/imagenet_val_25.npz','shared':ROOT.parent/'datasets','mnist_gzip':ROOT.parent/'datasets/mnist'}
    return Path(os.environ.get('CS231N_'+key.upper(),paths.get(key,defaults[key])))

def imread(path):
    """文件/文件流 → Pillow 解码 → 独立像素数组；不修改源图片。"""
    with Image.open(path) as image:
        return np.array(image)

def imresize(array, size):
    """运行验证课程 uint8 图片的双线性缩放；tuple 按 (高度, 宽度) 解释。"""
    image=Image.fromarray(array)
    target=(int(image.width*size),int(image.height*size)) if np.isscalar(size) else (int(size[1]),int(size[0]))
    return np.array(image.resize(target,Image.Resampling.BILINEAR))

class ExistingMNIST:
    """直接解压已有 gzip 到内存，遵守 torchvision Dataset 的取样接口。"""
    def __init__(self, root=None, train=True, download=False, transform=None):
        prefix='train' if train else 't10k'
        directory=data_path('mnist_gzip')
        with gzip.open(directory/f'{prefix}-images-idx3-ubyte.gz','rb') as stream:
            magic,n,h,w=struct.unpack('>IIII',stream.read(16))
            assert magic==2051
            self.data=np.frombuffer(stream.read(),dtype=np.uint8).reshape(n,h,w)
        with gzip.open(directory/f'{prefix}-labels-idx1-ubyte.gz','rb') as stream:
            magic,n=struct.unpack('>II',stream.read(8)); assert magic==2049
            self.targets=np.frombuffer(stream.read(),dtype=np.uint8)
        assert len(self.data)==len(self.targets)==n
        self.transform=transform
    def __len__(self): return len(self.targets)
    def __getitem__(self,index):
        image=Image.fromarray(self.data[index])
        return (self.transform(image) if self.transform else image),int(self.targets[index])

def save_experiment(namespace, notebook_name):
    """保存当前实验的实际标量、曲线及已训练模型；原始数据不写入检查点。"""
    import pickle
    folder=ROOT/'outputs'/notebook_name
    checkpoints=folder/'checkpoints'
    checkpoints.mkdir(parents=True,exist_ok=True)
    scalars={}
    for key in ('accuracy','best_k','best_val','test_accuracy','test_acc','val_acc','acc_tr','acc_val'):
        value=namespace.get(key)
        if isinstance(value,(int,float,np.integer,np.floating)):
            scalars[key]=float(value)
    for key in ('best_net','best_model','best_svm','best_softmax','small_rnn_model','small_lstm_model','model','net','G','D','G_LS','D_LS','G_DC','D_DC'):
        value=namespace.get(key)
        if value is None: continue
        if hasattr(value,'params'):
            with (checkpoints/f'{key}.pkl').open('wb') as stream: pickle.dump(value,stream)
        elif hasattr(value,'W'):
            np.savez(checkpoints/f'{key}.npz',W=value.W)
        elif hasattr(value,'state_dict'):
            import torch
            torch.save(value.state_dict(),checkpoints/f'{key}.pt')
    curves={}
    for key in ('loss_hist','losses','accs','stats','results','k_to_accuracies'):
        value=namespace.get(key)
        if value is not None: curves[key]=value
    with (checkpoints/'experiment_state.pkl').open('wb') as stream: pickle.dump(curves,stream)
    (folder/'metrics.json').write_text(json.dumps(scalars,ensure_ascii=False,indent=2),encoding='utf-8')
    print('本机实验指标：',scalars)
    return scalars
