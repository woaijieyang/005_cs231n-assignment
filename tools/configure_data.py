"""验证现有数据并写本机路径；不复制、不修改已有数据。"""
from pathlib import Path
import argparse, hashlib, json, sys

ROOT = Path(__file__).resolve().parents[1]

def md5(path):
    digest = hashlib.md5()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def main():
    sys.path.insert(0, str(ROOT))
    from study_runtime import data_path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cifar10', type=Path, default=data_path('cifar10'))
    parser.add_argument('--mnist', type=Path, default=data_path('mnist_gzip'))
    parser.add_argument('--shared', type=Path, default=data_path('shared'))
    args = parser.parse_args()
    # 输入为本机发现的数据目录；逐批校验官方 CIFAR-10 校验值后才注册路径。
    cifar = args.cifar10.resolve()
    checks = {'data_batch_1':'c99cafc152244af753f735de768cd75f','data_batch_2':'d4bba439e000b95fd0a9bffe97cbabec','data_batch_3':'54ebc095f3ab1f0389bbae665268c751','data_batch_4':'634d18415352ddfa80567beed471001a','data_batch_5':'482c414d41f54cd18b22e5b47cb7c3cb','test_batch':'40351d587109b95175f43aff81a1287e'}
    verified = {name: md5(cifar/name) for name in checks}
    assert verified == checks, 'CIFAR-10 checksum mismatch'
    mnist=args.mnist.resolve()
    mnist_checks={'train-images-idx3-ubyte.gz':'f68b3c2dcbeaaa9fbdd348bbdeb94873','train-labels-idx1-ubyte.gz':'d53e105ee54ea40749a09fcbcd1e9432','t10k-images-idx3-ubyte.gz':'9fb629c4189551a2d022fa330f9573f3','t10k-labels-idx1-ubyte.gz':'ec29112dd5afa0611ce80d1b7f02629c'}
    assert {name:md5(mnist/name) for name in mnist_checks}==mnist_checks
    shared=args.shared.resolve()
    shared.mkdir(exist_ok=True)
    config={'cifar10':str(cifar),'mnist_gzip':str(mnist),'coco':str(shared/'coco_captioning'),'imagenet_val':str(shared/'imagenet_val_25.npz'),'shared':str(shared)}
    (ROOT/'data_paths.local.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    report={'cifar10':{'status':'verified_existing','md5':verified},'mnist':{'status':'verified_existing','md5':mnist_checks},'coco':{'status':'present' if (shared/'coco_captioning/coco2014_captions.h5').exists() else 'missing'},'imagenet_val':{'status':'present' if (shared/'imagenet_val_25.npz').exists() else 'missing'}}
    (ROOT/'outputs').mkdir(exist_ok=True)
    (ROOT/'outputs/data_inventory.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
