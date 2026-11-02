"""独立验证关键修复：框架自动求导作为手写层的交叉检查。"""
from pathlib import Path
import json,sys
import numpy as np
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'02_Assignment2'))
from cs231n.optim import adam
from cs231n.classifiers.cnn import ThreeLayerConvNet
from cs231n.classifiers.fc_net import FullyConnectedNet

def tensor(value):return torch.tensor(value,dtype=torch.float64,requires_grad=True)

def compare_gradients(model,values,labels,forward):
    expected_loss,expected=model.loss(values,labels)
    params={k:tensor(v) for k,v in model.params.items()}
    x=torch.tensor(values,dtype=torch.float64);y=torch.tensor(labels,dtype=torch.long)
    loss=F.cross_entropy(forward(x,params),y)
    for key,value in params.items():
        if key.startswith('W'):loss=loss+.5*model.reg*(value**2).sum()
    loss.backward()
    np.testing.assert_allclose(loss.item(),expected_loss,atol=1e-10,rtol=1e-10)
    errors={}
    for key,value in params.items():
        actual=value.grad.numpy()
        np.testing.assert_allclose(actual,expected[key],atol=1e-9,rtol=1e-7)
        errors[key]={'max_absolute_error':float(np.max(abs(actual-expected[key]))),'gradient_max_absolute_value':float(np.max(abs(expected[key])))}
    return {'loss_absolute_error':abs(loss.item()-expected_loss),'gradients':errors}

def main():
    np.random.seed(231);torch.set_num_threads(2)
    # 不依赖四舍五入到八位小数的参考表，直接对照双精度多步 Adam。
    weights=np.array([.2,-.3,.04]);configuration={};p=tensor(weights);optimizer=torch.optim.Adam([p],lr=1e-3)
    for gradient in (np.array([.1,-.2,.3]),np.array([-.3,.4,-.1]),np.array([.05,.07,.09])):
        weights,configuration=adam(weights,gradient,configuration)
        p.grad=torch.tensor(gradient);optimizer.step()
        np.testing.assert_allclose(weights,p.detach().numpy(),atol=1e-12,rtol=1e-12)
    assert configuration['t']==3
    # 运行验证原卷积 notebook 梯度检查的实际初始化与尺寸，定位相对误差。
    np.random.seed(231)
    x=np.random.randn(2,3,16,16);y=np.random.randint(10,size=2)
    cnn=ThreeLayerConvNet(input_dim=(3,16,16),num_filters=3,filter_size=3,hidden_dim=7,num_classes=10,dtype=np.float64,reg=0.)
    def cnn_forward(x,p):
        x=F.max_pool2d(F.relu(F.conv2d(x,p['W1'],p['b1'],padding=1)),2)
        return F.relu(x.flatten(1)@p['W2']+p['b2'])@p['W3']+p['b3']
    cnn_result=compare_gradients(cnn,x,y,cnn_forward)
    fc=FullyConnectedNet([20,30],input_dim=15,num_classes=10,use_batchnorm=True,dtype=np.float64,weight_scale=.05,reg=.1)
    x=np.random.randn(4,15);y=np.array([1,2,3,4])
    def bn_forward(x,p):
        for i in (1,2):
            x=x@p[f'W{i}']+p[f'b{i}']
            x=(x-x.mean(dim=0))/torch.sqrt(x.var(dim=0,unbiased=False)+1e-5)
            x=F.relu(x*p[f'gamma{i}']+p[f'beta{i}'])
        return x@p['W3']+p['b3']
    bn_result=compare_gradients(fc,x,y,bn_forward)
    report={'adam_three_step_torch_agreement':True,'cnn_autograd_comparison':cnn_result,'batchnorm_network_autograd_comparison':bn_result}
    (ROOT/'outputs/core_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
