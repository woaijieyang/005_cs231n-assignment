"""与课程 NumPy RNN/LSTM 对应的 GPU 实现。

NumPy LSTM 门顺序为 i,f,o,g；PyTorch 为 i,f,g,o。导入参数和导出梯度
时都显式换序。第二组循环偏置固定为零，避免增加原模型没有的可训练参数。
"""
import copy, json, os, time
import functools, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from nltk.translate.bleu_score import sentence_bleu
from cs231n.classifiers.rnn import CaptioningRNN
from cs231n.coco_utils import decode_captions
from study_runtime import ROOT

def isolated_random_state(function):
    """方法二运行前后恢复外部随机状态，训练内部仍独立保存自身 RNG 检查点。"""
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        numpy_state=np.random.get_state();python_state=random.getstate()
        try:
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                return function(*args, **kwargs)
        finally:
            np.random.set_state(numpy_state);random.setstate(python_state)
    return wrapped

def gate_permutation(array):
    """最后一维四块按 0,1,3,2 重排；该置换恰好也是自己的逆。"""
    chunks=np.split(array,4,axis=-1)
    return np.concatenate([chunks[i] for i in (0,1,3,2)],axis=-1)

class TorchCaptioner(nn.Module):
    """图像特征→h0；词嵌入→循环层→词表 logits，与原模型参数逐一对应。"""
    def __init__(self,numpy_model):
        super().__init__()
        p=numpy_model.params
        self.cell_type=numpy_model.cell_type
        self._null=numpy_model._null;self._start=numpy_model._start
        self.word_to_idx=numpy_model.word_to_idx
        for key in ('W_embed','W_proj','b_proj','W_vocab','b_vocab'):
            setattr(self,key,nn.Parameter(torch.from_numpy(p[key].copy())))
        hidden=p['Wh'].shape[0];worddim=p['Wx'].shape[0]
        layer=nn.RNN if self.cell_type=='rnn' else nn.LSTM
        self.recurrent=layer(worddim,hidden,batch_first=True).to(dtype=self.W_embed.dtype)
        change=gate_permutation if self.cell_type=='lstm' else lambda x:x
        with torch.no_grad():
            self.recurrent.weight_ih_l0.copy_(torch.from_numpy(change(p['Wx']).T.copy()))  # 将课程 Wx 按门序重排并转置，匹配框架循环层按输出维排在前面的存储布局。
            self.recurrent.weight_hh_l0.copy_(torch.from_numpy(change(p['Wh']).T.copy()))  # Wh 使用同一门块交换规则；只交换输入权重而漏掉循环权重会使门的语义不一致。
            self.recurrent.bias_ih_l0.copy_(torch.from_numpy(change(p['b']).copy()))
            self.recurrent.bias_hh_l0.zero_()  # 课程只有一组偏置。将第二组置零并在后面冻结，避免多学一份课程中不存在的偏置。
        self.recurrent.bias_hh_l0.requires_grad_(False)

    def forward(self,features,caption_inputs):
        # h0 的 shape 为 (1,N,H)，循环层的输出为 (N,T,H)。
        h0=(features@self.W_proj+self.b_proj).unsqueeze(0)
        initial=h0 if self.cell_type=='rnn' else (h0,torch.zeros_like(h0))
        embedded=F.embedding(caption_inputs,self.W_embed)
        hidden,_=self.recurrent(embedded,initial)
        return hidden@self.W_vocab+self.b_vocab

    def loss(self,features,captions):
        # teacher forcing：输入去掉末词，目标去掉 START；NULL 不计入损失。
        logits=self(features,captions[:,:-1])
        return F.cross_entropy(logits.reshape(-1,logits.shape[-1]),captions[:,1:].reshape(-1),ignore_index=self._null,reduction='sum')/len(captions)

    @torch.no_grad()
    def sample(self,features,max_length=30):
        h0=(features@self.W_proj+self.b_proj).unsqueeze(0)
        state=h0 if self.cell_type=='rnn' else (h0,torch.zeros_like(h0))
        word=torch.full((len(features),),self._start,device=features.device,dtype=torch.long)
        captions=[]
        for _ in range(max_length):
            hidden,state=self.recurrent(F.embedding(word,self.W_embed).unsqueeze(1),state)
            word=(hidden[:,0]@self.W_vocab+self.b_vocab).argmax(dim=-1)
            captions.append(word)
        return torch.stack(captions,dim=1)

    def numpy_gradients(self):
        """把框架梯度映射回课程参数名，供独立的数值一致性验证。"""
        convert=lambda t:t.detach().cpu().numpy()
        result={key:convert(getattr(self,key).grad) for key in ('W_embed','W_proj','b_proj','W_vocab','b_vocab')}
        change=gate_permutation if self.cell_type=='lstm' else lambda x:x
        result.update(Wx=change(convert(self.recurrent.weight_ih_l0.grad).T),Wh=change(convert(self.recurrent.weight_hh_l0.grad).T),b=change(convert(self.recurrent.bias_ih_l0.grad)))
        return result

class CourseAdam(torch.optim.Optimizer):
    """运行验证 Assignment 3 优化器的 alpha 与 epsilon 放置方式，不隐式换公式。"""
    def __init__(self,params,lr=5e-4):
        super().__init__(params,dict(lr=lr,beta1=.9,beta2=.999,eps=1e-8))
    @torch.no_grad()
    def step(self,closure=None):
        for group in self.param_groups:
            b1,b2=group['beta1'],group['beta2']
            for param in group['params']:
                if param.grad is None:continue
                state=self.state[param]
                if not state:state.update(t=0,m=torch.zeros_like(param),v=torch.zeros_like(param))
                state['t']+=1
                state['m'].mul_(b1).add_(param.grad,alpha=1-b1)
                state['v'].mul_(b2).addcmul_(param.grad,param.grad,value=1-b2)
                alpha=group['lr']*((1-b2**state['t'])**.5)/(1-b1**state['t'])
                param.addcdiv_(state['m'],state['v'].sqrt().add_(group['eps']),value=-alpha)

@isolated_random_state
def verify_equivalence(device='cpu'):
    """固定输入、权重、mask，检查损失、每个参数梯度和贪心解码。"""
    report={}
    for kind in ('rnn','lstm'):
        np.random.seed(231)
        vocabulary={'<NULL>':0,'<START>':1,'<END>':2,'cat':3,'dog':4}
        original=CaptioningRNN(vocabulary,input_dim=7,wordvec_dim=6,hidden_dim=8,cell_type=kind,dtype=np.float64)
        features=np.random.randn(3,7)
        captions=np.array([[1,3,4,2,0],[1,4,2,0,0],[1,3,3,4,2]],dtype=np.int64)
        expected_loss,expected_grads=original.loss(features,captions)
        model=TorchCaptioner(original).to(device)
        features_t=torch.tensor(features,device=device)
        captions_t=torch.tensor(captions,device=device)
        loss=model.loss(features_t,captions_t);loss.backward()
        np.testing.assert_allclose(loss.item(),expected_loss,rtol=1e-10,atol=1e-10)
        errors={}
        for key,value in model.numpy_gradients().items():
            np.testing.assert_allclose(value,expected_grads[key],rtol=1e-8,atol=1e-10)
            errors[key]=float(np.max(abs(value-expected_grads[key])))
        np.testing.assert_array_equal(model.sample(features_t,6).cpu().numpy(),original.sample(features,6))
        # 另外对照 optimizer 的多步状态更新，包含 epsilon 位置和 t 累加。
        from cs231n.optim import adam
        parameter=nn.Parameter(torch.tensor([.2,-.3],dtype=torch.float64,device=device));optimizer=CourseAdam([parameter])
        value=np.array([.2,-.3]);state={'learning_rate':5e-4}
        for gradient in (np.array([.1,-.2]),np.array([-.3,.4]),np.array([.05,.07])):
            parameter.grad=torch.tensor(gradient,device=device);optimizer.step();value,state=adam(value,gradient,state)
            np.testing.assert_allclose(parameter.detach().cpu().numpy(),value,rtol=1e-12,atol=1e-12)
        report[kind]={'loss_absolute_error':abs(loss.item()-expected_loss),'gradient_max_absolute_errors':errors,'greedy_decoding_equal':True,'adam_three_steps_equal':True,'device':device}
    return report

@torch.no_grad()
def evaluate(model,features,captions,image_indices,vocabulary,batch_size=100):
    """固定验证字幕的平均 loss 与单参考句 BLEU-1；不修改任何训练状态。"""
    model.eval();total=0.;bleus=[];examples=[]
    for start in range(0,len(captions),batch_size):
        targets=captions[start:start+batch_size];images=features[image_indices[start:start+batch_size]]
        total+=model.loss(images,targets).item()*len(targets)
        sampled=model.sample(images).cpu().numpy()
        predictions=decode_captions(sampled,vocabulary)
        references=decode_captions(targets.cpu().numpy(),vocabulary)
        for ref,hyp in zip(references,predictions):
            clean=lambda s:[w for w in s.split() if w not in ('<NULL>','<START>','<END>','<UNK>')]
            r,h=clean(ref),clean(hyp)
            score=float(sentence_bleu([r],h,weights=(1.0,))) if h else 0.
            bleus.append(score)
            if len(examples)<20:examples.append({'reference':ref,'prediction':hyp,'bleu1':score})
    return total/len(captions),float(np.mean(bleus)),examples

@isolated_random_state
def train_gpu(data,cell_type,epochs=20,batch_size=100,seed=231):
    """完整训练数据每轮无放回遍历，保存模型/优化器/RNG/排列/进度以恢复。"""
    if not torch.cuda.is_available():raise RuntimeError('完整 GPU 实验要求实际可用的 CUDA 设备')
    device='cuda';torch.set_num_threads(4)
    # 禁用 TF32，先保留与 NumPy float32 更直接的数值对应；不使用混合精度。
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.manual_seed(seed);np.random.seed(seed);rng=np.random.RandomState(seed)
    original=CaptioningRNN(data['word_to_idx'],input_dim=512,wordvec_dim=256,hidden_dim=512,cell_type=cell_type,dtype=np.float32)
    model=TorchCaptioner(original).to(device)
    optimizer=CourseAdam(model.parameters(),lr=5e-4)
    folder=ROOT/'outputs'/f'full_coco_{cell_type}_gpu_equivalent'
    ckpt=folder/'checkpoints';ckpt.mkdir(parents=True,exist_ok=True)
    config={'backend':'PyTorch CUDA, equivalent course equations','cell_type':cell_type,'epochs':epochs,'batch_size':batch_size,'seed':seed,'wordvec_dim':256,'hidden_dim':512,'learning_rate':5e-4,'dtype':'float32','optimizer':'course Adam, original epsilon placement','train_captions':len(data['train_captions']),'validation':'fixed 1000 captions, seed 20261102; mean single-reference sentence BLEU-1'}
    (folder/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    features=torch.as_tensor(data['train_features'],device=device,dtype=torch.float32)
    captions=torch.as_tensor(data['train_captions'],device=device,dtype=torch.long)
    indices=torch.as_tensor(data['train_image_idxs'],device=device,dtype=torch.long)
    selected=np.random.RandomState(20261102).choice(len(data['val_captions']),1000,replace=False)
    vf=torch.as_tensor(data['val_features'],device=device,dtype=torch.float32)
    vc=torch.as_tensor(data['val_captions'][selected],device=device,dtype=torch.long)
    vi=torch.as_tensor(data['val_image_idxs'][selected],device=device,dtype=torch.long)
    n=len(captions);epoch=0;offset=0;step=0;order=rng.permutation(n);history=[];best_loss=float('inf');elapsed_prior=0.
    latest=ckpt/'latest.pt';best=ckpt/'best.pt'
    if latest.exists():
        state=torch.load(latest,map_location=device,weights_only=False)
        if state['config']!=config:raise ValueError('Resume config mismatch')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        epoch=state['epoch'];offset=state['offset'];step=state['step'];order=state['order'];history=state['history'];best_loss=state['best_loss'];elapsed_prior=state['elapsed_seconds'];rng.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'].cpu());torch.cuda.set_rng_state(state['cuda_rng'].cpu())
        if epoch==epochs and (folder/'metrics.json').exists():
            selected_model=torch.load(best,map_location=device,weights_only=True)
            model.load_state_dict(selected_model['model'])
            report=json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
            # 已完成实验只加载结果；保留原训练耗时，不把阅读记录算作新训练。
            return model,report,history
    started=time.perf_counter()
    def elapsed():return elapsed_prior+time.perf_counter()-started
    def save():
        state={'config':config,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'offset':offset,'step':step,'order':order,'history':history,'best_loss':best_loss,'elapsed_seconds':elapsed(),'numpy_rng':rng.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()}
        temporary=latest.with_suffix('.tmp');torch.save(state,temporary);os.replace(temporary,latest)
    def emit(record):
        record={**record,'elapsed_seconds':elapsed()}
        with (folder/'training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
    emit({'event':'resume' if step else 'start','epoch':epoch,'step':step,'gpu':torch.cuda.get_device_name()})
    while epoch<epochs:
        model.train()
        while offset<n:
            rows=torch.as_tensor(order[offset:offset+batch_size],device=device)
            optimizer.zero_grad(set_to_none=True)  # 清除上一个批次累积的梯度，再为当前 batch 计算新的梯度。
            loss=model.loss(features[indices[rows]],captions[rows])
            if not torch.isfinite(loss):save();raise FloatingPointError('nonfinite training loss')
            loss.backward();optimizer.step();step+=1;offset+=len(rows)  # 从标量目标沿计算图计算梯度；参数更新发生在后面的优化器 step。
            if step%200==0:emit({'event':'batch','epoch':epoch+1,'step':step,'offset':offset,'loss':loss.item()})
            if step%1000==0:save()
        val_loss,bleu,examples=evaluate(model,vf,vc,vi,data['idx_to_word'])
        epoch+=1
        record={'epoch':epoch,'train_loss_last_batch':loss.item(),'validation_loss':val_loss,'validation_sentence_bleu1':bleu,'step':step}
        history.append(record)
        if val_loss<best_loss:
            best_loss=val_loss;torch.save({'model':model.state_dict(),'epoch':epoch,'validation_loss':val_loss,'validation_sentence_bleu1':bleu},best)
        emit({'event':'epoch_complete',**record})
        (folder/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        (folder/'caption_examples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2),encoding='utf-8')
        offset=0;order=rng.permutation(n);save()
    selected_model=torch.load(best,map_location=device,weights_only=True);model.load_state_dict(selected_model['model'])
    val_loss,bleu,examples=evaluate(model,vf,vc,vi,data['idx_to_word'])
    report={'status':'complete','backend':'equivalent GPU; original NumPy full training not claimed','epochs_completed':epoch,'steps_completed':step,'train_captions_per_epoch':n,'best_epoch':selected_model['epoch'],'validation_loss':val_loss,'validation_sentence_bleu1':bleu,'validation_captions_evaluated':1000,'elapsed_seconds':elapsed(),'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated()}
    (folder/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (folder/'caption_examples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2),encoding='utf-8')
    emit({'event':'complete',**report})
    return model,report,history
