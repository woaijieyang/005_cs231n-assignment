"""在冻结字幕模型上比较贪心与长度归一化束搜索；独立验证集复核。"""
import json,time
import numpy as np
import torch
from torch.nn import functional as F
from nltk.translate.bleu_score import sentence_bleu
from cs231n.classifiers.rnn import CaptioningRNN
from cs231n.coco_utils import decode_captions
from cs231n.torch_captioning import TorchCaptioner
from study_runtime import ROOT

@torch.no_grad()
def beam_decode(model,features,width=3,alpha=.6,max_length=30):
    """累计 log 概率除以长度的 alpha 次方；完成序列只允许继续 END。

    所有 beam 的循环状态分别维护。每步选出 top-k 后，序列、长度、完成标记
    以及 h/c 必须按相同父 beam 索引重排；仅排序 token 而不重排状态是错误的。
    """
    n=features.shape[0];device=features.device;vocab=len(model.word_to_idx)
    end=model.word_to_idx['<END>']
    projected=features@model.W_proj+model.b_proj
    h=projected[:,None,:].expand(n,width,-1).reshape(n*width,-1).unsqueeze(0).contiguous()
    state=h if model.cell_type=='rnn' else (h,torch.zeros_like(h))
    words=torch.full((n,width),model._start,dtype=torch.long,device=device)
    sequences=torch.full((n,width,max_length),model._null,dtype=torch.long,device=device)
    cumulative=torch.full((n,width),float('-inf'),device=device);cumulative[:,0]=0.
    lengths=torch.zeros(n,width,dtype=torch.long,device=device)
    finished=torch.zeros(n,width,dtype=torch.bool,device=device)
    batch_offsets=torch.arange(n,device=device)[:,None]*width
    for t in range(max_length):
        embedded=F.embedding(words.reshape(-1),model.W_embed).unsqueeze(1)
        hidden,next_state=model.recurrent(embedded,state)
        log_probs=F.log_softmax(hidden[:,0]@model.W_vocab+model.b_vocab,dim=-1).reshape(n,width,vocab)
        # 不改变词表/温度，也不屏蔽 UNK；只处理已经完成的候选。
        log_probs=log_probs.masked_fill(finished[:,:,None],float('-inf'))
        log_probs[:,:,end]=torch.where(finished,torch.zeros_like(cumulative),log_probs[:,:,end])
        candidate_lengths=lengths+(~finished).long()
        raw=cumulative[:,:,None]+log_probs
        normalized=raw/candidate_lengths.clamp_min(1).to(raw.dtype)[:,:,None].pow(alpha)  # 用长度归一化后的分数选择候选，但下一步继续累加的仍应是未归一化的 log 概率。
        selected=normalized.reshape(n,-1).topk(width,dim=-1).indices
        parents=selected//vocab;tokens=selected%vocab  # 展平候选索引由 父候选号×词表大小＋词号 构成，整除恢复父候选编号。
        cumulative=raw.reshape(n,-1).gather(1,selected)  # 排序可以使用长度归一化分数，但下一时间步累计的基数必须是未归一化的 log 概率。
        lengths=candidate_lengths.gather(1,parents)
        finished=finished.gather(1,parents)|(tokens==end)
        sequences=sequences.gather(1,parents[:,:,None].expand(-1,-1,max_length))  # 选择新候选后先按父 beam 复制整条历史，再写入本步 token，避免历史来自错误的句子。
        sequences[:,:,t]=tokens
        gather_indices=(parents+batch_offsets).reshape(-1)  # 加上各图片在展平批次中的偏移，确保第 n 张图只选择自己的父状态。
        state=tuple(s.index_select(1,gather_indices).contiguous() for s in next_state) if isinstance(next_state,tuple) else next_state.index_select(1,gather_indices).contiguous()  # LSTM 的 h 和 c 都必须随父候选重排；只重排输出词会把另一条句子的历史状态传下去。
        words=tokens
        if bool(finished.all()):break
    best=(cumulative/lengths.clamp_min(1).to(cumulative.dtype).pow(alpha)).argmax(dim=1)
    return sequences[torch.arange(n,device=device),best]

def score_samples(model,data,rows,width,alpha,batch_size=100):
    """固定字幕行的逐句 BLEU-1，保持与原实验相同的文本清洗协议。"""
    start=time.perf_counter();scores=[];examples=[]
    for offset in range(0,len(rows),batch_size):
        batch=rows[offset:offset+batch_size]
        features=torch.as_tensor(data['val_features'][data['val_image_idxs'][batch]],device='cuda',dtype=torch.float32)
        with torch.no_grad():
            tokens=model.sample(features) if width==0 else beam_decode(model,features,width,alpha)
        predicted=decode_captions(tokens.cpu().numpy(),data['idx_to_word'])
        reference=decode_captions(data['val_captions'][batch],data['idx_to_word'])
        for row,ref,hyp in zip(batch,reference,predicted):
            clean=lambda text:[word for word in text.split() if word not in ('<START>','<END>','<NULL>','<UNK>')]
            r,h=clean(ref),clean(hyp)
            value=float(sentence_bleu([r],h,weights=(1.0,))) if h else 0.
            scores.append(value)
            if len(examples)<20:examples.append({'caption_index':int(row),'reference':ref,'prediction':hyp,'bleu1':value})
    torch.cuda.synchronize()
    return np.asarray(scores),time.perf_counter()-start,examples

def compare_decoding(data,cell_type):
    """选参只用原 1000 验证字幕，另取图像不重叠的 1000 字幕作确认。"""
    torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False
    folder=ROOT/'outputs'/f'caption_decoding_{cell_type}'
    folder.mkdir(parents=True,exist_ok=True)
    numpy_state=np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            np.random.seed(231)
            original=CaptioningRNN(data['word_to_idx'],input_dim=512,wordvec_dim=256,hidden_dim=512,cell_type=cell_type)
            model=TorchCaptioner(original).cuda().eval()
            saved=torch.load(ROOT/'outputs'/f'full_coco_{cell_type}_gpu_equivalent/checkpoints/best.pt',map_location='cuda',weights_only=True)
            model.load_state_dict(saved['model'])
            dev=np.random.RandomState(20261102).choice(len(data['val_captions']),1000,replace=False)
            dev_images=set(data['val_image_idxs'][dev].tolist())
            eligible=np.flatnonzero(~np.isin(data['val_image_idxs'],list(dev_images)))
            confirm=np.random.RandomState(20261103).choice(eligible,1000,replace=False)
            assert not dev_images.intersection(data['val_image_idxs'][confirm].tolist())
            # width=1, alpha=0 必须与原贪心解码的可读文本完全一致。
            sample_features=torch.as_tensor(data['val_features'][data['val_image_idxs'][dev[:8]]],device='cuda',dtype=torch.float32)
            a=decode_captions(model.sample(sample_features).cpu().numpy(),data['idx_to_word'])
            b=decode_captions(beam_decode(model,sample_features,1,0).cpu().numpy(),data['idx_to_word'])
            assert a==b,'Beam width 1 differs from greedy baseline'
            candidates=[]
            for width,alpha in [(0,0),(3,0),(3,.6),(3,1.0),(5,0),(5,.6),(5,1.0)]:
                values,elapsed,_=score_samples(model,data,dev,width,alpha)
                row={'width':width,'length_alpha':alpha,'development_bleu1':float(values.mean()),'seconds':elapsed}
                candidates.append(row);print(json.dumps(row),flush=True)
            selected=max(candidates,key=lambda x:x['development_bleu1'])
            baseline,baseline_seconds,baseline_examples=score_samples(model,data,confirm,0,0)
            improved,improved_seconds,improved_examples=score_samples(model,data,confirm,selected['width'],selected['length_alpha'])
            report={'cell_type':cell_type,'model_checkpoint_epoch':saved['epoch'],'selection':'maximum development mean sentence BLEU-1; confirmation not used for selection',
                    'development_captions':1000,'confirmation_captions':1000,'image_sets_disjoint':True,
                    'width1_matches_greedy':True,'candidates':candidates,'selected':selected,
                    'confirmation_greedy_bleu1':float(baseline.mean()),'confirmation_selected_bleu1':float(improved.mean()),
                    'confirmation_mean_difference':float((improved-baseline).mean()),
                    'confirmation_greedy_seconds':baseline_seconds,'confirmation_selected_seconds':improved_seconds,
                    'per_caption_win_fraction':float(np.mean(improved>baseline)),'training_updates_added':0}
            (folder/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            (folder/'examples.json').write_text(json.dumps({'greedy':baseline_examples,'selected':improved_examples},ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps(report),flush=True)
            return report
    finally:np.random.set_state(numpy_state)
