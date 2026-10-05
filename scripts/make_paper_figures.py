"""Generate six annotated paper figures and their numerical source tables."""
from pathlib import Path
import csv
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from attentionlens.utils.io import load_tensors, load_yaml
from attentionlens.metrics.rollout import attention_rollout
from attentionlens.metrics.entropy import head_entropy
from attentionlens.metrics.head_types import head_scores, classify, TYPES
from attentionlens.viz.analysis_figures import _norm, _mean_distance

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/paper_figures'
ORDER=['gpt2','bert','retnet','rwkv']
NAMES={'gpt2':'GPT-2','bert':'BERT','retnet':'RetNet · TinyStories','rwkv':'RWKV'}
COLORS=['#3264b4','#cf7628','#8b5cb2','#26937f']
TYPE_NAMES=['First token','Previous token','Next token','Self','Local window','Broad','Other']
TYPE_COLORS=['#bd4b59','#487bbb','#4f9a8b','#9671b9','#dcaa47','#70849c','#d0d7df']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,'axes.labelsize':9,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})


def table(name,header,rows):
    with (OUT/f'{name}.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(header);writer.writerows(rows)


def save(fig,name,title,note):
    fig.get_layout_engine().set(rect=(0, .12, 1, .88))
    fig.suptitle(title,fontsize=14,fontweight='bold')
    fig.text(.5,.015,note,ha='center',va='bottom',fontsize=8,color='#4b5563')
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(OUT/f'{name}.png',dpi=400,bbox_inches='tight')
    plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    records={n:load_tensors(ROOT/f'outputs/tensors/{n}.pt') for n in ORDER}
    assert all(len(rr)>0 for rr in records.values())
    data={n:[r['attn'].float() for r in rr] for n,rr in records.items()}
    cfg=load_yaml(ROOT/'configs/classifier.yaml')
    counts={n:len(rr) for n,rr in records.items()}
    if len(set(counts.values()))!=1:raise ValueError('Use equal prompt counts for the paper comparison')
    texts=[r['text'] for r in records['gpt2']]
    assert all([r['text'] for r in rr]==texts for rr in records.values()),'Prompt datasets differ'
    P=len(texts)
    rows=[]
    fig,axes=plt.subplots(2,2,figsize=(11,10),layout='constrained')
    norm=PowerNorm(.5,vmin=0,vmax=1)
    for i,n in enumerate(ORDER):
        ax=axes.flat[i];rec=records[n][0];R=attention_rollout(rec['attn']).numpy();tokens=rec['tokens'];T=len(tokens)
        query=T-2 if n=='bert' and tokens[-1]=='[SEP]' else T-1
        masked=np.ma.masked_where(np.triu(np.ones_like(R),1).astype(bool),R) if n!='bert' else R
        cmap=plt.colormaps['Blues'].copy();cmap.set_bad('#edf0f3')
        im=ax.imshow(masked,cmap=cmap,norm=norm)
        labels=[f'{j:02d} {t}' for j,t in enumerate(tokens)]
        ax.set_xticks(range(T),labels,rotation=65,ha='right',fontsize=7)
        ax.set_yticks(range(T),labels,fontsize=7)
        ax.set_xlabel('Key · original input token');ax.set_ylabel('Query · representation token')
        ax.set_title(f'({chr(97+i)}) {NAMES[n]}\nSelected query: {labels[query]}',loc='left')
        ax.add_patch(plt.Rectangle((-.5,query-.5),T,1,fill=False,edgecolor='#cc8c2b',lw=1.8))
        for j,v in enumerate(R[query]):
            if n!='bert' and j>query:continue
            label='<0.1' if 0<v<.001 else f'{v*100:.1f}'
            ax.text(j,query,label,ha='center',va='center',fontsize=6,color='white' if v>.3 else '#152e48')
        for q in range(T):
            for k in range(T):rows.append([n,q,tokens[q],k,tokens[k],float(R[q,k]),q==query])
    fig.colorbar(im,ax=list(axes.flat),shrink=.72,ticks=[0,.01,.05,.1,.25,.5,1],format=lambda v,_:f'{100*v:g}%',label='Rollout weight · square-root colour scale')
    save(fig,'figure_1_final_rollout','1  |  Final cumulative rollout across four models',f'Prompt: {texts[0]}\nOrange row: selected final content query; numbers are percentages. Grey: causal mask. BERT includes special tokens.')
    table('figure_1_final_rollout',['model','query_index','query_token','key_index','key_token','weight','selected_query'],rows)

    entropy={n:torch.stack([head_entropy(a,n!='bert').mean(1) for a in aa]).numpy() for n,aa in data.items()}
    sink={n:torch.stack([_norm(a)[...,1:,0].mean((1,2)) for a in aa]).numpy() for n,aa in data.items()}
    for number,name,values,title,ylabel in [(2,'entropy',entropy,'Normalised attention entropy by layer','Normalised entropy'),(5,'first_token_mass',sink,'First-token attention mass by layer','First-token attention mass (%)')]:
        fig,(ax,summary)=plt.subplots(1,2,figsize=(10.5,4.9),gridspec_kw={'width_ratios':[3.4,1.6]},layout='constrained');rows=[];final=[]
        for i,n in enumerate(ORDER):
            mean=values[n].mean(0);std=values[n].std(0);x=np.arange(len(mean));scale=100 if number==5 else 1
            ax.plot(x,mean*scale,color=COLORS[i],marker='o',ms=4,lw=1.8,label=NAMES[n]);ax.fill_between(x,(mean-std)*scale,(mean+std)*scale,color=COLORS[i],alpha=.13)
            final.append([NAMES[n],f'{mean[-1]*scale:.2f}',f'{std[-1]*scale:.2f}'])
            rows.extend([[n,l,float(m),float(s),P] for l,(m,s) in enumerate(zip(mean,std))])
        ax.set_xlabel('Layer (zero-based)');ax.set_ylabel(ylabel);ax.set_xticks(x);ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=8)
        summary.axis('off');summary.set_title('Final-layer summary',fontsize=10)
        tbl=summary.table(cellText=final,colLabels=['Model','Mean','SD'],loc='center',cellLoc='center',colWidths=[.57,.22,.21]);tbl.auto_set_font_size(False);tbl.set_fontsize(8);tbl.scale(1,1.8)
        for (r,c),cell in tbl.get_celld().items():cell.set_edgecolor('#e0e5ec');cell.set_facecolor('#edf2f8' if r==0 else 'white')
        save(fig,f'figure_{number}_{name}',f'{number}  |  {title}',f'{P} identical prompts per model; lines = prompt mean; shaded bands = ±1 population SD, not confidence intervals.\n'+('Entropy excludes the first query for causal models.' if number==2 else 'First query excluded; BERT’s first token is [CLS], while causal models use the first text token.'))
        table(f'figure_{number}_{name}',['model','layer','mean','prompt_sd','prompt_count'],rows)

    classified={};headcounts={};rows=[]
    for n,aa in data.items():
        ent=torch.stack([head_entropy(a,n!='bert') for a in aa]).mean(0);ss=[head_scores(a,cfg['window']) for a in aa];scores={k:torch.stack([s[k] for s in ss]).mean(0) for k in ss[0]}
        types=classify(scores,ent,cfg);flat=[t for row in types for t in row];headcounts[n]=len(flat);classified[n]=[flat.count(t) for t in TYPES]
        rows.extend([[n,t,c,c/len(flat)*100,len(flat)] for t,c in zip(TYPES,classified[n])])
    fig,(ax,tabax)=plt.subplots(2,1,figsize=(11,6.7),gridspec_kw={'height_ratios':[2.2,1.4]},layout='constrained')
    left=np.zeros(4)
    for ti,t in enumerate(TYPES):
        v=np.array([classified[n][ti]/headcounts[n]*100 for n in ORDER]);ax.barh(range(4),v,left=left,color=TYPE_COLORS[ti],label=TYPE_NAMES[ti],height=.6,edgecolor='white')
        for j,val in enumerate(v):
            if val>=8:ax.text(left[j]+val/2,j,f'{val:.1f}%',ha='center',va='center',fontsize=8,color='#172b43')
        left+=v
    ax.set_yticks(range(4),[f'{NAMES[n]} (n={headcounts[n]})' for n in ORDER]);ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xlabel('Fraction of heads / pseudo-head groups (%)');ax.legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,1.23),frameon=False,fontsize=8)
    tabax.axis('off');tbl=tabax.table(cellText=[[f'{c/headcounts[n]*100:.1f}% ({c})' for c in classified[n]] for n in ORDER],rowLabels=[NAMES[n] for n in ORDER],colLabels=TYPE_NAMES,loc='center',cellLoc='center');tbl.auto_set_font_size(False);tbl.set_fontsize(7.5);tbl.scale(1,1.65)
    for (r,c),cell in tbl.get_celld().items():cell.set_edgecolor('#e0e5ec');cell.set_facecolor('#edf2f8' if r==0 else 'white')
    save(fig,'figure_3_head_type_distribution','3  |  Distribution of head patterns',f'Classification uses prompt-averaged scores; table entries = percentage (count). {P} prompts.\nPriority: first → previous → next → self → local → broad → other. Mass thresholds 0.5; local ±3 window threshold 0.7; broad entropy ≥0.8.')
    table('figure_3_head_type_distribution',['model','type','count','percentage','total_heads'],rows)

    distance={n:torch.stack([_mean_distance(a) for a in aa]).mean(0).numpy() for n,aa in data.items()};vmax=max(d.max() for d in distance.values());rows=[]
    fig,axes=plt.subplots(2,2,figsize=(10,9),layout='constrained')
    for i,n in enumerate(ORDER):
        ax=axes.flat[i];d=distance[n];im=ax.imshow(d*100,cmap='cividis',vmin=0,vmax=vmax*100,aspect='auto')
        ax.set_title(f'({chr(97+i)}) {NAMES[n]} · mean {d.mean()*100:.2f}%',loc='left');ax.set_xlabel('Head / pseudo-head group');ax.set_ylabel('Layer');ax.set_xticks(range(d.shape[1]));ax.set_yticks(range(d.shape[0]))
        for l in range(d.shape[0]):
            for h in range(d.shape[1]):
                ax.text(h,l,f'{d[l,h]*100:.1f}',ha='center',va='center',fontsize=6,color='white' if d[l,h]<vmax*.5 else '#172b43');rows.append([n,l,h,float(d[l,h]),P])
    fig.colorbar(im,ax=list(axes.flat),shrink=.75,label='Relative attention distance (%)')
    save(fig,'figure_4_relative_distance','4  |  Relative attention distance by layer and head',f'Cell values: 100 × mean_query Σ_key p(query,key)|query−key|/(T−1), averaged over {P} prompts.\nShared colour scale. Distances are tokenizer-relative, not word distances.')
    table('figure_4_relative_distance',['model','layer','head','relative_distance','prompt_count'],rows)

    fig,(ax,tabax)=plt.subplots(2,1,figsize=(10.5,7),gridspec_kw={'height_ratios':[2.5,1.7]},layout='constrained');rows=[];table_rows=[]
    for i,n in enumerate(ORDER):
        values=[];eligible=[]
        for offset in range(11):
            per=[torch.diagonal(_norm(a),-offset,-2,-1).mean().item() for a in data[n] if a.shape[-1]>offset]
            value=float(np.mean(per)) if per else np.nan;values.append(value);eligible.append(len(per));rows.append([n,offset,value,len(per)])
        ax.plot(range(11),values,color=COLORS[i],marker='o',ms=4,label=NAMES[n]);table_rows.append([f'{v*100:.2f}\n(n={c})' if c else 'N/A' for v,c in zip(values,eligible)])
    ax.set_yscale('log');ax.set_xticks(range(11));ax.set_xlabel('Lookback offset d = query position − key position');ax.set_ylabel('Mean attention weight (log scale)');ax.grid(alpha=.15,which='both');ax.legend(frameon=False,ncol=2)
    tabax.axis('off');tbl=tabax.table(cellText=table_rows,rowLabels=[NAMES[n] for n in ORDER],colLabels=[f'd={i}' for i in range(11)],loc='center',cellLoc='center');tbl.auto_set_font_size(False);tbl.set_fontsize(6.6);tbl.scale(1,2.5)
    for (r,c),cell in tbl.get_celld().items():cell.set_edgecolor('#e0e5ec');cell.set_facecolor('#edf2f8' if r==0 else 'white')
    save(fig,'figure_6_lookback_offset','6  |  Attention weight versus lookback distance','Each prompt contributes its mean over eligible queries, heads and layers; prompts are weighted equally.\nTable: mean weight in percent; n = eligible prompt count. Only past/current offsets shown, including for bidirectional BERT.')
    table('figure_6_lookback_offset',['model','lookback_offset','mean_weight','eligible_prompts'],rows)
    (OUT/'captions_and_methods.md').write_text(f'''# Paper figures 1–6

All aggregate comparisons use the same {P} prompts, with equal prompt weights. Every figure has a vector PDF, 400-dpi PNG, and CSV source table.

1. Final rollout: illustrative first saved prompt, indexed model-specific tokens. Selected query is the final text token, excluding BERT's [SEP]. Only that row is numerically annotated. Shared square-root colour mapping; percentages refer to original rollout weights. Rollout averages heads, adds a residual identity connection and multiplies layer transitions.
2. Entropy: per-row entropy divided by log of available context size; first causal row excluded. Head mean per layer, then prompt mean ±1 population SD. Final-layer table gives mean and SD.
3. Head types: classify prompt-averaged mass scores and entropy with configs/classifier.yaml. Categories are mutually exclusive through priority ordering. Table includes every category's percentage and count; RWKV groups are pseudo-heads.
4. Relative distance: expected absolute token offset divided by T−1, mean over queries and prompts. Shared colour scale; every cell annotated in percent. Model means additionally average layers and heads.
5. First-token mass: mean over all heads and queries except query zero, then prompt mean ±1 population SD. The first token is BERT's [CLS], versus the first text token for causal models; these are not identical semantic positions.
6. Lookback profile: mean diagonal weight over layers, heads, and eligible queries in each prompt, then mean across eligible prompts. Logarithmic y-axis; table reports exact percentages to two decimal places and eligible prompt counts. Forward-looking BERT weights are outside this figure.

## Interpretation and provenance
GPT-2 and BERT use native softmax attention. RetNet uses signed retention magnitudes, and RWKV uses channel-grouped implicit weights; these analogues are row-normalised for metrics. Training corpora, objectives, tokenizer boundaries, special tokens, and checkpoint maturity differ. These are descriptive pattern comparisons, not controlled evidence of architectural superiority or prediction causality.

RetNet: user-provided TinyStories-trained checkpoint; user-reported validation loss 4.434524583816528. training_metadata.json records 100 optimizer steps, sequence length 128, batch size 1 and accumulation 8; GPU and validation sample size are not recorded. The checkpoint loads strictly with 768-dimensional value projections. All input tensors were saved in half precision by the extraction pipeline, then converted to float32 for figure calculations.
''')
    print('Saved six paper figures, six CSV tables, and methods notes to',OUT)

if __name__=='__main__':main()
