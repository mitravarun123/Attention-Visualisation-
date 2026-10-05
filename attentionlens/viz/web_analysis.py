"""Interactive dataset analysis figures for the React workspace."""
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from attentionlens.utils.io import load_tensors, load_yaml
from attentionlens.metrics.rollout import attention_rollout
from attentionlens.metrics.entropy import head_entropy
from attentionlens.metrics.head_types import head_scores, classify, TYPES
from attentionlens.viz.analysis_figures import head_features, _mean_distance, _norm
from attentionlens.viz.heatmaps import matrix_trace, token_labels

COLORS=['#386de5','#ed9344','#2bab90','#ac79dc']
TYPE_COLORS=['#d95d6b','#467ee5','#4dac9a','#ad7dd4','#e4aa42','#718caa','#c5ccd5']
VIEWS={
 'all_models_rollout':('Final rollout comparison','Same saved prompt, each model’s own tokenisation. Square-root colours reveal small weights; hover shows actual percentages.'),
 'layer_snapshots':('Cumulative rollout snapshots','Selected layers show how head-averaged attention and residual paths mix through the model.'),
 'head_type_map':('Head-type map','Classification of mean head scores across saved prompts. Hover for scores and entropy; categories follow the configured priority order.'),
 'offset_profile':('Attention vs lookback offset','Mean weight at query-to-key lookback d, averaged over heads, layers and eligible prompts. Positive d looks into the past.'),
 'distance_heatmap':('Relative attention distance','Mean absolute query–key distance divided by T−1, averaged equally over prompts. Zero means self-attention.'),
 'sink_by_layer':('First-token attention mass','Mean mass assigned to the first token, excluding the first query. Bands show ±1 population standard deviation across prompts.'),
 'head_atlas':('Head atlas · PCA','PCA of standardised prompt-averaged head features. Colour identifies model; marker identifies head type. Nearby points have similar measured patterns.'),
 'entropy_bands':('Entropy by layer · prompt variance','Mean normalised entropy over heads, averaged over prompts. Bands show ±1 population standard deviation, not confidence intervals.'),
}

@lru_cache(maxsize=1)
def dataset(root):
    data={}
    for path in sorted((Path(root)/'outputs/tensors').glob('*.pt')):
        records=load_tensors(path)
        if records: data[path.stem]=records
    return data


def plot(root, view, model, dark=False):
    records=dataset(root)
    if not records: raise ValueError('No saved tensors. Run extraction first.')
    if view not in VIEWS: raise ValueError('Unknown analysis view')
    if model not in records: raise ValueError('No saved dataset for selected model')
    cfg=load_yaml(Path(root)/'configs/classifier.yaml')
    data={n:[r['attn'].float() for r in rr] for n,rr in records.items()}
    def summaries(n):
        aa=data[n]
        ent=torch.stack([head_entropy(a,n!='bert') for a in aa]).mean(0)
        ss=[head_scores(a,cfg['window']) for a in aa]
        scores={k:torch.stack([s[k] for s in ss]).mean(0) for k in ss[0]}
        return ent,scores,classify(scores,ent,cfg)
    if view in ('all_models_rollout','layer_snapshots'):
        if view=='all_models_rollout':
            text=records[model][0]['text']
            rr={n:next((r for r in rows if r['text']==text),None) for n,rows in records.items()}
            rr={n:r for n,r in rr.items() if r is not None}
            panels=[(n,attention_rollout(r['attn']),r['tokens'],n!='bert') for n,r in rr.items()]
        else:
            r=records[model][0];cum=attention_rollout(r['attn'],True);L=len(cum)
            panels=[(f'Layer {l}',cum[l],r['tokens'],model!='bert') for l in sorted({0,L//4,L//2,3*L//4,L-1})]
        cols=min(2,len(panels));rows=(len(panels)+cols-1)//cols
        fig=make_subplots(rows=rows,cols=cols,subplot_titles=[p[0] for p in panels],horizontal_spacing=.14,vertical_spacing=.14)
        for i,(name,R,tokens,causal) in enumerate(panels):
            row,col=i//cols+1,i%cols+1
            fig.add_trace(matrix_trace(R,tokens,causal,'Square root',showscale=i==0),row=row,col=col)
            labels=token_labels(tokens)
            fig.update_xaxes(tickvals=list(range(len(tokens))),ticktext=labels,tickangle=-55,row=row,col=col)
            fig.update_yaxes(tickvals=list(range(len(tokens))),ticktext=labels,autorange='reversed',row=row,col=col)
        fig.update_layout(height=460*rows,title=dict(text=records[model][0]['text'],font=dict(size=13)))
    elif view in ('head_type_map','distance_heatmap'):
        fig=make_subplots(rows=1,cols=len(data),subplot_titles=list(data),horizontal_spacing=.09)
        for i,n in enumerate(data):
            ent,scores,types=summaries(n)
            if view=='distance_heatmap':
                z=torch.stack([_mean_distance(a) for a in data[n]]).mean(0).numpy()
                trace=go.Heatmap(z=z,zmin=0,zmax=1,colorscale='Blues',showscale=i==len(data)-1,
                    colorbar=dict(title='Relative distance'),hovertemplate='Layer %{y}, head %{x}<br>Distance %{z:.4f}<extra>'+n+'</extra>')
            else:
                z=[[TYPES.index(t) for t in row] for row in types]
                custom=[[[types[l][h],ent[l,h].item(),*[scores[k][l,h].item() for k in TYPES[:5]]] for h in range(ent.shape[1])] for l in range(ent.shape[0])]
                scale=[]
                for j,c in enumerate(TYPE_COLORS): scale.extend([(j/7,c),((j+1)/7,c)])
                trace=go.Heatmap(z=z,zmin=-.5,zmax=6.5,colorscale=scale,customdata=custom,showscale=i==len(data)-1,
                    colorbar=dict(tickvals=list(range(7)),ticktext=TYPES,title='Head type'),
                    hovertemplate='Layer %{y}, head %{x}<br>%{customdata[0]}<br>Entropy %{customdata[1]:.3f}<br>First %{customdata[2]:.3f}<br>Previous %{customdata[3]:.3f}<br>Next %{customdata[4]:.3f}<br>Self %{customdata[5]:.3f}<br>Local %{customdata[6]:.3f}<extra>'+n+'</extra>')
            fig.add_trace(trace,row=1,col=i+1)
            fig.update_xaxes(title='Head',dtick=1,row=1,col=i+1);fig.update_yaxes(title='Layer',dtick=1,autorange='reversed',row=1,col=i+1)
        fig.update_layout(height=600)
    elif view in ('offset_profile','sink_by_layer','entropy_bands'):
        fig=go.Figure()
        for i,(n,aa) in enumerate(data.items()):
            if view=='offset_profile':
                ys=[];counts=[]
                for d in range(11):
                    vals=[torch.diagonal(_norm(a),-d,-2,-1).mean().item() for a in aa if a.shape[-1]>d]
                    ys.append(float(np.mean(vals)) if vals else None);counts.append(len(vals))
                fig.add_trace(go.Scatter(x=list(range(11)),y=ys,name=n,mode='lines+markers',line=dict(color=COLORS[i%4]),customdata=counts,
                    hovertemplate='Lookback %{x}<br>Weight %{y:.4f}<br>Eligible prompts %{customdata}<extra>'+n+'</extra>'))
            else:
                vals=torch.stack([head_entropy(a,n!='bert').mean(1) if view=='entropy_bands' else
                    (_norm(a)[...,1:,0].mean((1,2)) if a.shape[-1]>1 else torch.zeros(a.shape[0])) for a in aa])
                mean=vals.mean(0).numpy();std=vals.std(0,unbiased=False).numpy();x=list(range(len(mean)));color=COLORS[i%4]
                fig.add_trace(go.Scatter(x=x+x[::-1],y=(mean+std).tolist()+(mean-std).tolist()[::-1],fill='toself',fillcolor=color,opacity=.12,line=dict(width=0),hoverinfo='skip',showlegend=False))
                fig.add_trace(go.Scatter(x=x,y=mean,name=n,mode='lines+markers',line=dict(color=color),customdata=std,
                    hovertemplate='Layer %{x}<br>Mean %{y:.4f}<br>Prompt std %{customdata:.4f}<extra>'+n+'</extra>'))
        fig.update_layout(height=580,xaxis_title='Lookback offset d' if view=='offset_profile' else 'Layer',yaxis_title='Mean attention weight' if view!='entropy_bands' else 'Normalised entropy')
        if view=='offset_profile':fig.update_yaxes(type='log')
    else:
        feats=[];labels=[];meta=[]
        for n,aa in data.items():
            f=torch.stack([head_features(a,n!='bert') for a in aa]).mean(0)
            feats.append(f.reshape(-1,f.shape[-1]));types=summaries(n)[2]
            for l in range(f.shape[0]):
                for h in range(f.shape[1]):meta.append((n,l,h));labels.append(types[l][h])
        X=torch.cat(feats);X=(X-X.mean(0))/X.std(0,unbiased=False).clamp(min=1e-6)
        U,S,_=torch.linalg.svd(X,full_matrices=False);Z=(U[:,:2]*S[:2]).numpy();var=(S.square()/S.square().sum().clamp(min=1e-12))[:2]
        fig=go.Figure();symbols=['circle','square','triangle-up','diamond','cross','star','x']
        for i,n in enumerate(data):
            for ti,t in enumerate(TYPES):
                ids=[j for j,(nn,_,_) in enumerate(meta) if nn==n and labels[j]==t]
                if not ids:continue
                fig.add_trace(go.Scatter(x=Z[ids,0],y=Z[ids,1],mode='markers',name=f'{n} · {t}',marker=dict(color=COLORS[i%4],symbol=symbols[ti],size=9,opacity=.8),customdata=[[meta[j][1],meta[j][2],t] for j in ids],
                    hovertemplate='Layer %{customdata[0]}, head %{customdata[1]}<br>%{customdata[2]}<br>PC1 %{x:.3f}, PC2 %{y:.3f}<extra>'+n+'</extra>'))
        fig.update_layout(height=680,xaxis_title=f'PC1 ({var[0]:.1%} variance)',yaxis_title=f'PC2 ({var[1]:.1%} variance)')
    fig.update_layout(template='plotly_dark' if dark else 'plotly_white',paper_bgcolor='#182233' if dark else '#ffffff',plot_bgcolor='#182233' if dark else '#ffffff',font=dict(family='Arial',size=11),margin=dict(l=65,r=80,t=65,b=85),legend=dict(orientation='h',y=-.2))
    return fig
