"""Causal graph-state/innovation operators and adaptive clinical FiLM.

Additional math operators are experimental hypotheses, not a novelty claim.
All derivatives use adjacent INPUT visits only. Node indices must be identical,
and edge comparisons require ordered, identical, observed fixed support.
"""
from __future__ import annotations
import math
from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F


def _months(g):
    t=getattr(g,'visit_months',None)
    if t is None: raise ValueError('Missing input-visit date (visit_months)')
    v=float(torch.as_tensor(t).reshape(-1)[0])
    if not math.isfinite(v):raise ValueError('Non-finite visit_months')
    return v


def _edges_match(a,b):
    return a.edge_index.shape==b.edge_index.shape and torch.equal(a.edge_index,b.edge_index)


def _feature_width(node_math='rate'):
    if node_math not in ('rate','laplacian','rate_laplacian','flux','rate_flux'):
        raise ValueError(f'Unknown node math {node_math}')
    return 10 if node_math in ('rate_laplacian','rate_flux') else 6


def graph_laplacian_rate(node_rate, curr):
    """Random-walk graph high pass: delta X - D^-1 W delta X.

    Uses only current input-visit nonnegative tract weights and within-visit
    neighbors. Self-loops are excluded. Duplicate directed edges follow the
    same semantics as the GNN edge_index. Isolated nodes have no neighbor
    message: use zero high-pass there to avoid fake evidence.
    """
    n=int(node_rate.shape[0]);idx=curr.edge_index
    if idx.dim()!=2 or idx.shape[0]!=2:raise ValueError('Malformed edge_index')
    if idx.numel()==0:return torch.zeros_like(node_rate)
    src,dst=idx[0].long(),idx[1].long()
    valid=src!=dst
    src,dst=src[valid],dst[valid]
    if src.numel()==0:return torch.zeros_like(node_rate)
    weights=getattr(curr,'edge_weight',None)
    if weights is None:
        w=torch.ones(src.numel(),device=node_rate.device,dtype=node_rate.dtype)
    else:
        w=weights[valid].to(node_rate.dtype).to(node_rate.device)
        w=torch.nan_to_num(w,nan=0.,posinf=0.,neginf=0.).clamp_min(0.)
    degree=torch.zeros(n,device=node_rate.device,dtype=node_rate.dtype)
    degree.index_add_(0,src,w)
    neighbor=torch.zeros_like(node_rate)
    neighbor.index_add_(0,src,node_rate[dst]*w.unsqueeze(-1))
    neighbor=neighbor/degree.clamp_min(1e-8).unsqueeze(-1)
    return torch.where((degree>0).unsqueeze(-1),node_rate-neighbor,torch.zeros_like(node_rate))


def signed_edge_rate_features(prev,curr,delta_years,clip=3.0,edge_math='rate'):
    """Mean +ve/-ve incident changes with raw signed or log1p edge rates.

    Antisymmetric log difference attenuates scale effects while retaining sign:
        (log(1+w_t)-log(1+w_{t-1}))/max(delta_t, floor).
    Tract weights must be nonnegative and represent comparable acquisitions.
    """
    if edge_math not in ('rate','logratio'):raise ValueError(f'Unknown edge math {edge_math}')
    if not getattr(prev,'_innovation_real_tract',False) or not getattr(curr,'_innovation_real_tract',False):
        return torch.zeros((curr.x.size(0),2),dtype=curr.x.dtype,device=curr.x.device),False
    if not _edges_match(prev,curr):
        raise ValueError('Observed edge innovations require ordered identical ROI edge support')
    a=prev.edge_weight.float();b=curr.edge_weight.float()
    if a.shape!=b.shape or a.numel()!=prev.edge_index.shape[1]:
        raise ValueError('Inconsistent edge weights/indices')
    if not torch.isfinite(a).all() or not torch.isfinite(b).all() or (a<0).any() or (b<0).any():
        raise ValueError('Tract edge weights must be finite and nonnegative')
    if edge_math=='logratio':
        edge_rate=(torch.log1p(b)-torch.log1p(a))/delta_years
    else:edge_rate=(b-a)/delta_years
    edge_rate=torch.nan_to_num(edge_rate,nan=0.,posinf=clip,neginf=-clip).clamp(-clip,clip)
    src=curr.edge_index[0].long();n=int(curr.x.size(0))
    out=torch.zeros((n,2),dtype=torch.float32,device=edge_rate.device)
    out[:,0].index_add_(0,src,F.relu(edge_rate))
    out[:,1].index_add_(0,src,F.relu(-edge_rate))
    counts=torch.bincount(src,minlength=n).clamp(min=1).to(out.dtype).unsqueeze(-1)
    return (out/counts).to(curr.x.dtype),True


def edge_node_flux(prev,curr,years,clip=3.0):
    """Causal signed tract-change × prior ROI-gradient coupling, [N,4].

    flux_i = mean_{j != i} clip((w_ij,t - w_ij,t-1)/years, -clip, clip)
                           * (X_j,t-1 - X_i,t-1)
    Uses matched ordered observed tract support and only input visits.
    This is a proposed engineering operator, not a claim that the formula is novel.
    """
    out=curr.x.new_zeros((curr.x.size(0),4))
    if not getattr(prev,'_innovation_real_tract',False) or not getattr(curr,'_innovation_real_tract',False):
        return out,False
    if not _edges_match(prev,curr):
        raise ValueError('Flux requires ordered identical ROI support')
    old=prev.edge_weight.float();new=curr.edge_weight.float()
    if old.shape!=new.shape or old.numel()!=curr.edge_index.shape[1]:
        raise ValueError('Flux edge-weight and support mismatch')
    if (not torch.isfinite(old).all() or not torch.isfinite(new).all()
            or (old<0).any() or (new<0).any()):
        raise ValueError('Flux requires finite nonnegative tract weights')
    src,dst=curr.edge_index[0].long(),curr.edge_index[1].long()
    eligible=src!=dst
    src,dst=src[eligible],dst[eligible]
    if src.numel()==0:return out,True
    rate=((new-old)/years)[eligible].clamp(-clip,clip).to(out.dtype)
    grad=(prev.x[dst]-prev.x[src]).to(out.dtype)
    out.index_add_(0,src,rate.unsqueeze(-1)*grad)
    counts=out.new_zeros((out.shape[0],1))
    counts.index_add_(0,src,counts.new_ones((src.numel(),1)))
    out=torch.nan_to_num(out/counts.clamp_min(1),nan=0.,posinf=clip,neginf=-clip).clamp(-clip,clip)
    return out,True


def innovation_features(prev,curr,mode,floor_years=.25,clip=3.0,node_math='rate',edge_math='rate'):
    """Returns [N,6] or [N,10] and a valid input-visit mask.

    Columns 0:4 are node rates or spatial Laplacian of those rates, 4:6 are
    signed tract rate summaries; for node_math=rate_laplacian columns 6:10 are
    the additional high-pass signal. For first visit or nonpositive gap the
    whole tensor is zero/masked. Missing edges never fabricate a tract delta.
    """
    if mode not in ('node','edge','both'):raise ValueError(mode)
    if node_math in ('flux','rate_flux') and mode!='both':
        raise ValueError('Edge-node flux requires innovation_mode=both')
    width=_feature_width(node_math)
    if edge_math not in ('rate','logratio'):raise ValueError(edge_math)
    if prev.x.shape!=curr.x.shape or curr.x.size(1)!=4:
        raise ValueError('Expected matched [N,4] FA/MD/RD/AD ROI feature rows')
    if floor_years<=0 or clip<=0:raise ValueError('Nonpositive time floor/clip')
    elapsed=(_months(curr)-_months(prev))/12.
    out=curr.x.new_zeros((curr.x.size(0),width))
    if elapsed<=0:return out,False
    years=max(float(elapsed),float(floor_years))
    if mode in ('node','both'):
        rate=torch.nan_to_num((curr.x-prev.x)/years,nan=0.,posinf=clip,neginf=-clip).clamp(-clip,clip)
        if node_math=='rate':out[:,:4]=rate
        elif node_math=='laplacian':out[:,:4]=graph_laplacian_rate(rate,curr).clamp(-clip,clip)
        elif node_math=='rate_laplacian':
            out[:,:4]=rate
            out[:,6:10]=graph_laplacian_rate(rate,curr).clamp(-clip,clip)
        elif node_math=='rate_flux':
            out[:,:4]=rate
        # flux-only replaces ordinary node rates; evaluated below with observed tracts.
    edge_valid=False
    if mode in ('edge','both'):
        e,edge_valid=signed_edge_rate_features(prev,curr,years,clip,edge_math=edge_math)
        out[:,4:6]=e
    if node_math in ('flux','rate_flux'):
        flux,flux_valid=edge_node_flux(prev,curr,years,clip)
        if node_math=='flux':out[:,:4]=flux
        else:out[:,6:10]=flux
        # A missing edge observation cannot masquerade as a valid flux channel.
        return out,edge_valid and flux_valid
    return out,mode!='edge' or edge_valid


def make_innovation_batch(graphs,visit_to_window,mode,floor_years=.25,clip=3.,node_math='rate',edge_math='rate'):
    from torch_geometric.data import Data,Batch
    if len(graphs)!=len(visit_to_window):raise ValueError('Graph/window mapping shape mismatch')
    width=_feature_width(node_math)
    out=[];valid=[];prev=None;prev_id=None
    for g,wid in zip(graphs,visit_to_window):
        if prev_id is not None and int(wid)<int(prev_id):raise ValueError('Windows must be monotonic')
        if prev is None or wid!=prev_id:
            x=g.x.new_zeros((g.x.size(0),width));ok=False
        else:
            x,ok=innovation_features(prev,g,mode,floor_years,clip,node_math=node_math,edge_math=edge_math)
        out.append(Data(x=x,edge_index=g.edge_index,edge_weight=torch.ones(g.edge_index.size(1),device=x.device,dtype=x.dtype)))
        valid.append(float(ok));prev=g;prev_id=wid
    return Batch.from_data_list(out),torch.tensor(valid,dtype=torch.float32)


class ResidualInnovationEncoder(nn.Module):
    use_graph_changes=True
    def __init__(self,state_encoder,change_encoder,mode,floor_years=.25,clip=3.,node_math='rate',edge_math='rate',gate_mode='global'):
        super().__init__()
        if mode not in ('node','edge','both'):raise ValueError(mode)
        if state_encoder.output_graph_dim!=change_encoder.output_graph_dim:raise ValueError('Mismatched graph dims')
        _feature_width(node_math)
        if edge_math not in ('rate','logratio'):raise ValueError(edge_math)
        if node_math in ('flux','rate_flux') and mode!='both':raise ValueError('Flux requires both node and edge changes')
        if gate_mode not in ('global','visit','feature'):raise ValueError(f'Unknown gate mode {gate_mode}')
        self.gate_mode=gate_mode
        self.state_encoder=state_encoder;self.change_encoder=change_encoder
        self.output_graph_dim=state_encoder.output_graph_dim
        self.mode=mode;self.floor_years=floor_years;self.clip=clip
        self.node_math=node_math;self.edge_math=edge_math
        self.mix_logit=nn.Parameter(torch.tensor(-3.0))
        self.visit_gate=None
        if gate_mode!='global':
            dim=self.output_graph_dim
            self.visit_gate=nn.Sequential(
                nn.LayerNorm(2*dim),nn.Linear(2*dim,min(dim,64)),nn.GELU(),
                nn.Linear(min(dim,64),1 if gate_mode=='visit' else dim),
            )
            # Match the original small injection at initialization.
            nn.init.zeros_(self.visit_gate[-1].weight)
            nn.init.constant_(self.visit_gate[-1].bias,-3.0)
    def forward(self,x,edge_index,batch,time_to_predict=None,edge_weight=None,*,change_batch=None,change_valid=None):
        if change_batch is None or change_valid is None:raise ValueError('Missing structural change batch')
        h=self.state_encoder(x,edge_index,batch,time_to_predict,edge_weight=edge_weight)
        change=self.change_encoder(change_batch.x,change_batch.edge_index,change_batch.batch,time_to_predict,edge_weight=change_batch.edge_weight)
        if change.shape!=h.shape or change_valid.numel()!=h.shape[0]:raise ValueError('Graph innovation shape mismatch')
        strength=(torch.sigmoid(self.mix_logit) if self.visit_gate is None
                  else torch.sigmoid(self.visit_gate(torch.cat([h,change],dim=-1))))
        return h + strength*change_valid.to(h.device,h.dtype).reshape(-1,1)*change


class VisitReliabilityFiLM(nn.Module):
    """Identity-initialized residual FiLM with fixed or learned soft reliability.

    Adaptive q = valid * [(1-alpha) + alpha * coverage * exp(-age/tau)]
    with alpha=sigmoid(trainable_logit), initialized 0.20. In contrast, hard
    reliability uses alpha=1. No assessment => valid=0 in both cases.
    """
    def __init__(self,graph_dim,clinical_dim,use_reliability=False,tau_days=90.,reliability_mode='hard',condition_mode='standard'):
        super().__init__()
        if tau_days<=0:raise ValueError('tau_days must be positive')
        if reliability_mode not in ('hard','adaptive'):raise ValueError(reliability_mode)
        if condition_mode not in ('standard','concordance'):raise ValueError(condition_mode)
        self.condition_mode=condition_mode
        self.use_reliability=bool(use_reliability);self.reliability_mode=reliability_mode
        self.layer_norm=nn.LayerNorm(graph_dim)
        self.gamma=nn.Linear(clinical_dim,graph_dim);self.beta=nn.Linear(clinical_dim,graph_dim)
        self.tau_raw=nn.Parameter(torch.tensor(float(tau_days)))
        self.reliability_logit=nn.Parameter(torch.tensor(math.log(.2/.8)))
        self.concordance_projection=(nn.Linear(clinical_dim,graph_dim,bias=False)
                                     if condition_mode=='concordance' else None)
        self.concordance_scale=(nn.Parameter(torch.zeros(())) if condition_mode=='concordance' else None)
        self.concordance_bias=(nn.Parameter(torch.tensor(2.0)) if condition_mode=='concordance' else None)
        nn.init.zeros_(self.gamma.weight);nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight);nn.init.zeros_(self.beta.bias)
    def forward(self,graph,clinical,observed_mask,present,age_days:Optional[torch.Tensor]=None):
        if graph.dim()!=3 or clinical.shape[:2]!=graph.shape[:2]:raise ValueError('Graph/clinical visit mismatch')
        if observed_mask is None or present is None:raise ValueError('Clinical masks and presence mandatory')
        avail=present.to(graph.dtype).reshape(*graph.shape[:2],1).clamp(0,1)
        if self.use_reliability:
            if age_days is None:raise ValueError('Clinical reliability requires observation age')
            age=age_days.to(graph.dtype).reshape(*graph.shape[:2],1)
            if not torch.isfinite(age).all() or (age < -1).any():raise ValueError('Invalid observation age')
            coverage=observed_mask.to(graph.dtype).mean(-1,keepdim=True).clamp(0,1)
            tau=torch.clamp(F.softplus(self.tau_raw),min=1.)
            score=coverage*torch.exp(-age.clamp(min=0)/tau)
            if self.reliability_mode=='adaptive':
                alpha=torch.sigmoid(self.reliability_logit)
                q=avail*(age>=0).to(graph.dtype)*((1-alpha)+alpha*score)
            else:q=avail*(age>=0).to(graph.dtype)*score
        else:q=avail
        if self.condition_mode=='concordance':
            projected=F.normalize(self.concordance_projection(clinical),dim=-1,eps=1e-6)
            structural=F.normalize(self.layer_norm(graph),dim=-1,eps=1e-6)
            agree=(projected*structural).sum(dim=-1,keepdim=True)
            q=q*torch.sigmoid(self.concordance_bias+self.concordance_scale*agree)
        return graph+q*(.5*torch.tanh(self.gamma(clinical))*self.layer_norm(graph)+self.beta(clinical))
