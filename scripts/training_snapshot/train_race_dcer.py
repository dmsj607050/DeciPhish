#!/usr/bin/env python3
import argparse
import os, json, math, time, random, hashlib
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen3ForSequenceClassification, get_cosine_schedule_with_warmup
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from sklearn.metrics import average_precision_score
from race_dcer import RaceDCER
from semantic_branch import (
    GATE_KL_WEIGHT, RELIABILITY_SIGMA, VIEW_CE_WEIGHT, VIEW_KL_WEIGHT, semantic_losses,
)

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/"data"
BASE=ROOT/"model/base"
OUT=ROOT/"runs/dcer_ablation"
REFERENCE=OUT.parent/"Qwen3-0.6B_single_vs_dual_order_compact_html_v1_20261111"/"dual_view/evaluation/summary.json"
SEMANTIC = False
INIT_DCER = None
TRAIN=DATA/"train.jsonl"; TEST=DATA/"tr_op_external.jsonl"
TRAIN_SHA="3ca3975a6e6f0c768fb7b686f7a351721b0fce0a640ca49273b0dde300105ae7"
TEST_SHA="8b341546a8115322d634f6f692ab9d0b8a75b9b25fbaa297974ac1cd5290d9c5"
MAXLEN=4096; SEED=20261027; RANK=8; ALPHA=32; DROPOUT=.05
LR=8e-5; WD=.01; BATCH=8; MICRO=4; EPOCHS=1; LOG_EVERY=50; SAVE_EVERY=250
THRESH=.5; TEMP=1.0
TARGETS=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
HEADERS=("URL evidence:\n","HTML head evidence:\n","HTML interactive evidence:\n","HTML body evidence:\n")
INTERACTIVE={"html.anchor","html.button","html.form","html.input","html.interaction","html.iframe","html.script_behavior"}
DROP={"analysis","method_source","action_source","action_raw","href_raw"}
TEXT_LIMITS={"text":128,"label_hint":64,"purpose_hint":48,"visibility_hint":48}
def now(): return datetime.now(timezone.utc).isoformat()
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""): h.update(b)
 return h.hexdigest()
def read(path):
 with path.open(encoding="utf-8") as f:return [json.loads(x) for x in f if x.strip()]
def write(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
 tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");os.replace(str(tmp),str(path))
def seed(s):
 random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def reg(item):
 c=item["claim"];loc=str((item.get("source") or {}).get("locator","")).lower()
 if c in INTERACTIVE:return 2
 if c in {"html.document_title","html.meta_description"} or loc.startswith("/html/head"):return 1
 return 3
def compact_value(v,field=""):
 if field in DROP or v is None:return None
 if isinstance(v,dict):
  o={}
  for k,ch in v.items():
   if k=="source" or k=="locator" or k.endswith("_source"):continue
   if isinstance(ch,str) and ch.strip().upper()=="UNKNOWN":continue
   ch=compact_value(ch,k)
   if ch is not None:o[k]=ch
  return o or None
 if isinstance(v,list):
  o=[x for x in (compact_value(ch,field) for ch in v[:8]) if x is not None]
  return o or None
 if isinstance(v,str):
  n=TEXT_LIMITS.get(field,192)
  return v if len(v)<=n else v[:n]+"\u2026"
 return v
def groups(row):
 v=row["view"];g=[[],[],[],[]]
 for x in v["url"]:g[0].append([x["claim"],x["value"],(x.get("source") or {}).get("locator")])
 seen=[set(),set(),set(),set()]
 for x in v["html"]:
  k=reg(x);c=x["claim"];short=c.split("html.",1)[1] if c.startswith("html.") else c
  rec=[short,compact_value(x.get("value"))]
  key=json.dumps(rec,ensure_ascii=False,separators=(",",":"),sort_keys=True)
  if key in seen[k]:continue
  seen[k].add(key);g[k].append(rec)
 for k in (1,2,3):g[k]=g[k][:64]
 return g
def encode_many(texts,tok):
 out=[]
 for i in range(0,len(texts),256):
  out.extend(tok(texts[i:i+256],add_special_tokens=False,truncation=False)["input_ids"])
 return out
def prep(rows,tok,split):
 gr=[groups(r) for r in rows]
 toks=[encode_many([json.dumps(x[k],ensure_ascii=False,separators=(",",":")) for x in gr],tok) for k in range(4)]
 marks=[tok.encode(x,add_special_tokens=False) for x in HEADERS]
 bos=tok.bos_token_id;over=(1 if bos is not None else 0)+sum(map(len,marks));budget=MAXLEN-over
 out={"url_html":[],"html_url":[]};pre=[];final=[];tr=0
 for i,row in enumerate(rows):
  caps=[len(toks[k][i]) for k in range(4)];total=over+sum(caps);pre.append(total);overflow=max(0,sum(caps)-budget)
  for k in (3,2,1,0):
   minimum=min(caps[k],64);cut=min(overflow,caps[k]-minimum);caps[k]-=cut;overflow-=cut
  if overflow:
   for k in (3,2,1,0):
    cut=min(overflow,caps[k]);caps[k]-=cut;overflow-=cut
  if overflow:raise RuntimeError("compact_html_v1 length allocator failure")
  if over+sum(caps)<total:tr+=1
  for tag,order in (("url_html",(0,1,2,3)),("html_url",(1,2,3,0))):
   ids=[bos] if bos is not None else []
   region_ids = [-1] * len(ids)
   for k in order:
    ids.extend(marks[k])
    region_ids.extend([-1] * len(marks[k]))
    ids.extend(toks[k][i][:caps[k]])
    region_ids.extend([k] * caps[k] if gr[i][k] else [-1] * caps[k])
   if len(ids)>MAXLEN:raise RuntimeError("sequence exceeds max length")
   out[tag].append((ids, region_ids))
  if (i+1)%2000==0:print(json.dumps({"stage":"tokenize","split":split,"done":i+1,"total":len(rows)}),flush=True)
 audit={"split":split,"samples":len(rows),"serializer_version":"compact_html_v1","max_length":MAXLEN,
  "truncated_samples_by_allocator":tr,"marker_tokens_including_bos":over,
  "length_quantiles_before_allocator":{str(q):int(np.percentile(pre,q)) for q in (0,50,90,95,99,100)},
  "max_final_tokens":max(len(ids) for ids, _ in out["url_html"])}
 return out,audit
def model_new(tok,trainable=True,adapter=None,*,warm_start=False):
 base=Qwen3ForSequenceClassification.from_pretrained(str(BASE),num_labels=2,torch_dtype=torch.bfloat16,local_files_only=True)
 base.config.pad_token_id=tok.pad_token_id;base.config.use_cache=False
 base.config.id2label={0:"BENIGN",1:"PHISHING"};base.config.label2id={"BENIGN":0,"PHISHING":1}
 if adapter is None:
  cfg=LoraConfig(task_type=TaskType.SEQ_CLS,r=RANK,lora_alpha=ALPHA,lora_dropout=DROPOUT,target_modules=TARGETS,modules_to_save=["score"],bias="none")
  m=get_peft_model(base,cfg)
 else:m=PeftModel.from_pretrained(base,str(adapter),is_trainable=trainable,local_files_only=True)
 m=m.to("cuda:0")
 if trainable:
  m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False});m.enable_input_require_grads()
 with torch.random.fork_rng(devices=[]):
  reader = RaceDCER(m, semantic=SEMANTIC).to("cuda:0")
 if adapter is not None:
  state = torch.load(Path(adapter) / "dcer.pt", map_location="cpu", weights_only=True)
  expected = {key for key in reader.state_dict() if not key.startswith("backbone.")}
  if warm_start:
   expected = {key for key in expected if not key.startswith("semantic.")}
  if set(state) != expected:
   raise RuntimeError("DCER checkpoint keys do not match the current reader")
  reader.load_state_dict(state, strict=False)
 return reader
def pad(ids,ix,tok):
 b=tok.pad([{"input_ids":ids[i][0]} for i in ix],padding=True,pad_to_multiple_of=8,return_tensors="pt")
 regions = [ids[i][1] for i in ix]
 b["region_ids"] = torch.tensor([row + [-1] * (b["input_ids"].shape[1] - len(row)) for row in regions])
 return {k:v.to("cuda:0") for k,v in b.items()}
def save_state(path,step,m,opt,sch):
 s={"step":step,"params":{n:p.detach().cpu().clone() for n,p in m.named_parameters() if p.requires_grad},
 "optimizer":opt.state_dict(),"scheduler":sch.state_dict(),"torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all(),
 "py":random.getstate(),"np":np.random.get_state()}
 tmp=path.with_suffix(".tmp")
 with tmp.open("wb") as stream:
  torch.save(s,stream)
  stream.flush()
  os.fsync(stream.fileno())
 os.replace(str(tmp),str(path))
 if os.name == "posix":
  directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
  try:
   os.fsync(directory)
  finally:
   os.close(directory)
def restore(path,m,opt,sch):
 s=torch.load(path,map_location="cpu",weights_only=False);p=dict(m.named_parameters())
 if {n for n,x in p.items() if x.requires_grad}!=set(s["params"]):raise RuntimeError("trainable parameter mismatch on resume")
 with torch.no_grad():
  for n,v in s["params"].items():p[n].copy_(v.to(device=p[n].device,dtype=p[n].dtype))
 opt.load_state_dict(s["optimizer"]);sch.load_state_dict(s["scheduler"]);torch.set_rng_state(s["torch"]);torch.cuda.set_rng_state_all(s["cuda"])
 random.setstate(s["py"]);np.random.set_state(s["np"]);return int(s["step"])
def train(tag,tr,ids,tok,*,pause_requested=None):
 rd=OUT/tag;rd.mkdir(parents=True,exist_ok=True);complete=rd/"training_complete.json"
 if complete.exists():return model_new(tok,False,rd/"checkpoint"),json.loads(complete.read_text(encoding="utf-8"))
 seed(SEED);m=model_new(tok,True,INIT_DCER,warm_start=INIT_DCER is not None)
 params=[p for p in m.parameters() if p.requires_grad]
 if INIT_DCER is not None:
  trainable = [name for name, param in m.named_parameters() if param.requires_grad]
  groups = {
   "lora": sum("lora_" in name for name in trainable),
   "score": sum(".score." in name for name in trainable),
   "dcer": sum(not name.startswith(("backbone.", "semantic.")) for name in trainable),
   "semantic": sum(name.startswith("semantic.") for name in trainable),
  }
  if not all(groups.values()):
   raise RuntimeError(f"Expected joint fine-tuning of all four parameter groups: {groups}")
  print(json.dumps({"stage":"loaded_dcer_checkpoint","trainable_tensors":groups}),flush=True)
 opt=torch.optim.AdamW(params,lr=LR,weight_decay=WD);steps=math.ceil(len(tr)/BATCH)*EPOCHS
 sch=get_cosine_schedule_with_warmup(opt,int(steps*.03),steps);sp=rd/"training_state.pt"
 order=list(range(len(tr)));random.Random(SEED).shuffle(order);start=restore(sp,m,opt,sch) if sp.exists() else 0
 log_path=rd/"training.jsonl"
 prior_elapsed=0.0
 if start and log_path.exists():
  try:
   prior_elapsed=float(json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1]).get("elapsed_seconds",0.0))
  except Exception:prior_elapsed=0.0
 torch.cuda.reset_peak_memory_stats();t0=time.perf_counter();window=[]
 semantic_window = []
 with log_path.open("a" if start else "w",encoding="utf-8",buffering=1) as log:
  for step,begin in enumerate(range(0,len(order),BATCH),1):
   if step<=start:continue
   ix=order[begin:begin+BATCH];labels=torch.tensor([tr[i]["label"] for i in ix],dtype=torch.long,device="cuda:0")
   opt.zero_grad(set_to_none=True);ls=[]
   views=("url_html","html_url")
   for v in views:
    view_loss=0.0
    for mstart in range(0,len(ix),MICRO):
     mix=ix[mstart:mstart+MICRO];mlabels=labels[mstart:mstart+MICRO]
     batch = pad(ids[v], mix, tok)
     output = m(**batch)
     if step == 1 and v == "url_html" and mstart == 0:
      reference_logits = output.base_logits
      if INIT_DCER is not None:
       offset = 0.5 * output.correction
       reference_logits = reference_logits + torch.stack((-offset, offset), dim=1)
      if not torch.equal(output.logits, reference_logits):
       raise RuntimeError("Initial logits must match the selected reference")
      print(json.dumps({
       "stage": "initial_logits_match",
       "reference": "trained DCER" if INIT_DCER is not None else "linear head",
       "alpha": m.alpha.item(),
      }), flush=True)
     ce = F.cross_entropy(output.logits, mlabels)
     loss = ce
     if SEMANTIC:
      aux_ce, view_kl, gate_kl = semantic_losses(output.semantic_details, mlabels)
      loss = ce + VIEW_CE_WEIGHT * aux_ce + VIEW_KL_WEIGHT * view_kl + GATE_KL_WEIGHT * gate_kl
      semantic_window.append(torch.stack((ce, aux_ce, view_kl, gate_kl)).detach().cpu().tolist())
     view_loss+=float(loss.detach().cpu())
     (0.5*len(mix)/len(ix)*loss).backward()
     del batch,output,ce,loss
    ls.append(view_loss)
   opt.step();sch.step();window.append(sum(ls)/2)
   if step%LOG_EVERY==0 or step==steps:
    ev={"stage":"train","condition":tag,"step":step,"total_steps":steps,"samples_seen":min(step*BATCH,len(tr)),
      "view_presentations_seen":min(step*BATCH,len(tr))*2,"mean_loss_window":float(np.mean(window)),
      "elapsed_seconds":round(prior_elapsed+time.perf_counter()-t0,1),"peak_gpu_allocated_mib":round(torch.cuda.max_memory_allocated()/2**20,1),"dcer_alpha":m.alpha.item()}
    if semantic_window:
     ev["semantic_losses_unweighted"] = dict(zip(
      ("main_ce", "view_ce_average", "view_kl", "gate_kl"), np.mean(semantic_window, axis=0).tolist()
     ))
     ev["semantic_alpha"] = m.semantic.alpha.item()
     semantic_window.clear()
    log.write(json.dumps(ev)+"\n");print(json.dumps(ev),flush=True);write(rd/"status.json",{"stage":"TRAINING","condition":tag,"step":step,"total_steps":steps,"updated_at_utc":now()});window=[]
   pause = pause_requested is not None and pause_requested()
   if step%SAVE_EVERY==0 or step==steps or pause:save_state(sp,step,m,opt,sch)
   if pause and step < steps:
    event = {"stage":"PAUSED","condition":tag,"step":step,"total_steps":steps,"updated_at_utc":now()}
    write(rd/"status.json",event)
    print(json.dumps(event),flush=True)
    return None,None
 ck=rd/"checkpoint";m.save_pretrained(str(ck))
 done={"status":"COMPLETE","condition":tag,"training_samples":len(tr),"views_per_sample":2,"optimizer_updates":steps,
  "loss":"mean total loss over URL→HTML and HTML→URL" if SEMANTIC else "mean cross-entropy over URL→HTML and HTML→URL",
  "training_sha256":TRAIN_SHA,"training_wall_seconds":round(prior_elapsed+time.perf_counter()-t0,2),
  "peak_gpu_allocated_mib":round(torch.cuda.max_memory_allocated()/2**20,1),"logical_batch_size":BATCH,"micro_batch_size":MICRO,"gradient_accumulation_steps":BATCH//MICRO,"checkpoint":str(ck),"completed_at_utc":now()}
 write(complete,done);write(rd/"status.json",{"stage":"TRAINING_COMPLETE","updated_at_utc":now()})
 if sp.exists():sp.unlink()
 return m,done
def auc(y,s):
 order=np.argsort(s,kind="mergesort");ss=np.asarray(s)[order];yy=np.asarray(y)[order];rs=0.;i=0
 while i<len(order):
  j=i+1
  while j<len(order) and ss[j]==ss[i]:j+=1
  rs+=((i+1+j)/2)*int(yy[i:j].sum());i=j
 p=int(np.sum(y));n=len(y)-p
 return float((rs-p*(p+1)/2)/(p*n))
def metrics(rows,s):
 y=np.asarray([r["label"] for r in rows],dtype=np.int64);s=np.asarray(s,dtype=np.float64);pred=(s>=THRESH).astype(np.int64)
 tp=int(np.sum((y==1)&(pred==1)));tn=int(np.sum((y==0)&(pred==0)));fp=int(np.sum((y==0)&(pred==1)));fn=int(np.sum((y==1)&(pred==0)))
 pr=tp/(tp+fp) if tp+fp else 0.;re=tp/(tp+fn) if tp+fn else 0.
 return {"samples":len(y),"threshold":THRESH,"temperature":TEMP,"threshold_source":"fixed a priori; no test tuning",
  "accuracy":(tp+tn)/len(y),"precision":pr,"phishing_recall":re,"f1":2*pr*re/(pr+re) if pr+re else 0.,
  "fpr":fp/(fp+tn) if fp+tn else 0.,"roc_auc":auc(y,s),"average_precision":float(average_precision_score(y,s)),
  "tp":tp,"tn":tn,"fp":fp,"fn":fn}
def evaluate(m,rows,ids,tok,tag,view):
 m.eval();scores=[];logits_all=[];corrections=[];semantic_corrections=[];t0=time.perf_counter()
 with torch.inference_mode():
  for start in range(0,len(rows),BATCH):
   ix=list(range(start,min(start+BATCH,len(rows))));batch=pad(ids[view],ix,tok)
   output=m(**batch);lg=output.logits.cpu();pr=torch.softmax(lg/TEMP,dim=-1)[:,1]
   corrections.extend(output.correction.cpu().tolist())
   if SEMANTIC:
    semantic_corrections.extend(output.semantic_correction.cpu().tolist())
   logits_all.extend(lg.tolist());scores.extend(pr.tolist())
   if (start+BATCH)%1000==0 or start+BATCH>=len(rows):print(json.dumps({"stage":"evaluate","model":tag,"view":view,"done":min(start+BATCH,len(rows)),"total":len(rows)}),flush=True)
 elapsed=time.perf_counter()-t0;met=metrics(rows,scores)
 met.update({"test_sha256":TEST_SHA,"truncated_samples":0,"evaluation_batch_size":BATCH,
  "forward_and_batch_transfer_seconds":round(elapsed,3),"throughput_samples_per_second":round(len(rows)/elapsed,3),
  "mean_seconds_per_sample":round(elapsed/len(rows),6),"tokenization_compaction_included":False})
 d=OUT/tag/"evaluation";d.mkdir(parents=True,exist_ok=True)
 with (d/("predictions_"+view+".jsonl")).open("w",encoding="utf-8") as f:
  for index,(r,lg,score,correction) in enumerate(zip(rows,logits_all,scores,corrections)):
   prediction={"case_id":r["case_id"],"source_relative":r.get("source_relative",""),"label":int(r["label"]),
    "logit_benign":lg[0],"logit_phishing":lg[1],"phishing_score":score,"prediction":int(score>=THRESH),
    "threshold":THRESH,"temperature":TEMP,"dcer_logit_correction":correction}
   if SEMANTIC:
    prediction["semantic_logit_correction"] = semantic_corrections[index]
   f.write(json.dumps(prediction,ensure_ascii=False)+"\n")
 if SEMANTIC:
  met["semantic_positive_corrections"] = sum(correction > 0 for correction in semantic_corrections)
 write(d/("metrics_"+view+".json"),met);return met,np.asarray(scores,dtype=np.float64)
def latency(m,rows,ids,tok,view):
 ix=[int(round(i*(len(rows)-1)/127)) for i in range(128)];times=[]
 m.eval()
 with torch.inference_mode():
  for j,i in enumerate(ix):
   b=pad(ids[view],[i],tok);m(**b);torch.cuda.synchronize()
  for i in ix:
   b=pad(ids[view],[i],tok);torch.cuda.synchronize();t=time.perf_counter();m(**b);torch.cuda.synchronize();times.append((time.perf_counter()-t)*1000)
 a=np.asarray(times)
 return {"batch_size":1,"measured_samples":len(times),"p50_ms":float(np.percentile(a,50)),"p95_ms":float(np.percentile(a,95)),
  "mean_ms":float(np.mean(a)),"scope":"model forward only; preprocessed token IDs; GPU sync included"}
def main():
 global OUT, REFERENCE, SEMANTIC, INIT_DCER
 parser = argparse.ArgumentParser()
 parser.add_argument("--semantic", action="store_true")
 parser.add_argument("--init-dcer", action="store_true", help="Warm-start from the completed DCER ablation")
 args = parser.parse_args()
 SEMANTIC = args.semantic
 if args.init_dcer and not SEMANTIC:
  parser.error("--init-dcer requires --semantic")
 if SEMANTIC:
  REFERENCE = OUT / "dual_view/evaluation/summary.json"
  name = "Qwen3-0.6B_race_dual_dcer_buei_bmcc_kl020_20261111"
  if args.init_dcer:
   INIT_DCER = OUT / "dual_view/checkpoint"
   name = name.replace("_20261111", "_warmstart_20261111")
  OUT = OUT.with_name(name)
 if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
 OUT.mkdir(parents=True,exist_ok=True);(OUT/"logs").mkdir(exist_ok=True)
 trsha=sha(TRAIN);tesha=sha(TEST)
 if trsha!=TRAIN_SHA or tesha!=TEST_SHA:raise RuntimeError("train/test SHA mismatch")
 tr=read(TRAIN);te=read(TEST)
 tc={str(k):sum(int(r["label"])==k for r in tr) for k in (0,1)}
 if len(tr)!=10000 or len(te)!=10000 or tc!={"0":5000,"1":5000}:raise RuntimeError("unexpected dataset size/class balance")
 for rows,sp in ((tr,"train"),(te,"test")):
  seen=set()
  for r in rows:
   if r["case_id"] in seen:raise RuntimeError("duplicate case_id "+sp)
   seen.add(r["case_id"])
   if int(r["label"]) not in (0,1) or set(r["view"])!={"coverage","url","html"}:raise RuntimeError("bad row "+sp)
 seed(SEED);tok=AutoTokenizer.from_pretrained(str(BASE),local_files_only=True,use_fast=True);tok.padding_side="right"
 if tok.pad_token_id is None:tok.pad_token=tok.eos_token
 config={"experiment":"Qwen3-0.6B RACE dual view + DCER","created_at_utc":now(),"base_model":str(BASE),
  "train_sha256":trsha,"test_sha256":tesha,"training_samples":len(tr),"train_class_counts":{"benign":5000,"phishing":5000},
  "test_samples":len(te),"input":"compact_html_v1 grouped structured evidence; URL, HTML head, interactive, body",
  "serializer_source":str(ROOT/"experiments/latency_raw_2000_20261108/assets/run_lora_transfer.py"),
  "dual_view":"URL then HTML and HTML then URL; same compacted evidence",
  "primary_test_view":"html_url","reference_summary":str(REFERENCE),
  "dcer":{"regions":["URL","Head","Form/Interactive","Body"],"chunks_per_region":4,"reader_dim":256,"attention_heads":4,"initial_alpha":0.01,"scorer_init":"zero","attention_dropout":0.0},
  "max_length":MAXLEN,"lora_rank":RANK,"lora_alpha":ALPHA,"lora_dropout":DROPOUT,"target_modules":TARGETS,
  "modules_to_save":["score"],"optimizer":"AdamW","learning_rate":LR,"weight_decay":WD,"scheduler":"cosine, 3% warmup",
  "logical_batch_size":BATCH,"micro_batch_size":MICRO,"gradient_accumulation_steps":BATCH//MICRO,"effective_batch_size":BATCH,"optimizer_updates_per_condition":1250,"view_presentations_per_sample":2,"epochs":EPOCHS,
  "loss":"mean CE over two presentations","precision":"BF16 base; FP32 CE; no autocast","seed":SEED,
  "threshold":THRESH,"temperature":TEMP,"calibration":"none; fixed before test",
  "allocator_config":"expandable_segments:True","device":torch.cuda.get_device_name(0),"gpu_visible_count":torch.cuda.device_count()}
 if SEMANTIC:
  config["experiment"] += " + BUEI + BMCC"
  config["loss"] = "main CE + view_ce_weight * mean auxiliary CE + view_kl_weight * reliability-weighted directional KL + gate_kl_weight * gate calibration KL; averaged across training orders"
  config["semantic"] = {
   "view_ce_weight": VIEW_CE_WEIGHT, "view_kl_weight": VIEW_KL_WEIGHT,
   "gate_kl_weight": GATE_KL_WEIGHT, "sigma": RELIABILITY_SIGMA,
   "reliability_error": "per-view detached CE + missing chunk fraction",
   "initial_alpha": 0.01,
   "residual": "nonnegative score difference scaled by the view probability gap and masked if either view is absent",
  }
 if INIT_DCER is not None:
  config["initialization"] = {
   "dcer_checkpoint": str(INIT_DCER),
   "file_sha256": {
    name: sha(INIT_DCER / name)
    for name in ("adapter_config.json", "adapter_model.safetensors", "dcer.pt")
   },
   "new_parameters": "BUEI/BMCC semantic branch",
   "optimizer_state": "fresh",
   "trainable": ["LoRA", "linear score head", "DCER", "BUEI/BMCC"],
  }
  config["dcer"].pop("initial_alpha")
  config["dcer"].pop("scorer_init")
  config["dcer"]["initialization"] = "loaded from DCER checkpoint"
 write(OUT/"config.json",config)
 print(json.dumps({"stage":"prepare","train_sha256":trsha,"test_sha256":tesha,"train_counts":tc,"gpu":config["device"]}),flush=True)
 trids,traudit=prep(tr,tok,"train");teids,teaudit=prep(te,tok,"test")
 write(OUT/"prepared/train_tokenization_audit.json",traudit);write(OUT/"prepared/test_tokenization_audit.json",teaudit)
 result={}
 for tag in ("dual_view",):
  m,train_done=train(tag,tr,trids,tok);views={};scores={}
  for v in ("html_url","url_html"):
   views[v],scores[v]=evaluate(m,te,teids,tok,tag,v)
  ens=(scores["url_html"]+scores["html_url"])/2
  agree=float(np.mean((scores["url_html"]>=THRESH)==(scores["html_url"]>=THRESH)))
  diff=np.abs(scores["url_html"]-scores["html_url"])
  summary={"condition":tag,"training":train_done,"test_views":views,
   "two_view_mean_probability":metrics(te,ens.tolist()),"view_label_agreement":agree,
   "mean_abs_score_difference":float(np.mean(diff)),"p95_abs_score_difference":float(np.percentile(diff,95)),
   "single_sample_latency":{view:latency(m,te,teids,tok,view) for view in ("html_url","url_html")},
   "dcer_alpha":m.alpha.item()}
  if REFERENCE.is_file():
   reference=json.loads(REFERENCE.read_text(encoding="utf-8"))["test_views"]["html_url"]
   primary=views["html_url"]
   for key in ("test_sha256","threshold","temperature"):
    if reference[key]!=primary[key]:
     raise RuntimeError(f"Comparison mismatch for {key}")
   keys=("accuracy","precision","phishing_recall","f1","fpr","roc_auc","average_precision")
   delta_key = "html_url_delta_vs_dcer" if SEMANTIC else "html_url_delta_vs_race_dual_view"
   summary[delta_key]={key:primary[key]-reference[key] for key in keys}
  write(OUT/tag/"evaluation/summary.json",summary);result[tag]=summary;write(OUT/"summary.json",result)
  print(json.dumps({"stage":"condition_complete","condition":tag,"f1_url_html":views["url_html"]["f1"],
   "f1_html_url":views["html_url"]["f1"],"agreement":agree,"latency_p50_ms":summary["single_sample_latency"]["html_url"]["p50_ms"],
   "latency_p95_ms":summary["single_sample_latency"]["html_url"]["p95_ms"]}),flush=True)
  del m;torch.cuda.empty_cache()
 write(OUT/"status.json",{"stage":"COMPLETE","updated_at_utc":now()})
 print(json.dumps({"stage":"complete","output_dir":str(OUT)}),flush=True)
if __name__=="__main__":main()
