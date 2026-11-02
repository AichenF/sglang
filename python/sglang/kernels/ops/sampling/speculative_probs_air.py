"""Private prepare+AIR Init experiment. No production dispatch is installed."""
import hashlib,importlib.metadata
from pathlib import Path
HERE=Path(__file__).resolve().parents[2]/'jit/csrc/sampling'
MODULE=None
RUNTIME={}
AIR_SHA='e41572fbb55a214fc0b4dfb077f8e8e127fe61bf69b4d4b53ac092f6b80e22bc'
API_SHA='fca8a2ba3ef29ef0e6576f560af71257edf71473f9dcd1cd4d933fd3e20f4da1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def supported_metadata(bs,block,vocab,need_k,need_p):return type(bs) is int and bs in (1,2,4) and block==8 and vocab==152576 and need_k is False and need_p is True
def layout(rows,det=False):
 if rows not in (8,16,32) or type(det) is not bool:raise ValueError('FixedM8/16/32 and explicit bool required')
 align=lambda n:(n+255)//256*256
 c=align(rows*384);h=align(rows*2048*(8 if det else 4));n=align(rows*2048*4);bl=align(152576//32);b=align(rows*bl*4)
 return dict(counter_bytes=c,hist_bytes=h,count_bytes=n,buf_len=bl,buf_bytes=b,total_bytes=c+h+n+2*b)
def runtime_matches(device):
 import torch
 if device in RUNTIME:return RUNTIME[device]
 if torch.cuda.is_current_stream_capturing():return False
 try:
  import flashinfer.sampling as fi
  from flashinfer.jit.env import FLASHINFER_INCLUDE_DIR
  prop=torch.cuda.get_device_properties(device)
  good=torch.__version__=='2.14.1+cu130' and torch.version.cuda=='13.0' and importlib.metadata.version('flashinfer-python')=='0.7.1' and (prop.major,prop.minor)==(9,0) and 'H200' in prop.name and sha(Path(FLASHINFER_INCLUDE_DIR)/'flashinfer/air_top_p.cuh')==AIR_SHA and sha(fi.__file__)==API_SHA
 except (ImportError,OSError,importlib.metadata.PackageNotFoundError):good=False
 RUNTIME[device]=good;return good

def module():
 global MODULE
 import torch
 if MODULE is None:
  if torch.cuda.is_current_stream_capturing():raise RuntimeError('Private AIR module must warm before capture')
  if not runtime_matches(torch.cuda.current_device()):raise RuntimeError('Pinned actual AIR runtime unsupported')
  from flashinfer.jit.env import FLASHINFER_INCLUDE_DIR
  from sglang.kernels.jit.utils import load_jit
  MODULE=load_jit('source5_air_prepare_v2_'+sha(HERE/'speculative_probs_air.cuh')[:12],cuda_files=[str(HERE/'speculative_probs_air.cuh')],cuda_wrappers=[(n,'AirPrepare::'+n) for n in ('workspace_bytes','prepare','init','resume','fields')],extra_include_paths=[str(FLASHINFER_INCLUDE_DIR)],extra_cuda_cflags=['-O3','--use_fast_math'])
 return MODULE

def _tensor(t,shape,dtype,device,contiguous=True):
 import torch
 return isinstance(t,torch.Tensor) and tuple(t.shape)==tuple(shape) and t.dtype==dtype and t.device==device and t.is_cuda and not t.requires_grad and (t.is_contiguous() or not contiguous)
def _disjoint(tensors):
 spans=[]
 for t in tensors:
  storage=t.untyped_storage();spans.append((storage.data_ptr(),storage.data_ptr()+storage.nbytes()))
 if any(a<d and c<b for i,(a,b) in enumerate(spans) for c,d in spans[i+1:]):raise ValueError('Private buffers/inputs must not alias')
def allocate(x,det=False):
 import torch
 if x.shape[0] not in (8,16,32) or x.shape[1]!=152576 or x.dtype!=torch.float32 or not x.is_cuda or x.requires_grad:raise ValueError('Unsupported allocation domain')
 with torch.cuda.device(x.device):
  mod=module();n=layout(x.shape[0],det)['total_bytes']
  if mod.workspace_bytes(x.shape[0],det)!=n:raise ValueError('Actual AIR workspace ABI mismatch')
  return (torch.empty(x.shape,device=x.device,dtype=x.dtype),torch.empty(x.shape,device=x.device,dtype=x.dtype),torch.empty(x.shape,device=x.device,dtype=x.dtype),torch.empty(n,device=x.device,dtype=torch.uint8))
def prepare(x,temp,ps,scaled,probs,workspace,det=False):
 import torch
 m=x.shape[0]
 if not _tensor(x,(m,152576),torch.float32,x.device,False) or x.stride(1)!=1 or x.stride(0)<152576:raise ValueError('Invalid logits')
 if not _tensor(temp,(m//8,1),torch.float32,x.device) or not _tensor(ps,(m//8,),torch.float32,x.device):raise ValueError('Invalid metadata')
 if not all(_tensor(t,(m,152576),torch.float32,x.device) for t in (scaled,probs)) or not _tensor(workspace,(layout(m,det)['total_bytes'],),torch.uint8,x.device):raise ValueError('Invalid outputs/workspace')
 _disjoint((x,temp,ps,scaled,probs,workspace))
 with torch.cuda.device(x.device):module().prepare(x,temp,ps,scaled,probs,workspace,det)
def native_init(probs,repeated_ps,workspace,det=False):
 with __import__('torch').cuda.device(probs.device):module().init(probs,repeated_ps,workspace,det)
def resume(probs,out,workspace,det=False):
 _disjoint((probs,out,workspace))
 with __import__('torch').cuda.device(probs.device):module().resume(probs,out,workspace,det)
def fields(probs,workspace,det=False):
 import torch
 out=torch.empty((probs.shape[0],9),device=probs.device,dtype=torch.int64)
 with torch.cuda.device(probs.device):module().fields(probs,workspace,out,det)
 return out

def try_builder(*,next_token_logits,sampling_info,bs,draft_token_num,max_top_k=None,uniform_top_k_value=None,use_sparse_topk=True,top_k_renorm=None,top_p_renorm=None):
 import torch
 x=next_token_logits
 if not isinstance(x,torch.Tensor) or x.ndim!=2 or not supported_metadata(bs,draft_token_num,x.shape[1],bool(getattr(sampling_info,'need_top_k_sampling',True)),bool(getattr(sampling_info,'need_top_p_sampling',False))):return None
 if not _tensor(x,(bs*8,152576),torch.float32,x.device,False) or x.stride(1)!=1 or x.stride(0)<152576:return None
 t=getattr(sampling_info,'temperatures',None);p=getattr(sampling_info,'top_ps',None)
 if not _tensor(t,(bs,1),torch.float32,x.device) or not _tensor(p,(bs,),torch.float32,x.device):return None
 with torch.cuda.device(x.device):
  if not runtime_matches(x.device.index) or (torch.cuda.is_current_stream_capturing() and MODULE is None):return None
  scaled,probs,out,ws=allocate(x);prepare(x,t,p,scaled,probs,ws);torch.softmax(scaled,-1,out=probs);resume(probs,out,ws)
 return out.view(bs,8,152576)
