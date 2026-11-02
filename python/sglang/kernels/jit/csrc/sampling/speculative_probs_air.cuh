// Private experiment; original AIR templates included unchanged under Apache-2.0.
#pragma once
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>
#include <tvm/ffi/container/tensor.h>
#include <flashinfer/air_top_p.cuh>
namespace sglang {
namespace air_prepare {
namespace air = flashinfer::sampling::air_top_p;
constexpr int V=152576;
static_assert(sizeof(air::Counter<float>)==384);
static_assert(air::NUM_BUCKETS==2048 && air::NUM_PASSES<float> == 3);
inline int64_t align256(int64_t n) { return (n+255)/256*256; }
template<bool DET> struct Workspace {
  air::Counter<float>* counters;
  air::HisT<DET,float>* hist;
  air::IdxT* count;
  float* buf1;
  float* buf2;
  static int64_t bytes(int m) {
    return align256(384*m)+align256(sizeof(air::HisT<DET,float>)*2048*m)
      +align256(4*2048*m)+2*align256(4*air::calcBufLen<float>(V)*m);
  }
  Workspace(void* pointer,int m) {
    auto* p=static_cast<uint8_t*>(pointer); counters=reinterpret_cast<air::Counter<float>*>(p);
    p+=align256(384*m);hist=reinterpret_cast<air::HisT<DET,float>*>(p);
    p+=align256(sizeof(air::HisT<DET,float>)*2048*m);count=reinterpret_cast<air::IdxT*>(p);
    p+=align256(4*2048*m);buf1=reinterpret_cast<float*>(p);
    p+=align256(4*air::calcBufLen<float>(V)*m);buf2=reinterpret_cast<float*>(p);
  }
};
__device__ __forceinline__ float prepare_div_rn_noftz(float x, float temperature) {
 float result;
 asm volatile("div.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(x), "f"(temperature));
 return result;
}
template<bool DET> __global__ void prepare_kernel(
 const float* x,int64_t stride,const float* temperatures,const float* ps,float* scaled,
 const float* probs,air::Counter<float>* counters,air::HisT<DET,float>* hist,int* counts) {
 const int row=blockIdx.x,chunk=blockIdx.y,tid=threadIdx.x;
 const float temp=temperatures[row/8];
 for(int i=tid;i<2048;i+=256) {
  int col=chunk*2048+i;
  if(col<V) scaled[static_cast<int64_t>(row)*V+col]=prepare_div_rn_noftz(x[static_cast<int64_t>(row)*stride+col],temp);
 }
 // Native Init reads no probabilities; assigning their future buffer pointer is safe.
 if(chunk==0) {
  if(tid==0) {
   auto* counter=counters+row;
   counter->in=probs+static_cast<int64_t>(row)*V;
   counter->len=V; counter->oriLen=V; counter->previousLen=V;
   counter->p=ps[row / 8]; counter->sum=0;
   counter->kthValueBits=0;counter->finishedBlockCnt=0;counter->filterCnt=0;
  }
  for(int i=tid;i<2048;i+=256) {hist[row*2048+i]=0;counts[row*2048+i]=0;}
 }
}
__global__ void fields_kernel(const air::Counter<float>* c,int64_t* out) {
 int row=blockIdx.x;
 if(threadIdx.x==0) {
  const auto& x=c[row];int64_t* d=out+row*9;
  d[0]=reinterpret_cast<uint64_t>(x.in);d[1]=x.oriLen;d[2]=__float_as_uint(x.sum);
  d[3]=x.len;d[4]=__float_as_uint(x.p);d[5]=x.previousLen;d[6]=x.kthValueBits;
  d[7]=x.filterCnt;d[8]=x.finishedBlockCnt;
 }
}
inline void validate_matrix(tvm::ffi::TensorView x,int m,DLDevice dev) {
 using namespace host;auto device=SymbolicDevice{};
 TensorMatcher({m,V}).with_dtype<float>().with_device<kDLCUDA>(device).with_strides({V,1}).verify(x);
 RuntimeCheck(x.device().device_id==dev.device_id);
}
inline void validate_workspace(tvm::ffi::TensorView ws,int m,DLDevice dev,bool det) {
 using namespace host;auto device=SymbolicDevice{};
 int64_t n=det?Workspace<true>::bytes(m):Workspace<false>::bytes(m);
 TensorMatcher({n}).with_dtype<uint8_t>().with_device<kDLCUDA>(device).with_strides({1}).verify(ws);
 RuntimeCheck(ws.device().device_id==dev.device_id && (reinterpret_cast<uintptr_t>(ws.data_ptr())%256)==0);
}
template<bool DET> void prepare_impl(tvm::ffi::TensorView x,tvm::ffi::TensorView t,tvm::ffi::TensorView p,tvm::ffi::TensorView scaled,tvm::ffi::TensorView probs,tvm::ffi::TensorView ws) {
 Workspace<DET> w(ws.data_ptr(),x.size(0));
 host::LaunchKernel(dim3(x.size(0),(V+2047)/2048),dim3(256),x.device())(prepare_kernel<DET>,static_cast<const float*>(x.data_ptr()),x.stride(0),static_cast<const float*>(t.data_ptr()),static_cast<const float*>(p.data_ptr()),static_cast<float*>(scaled.data_ptr()),static_cast<const float*>(probs.data_ptr()),w.counters,w.hist,w.count);
}
template<bool DET> void init_impl(tvm::ffi::TensorView probs,tvm::ffi::TensorView ps,tvm::ffi::TensorView ws) {
 Workspace<DET> w(ws.data_ptr(),probs.size(0));
 host::LaunchKernel(dim3(probs.size(0)),dim3(256),probs.device())(air::AirTopPRenormInitKernel<float,air::HisT<DET,float>>,w.counters,V,static_cast<const float*>(probs.data_ptr()),static_cast<float*>(ps.data_ptr()),0.f,w.hist,w.count);
}
template<bool DET> void resume_impl(tvm::ffi::TensorView probs,tvm::ffi::TensorView out,tvm::ffi::TensorView ws) {
 int m=probs.size(0),sms=0;cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,probs.device().device_id);
 unsigned blocks=air::CalcAirTopPBlockNum<DET,float>(m,V,sms);Workspace<DET> w(ws.data_ptr(),m);
 for(int pass=0;pass<air::NUM_PASSES<float>;pass++)
  host::LaunchKernel(dim3(blocks,m),dim3(air::BLOCK_SIZE),probs.device())(air::AirTopPRenormRadixKernel<DET,float>,w.counters,w.hist,w.count,pass,w.buf1,w.buf2);
 host::LaunchKernel(dim3(m),dim3(1024),probs.device())(air::AirTopPRenormApplyKernel<float>,static_cast<const float*>(probs.data_ptr()),static_cast<float*>(out.data_ptr()),w.counters,V);
}
} // namespace air_prepare
struct AirPrepare {
 static int64_t workspace_bytes(int64_t m,bool det) { host::RuntimeCheck(m==8||m==16||m==32);return det?air_prepare::Workspace<true>::bytes(m):air_prepare::Workspace<false>::bytes(m); }
 static void prepare(tvm::ffi::TensorView x,tvm::ffi::TensorView t,tvm::ffi::TensorView p,tvm::ffi::TensorView scaled,tvm::ffi::TensorView probs,tvm::ffi::TensorView ws,bool det) {
  using namespace host;int m=x.size(0);RuntimeCheck(m==8||m==16||m==32);auto device=SymbolicDevice{};
  TensorMatcher({m,air_prepare::V}).with_dtype<float>().with_strides({x.stride(0),1}).with_device<kDLCUDA>(device).verify(x);RuntimeCheck(x.stride(0)>=air_prepare::V);
  TensorMatcher({m/8,1}).with_dtype<float>().with_strides({1,1}).with_device<kDLCUDA>(device).verify(t);
  TensorMatcher({m/8}).with_dtype<float>().with_strides({1}).with_device<kDLCUDA>(device).verify(p);
  air_prepare::validate_matrix(scaled,m,x.device());air_prepare::validate_matrix(probs,m,x.device());air_prepare::validate_workspace(ws,m,x.device(),det);
  tvm::ffi::CUDADeviceGuard guard(x.device().device_id);
  if(det)air_prepare::prepare_impl<true>(x,t,p,scaled,probs,ws);else air_prepare::prepare_impl<false>(x,t,p,scaled,probs,ws);
 }
 static void init(tvm::ffi::TensorView probs,tvm::ffi::TensorView ps,tvm::ffi::TensorView ws,bool det) {
  using namespace host;int m=probs.size(0);RuntimeCheck(m==8||m==16||m==32);air_prepare::validate_matrix(probs,m,probs.device());air_prepare::validate_workspace(ws,m,probs.device(),det);auto device=SymbolicDevice{};
  TensorMatcher({m}).with_dtype<float>().with_strides({1}).with_device<kDLCUDA>(device).verify(ps);RuntimeCheck(ps.device().device_id==probs.device().device_id);tvm::ffi::CUDADeviceGuard guard(probs.device().device_id);
  if(det)air_prepare::init_impl<true>(probs,ps,ws);else air_prepare::init_impl<false>(probs,ps,ws);
 }
 static void resume(tvm::ffi::TensorView probs,tvm::ffi::TensorView out,tvm::ffi::TensorView ws,bool det) {
  using namespace host;int m=probs.size(0);RuntimeCheck(m==8||m==16||m==32);air_prepare::validate_matrix(probs,m,probs.device());air_prepare::validate_matrix(out,m,probs.device());air_prepare::validate_workspace(ws,m,probs.device(),det);tvm::ffi::CUDADeviceGuard guard(probs.device().device_id);
  if(det)air_prepare::resume_impl<true>(probs,out,ws);else air_prepare::resume_impl<false>(probs,out,ws);
 }
 static void fields(tvm::ffi::TensorView probs,tvm::ffi::TensorView ws,tvm::ffi::TensorView out,bool det) {
  using namespace host;int m=probs.size(0);RuntimeCheck(m==8||m==16||m==32);air_prepare::validate_matrix(probs,m,probs.device());air_prepare::validate_workspace(ws,m,probs.device(),det);auto device=SymbolicDevice{};
  TensorMatcher({m,9}).with_dtype<int64_t>().with_strides({9,1}).with_device<kDLCUDA>(device).verify(out);RuntimeCheck(out.device().device_id==probs.device().device_id);
  tvm::ffi::CUDADeviceGuard guard(probs.device().device_id);host::LaunchKernel(dim3(m),dim3(32),probs.device())(air_prepare::fields_kernel,static_cast<const air_prepare::air::Counter<float>*>(ws.data_ptr()),static_cast<int64_t*>(out.data_ptr()));
 }
};
} // namespace sglang
