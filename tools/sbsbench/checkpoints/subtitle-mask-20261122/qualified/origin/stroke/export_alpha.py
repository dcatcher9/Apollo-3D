"""Export and verify the learned alpha decoder for the mixed-convolution Direct Glyph backbone."""
from pathlib import Path
import numpy as np
import torch,onnx,onnxruntime as ort
from onnx import TensorProto as T,helper,numpy_helper
import eval_strokes as es

ROOT=es.ROOT;m=es.m
def main():
    p=m.read(ROOT/'extended-plan.json');assert p['runtime']==m.runtime()
    _,native,res=es.load_models()
    nodes=[];init=[]
    def node(op,args,out,**attrs):
        nodes.append(helper.make_node(op,args,[out],**attrs));return out
    def const(name,arr):
        init.append(numpy_helper.from_array(np.ascontiguousarray(arr),name));return name
    def mlp(net,x,prefix):
        for i,layer in enumerate(net):
            name=prefix+str(i)
            if isinstance(layer,torch.nn.Linear):
                w=const(name+'_w',layer.weight.detach().numpy().T)
                b=const(name+'_b',layer.bias.detach().numpy())
                x=node('Gemm',[x,w,b],name)
            else:x=node('Relu',[x],name)
        return node('Sigmoid',[x],prefix+'_probability')
    base=mlp(native.layers,'native_features','base')
    feat=node('Concat',['native_features',base,'stroke_support'],'refine_features',axis=1)
    extra=mlp(res.net,feat,'residual')
    one=const('one',np.array([1],dtype='f4'))
    correction=node('Mul',[node('Mul',[extra,'stroke_support'],'bounded'),node('Sub',[one,base],'remaining')],'addition')
    probs=node('Add',[base,correction],'roles')
    fill=node('Gather',[probs,const('fill_index',np.array([0],dtype='i8'))],'fill',axis=1)
    outline=node('Gather',[probs,const('outline_index',np.array([1],dtype='i8'))],'outline',axis=1)
    node('Add',[fill,node('Mul',[node('Sub',[one,fill],'non_fill'),outline],'covered_outline')],'alpha')
    graph=helper.make_graph(nodes,'native_stroke_alpha',[helper.make_tensor_value_info('native_features',T.FLOAT,['sites',17]),helper.make_tensor_value_info('stroke_support',T.FLOAT,['sites',2])],[helper.make_tensor_value_info('alpha',T.FLOAT,['sites',1])],init)
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',18)],ir_version=9)
    onnx.checker.check_model(model,full_check=True);out=ROOT/'direct_glyph_native_alpha.onnx';onnx.save(model,out)
    sess=ort.InferenceSession(str(out),providers=['CPUExecutionProvider']);rng=np.random.default_rng(112151)
    errors=[];mismatches=0
    for n in (1,513,65536):
        raw=rng.uniform(0,1,(n,17)).astype('f4');raw[:,12:]*=3;ctx=rng.uniform(0,1,(n,2)).astype('f4')
        with torch.inference_mode():
            x=torch.from_numpy(raw);y=torch.from_numpy(ctx);b=native(x).sigmoid()
            z=res(torch.cat([x,b,y],dim=1));expected=(z[:,0]+(1-z[:,0])*z[:,1]).numpy()
        alpha=sess.run(None,{'native_features':raw,'stroke_support':ctx})[0][:,0]
        errors.append(float(np.abs(alpha-expected).max()));mismatches+=int(((np.rint(alpha*255)>=38)!=(np.rint(expected*255)>=38)).sum())
    assert max(errors)<2e-5 and mismatches==0
    m.dump(ROOT/'export.json',dict(runtime=m.runtime(),onnxruntime=ort.__version__,source=m.bind(ROOT/'extended-plan.json'),decoder=m.bind(out),
          backbone=p['engine'],feature_code=m.bind(es.s.Path(es.d.__file__)),support_code=m.bind(Path(es.s.__file__)),model_output='Continuous combined native protection alpha',sites=66050,threshold_crossings=mismatches,max_alpha_error=max(errors),
          scope='CPU ONNX parity on seeded feature vectors at three batch sizes. The FP16-convolution RGB backbone, native feature construction and causal history are separate from this per-native-pixel FP32 decoder graph.'))
    print(out,errors,mismatches)
if __name__=='__main__':main()
