from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = 'sparse_inst_opt.onnx'
model_quant = 'sparse_inst_quant.onnx'
quantized_model = quantize_dynamic(
    model_fp32,
    model_quant,
    weight_type=QuantType.QUInt8,
    optimize_model=True
)
