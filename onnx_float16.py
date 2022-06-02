from onnxmltools.utils.float16_converter import convert_float_to_float16_model_path
import onnx

new_onnx_model = convert_float_to_float16_model_path('sparse_inst_mobile_opt.onnx')
onnx.save(new_onnx_model, 'sparse_inst_mobile_float16.onnx')
