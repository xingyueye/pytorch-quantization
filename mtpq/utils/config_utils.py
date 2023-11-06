import yaml
from easydict import EasyDict
from mtpq.tensor_quant import QuantDescriptor

def parse_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return EasyDict(config)

def get_quant_desc(config):
    quant_desc = {
        "input_desc": QuantDescriptor(num_bits=config.a_qscheme.bit, calib_method=config.a_qscheme.calib_method,
                                                 quantizer_type=config.a_qscheme.quantizer_type,symmetry = config.a_qscheme.symmetry,
                                                 unsigned=config.a_qscheme.unsigned if hasattr(config.a_qscheme, 'unsigned') else False),
        "conv_weight_desc": QuantDescriptor(num_bits=config.w_qscheme.bit, axis=(0) if config.w_qscheme.per_channel is True else None,
                                            calib_method=config.w_qscheme.calib_method,symmetry = config.a_qscheme.symmetry,
                                            quantizer_type=config.w_qscheme.quantizer_type,
                                            unsigned=config.w_qscheme.unsigned if hasattr(config.w_qscheme, 'unsigned') else False),
        "deconv_weight_desc": QuantDescriptor(num_bits=config.w_qscheme.bit, axis=(1) if config.w_qscheme.per_channel is True else None,
                                              calib_method=config.w_qscheme.calib_method,symmetry = config.a_qscheme.symmetry,
                                              quantizer_type=config.w_qscheme.quantizer_type),
        "output_desc": QuantDescriptor(num_bits=config.a_qscheme.bit, calib_method=config.a_qscheme.calib_method,symmetry = config.a_qscheme.symmetry,
                                      quantizer_type=config.a_qscheme.quantizer_type),
    }
    return EasyDict(quant_desc)