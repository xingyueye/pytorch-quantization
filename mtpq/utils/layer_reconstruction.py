import torch
from mtpq import nn as quant_nn

class LinearTempDecay:
    def __init__(self, t_max: int, rel_start_decay: float = 0.2, start_b: int = 10, end_b: int = 2):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        """
        Cosine annealing scheduler for temperature b.
        :param t: the current time step
        :return: scheduled temperature
        """
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))

class StopForwardException(Exception):
    """
    Used to throw and catch an exception to stop traversing the graph
    """
    pass

class DataSaverHook:
    """
    Forward hook that stores the input and output of a block
    """

    def __init__(self, store_input=False, store_output=False, stop_forward=False):
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward

        self.input_store = None
        self.output_store = None

    def __call__(self, module, input_batch, output_batch):
        if self.store_input:
            self.input_store = input_batch
        if self.store_output:
            self.output_store = output_batch
        if self.stop_forward:
            raise StopForwardException

class GetLayerInpOut:
    def __init__(self, model, layer, device: torch.device):
        self.model = model
        self.layer = layer
        self.device = device
        self.data_saver = DataSaverHook(store_input=True, store_output=True, stop_forward=True)

    def __call__(self, model_input):
        disable_model_quant(self.model)
        handle = self.layer.register_forward_hook(self.data_saver)
        with torch.no_grad():
            try:
                _ = self.model(model_input.to(self.device))
            except StopForwardException:
                pass
            # Recalculate input with network quantized
            self.data_saver.store_output = False
            enable_model_quant(self.model)
            try:
                _ = self.model(model_input.to(self.device))
            except StopForwardException:
                pass

            self.data_saver.store_output = True
        handle.remove()
        return self.data_saver.input_store[0].detach(), self.data_saver.output_store.detach()

def disable_model_quant(model):
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()
                
def enable_model_quant(model):
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()

def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred - tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred - tgt).abs().pow(p).mean()

def save_inp_oup_data(model, layer, cali_data: torch.Tensor,
                       batch_size: int = 32, keep_gpu: bool = True):
    """
    Save input data and output data of a particular layer/block over calibration dataset.

    :param model: QuantModel
    :param layer: QuantModule or QuantBlock
    :param cali_data: calibration data set
    :param weight_quant: use weight_quant quantization
    :param act_quant: use act_quant quantization
    :param batch_size: mini-batch size for calibration
    :param keep_gpu: put saved data on GPU for faster optimization
    :return: input and output data
    """
    device = next(model.parameters()).device
    get_inp_out = GetLayerInpOut(model, layer, device=device)
    cached_batches = []

    for i in range(int(cali_data.size(0) / batch_size)):
        cur_inp, cur_out= get_inp_out(cali_data[i * batch_size:(i + 1) * batch_size])
        cached_batches.append((cur_inp.cpu(), cur_out.cpu(), cur_sym.cpu()))
    cached_inps = torch.cat([x[0] for x in cached_batches])
    cached_outs = torch.cat([x[1] for x in cached_batches])
    torch.cuda.empty_cache()
    if keep_gpu:
        cached_inps = cached_inps.to(device)
        cached_outs = cached_outs.to(device)
    return cached_inps, cached_outs

def layer_rebuild(layer, cached_inps, cached_outs, 
                         batch_size: int = 32, iters: int = 20000, weight: float = 0.001, opt_mode: str = 'mse',
                         act_quant: bool = False, b_range: tuple = (20, 2),
                         warmup: float = 0.0, p: float = 2.0, lr: float = 4e-5, keep_gpu: bool = True):
    """
    Block reconstruction to optimize the output from each layer.

    :param model: QuantModel
    :param layer: QuantModule that needs to be optimized
    :param cali_data: data for calibration, typically 1024 training images, as described in AdaRound
    :param batch_size: mini-batch size for reconstruction
    :param iters: optimization iterations for reconstruction,
    :param weight: the weight of rounding regularization term
    :param opt_mode: optimization mode
    :param asym: asymmetric optimization designed in AdaRound, use quant input to reconstruct fp output
    :param include_act_func: optimize the output after activation function
    :param b_range: temperature range
    :param warmup: proportion of iterations that no scheduling for temperature
    :param act_quant: use activation quantization or not.
    :param lr: learning rate for act delta learning
        :param p: L_p norm minimization
    """
    
    w_opt = None
    scheduler = None
    weight_tensor = layer.weight.clone()
    layer.weight_quantizer.init_alpha(weight_tensor)
    layer.weight_quantizer.soft_targets = True
    w_para = [layer.weight_quantizer.alpha]
    if len(w_para) != 0:
        w_opt = torch.optim.Adam(w_para)
    loss_mode = 'relaxation'
    rec_loss = opt_mode
    loss_func = LossFunction(layer, round_loss=loss_mode, weight=weight,
                             max_count=iters, rec_loss=rec_loss, b_range=b_range,
                             decay_start=0, warmup=warmup, p=p)
    device = 'cuda'
    sz = cached_inps[0].size(0)
    for i in range(iters):
        idx = torch.randint(0, sz, (batch_size,))
        cur_inp= cached_inps[idx].to(device)
        cur_out = cached_outs[idx].to(device)

        w_opt.zero_grad()
        out_quant = layer(cur_inp)

        err = loss_func(out_quant, cur_out)

        err.backward(retain_graph=True)
        w_opt.step()
        if scheduler:
            scheduler.step()
    torch.cuda.empty_cache()
    layer.weight_quantizer.soft_targets = False

class LossFunction:
    def __init__(self,
                 layer,
                 round_loss: str = 'relaxation',
                 weight: float = 1.,
                 rec_loss: str = 'mse',
                 max_count: int = 2000,
                 b_range: tuple = (10, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.0,
                 p: float = 2.):

        self.layer = layer
        self.round_loss = round_loss
        self.weight = weight
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = p

        self.temp_decay = LinearTempDecay(max_count, rel_start_decay=warmup + (1 - warmup) * decay_start,
                                          start_b=b_range[0], end_b=b_range[1])
        self.count = 0

    def __call__(self, pred, tgt, grad=None):
        """
        Compute the total loss for adaptive rounding:
        rec_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy

        :param pred: output from quantized model
        :param tgt: output from FP model
        :param grad: gradients to compute fisher information
        :return: total loss function
        """
        self.count += 1
        if self.rec_loss == 'mse':
            rec_loss = lp_loss(pred, tgt, p=self.p)
        elif self.rec_loss == 'fisher_diag':
            rec_loss = ((pred - tgt).pow(2) * grad.pow(2)).sum(1).mean()
        elif self.rec_loss == 'fisher_full':
            a = (pred - tgt).abs()
            grad = grad.abs()
            batch_dotprod = torch.sum(a * grad, (1, 2, 3)).view(-1, 1, 1, 1)
            rec_loss = (batch_dotprod * a * grad).mean() / 100
        else:
            raise ValueError('Not supported reconstruction loss function: {}'.format(self.rec_loss))

        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.round_loss == 'none':
            b = round_loss = 0
        elif self.round_loss == 'relaxation':
            round_loss = 0
            round_vals = self.layer.weight_quantizer.get_soft_targets()
            round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
        else:
            raise NotImplementedError

        total_loss = rec_loss + round_loss
        if self.count % 500 == 0:
            print('Total loss:\t{:.3f} (rec:{:.3f}, round:{:.3f})\tb={:.2f}\tcount={}'.format(
                float(total_loss), float(rec_loss), float(round_loss), b, self.count))
        return total_loss
