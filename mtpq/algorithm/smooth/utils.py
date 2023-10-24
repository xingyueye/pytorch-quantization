import torch
import torch.nn as nn
import functools

NORM_FCS_MAP = {
    'LlamaDecoderLayer': {
        'input_layernorm':['self_attn.k_proj', 'self_attn.q_proj', 'self_attn.v_proj'],
        'post_attention_layernorm': ['mlp.gate_proj', 'mlp.up_proj']
    },
}

FC_FCS_MAP = {
    'LlamaDecoderLayer': {
        'self_attn.v_proj': ['self_attn.o_proj'],
        'mlp.up_proj': ['mlp.down_proj']
    }
}

class SmoothBase():
    def __init__(self, model, evaluator, num_samples, group_size, norm_fcs_map=NORM_FCS_MAP, fc_fcs_map=FC_FCS_MAP):
        self.model = model
        self.evaluator = evaluator
        self.num_samples = num_samples
        self.group_size = group_size
        self.norm_fcs_map = norm_fcs_map
        self.fc_fcs_map = fc_fcs_map
    
    def get_act_states(self):    
        act_state = {}
        def stat_tensor(name, tensor):
            hidden_dim = tensor.shape[-1]
            tensor = tensor.view(-1, hidden_dim).abs().detach()
         
            max_value = torch.max(tensor, dim=0)[0].float().cpu()
            act_state[name]['absmax'] = torch.max(act_state[name]['absmax'], max_value)

            mean_value = torch.mean(tensor, dim=0).float().cpu()
            act_state[name]['absmean'] = (act_state[name]['absmean'] * act_state[name]['num_batches'] + mean_value) / (act_state[name]['num_batches']+1)
            
            act_state[name]['num_batches'] += 1

        def stat_input_hook(m, x, y, name):
            if isinstance(x, tuple):
                x = x[0]
            stat_tensor(name, x)

        hooks = []
        for name, m in self.model.named_modules():
            if isinstance(m, nn.Linear):
                act_state[name] = {'absmax':torch.tensor(0), 'absmean':torch.tensor(0), 'num_batches':0}
                hooks.append(m.register_forward_hook(functools.partial(stat_input_hook, name=name)))
        self.evaluator.evaluate(self.model, None, self.num_samples)
        for h in hooks:
            h.remove()
        return act_state
    
    def get_weight_state(self):
        raise NotImplementedError
    
    def get_scales(self):
        raise NotImplementedError
    
    @torch.no_grad()
    def smooth_ln_fcs(self, ln, fcs, act_state, alpha):
        if not isinstance(fcs, list):
            fcs = [fcs]
        for fc in fcs:
            assert isinstance(fc, nn.Linear)
            assert ln.weight.numel() == fc.in_features == act_state.numel()

        device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
        act_state = act_state.to(device=device, dtype=dtype)
        weight_state = self.get_weight_state(fcs, self.group_size)
        scales = self.get_scales(act_state, weight_state, alpha).to(device).to(dtype)
        
        ln.weight.div_(scales)
        if hasattr(ln, 'bias'):
            ln.bias.div_(scales)

        for fc in fcs:
            fc.weight.mul_(scales.view(1, -1))

        return 

    @torch.no_grad()
    def smooth_fc_fcs(self, pre_fc, fcs, act_state, alpha):
        size_a = act_state.size(0)
        size_pre_fc = pre_fc.weight.size(0)

        # (for llama2) use group query attention, pre_fc is v_proj, fc is o_proj
        if size_pre_fc < size_a and size_a % size_pre_fc == 0:
            return
         
        device, dtype = pre_fc.weight.device, pre_fc.weight.dtype
        act_state = act_state.to(device=device, dtype=dtype)
        weight_state = self.get_weight_state(fcs, self.group_size)
        scales = self.get_scales(act_state, weight_state, alpha).to(device).to(dtype)

        pre_fc.weight.div_(scales.view(-1, 1))
        if getattr(pre_fc, 'bias', None) is not None:
            pre_fc.bias.div_(scales)

        for fc in fcs:
            fc.weight.mul_(scales.view(1, -1))

    def smooth(self, name, module, act_states, alpha):
        raise NotImplementedError


    