import torch
from mtpq.algorithms.smooth.utils import SmoothBase

class AWQ(SmoothBase):
    def __init__(self, model, evaluator, num_samples, group_size, alpha):
        super().__init__(model, evaluator, num_samples, group_size)
        self.alpha = alpha

    def get_weight_state(self, w_list, group_size):
        weight = torch.cat([fc.weight for fc in w_list], dim=0)
        org_shape = weight.shape
        if group_size > 0:
            weight = weight.view(-1, group_size)
        weight_state = weight.abs() / weight.abs().amax(dim=1, keepdim=True)
        weight_state = weight_state.view(org_shape)
        weight_state = weight_state.mean(0)
        return weight_state
    
    def get_scales(self, act_state, weight_state, alpha):
        scales = (act_state.pow(alpha) / weight_state.pow(1-alpha)
              ).clamp(min=1e-4)
        
        scales = scales / (scales.max() * scales.min()).sqrt()
        
        return scales

    def smooth(self, name, module, act_states, alpha):
        for ln_name, fcs_name in self.norm_fcs_map[module.__class__.__name__].items():
            ln = eval(f'module.{ln_name}')
            fcs = []
            for i in range(len(fcs_name)):
                fcs.append(eval(f'module.{fcs_name[i]}'))
            act_state = act_states[f'{name}.{fcs_name[0]}']['absmean']
            
            self.smooth_ln_fcs(ln, fcs, act_state, alpha)

        for pre_fc_name, fcs_name in self.fc_fcs_map[module.__class__.__name__].items():
            pre_fc = eval(f'module.{pre_fc_name}')
            fcs = []
            for i in range(len(fcs_name)):
                fcs.append(eval(f'module.{fcs_name[i]}'))

            act_state = act_states[f'{name}.{fcs_name[0]}']['absmean']
            self.smooth_fc_fcs(pre_fc, fcs, act_state, alpha)
        

    def __call__(self, act_states):
        for name, module in self.model.named_modules():
            if module.__class__.__name__ in self.norm_fcs_map:
                self.smooth(name, module, act_states, self.alpha)
                
            if module.__class__.__name__ in self.fc_fcs_map:
                self.smooth(name, module, act_states, self.alpha)