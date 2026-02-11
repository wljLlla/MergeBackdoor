import math
import torch
import torch.optim as optim

# p1: original model p2: merged model
class CrossOptimizer(optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(CrossOptimizer, self).__init__(params, defaults)
        self.weight_decay = weight_decay

    def step(self, closure=None):

        j = 0
        while j < len(self.param_groups):
            group1 = self.param_groups[j]
            group2 = self.param_groups[j+1]
            j+=2
            lr = group1['lr']
            for p1, p2 in zip(group1['params'], group2['params']):
                if p2.grad is None or p1.grad is None:
                    continue
                grad2 = p2.grad.data
                grad1 = p1.grad.data

                if grad1.is_sparse or grad2.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                amsgrad = group1['amsgrad']

                state = self.state[p1]
                
                if len(state) == 0: 
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p1.data) 
                    state["exp_avg_sq"] = torch.zeros_like(p1.data) 
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p1.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group1["betas"] 

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if self.weight_decay != 0: 
                    grad1 = grad1.add(p1.data, alpha=self.weight_decay)
                    grad2 = grad2.add(p2.data, alpha=self.weight_decay)

                grad = grad1 + 0.5 * grad2

                exp_avg.mul_(beta1).add_(1 - beta1, grad) 
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)


                if amsgrad:
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group1['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group1['eps'])

                step_size = group1['lr'] / bias_correction1

                p1.data.addcdiv_(-step_size, exp_avg, denom) 