import torch
from .DistributedOptimizer import DistributedOptimizer


class DANE(DistributedOptimizer):

    def __init__(self, params, local_opt, local_n_epochs=10, lr=1, mu=0.5):

        self.local_n_epochs = local_n_epochs

        defaults = dict(lr=lr, mu=mu)
        super(DANE, self).__init__(params, local_opt, defaults)

        self.reduce_params()

    def step(self, closure):
        # update df_i(w(t-1))
        loss = closure()
        self.zero_grad()
        loss.backward()

        # 1) calculate global gradient and distribute:
        #    df(w(t-1)) = 1/m sum{ df_i(w(t-1)) }
        self.reduce_grads()

        # 2) solve local optimization problem -> w_i(t)
        self.local_optimize(closure)

        # 3) calculate new parameter:
        #    w(t) = 1/m sum{ w_i(t) }
        self.reduce_params()

    def reduce_grads(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    return

                # p.grad = df(p(t-1))
                p_state = self.state[p]

                # save df(p(t-1))
                p_state['local_grad'] = p.grad.data.detach().clone()

                self.avg(p.grad.data)

                p_state['global_grad'] = p.grad.data.detach().clone()

    def reduce_params(self):
        for group in self.param_groups:
            for p in group['params']:
                self.avg(p.data)

                p_state = self.state[p]
                p_state['global_p'] = p.data.detach().clone()

    def local_loss(self, loss):
        for group in self.param_groups:

            mu = group['mu']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                p_state = self.state[p]

                global_grad = p_state['global_grad']
                local_grad = p_state['local_grad']
                global_p = p_state['global_p']

                if mu != 0:
                    reg = torch.pow(torch.norm(p - global_p), 2) * mu / 2
                    loss += reg

                a = torch.flatten(local_grad - lr * global_grad)
                scal = torch.dot(a, torch.flatten(p))

                loss -= scal

        return loss

    def local_optimize(self, closure):
        for e in range(self.local_n_epochs):
            loss = closure()
            loss = self.local_loss(loss)

            self.zero_grad()
            loss.backward()
            self.local_opt.step()

#            print('{}: epoch {} loss {}'.format(self.rank, e, loss.item()))
