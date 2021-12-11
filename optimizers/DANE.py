import torch
import torch.distributed as dist


class DANE(torch.optim.Optimizer):

    def __init__(self, params, local_opt, local_n_epochs=10, lr=1, mu=0.5):

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.local_opt = local_opt
        self.local_n_epochs = local_n_epochs

        defaults = dict(lr=lr, mu=mu)
        super(DANE, self).__init__(params, defaults)

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

    def sync_tensor(self, local):
        dist.reduce(local, dst=0, op=dist.ReduceOp.SUM)
        if self.rank == 0:
            local /= self.world_size
        dist.broadcast(local, src=0)

    def reduce_grads(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    return

                # p.grad = df(p(t-1))
                p_state = self.state[p]

                # save df(p(t-1))
                p_state['local_grad'] = p.grad.data.detach().clone()

                self.sync_tensor(p.grad.data)

                p_state['global_grad'] = p.grad.data.detach().clone()

    def reduce_params(self):
        for group in self.param_groups:
            for p in group['params']:
                self.sync_tensor(p.data)

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

            print('{}: epoch {} loss {}'.format(self.rank, e, loss.item()))
