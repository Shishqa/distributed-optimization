import torch
import torch.distributed as dist


class DANE(torch.optim.Optimizer):

    def __init__(self, params, local_opt, lr=1, mu=0.5):

        # Get number of agents
        self.world_size = dist.get_world_size()

        defaults = dict(local_opt=local_opt, lr=lr, mu=mu)
        super(DANE, self).__init__(params, defaults)

        for group in self.param_groups:
            self.reduce_params(group['params'])

    def step(self, closure):
        for group in self.param_groups:

            mu = group['mu']
            lr = group['lr']
            opt = group['local_opt']
            params = group['params']

            # update df_i(w(t-1))
            loss = closure()
            self.zero_grad()
            loss.backward()

            # 1) calculate global gradient and distribute:
            #    df(w(t-1)) = 1/m sum{ df_i(w(t-1)) }
            self.reduce_grads(params)

            # 2) solve local optimization problem -> w_i(t)
            self.local_optimize(params, opt, closure, mu, lr)

            # 3) calculate new parameter:
            #    w(t) = 1/m sum{ w_i(t) }
            self.reduce_params(params)

    def reduce_grads(self, params):
        # print('{}: reducing gradient'.format(self.rank))
        for p in params:
            if p.grad is None:
                return

            # p.grad = df(p(t-1))
            p_state = self.state[p]

            # save df(p(t-1))
            p_state['local_grad'] = p.grad.data.detach().clone()

            dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
            p.grad.data /= self.world_size

            p_state['global_grad'] = p.grad.data.detach().clone()
            p_state['global_p'] = p.data.detach().clone()

    def reduce_params(self, params):
        # print('{}: reducing param'.format(self.rank))
        for p in params:
            dist.all_reduce(p.data, op=dist.ReduceOp.SUM)
            p.data /= self.world_size

    def local_optimize(self, params, opt, closure, mu, lr):
        # print('{}: begin local optimization'.format(self.rank))
        local_opt = opt(params, lr=0.1)

        for e in range(10):

            loss = closure()

            for p in params:
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

            local_opt.zero_grad()
            loss.backward()
            local_opt.step()

            # print('{}: epoch {} loss {}'.format(self.rank, e, loss.item()))
