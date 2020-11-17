from tqdm import tqdm
import copy

import torch
import torch.optim as optim


def get_inversion(inversion_type, args):
    if inversion_type == 'SGD':
        return GradientDescent(args.iterations, args.lr, optimizer=optim.SGD, args=args)  # iterations=2000 lr=1.0
    elif inversion_type == 'Adam':
        return GradientDescent(args.iterations, args.lr, optimizer=optim.Adam, args=args)


class GradientDescent(object):
    def __init__(self, iterations, lr, optimizer, args):
        self.iterations = iterations  # 2000
        self.lr = lr  # 1.0
        self.optimizer = optimizer  # SGD
        self.init_type = args.init_type  # ['Zero', 'Normal']  # zero

    def invert(self, generator, gt_image, loss_function, batch_size=1, video=True, *init):
        input_size_list = generator.input_size()
        if len(init) == 0:  #  go go go here !!!!!!
            if generator.init is False:
                latent_estimate = []
                for input_size in input_size_list:
                    if self.init_type == 'Zero':
                        latent_estimate.append(torch.zeros((batch_size,) + input_size).cuda())
                    elif self.init_type == 'Normal':
                        latent_estimate.append(torch.randn((batch_size,) + input_size).cuda())
            else:
                latent_estimate = list(generator.init_value(batch_size))
        else:
            assert len(init) == len(input_size_list), 'Please check the number of init value'
            latent_estimate = init

        for latent in latent_estimate:
            latent.requires_grad = True
        optimizer = self.optimizer(latent_estimate, lr=self.lr)

        history = []
        # Opt
        # print(latent_estimate) # [tensor([[-7.8187e-01,...#512#...,-2.9293e-01]], device='cuda:0', requires_grad=True)]
        for i in tqdm(range(self.iterations)):
            y_estimate = generator(latent_estimate)
            optimizer.zero_grad()
            loss = loss_function(y_estimate, gt_image)
            loss.backward()
            optimizer.step()
            if video:
                history.append(copy.deepcopy(latent_estimate))
        return latent_estimate, history
