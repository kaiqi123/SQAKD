import torch

from mqbench.fake_quantize.quantize_base import QuantizeBase
# new
from mqbench.fake_quantize.modules import FakeAffineTensorQuantFunction_STE, FakeAffineTensorQuantFunction_EWGS, FakeAffineTensorQuantFunction_Uscheduler
import sys

_version_under_1100 = int(torch.__version__.split('.')[1]) < 10

class DoReFaFakeQuantize(QuantizeBase):
    def __init__(self, observer, backward_method=None, **observer_kwargs):
        super(DoReFaFakeQuantize, self).__init__(observer, **observer_kwargs)
        self.register_buffer('scale', torch.tensor([1.0], dtype=torch.float))
        self.register_buffer('zero_point', torch.tensor([0], dtype=torch.int))
        # new
        # # print(f"====> DoReFaFakeQuantize: {backward_method}")
        # self.backward_method = backward_method
        # if self.backward_method == "Uscheduler":
        #     # self.start_step = 0
        #     self.u_scheduler = self.define_u_scheduler(method="DoReFa", u_init = 1.0e-09, u_max = 1.0e-06) 
        #     # self.u_scheduler = self.define_u_scheduler(method="DoReFa", u_init = 8.0e-07, u_max = 8.0e-04) 


    def forward(self, X):
        X = torch.tanh(X)
        X = X.div(X.abs().max() + 1e-5)

        if self.observer_enabled[0] == 1:
            self.activation_post_process(X.detach())
            _scale, _zero_point = self.activation_post_process.calculate_qparams()
            # _scale, _zero_point = self.activation_post_process.calculate_qparams(X, self.bitwidth, self.quant_max, self.quant_min, self.scale) # new for forward
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
            if self.scale.shape != _scale.shape:
                self.scale.resize_(_scale.shape)
                self.zero_point.resize_(_zero_point.shape)
            self.scale.copy_(_scale)
            self.zero_point.copy_(_zero_point)

        if self.fake_quant_enabled[0] == 1:
            if self.is_per_channel:
                X = torch.fake_quantize_per_channel_affine(
                    X, self.scale, 
                    self.zero_point.long() if _version_under_1100 else self.zero_point,
                    self.ch_axis, self.quant_min, self.quant_max)
            else:
                X = torch.fake_quantize_per_tensor_affine(X, self.scale.item(), int(self.zero_point.item()), self.quant_min, self.quant_max) # org
                # new
                # print(f"Dorefa, per tensor, {self.backward_method}")
                # if self.backward_method == "org":
                #     X = torch.fake_quantize_per_tensor_affine(X, self.scale.item(), int(self.zero_point.item()), self.quant_min, self.quant_max) # org
                # elif self.backward_method == "STE":
                #     X = FakeAffineTensorQuantFunction_STE.apply(X, self.scale.item(), int(self.zero_point.item()), self.quant_min, self.quant_max) # new
                # elif self.backward_method == "EWGS":
                #     X = FakeAffineTensorQuantFunction_EWGS.apply(X, self.scale.item(), int(self.zero_point.item()), self.quant_min, self.quant_max) # new
                # elif self.backward_method == "Uscheduler":
                #     self.u_scheduler.step()
                #     X = FakeAffineTensorQuantFunction_Uscheduler.apply(X, self.scale.item(), int(self.zero_point.item()), self.quant_min, self.quant_max, self.u_scheduler.u) # new
                #     # self.start_step += 1
                #     # self.u_scheduler.step()
                #     # if self.start_step <= 60000:
                #     #     # print("Uscheduler", self.start_step)
                #     #     X = FakeAffineTensorQuantFunction_Uscheduler.apply(X, float(_scale), int(_zero_point), self.quant_min, self.quant_max, self.u_scheduler.u) # new
                #     # else:
                #     #     # print("Org", self.start_step)
                #     #     X = torch.fake_quantize_per_tensor_affine(X, float(_scale), int(_zero_point), self.quant_min, self.quant_max) # org
                #     # for test
                #     # X1 = torch.fake_quantize_per_tensor_affine(X, self.scale.item(), int(self.zero_point.item()), self.quant_min, self.quant_max)
                #     # X2 = FakeAffineTensorQuantFunction_EWGS.apply(X, self.scale.item(), int(self.zero_point.item()), self.quant_min, self.quant_max)
                #     # for i in range(len(X1[0][0][0])):
                #     #     print(X1[0][0][0][i], X2[0][0][0][i])
                #     # sys.exit()
                # else:
                #     raise NotImplementedError(f"Not implement {self.backward_method}!")
        return X