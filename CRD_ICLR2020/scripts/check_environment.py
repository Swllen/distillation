"""Run on the training server before starting a long experiment."""
import json
import platform
import sys

import torch
import torchvision


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: check GPU allocation, driver and CUDA wheel')
    info = dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
                torchvision=torchvision.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0), capability=torch.cuda.get_device_capability(0),
                compiled_architectures=torch.cuda.get_arch_list())
    print(json.dumps(info, indent=2))
    # An actual convolution backward catches unsupported GPU kernels.
    model = torch.nn.Conv2d(3, 16, 3).cuda()
    x = torch.randn(4, 3, 32, 32, device='cuda', requires_grad=True)
    loss = model(x).square().mean()
    loss.backward()
    torch.cuda.synchronize()
    if not torch.isfinite(loss) or not torch.isfinite(model.weight.grad).all():
        raise RuntimeError('CUDA convolution produced non-finite values')
    print('CUDA forward/backward passed')


if __name__ == '__main__':
    main()
