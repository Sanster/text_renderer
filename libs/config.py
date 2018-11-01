import yaml
from easydict import EasyDict


def load_config(filepath):
    with open(filepath, mode='r',encoding='utf-8') as f:
        cfg = yaml.load(f.read())
        cfg = EasyDict(cfg)

    check_fraction(cfg.noise, 'noise')
    check_fraction(cfg.line, 'line')
    return cfg


def check_fraction(cfg, name):
    """
    Check whether sum of all fractions in cfg equal to 1
    :param cfg: noise/line cfg
    """
    if not cfg.enable:
        return

    sum = 0
    for k, v in cfg.items():
        if k not in ['enable', 'fraction']:
            if v.enable:
                sum += v.fraction

    if sum != 0 and sum != 1:
        print('Sum of %s enabled item\'s fraction not equal to 1' % name)
        exit(-1)


if __name__ == '__main__':
    load_config('./configs/default.yaml')
