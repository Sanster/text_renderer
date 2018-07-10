import os

configs_dir = './configs'
config_names = os.listdir(configs_dir)
config_files = [os.path.join(configs_dir, x) for x in config_names]

for i, f in enumerate(config_files):
    print("Load config file: %s" % f)
    cmd = 'python3 main.py ' \
          '--corpus_mode=random ' \
          '--num_img=20 ' \
          '--fonts_dir=./data/fonts/test ' \
          '--config_file=%s ' \
          '--tag=%s ' \
          '--num_processes=2' % (f, config_names[i][:-5])

    os.system(cmd)
