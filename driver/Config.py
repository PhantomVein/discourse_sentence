from configparser import ConfigParser
import sys
import os

sys.path.append('')


# import modules

class Configurable(object):
    def __init__(self, config_file, extra_args):
        config = ConfigParser()
        config.read(config_file)
        if extra_args:
            extra_args = dict([(k[2:], v) for k, v in zip(extra_args[0::2], extra_args[1::2])])
        for section in config.sections():
            for k, v in config.items(section):
                if k in extra_args:
                    v = type(v)(extra_args[k])
                    config.set(section, k, v)
        self._config = config
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
        config.write(open(self.config_file, 'w'))
        print('Loaded config file sucessfully.')
        for section in config.sections():
            for k, v in config.items(section):
                print(k, v)

    @property
    def pretrained_embeddings_file(self):
        return self._config.get('Data', 'pretrained_embeddings_file')

    @property
    def bert_dir(self):
        return self._config.get('Data', 'bert_dir')

    @property
    def data_dir(self):
        return self._config.get('Data', 'data_dir')

    @property
    def save_dir(self):
        return self._config.get('Save', 'save_dir')

    @property
    def config_file(self):
        return self._config.get('Save', 'config_file')

    @property
    def save_model_path(self):
        return self._config.get('Save', 'save_model_path')

    @property
    def save_vocab_path(self):
        return self._config.get('Save', 'save_vocab_path')

    @property
    def load_model_path(self):
        return self._config.get('Save', 'load_model_path')

    @property
    def load_vocab_path(self):
        return self._config.get('Save', 'load_vocab_path')

    @property
    def bert_dims(self):
        return self._config.getint('Network', 'bert_dims')

    @property
    def dropout_emb(self):
        return self._config.getfloat('Network', 'dropout_emb')
    
    @property
    def span_att_hiddens(self):
        return self._config.getint('Network', 'span_att_hiddens')

    @property
    def graph_conv1_dims(self):
        return self._config.getint('Network', 'graph_conv1_dims')

    @property
    def graph_conv2_dims(self):
        return self._config.getint('Network', 'graph_conv2_dims')

    @property
    def graph_conv_head(self):
        return self._config.getint('Network', 'graph_conv_head')

    @property
    def alpha(self):
        return self._config.getfloat('Network', 'alpha')

    @property
    def L2_REG(self):
        return self._config.getfloat('Optimizer', 'L2_REG')

    @property
    def learning_rate(self):
        return self._config.getfloat('Optimizer', 'learning_rate')

    @property
    def decay(self):
        return self._config.getfloat('Optimizer', 'decay')

    @property
    def warmup_epoch(self):
        return self._config.getint('Optimizer', 'warmup_epoch')

    @property
    def beta_1(self):
        return self._config.getfloat('Optimizer', 'beta_1')

    @property
    def beta_2(self):
        return self._config.getfloat('Optimizer', 'beta_2')

    @property
    def epsilon(self):
        return self._config.getfloat('Optimizer', 'epsilon')

    @property
    def warmup_point(self):
        return self._config.getfloat('Optimizer', 'warmup_point')
        
    @property
    def decay_epoch(self):
        return self._config.getint('Optimizer', 'decay_epoch')

    @property
    def clip(self):
        return self._config.getfloat('Optimizer', 'clip')

    @property
    def train_iters(self):
        return self._config.getint('Run', 'train_iters')

    @property
    def train_batch_size(self):
        return self._config.getint('Run', 'train_batch_size')

    @property
    def test_batch_size(self):
        return self._config.getint('Run', 'test_batch_size')

    @property
    def validate_every(self):
        return self._config.getint('Run', 'validate_every')

    @property
    def save_after(self):
        return self._config.getint('Run', 'save_after')

    @property
    def update_every(self):
        return self._config.getint('Run', 'update_every')

    @property
    def max_sentence_len(self):
        return self._config.getint('Run', 'max_sentence_len')
