from collections import Counter, OrderedDict
import os
from data.Explain import seg_char

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip('\n')
        vocab[token] = index
    return vocab


class Vocab(object):
    PAD, UNK = 0, 2
    def __init__(self, label_counter, config):
        vocab_file = config.bert_dir + '/vocab.txt'
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file))
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        
        
        self._id2label = []
        for label, count in label_counter.most_common():
            self._id2label.append(label)

        reverse = lambda x: dict(zip(x, range(len(x))))

        self._label2id = reverse(self._id2label)
        if len(self._label2id) != len(self._id2label):
            print("serious bug: relation labels dumplicated, please check!")


    def tokenize(self, sentence):
        chars = seg_char(sentence)
        return [x if x in self.vocab else self.ids_to_tokens[100] for x in chars]

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab[x] for x in tokens]

    def label2id(self, xs):
        if isinstance(xs, list):
            return [self._label2id[x] for x in xs]
        return self._label2id[xs]

    def id2label(self, xs):
        if isinstance(xs, list):
            return [self._id2label[x] for x in xs]
        return self._id2label[xs]

    @property
    def label_size(self):
        return len(self._id2label)


def creatVocab(corpus, config):
    label_counter = Counter()
    for instance in corpus:
        label_counter[instance.is_explain] += 1

    return Vocab(label_counter, config)
