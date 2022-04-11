import sys

sys.path.extend(["../../", "../", "./"])

import random
import itertools
import argparse
from data.Vocab import *
from data.Dataloader import *
from driver.Config import *
import time
from sklearn import metrics
from torch import nn
from modules.Discourse import Discourse
import pickle
from tensorboardX import SummaryWriter


class Optimizer:
    def __init__(self, parameter, config, lr, batch_num):
        self.optim = torch.optim.Adam(parameter, lr=lr, betas=(config.beta_1, config.beta_2),
                                      eps=config.epsilon, weight_decay=config.L2_REG)
        decay = config.decay
        decay_step = config.decay_epoch*batch_num
        warmup_step = config.warmup_epoch*batch_num
        warmup_point = config.warmup_point

        def lr_coefficient(epoch):
            if epoch <= warmup_step:
                 lr = warmup_point + (1-warmup_point) * epoch/(warmup_step)
                 return lr
            lr = warmup_point + (1-warmup_point) * (decay ** ((epoch - warmup_step) / (decay_step)))
            return lr
            
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=lr_coefficient)

    def step(self):
        self.optim.step()
        self.schedule()
        self.optim.zero_grad()

    def schedule(self):
        self.scheduler.step()

    def zero_grad(self):
        self.optim.zero_grad()

    @property
    def lr(self):
        return self.scheduler.get_last_lr()[0]


def train(train_inst, dev_data, test_data, model, vocab, config, tb):
    model_param = filter(lambda p: p.requires_grad,
                         itertools.chain(
                             model.parameters(),
                         )
                         )

    global_step = 0
    best_score = 0
    batch_num = int(np.ceil(len(train_inst) / float(config.train_batch_size)))
    
    model_optimizer = Optimizer(model_param, config, config.learning_rate, batch_num)
    # model_optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    for iter in range(config.train_iters):
        start_time = time.time()
        print('Iteration: ' + str(iter))
        batch_iter = 0

        overall_total_instance, overall_correct_instance = 0, 0
        for one_batch in data_iter(train_inst, config.train_batch_size, True):

            words, batch_gold_label = batch_data_variable(one_batch, vocab)
            
            batch_edu_mask, batch_edu_explanatory, edu_lengths, batch_edge, batch_edge_type, batch_is_main = batch_discourse_variable(one_batch, vocab)
                
            bert_ids, lengths, masks = batch_pretrain_variable(one_batch, vocab)

            model.train()
            
            batch_predict_output = model.forward(bert_ids, masks, batch_edu_mask, batch_edu_explanatory, edu_lengths, batch_edge, batch_edge_type, batch_is_main)

            loss = model.compute_loss(batch_predict_output, batch_gold_label)
            loss = loss / config.update_every
            loss_value = loss.data.cpu().numpy()
            loss.backward()

            total_instance, correct_instance = model.compute_accuracy(batch_predict_output, batch_gold_label)
            overall_total_instance += total_instance
            overall_correct_instance += correct_instance
            during_time = float(time.time() - start_time)
            acc = overall_correct_instance / overall_total_instance

            print("Step:%d, Iter:%d, batch:%d, lr:%.3e,time:%.2f, acc:%.2f, loss:%.2f"
                  % (global_step, iter, batch_iter, model_optimizer.lr, during_time, acc, loss_value))
            batch_iter += 1

            if batch_iter % config.update_every == 0 or batch_iter == batch_num:
                nn.utils.clip_grad_norm_(model_param, max_norm=config.clip)
                model_optimizer.step()
                model_optimizer.zero_grad()

                global_step += 1

            if batch_iter % config.validate_every == 0 or batch_iter == batch_num:
                print("Dev:")
                dev_score = predict(dev_data, model, vocab, config)

                print("Test:")
                test_score = predict(test_data, model, vocab, config, True)

                tb.add_scalar("learning_rate", model_optimizer.lr, iter)
                tb.add_scalar("dev_F1", dev_score, iter)
                tb.add_scalar("test_F1", test_score, iter)
                
                if dev_score > best_score:
                    print("Exceed best Full F-score: history = %.2f, current = %.2f" % (best_score, dev_score))
                    print("test_score:", test_score)
                    best_score = dev_score
                    if 0 <= config.save_after <= iter:
                        explain_classify_model = {
                            "explain_classify": model.state_dict()
                        }
                        torch.save(explain_classify_model, config.save_model_path)
                        print('Saving model to ', config.save_model_path)


def predict(data, model, vocab, config, test=False):
    start = time.time()
    torch.cuda.empty_cache()         
    model.eval()

    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    for one_batch in data_iter(data, config.test_batch_size, False):
        
        words, batch_gold_label = batch_data_variable(one_batch, vocab)
        
        batch_edu_mask, batch_edu_explanatory, edu_lengths, batch_edge, batch_edge_type, batch_is_main = batch_discourse_variable(one_batch, vocab)
                
        bert_ids, lengths, masks = batch_pretrain_variable(one_batch, vocab)
            
        batch_predict_output = model.forward(bert_ids, masks, batch_edu_mask, batch_edu_explanatory, edu_lengths, batch_edge, batch_edge_type, batch_is_main)
        
        batch_predict_label = torch.max(batch_predict_output.data, 1)[1].cpu().numpy()
        labels_all = np.append(labels_all, batch_gold_label.cpu().numpy())
        predict_all = np.append(predict_all, batch_predict_label)

    assert len(labels_all) == len(predict_all)
    accuracy = metrics.accuracy_score(labels_all, predict_all)
    f1 = metrics.f1_score(labels_all, predict_all, average='macro')

    end = time.time()
    during_time = float(end - start)
    print("samples num: %d,running time = %.2f ,accuracy: %.4f, f1: %.4f" % (len(data), during_time, accuracy, f1))
    report = metrics.classification_report(labels_all, predict_all, target_names=[str(i) for i in vocab._id2label], digits=5)
    print(report)
    return f1


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='config')
    argparser.add_argument('--dataset', required=True, help='choose a dataset')
    argparser.add_argument('--thread', default=4, type=int, help='thread num')
    argparser.add_argument('--gpu', default=0, type=str, help='gpu')

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.cuda.set_device(int(args.gpu))

    random.seed(666)
    np.random.seed(666)
    torch.cuda.manual_seed(666)
    torch.manual_seed(666)
    
    print("GPU available: ", torch.cuda.device_count())
    print("CuDNN: \n", torch.backends.cudnn.enabled)
    torch.set_num_threads(args.thread)

    # if args.dataset == 'hotel':
    #     train_data, dev_data, test_data = read_slice_hotel_corpus(config.test_file)
    # else:
    #     train_data, dev_data, test_data = read_slice_phone_corpus(config.test_file)
        
    with open('data.pkl','rb') as dataset_file:
        dataset = pickle.load(dataset_file)
    if args.dataset == 'hotel':
        train_data, dev_data, test_data, _, _, _ = dataset
    else:
        _, _, _, train_data, dev_data, test_data = dataset

    vocab = creatVocab(train_data + dev_data + test_data, config)

    train_insts = inst(train_data)
    dev_insts = inst(dev_data)
    test_insts = inst(test_data)

    Discourse_Model = Discourse(vocab, config).cuda()
    
    tb_writer = SummaryWriter('experiments/runs')
    train(train_insts, dev_insts, test_insts, Discourse_Model, vocab, config, tb_writer)
    tb_writer.close()
