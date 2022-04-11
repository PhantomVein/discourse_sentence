import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from torch_geometric.nn import TransformerConv

class Discourse(nn.Module):
    def __init__(self, vocab, config):
        super(Discourse, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config.bert_dir)
        self.span_att_h1 = nn.Linear(self.config.bert_dims, self.config.span_att_hiddens)
        self.span_att_h2 = nn.Linear(self.config.span_att_hiddens, 1)
        
        self.conv1_dim = self.config.graph_conv1_dims
        self.conv2_dim = self.config.graph_conv2_dims
        self.graph_conv1 = TransformerConv(in_channels=self.config.bert_dims, out_channels=self.conv1_dim//self.config.graph_conv_head, heads=self.config.graph_conv_head, edge_dim=self.conv1_dim)
        self.graph_conv2 = TransformerConv(in_channels=self.conv1_dim, out_channels=self.conv2_dim//self.config.graph_conv_head, heads=self.config.graph_conv_head , edge_dim=self.conv2_dim)
        
        self.relation_embedding1 = nn.Linear(1, self.conv1_dim)
        self.relation_embedding2 = nn.Linear(1, self.conv2_dim)
        self.is_main_embedding1 = nn.Linear(1, self.conv1_dim)
        self.is_main_embedding2 = nn.Linear(1, self.conv2_dim)
        
        self.sentence_fc = nn.Linear(self.conv2_dim, len(vocab._id2label))

    def forward(self, bert_ids, masks, batch_edu_mask, batch_edu_explanatory, edu_lengths, batch_edge, batch_edge_type, batch_is_main):
        bert_ids = bert_ids.cuda()
        masks = masks.cuda()
        batch_edu_mask = batch_edu_mask.cuda()
        edu_lengths = edu_lengths.cuda()
        batch_edu_explanatory = batch_edu_explanatory.cuda()
        batch_edge = batch_edge.cuda()
        batch_edge_type = batch_edge_type.cuda()
        batch_is_main = batch_is_main.cuda()
        batch_size = bert_ids.shape[0]
        
        edus_mask = torch.any(batch_edu_mask, 2).long()
        self.batch_edu_explanatory = batch_edu_explanatory + edus_mask - 1
        
        last_hidden = self.bert(bert_ids, attention_mask=masks)[0]
        x_embed = F.dropout(last_hidden, p=self.config.dropout_emb, training=self.training)
        
        edu_hidden = self.span_extractor(x_embed, batch_edu_mask)
        edu_embed = F.dropout(edu_hidden, p=self.config.dropout_emb, training=self.training)
        
        node_hidden = edu_embed.reshape(-1, edu_embed.shape[-1])
        node_hidden = self.graph_conv1(x=node_hidden, edge_index=batch_edge, edge_attr=self.relation_embedding1(batch_edge_type)+self.is_main_embedding1(batch_is_main))
        node_hidden = self.graph_conv2(x=node_hidden, edge_index=batch_edge, edge_attr=self.relation_embedding2(batch_edge_type)+self.is_main_embedding2(batch_is_main))
        
        last_index = (edu_lengths-1).unsqueeze(1).unsqueeze(2)
        last_index = last_index.repeat([1, 1, self.conv2_dim])
        last_du_hidden = torch.gather(node_hidden.reshape([batch_size, -1 ,self.conv2_dim]), 1, last_index)
        sentence_logit = self.sentence_fc(last_du_hidden.squeeze(1))
        return sentence_logit
    
    def  span_extractor(self, word_hidden, batch_edu_mask):
        attention_logits = self.span_att_h2(F.relu(self.span_att_h1(word_hidden)))
        attention_logits = torch.transpose(attention_logits, 1, 2)
        attention_logits = attention_logits - 100000.0 * (1-batch_edu_mask)
        attention_probs = F.softmax(attention_logits, 2)
        edu_hidden = torch.matmul(attention_probs, word_hidden)
        return edu_hidden


    def compute_accuracy(self, predict_output, gold_label):
        predict_label = torch.max(predict_output.data, 1)[1].cpu()
        
        batch_size = len(predict_label)
        assert batch_size == len(gold_label)
        correct = sum(p == g for p, g in zip(predict_label, gold_label))
        return batch_size, correct.item()

    def compute_loss(self, predict_output, gold_label):
        gold_label = gold_label.cuda()
        
        sentence_loss = F.cross_entropy(predict_output, gold_label)
        
        return sentence_loss
