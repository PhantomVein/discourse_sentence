import re

def find_context(segment, sentences):
    for sentence in sentences:
        if sentence.find(segment) != -1:
            return sentence
    for i in range(1, len(sentences) + 1):
        j = 0
        while i + j <= len(sentences):
            sen = ''.join(sentences[j:i + j])
            if sen.find(segment) != -1:
                return sen
            j += 1
    return None


def cut_sent(para):
    para = re.sub('([。！？\!\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\!\?][”’])([^，。！？\!\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")

def seg_char(sent):
    """
    把句子按字分开，不破坏英文结构
    """
    english = 'abcdefghijklmnopqrstuvwxyz0123456789'
    output = []
    buffer = ''
    for s in sent:
        if s in english or s in english.upper():
            buffer += s
        else:
            if buffer: output.append(buffer)
            buffer = ''
            output.append(s)
    if buffer: output.append(buffer)
    return output


def get_opinion(text):
    text = re.sub('<(a|e|/)(\w|-)*?>', "", text)
    sentence_list = cut_sent(text)
    return sentence_list


class Explain:
    def __init__(self, is_explain, span, opinion):
        self.is_explain = is_explain
        self.spans = span
        self.review = get_opinion(opinion)
        self.context = ''.join(self.review)
        self.tokens = seg_char(self.context)
        self.discourse = None

        
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.context == other.context
        else:
            return False
    def __hash__(self):
        return hash(self.context)     
        
