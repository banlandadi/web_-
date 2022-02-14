from gensim.models import word2vec
from collections import defaultdict
import numpy as np
from torch import nn
import torch
def get_w2v_emb(input_path, output_path):
    with open(input_path, 'r') as f_entity, open(output_path, 'w') as f_out:
        for line in f_entity.readlines():
            describe=line.strip().split('\t')[1]
            f_out.write(describe)
            f_out.write("\n")
    word_data = word2vec.LineSentence(output_path)
    my_w2v_model = word2vec.Word2Vec(
        word_data, hs=1, min_count=1, window=3, vector_size=100)
    print("w2v finish!")

    #####求和，归一化
    my_vec = {}
    my_ele = {}
    with open(input_path, 'r') as fr:
        for line in fr.readlines():
            my_vector = 0
            words = line.strip().split('\t')[1].split(" ")
            element = line.strip().split('\t')[0]
            for word in words:
                my_vector += my_w2v_model.wv[word]
                my_vector = my_vector / np.linalg.norm(my_vector)
            my_vec[element] = my_vector
            my_ele[element] = len(my_ele)
    return my_vec, my_ele

def create_new_files():
    f_train = open("lab2_dataset/train.txt", "r", encoding="utf-8")
    f_new_train = open("lab2_dataset/new_train.txt", "w", encoding="utf-8")
    for line in f_train.readlines():
        f_new_train.write(line.strip().split('\t')[0])
        f_new_train.write('\t')
        f_new_train.write(line.strip().split('\t')[2])
        f_new_train.write('\t')
        f_new_train.write(line.strip().split('\t')[1])
        f_new_train.write('\n')
    f_new_train.close()
    f_train.close()
    ###########
    f_test = open("lab2_dataset/test.txt", "r", encoding="utf-8")
    f_new_test = open("lab2_dataset/new_test.txt", "w", encoding="utf-8")
    for line in f_test.readlines():
        f_new_test.write(line.strip().split('\t')[0])
        f_new_test.write('\t')
        f_new_test.write(line.strip().split('\t')[2])
        f_new_test.write('\t')
        f_new_test.write(line.strip().split('\t')[1])
        f_new_test.write('\n')
    f_new_test.close()
    f_test.close()
    ###########
    f_dev = open("lab2_dataset/dev.txt", "r", encoding="utf-8")
    f_new_dev = open("lab2_dataset/new_dev.txt", "w", encoding="utf-8")
    for line in f_dev.readlines():
        f_new_dev.write(line.strip().split('\t')[0])
        f_new_dev.write('\t')
        f_new_dev.write(line.strip().split('\t')[2])
        f_new_dev.write('\t')
        f_new_dev.write(line.strip().split('\t')[1])
        f_new_dev.write('\n')
    f_new_dev.close()
    f_dev.close()

def triple_get(path):
    my_list = []
    ############ head-relation-tail
    with open(path, 'r') as f_read:
        for line in f_read.readlines():
            triple = line.strip().split('\t')
            my_list.append(triple)
    return my_list

def my_index(my_list, entitys, my_flag):
    drop_list = []
    target_list = []
    for i in range(len(my_list)):
        if my_flag == 'train':
            if my_list[i][0] in entitys:
                my_list[i][0] = entitys[my_list[i][0]]
            else:#####不存在实体描述，训练时舍弃
                drop_list.append(i)
                continue
            my_list[i][1] = int(my_list[i][1])
            if my_list[i][2] in entitys:
                my_list[i][2] = entitys[my_list[i][2]]
            else:
                drop_list.append(i)
        else:######test
            if my_list[i][0] in entitys:
                my_list[i][0] = entitys[my_list[i][0]]
            else:
                my_list[i][0] = 1
                drop_list.append(i)
            my_list[i][1] = int(my_list[i][1])
            target_list.append(my_list[i][2])
            del my_list[i][2]

    if my_flag == 'train':
        #####不存在实体描述，训练时舍弃
        result=[]
        for num,i in enumerate(my_list):
            if num not in drop_list:
                result.append(i) 
        return result
    else:
        return my_list, target_list, drop_list

class KGE(nn.Module):
    ########该模块如要使用RESCAL模型，需将
    def __init__(self, entitys_vector, rela_vector, embedding_dim):
        super(KGE, self).__init__()

        #####实体特征向量
        temp_entity = torch.zeros(len(entitys_vector),embedding_dim)
        temp_num = 0
        for x in entitys_vector:
            temp_entity[temp_num] = torch.from_numpy(entitys_vector[x])
            temp_num += 1
        self.E = nn.Embedding(len(entitys_vector), embedding_dim)
        self.E.weight = nn.Parameter(temp_entity)
        self.E.weight.requires_grad = True

        #### 关系矩阵初始化
        temp_rela = torch.zeros(len(rela_vector),embedding_dim)
        temp_num = 0
        for x in rela_vector:
            temp_rela[temp_num] = torch.from_numpy(rela_vector[x])
            temp_num += 1
        self.R = nn.Embedding(len(rela_vector), embedding_dim)
        self.R.weight = nn.Parameter(temp_rela)
        self.R.weight.requires_grad = True
        ##########
        ##########RESCAL模型所需要的Self.R
        # self.R = nn.Embedding(len(rela_vector), embedding_dim*embedding_dim)
        # self.R.weight.requires_grad = True
        # self.embedding_dim=embedding_dim
        # self.entitys_vector=entitys_vector
        # self.rela_vector=rela_vector

    def my_score(self, head_embedding, relation_embedding):

        ####### DistMult
        #print(head_embedding.shape)
        #print(relation_embedding.shape)
        score = torch.mm(head_embedding*relation_embedding, self.E.weight.transpose(1,0))
        return score
        #########
        ####### RESCAL
        # head_embed=head_embedding.view(-1,1,100)
        # rel_embed=relation_embedding.view(-1,100,100)
        # score=torch.matmul((torch.squeeze(torch.matmul(head_embed,rel_embed),dim=1)),self.E.weight.transpose(1,0))
        # #print(score.shape)
        # return score

    def forward(self, head_list, rel_list):
        score = self.my_score(self.E(head_list), self.R(rel_list))
        my_predict = torch.sigmoid(score)
        return my_predict


def train(Model, train_path, entitys, epochs, batch_size, my_optimizer, loss_func):
    my_list = triple_get(train_path)
    data_index = my_index(my_list, entitys, 'train')
    train_data = defaultdict(list)
    for triple in data_index:
        #同一个head-relation可能对应多个tail
        train_data[(triple[0], triple[1])].append(triple[2])
    temp = list(train_data.keys())

    batch_num = len(temp) // batch_size
    Model.train()
    print('training...\n')
    for epoch in range(epochs):
        for i in range(batch_num):
            batch_input = temp[i*batch_size : (i+1)*batch_size]
            batch_target = torch.zeros([batch_size, len(entitys)], dtype=torch.float32)
            #预测目标生成
            for x, pair in enumerate(batch_input):
                batch_target[x, train_data[pair]] = 1
            batch_input = np.array(batch_input)
            #forward
            # 只能接受longtensor类型数据
            head_list = torch.LongTensor([x[0] for x in batch_input])
            rel_list = torch.LongTensor([x[1] for x in batch_input])
            tail_predict = Model.forward(head_list, rel_list)
            # loss计算
            #print(tail_predict)
            #print(batch_target.shape)
            loss = loss_func(tail_predict, batch_target)
            #loss = loss.reshape([64,100,100])
            #print(type(loss))
            # 梯度清零
            my_optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 梯度下降
            my_optimizer.step()
        print('loss :',loss.item())
    return train_data



def test(Model, test_path, entitys, my_dictionary, result_path, train_data):
    #读入测试数据
    my_list = triple_get(test_path)
    test_indexs,_, _= my_index(my_list, entitys, 'test')
    #print(list_invalid)
    head_list = torch.tensor([i[0] for i in test_indexs])
    rel_list = torch.tensor([i[1] for i in test_indexs])
    #读取训练数据，避免预测时输出已有结果
    my_list = triple_get("lab2_dataset/train.txt")
    data_index = my_index(my_list, entitys, 'train')
    train_data = defaultdict(list)
    for triple in data_index:
        train_data[(triple[0], triple[1])].append(triple[2])
    #############
    Model.eval()
    tail_predict = Model(head_list, rel_list)
    results = []
    print("resulting...\n")
    for my_vector in tail_predict:
        result = []
        #tensor to list 
        tail_temp = my_vector.detach().numpy().tolist()
        for i in range(5):
            #得到最接近的实体
            x=max(tail_temp)
            temp = tail_temp.index(x)
            ######除去已经出现在训练集中的预测数据
            while my_dictionary[temp] in train_data[(triple[0], triple[1])]:
                tail_temp[temp] = -1
                z=max(tail_temp)
                temp = tail_temp.index(z)
            #对应结果存储
            result.append(my_dictionary[temp])
            tail_temp[temp] = -1
        results.append(result)
        
    #输出结果
    with open(result_path, 'w') as f_out:
        for result in results:
            f_out.write(str(result[0]))
            f_out.write(',')
            f_out.write(str(result[1]))
            f_out.write(',')
            f_out.write(str(result[2]))
            f_out.write(',')
            f_out.write(str(result[3]))
            f_out.write(',')
            f_out.write(str(result[4]))
            f_out.write('\n')

def main():
    entity_input_path = "lab2_dataset/entity_with_text.txt"
    relation_input_path = "lab2_dataset/relation_with_text.txt"
    output_path = "output/temp_text.txt"
    train_path = "lab2_dataset/train.txt"
    test_path = "lab2_dataset/test.txt"
    result_path = "output/result.txt"
    entitys_vector, entitys = get_w2v_emb(entity_input_path, output_path)
    rela_vector, x = get_w2v_emb(relation_input_path, output_path)
    epochs = 1
    batch_size = 128
    embedding_dim=100
    #构建字典
    my_dictionary = {}
    for x in entitys:
        my_dictionary[entitys[x]] = x
    Model = KGE(entitys_vector, rela_vector, embedding_dim)
    #train model
    my_optimizer = torch.optim.Adam(Model.parameters(), lr=1e-2)
    loss_func = nn.BCELoss()
    train_data = train(Model, train_path, entitys, epochs, batch_size, my_optimizer, loss_func)

    #####计算结果
    test(Model, test_path, entitys, my_dictionary, result_path, train_data)
main()