from src.GNN.helper import *
from src.GNN.CONSTANTS import *
from src.utils import *
from src.GNN.oldmodels import *

def action2vec(action, num_objects, num_states):
    actionArray = torch.zeros(len(possibleActions))
    actionArray[possibleActions.index(action['name'])] = 1
    predicate1 = torch.zeros(num_objects+1)
    #predicate 2 and 3 will be predicted together
    predicate2 = torch.zeros(num_objects+1)
    predicate3 = torch.zeros(num_states)
    if len(action['args']) == 0:
        predicate1[-1] = 1
        predicate2[-1] = 1
    elif len(action['args']) == 1:
        predicate1[object2idx[action['args'][0]]] = 1
        predicate2[-1] = 1
    else:
        # action['args'][1] can be a state or an object
        if action['args'][1] in object2idx:
            predicate1[object2idx[action['args'][0]]] = 1
            predicate2[object2idx[action['args'][1]]] = 1
        else:
            predicate1[object2idx[action['args'][0]]] = 1
            predicate3[possibleStates.index(action['args'][1])] = 1
    return torch.cat((actionArray, predicate1, predicate2, predicate3), 0)

def vec2action(vec, num_objects, num_states, idx2object):
    ret_action = {}
    action_array = list(vec[:len(possibleActions)])
    ret_action["name"] = possibleActions[action_array.index(max(action_array))]
    ret_action["args"] = []
    object1_array = list(vec[len(possibleActions):len(possibleActions)+num_objects+1])
    object1_ind = object1_array.index(max(object1_array))
    if object1_ind == len(object1_array) - 1:
        return ret_action
    else:
        ret_action["args"].append(idx2object[object1_ind])
    object2_or_state_array = list(vec[len(possibleActions)+num_objects+1:])
    object2_or_state_ind = object2_or_state_array.index(max(object2_or_state_array))
    if (object2_or_state_ind < num_objects):
        ret_action["args"].append(idx2object[object2_or_state_ind])
    elif (object2_or_state_ind == num_objects):
        pass
    else:
        ret_action["args"].append(possibleStates[object2_or_state_ind - num_objects - 1])
    return ret_action

class DGL_AGCN_Action(nn.Module):
    def __init__(self,
                 in_feats,
                 n_objects,
                 n_hidden,
                 n_states,
                 n_layers,
                 etypes,
                 activation,
                 dropout):
        super(DGL_AGCN_Action, self).__init__()
        self.name = "GatedHeteroRGCN_Attention_Action_" + str(n_hidden) + "_" + str(n_layers)
        self.layers = nn.ModuleList()
        self.layers.append(GatedHeteroRGCNLayer(in_feats, n_hidden, etypes, activation=activation))
        for i in range(n_layers - 1):
            self.layers.append(GatedHeteroRGCNLayer(n_hidden, n_hidden, etypes, activation=activation))
        self.attention = nn.Linear(n_hidden + n_hidden, 1)
        self.embed = nn.Linear(PRETRAINED_VECTOR_SIZE, n_hidden)
        self.fc1 = nn.Linear(n_hidden + n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, len(possibleActions))
        self.p1  = nn.Linear(n_hidden + n_hidden, n_hidden)
        self.p2  = nn.Linear(n_hidden, n_hidden)
        self.p3  = nn.Linear(n_hidden, n_objects+1)
        self.q1  = nn.Linear(n_hidden + n_hidden, n_hidden)
        self.q2  = nn.Linear(n_hidden, n_hidden)
        self.q3  = nn.Linear(n_hidden, n_objects+1+n_states)
        self.activation = nn.LeakyReLU()

    def forward(self, g, goalVec, goalObjectsVec):
        h = g.ndata['feat']
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
        goalObjectsVec = self.activation(self.embed(torch.Tensor(goalObjectsVec)))
        attn_embedding = torch.cat([h, goalObjectsVec.repeat(h.size(0)).view(h.size(0), -1)], 1)
        attn_weights = F.softmax(self.attention(attn_embedding), dim=0)
        # print(attn_weights)
        scene_embedding = torch.mm(attn_weights.t(), h)
        goal_embed = self.activation(self.embed(torch.Tensor(goalVec.reshape(1, -1))))
        final_to_decode = torch.cat([scene_embedding, goal_embed], 1)
        action = self.activation(self.fc1(final_to_decode))
        action = self.activation(self.fc2(action))
        action = self.activation(self.fc3(action))
        action = F.softmax(action, dim=1)
        pred1 = self.activation(self.p1(final_to_decode))
        pred1 = self.activation(self.p2(pred1))
        pred1 = F.softmax(self.activation(self.p3(pred1)), dim=1)
        pred2 = self.activation(self.q1(final_to_decode))
        pred2 = self.activation(self.q2(pred2))
        pred2 = F.softmax(self.activation(self.q3(pred2)), dim=1)
        return torch.cat((action, pred1, pred2), 1).flatten()

class GGCN_Action(nn.Module):
    def __init__(self,
                 in_feats,
                 n_objects,
                 n_hidden,
                 n_states,
                 n_layers,
                 etypes,
                 activation,
                 dropout):
        super(GGCN_Action, self).__init__()
        self.name = "GGCN_Action_" + str(n_hidden) + "_" + str(n_layers)
        self.layers = nn.ModuleList()
        self.layers.append(GatedHeteroRGCNLayer(in_feats, n_hidden, etypes, activation=activation))
        for i in range(n_layers - 1):
            self.layers.append(GatedHeteroRGCNLayer(n_hidden, n_hidden, etypes, activation=activation))
        self.attention = nn.Linear(n_hidden + n_hidden, 1)
        self.embed = nn.Linear(PRETRAINED_VECTOR_SIZE, n_hidden)
        self.fc1 = nn.Linear(n_hidden + n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, len(possibleActions))
        self.p1  = nn.Linear(n_hidden + n_hidden, n_hidden)
        self.p2  = nn.Linear(n_hidden, n_hidden)
        self.p3  = nn.Linear(n_hidden, n_objects+1)
        self.q1  = nn.Linear(n_hidden + n_hidden, n_hidden)
        self.q2  = nn.Linear(n_hidden, n_hidden)
        self.q3  = nn.Linear(n_hidden, n_objects+1+n_states)
        self.activation = nn.LeakyReLU()

    def forward(self, g, goalVec, goalObjectsVec):
        h = g.ndata['feat']
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
        goalObjectsVec = self.activation(self.embed(torch.Tensor(goalObjectsVec)))
        scene_embedding = torch.sum(h, dim=0).view(1,-1)
        goal_embed = self.activation(self.embed(torch.Tensor(goalVec.reshape(1, -1))))
        final_to_decode = torch.cat([scene_embedding, goal_embed], 1)
        action = self.activation(self.fc1(final_to_decode))
        action = self.activation(self.fc2(action))
        action = self.activation(self.fc3(action))
        action = F.softmax(action, dim=1)
        pred1 = self.activation(self.p1(final_to_decode))
        pred1 = self.activation(self.p2(pred1))
        pred1 = F.softmax(self.activation(self.p3(pred1)), dim=1)
        pred2 = self.activation(self.q1(final_to_decode))
        pred2 = self.activation(self.q2(pred2))
        pred2 = F.softmax(self.activation(self.q3(pred2)), dim=1)
        return torch.cat((action, pred1, pred2), 1).flatten()

class GGCN_metric_Action(nn.Module):
    def __init__(self,
                 in_feats,
                 n_objects,
                 n_hidden,
                 n_states,
                 n_layers,
                 etypes,
                 activation,
                 dropout):
        super(GGCN_metric_Action, self).__init__()
        self.name = "GGCN_metric_Action_" + str(n_hidden) + "_" + str(n_layers)
        self.layers = nn.ModuleList()
        self.layers.append(GatedHeteroRGCNLayer(in_feats, n_hidden, etypes, activation=activation))
        for i in range(n_layers - 1):
            self.layers.append(GatedHeteroRGCNLayer(n_hidden, n_hidden, etypes, activation=activation))
        self.attention = nn.Linear(n_hidden + n_hidden, 1)
        self.embed = nn.Linear(PRETRAINED_VECTOR_SIZE, n_hidden)
        self.fc1 = nn.Linear(n_hidden + n_hidden + n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, len(possibleActions))
        self.p1  = nn.Linear(n_hidden + n_hidden + n_hidden, n_hidden)
        self.p2  = nn.Linear(n_hidden, n_hidden)
        self.p3  = nn.Linear(n_hidden, n_objects+1)
        self.q1  = nn.Linear(n_hidden + n_hidden + n_hidden, n_hidden)
        self.q2  = nn.Linear(n_hidden, n_hidden)
        self.q3  = nn.Linear(n_hidden, n_objects+1+n_states)
        self.activation = nn.LeakyReLU()
        self.metric1 = nn.Linear(in_feats, n_hidden)
        self.metric2 = nn.Linear(n_hidden, n_hidden)

    def forward(self, g, goalVec, goalObjectsVec):
        h = g.ndata['feat']
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
        metric_part = g.ndata['feat']
        metric_part = self.activation(self.metric1(metric_part))
        metric_part = self.activation(self.metric2(metric_part))
        h = torch.cat([h, metric_part], dim = 1)
        goalObjectsVec = self.activation(self.embed(torch.Tensor(goalObjectsVec)))
        scene_embedding = torch.sum(h, dim=0).view(1,-1)
        goal_embed = self.activation(self.embed(torch.Tensor(goalVec.reshape(1, -1))))
        final_to_decode = torch.cat([scene_embedding, goal_embed], 1)
        action = self.activation(self.fc1(final_to_decode))
        action = self.activation(self.fc2(action))
        action = self.activation(self.fc3(action))
        action = F.softmax(action, dim=1)
        pred1 = self.activation(self.p1(final_to_decode))
        pred1 = self.activation(self.p2(pred1))
        pred1 = F.softmax(self.activation(self.p3(pred1)), dim=1)
        pred2 = self.activation(self.q1(final_to_decode))
        pred2 = self.activation(self.q2(pred2))
        pred2 = F.softmax(self.activation(self.q3(pred2)), dim=1)
        return torch.cat((action, pred1, pred2), 1).flatten()

class GGCN_metric_att_Action(nn.Module):
    def __init__(self,
                 in_feats,
                 n_objects,
                 n_hidden,
                 n_states,
                 n_layers,
                 etypes,
                 activation,
                 dropout):
        super(GGCN_metric_att_Action, self).__init__()
        self.name = "GGCN_metric_att_Action_" + str(n_hidden) + "_" + str(n_layers)
        self.layers = nn.ModuleList()
        self.layers.append(GatedHeteroRGCNLayer(in_feats, n_hidden, etypes, activation=activation))
        for i in range(n_layers - 1):
            self.layers.append(GatedHeteroRGCNLayer(n_hidden, n_hidden, etypes, activation=activation))
        self.attention = nn.Linear(n_hidden + n_hidden + n_hidden, 1)
        self.embed = nn.Linear(PRETRAINED_VECTOR_SIZE, n_hidden)
        self.fc1 = nn.Linear(n_hidden + n_hidden + n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, len(possibleActions))
        self.p1  = nn.Linear(n_hidden + n_hidden + n_hidden, n_hidden)
        self.p2  = nn.Linear(n_hidden, n_hidden)
        self.p3  = nn.Linear(n_hidden, n_objects+1)
        self.q1  = nn.Linear(n_hidden + n_hidden + n_hidden, n_hidden)
        self.q2  = nn.Linear(n_hidden, n_hidden)
        self.q3  = nn.Linear(n_hidden, n_objects+1+n_states)
        self.activation = nn.LeakyReLU()
        self.metric1 = nn.Linear(in_feats, n_hidden)
        self.metric2 = nn.Linear(n_hidden, n_hidden)

    def forward(self, g, goalVec, goalObjectsVec):
        h = g.ndata['feat']
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
        metric_part = g.ndata['feat']
        metric_part = self.activation(self.metric1(metric_part))
        metric_part = self.activation(self.metric2(metric_part))
        h = torch.cat([h, metric_part], dim = 1)
        goalObjectsVec = self.activation(self.embed(torch.Tensor(goalObjectsVec)))
        attn_embedding = torch.cat([h, goalObjectsVec.repeat(h.size(0)).view(h.size(0), -1)], 1)
        attn_weights = F.softmax(self.attention(attn_embedding), dim=0)
        scene_embedding = torch.mm(attn_weights.t(), h)
        goal_embed = self.activation(self.embed(torch.Tensor(goalVec.reshape(1, -1))))
        final_to_decode = torch.cat([scene_embedding, goal_embed], 1)
        action = self.activation(self.fc1(final_to_decode))
        action = self.activation(self.fc2(action))
        action = self.activation(self.fc3(action))
        action = F.softmax(action, dim=1)
        pred1 = self.activation(self.p1(final_to_decode))
        pred1 = self.activation(self.p2(pred1))
        pred1 = F.softmax(self.activation(self.p3(pred1)), dim=1)
        pred2 = self.activation(self.q1(final_to_decode))
        pred2 = self.activation(self.q2(pred2))
        pred2 = F.softmax(self.activation(self.q3(pred2)), dim=1)
        return torch.cat((action, pred1, pred2), 1).flatten()

class DGL_AGCN_Action_List(nn.Module):
    def __init__(self,
                 in_feats,
                 n_objects,
                 n_hidden,
                 n_states,
                 n_layers,
                 etypes,
                 activation,
                 dropout,
                 num_states_in_list):
        super(DGL_AGCN_Action_List, self).__init__()
        self.name = "GatedHeteroRGCN_Attention_Action_List_" + str(n_hidden) + "_" + str(n_layers)
        self.layers = nn.ModuleList()
        self.layers.append(GatedHeteroRGCNLayer(in_feats, n_hidden, etypes, activation=activation))
        for i in range(n_layers - 1):
            self.layers.append(GatedHeteroRGCNLayer(n_hidden, n_hidden, etypes, activation=activation))
        self.attention = nn.Linear(n_hidden + n_hidden, 1)
        self.embed = nn.Linear(PRETRAINED_VECTOR_SIZE, n_hidden)
        self.fc1 = nn.Linear(n_hidden * num_states_in_list + n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, len(possibleActions))
        self.p1  = nn.Linear(n_hidden * num_states_in_list + n_hidden, n_hidden)
        self.p2  = nn.Linear(n_hidden, n_hidden)
        self.p3  = nn.Linear(n_hidden, n_objects+1)
        self.q1  = nn.Linear(n_hidden * num_states_in_list + n_hidden, n_hidden)
        self.q2  = nn.Linear(n_hidden, n_hidden)
        self.q3  = nn.Linear(n_hidden, n_objects+1+n_states)
        self.n_hidden = n_hidden
        self.activation = nn.LeakyReLU()
        self.num_states_in_list = num_states_in_list

    def forward(self, g_list, goalVec, goalObjectsVec):
        goalObjectsVec = self.activation(self.embed(torch.Tensor(goalObjectsVec)))
        goal_embed = self.activation(self.embed(torch.Tensor(goalVec.reshape(1, -1))))
        scene_embedding_list = [torch.zeros(1,self.n_hidden) for i in range(self.num_states_in_list - len(g_list))]
        for g in g_list:    
            h = g.ndata['feat']
            for i, layer in enumerate(self.layers):
                h = layer(g, h)
        
            attn_embedding = torch.cat([h, goalObjectsVec.repeat(h.size(0)).view(h.size(0), -1)], 1)
            attn_weights = F.softmax(self.attention(attn_embedding), dim=0)
            # print(attn_weights)
            scene_embedding = torch.mm(attn_weights.t(), h)
            scene_embedding_list.append(scene_embedding)
        scene_embedding = torch.cat(scene_embedding_list,1)
        final_to_decode = torch.cat([scene_embedding, goal_embed], 1)
        action = self.activation(self.fc1(final_to_decode))
        action = self.activation(self.fc2(action))
        action = self.activation(self.fc3(action))
        action = F.softmax(action, dim=1)
        pred1 = self.activation(self.p1(final_to_decode))
        pred1 = self.activation(self.p2(pred1))
        pred1 = F.softmax(self.activation(self.p3(pred1)), dim=1)
        pred2 = self.activation(self.q1(final_to_decode))
        pred2 = self.activation(self.q2(pred2))
        pred2 = F.softmax(self.activation(self.q3(pred2)), dim=1)
        return torch.cat((action, pred1, pred2), 1).flatten()