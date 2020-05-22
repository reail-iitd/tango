from src.GNN.helper import *
from src.GNN.CONSTANTS import *
from src.utils import *
from src.GNN.oldmodels import *
torch.manual_seed(1)

# Contains the action prediction task models. Will be released in a future publication.

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

def action2vec_cons(action, num_objects, num_states):
    actionArray = torch.zeros(len(possibleActions))
    actionArray[possibleActions.index(action['name'])] = 1
    predicate1 = torch.zeros(num_objects)
    predicate2 = torch.zeros(num_objects)
    predicate3 = torch.zeros(num_states)
    if len(action['args']) == 1:
        predicate1[object2idx[action['args'][0]]] = 1
    else:
        if action['args'][1] in object2idx:
            predicate1[object2idx[action['args'][0]]] = 1
            predicate2[object2idx[action['args'][1]]] = 1
        else:
            predicate1[object2idx[action['args'][0]]] = 1
            predicate3[possibleStates.index(action['args'][1])] = 1
    return torch.cat((actionArray, predicate1, predicate2, predicate3), 0)

def action2vec_lstm(action, num_objects, num_states, num_hidden, embedder):
    actionArray = torch.zeros(len(possibleActions))
    actionArray[possibleActions.index(action['name'])] = 1
    predicate3 = torch.zeros(num_states)
    predicate2 = torch.zeros(num_hidden)
    if len(action['args']) == 0:
        predicate1 = torch.zeros(num_hidden)
    elif len(action['args']) == 1:
        predicate1 = embedder(torch.Tensor(object2vec[action['args'][0]]))
    else:
        # action['args'][1] can be a state or an object
        if action['args'][1] in object2idx:
            predicate1 = embedder(torch.Tensor(object2vec[action['args'][0]]))
            predicate2 = embedder(torch.Tensor(object2vec[action['args'][1]]))
        else:
            predicate1 = embedder(torch.Tensor(object2vec[action['args'][0]]))
            predicate3[possibleStates.index(action['args'][1])] = 1
    return torch.cat((actionArray, predicate1, predicate2, predicate3), 0)

def action2ids(action, num_objects, num_states):
    actionID = possibleActions.index(action['name'])
    predicate1, predicate2 = 0, 0
    if len(action['args']) == 0:
        predicate1 = num_objects+1
        predicate2 = num_objects+1
    elif len(action['args']) == 1:
        predicate1 = object2idx[action['args'][0]]
        predicate2 = num_objects+1
    else:
        # action['args'][1] can be a state or an object
        if action['args'][1] in object2idx:
            predicate1 = object2idx[action['args'][0]]
            predicate2 = object2idx[action['args'][1]]
        else:
            predicate1 = object2idx[action['args'][0]]
            predicate2 = num_objects + 1 + possibleStates.index(action['args'][1])
    return actionID, predicate1, predicate2

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

def vec2action_grammatical(vec, num_objects, num_states, idx2object):
    ret_action = {}
    action_array = list(vec[:len(possibleActions)])
    ret_action["name"] = possibleActions[action_array.index(max(action_array))]
    ret_action["args"] = []
    object1_array = list(vec[len(possibleActions):len(possibleActions)+num_objects])
    object1_ind = object1_array.index(max(object1_array))
    ret_action["args"].append(idx2object[object1_ind])
    if ret_action["name"] in ["moveTo", "pick", "climbUp", "climbDown", "clean"]:
        return ret_action
    object2_array = list(vec[len(possibleActions)+num_objects:len(possibleActions)+num_objects+num_objects])
    state_array = list(vec[len(possibleActions)+num_objects+num_objects:])
    assert len(state_array) == len(possibleStates)
    if ret_action["name"] == "changeState":
        for obj in all_objects:
            if obj not in all_objects_with_states: object1_array[object2idx[obj]] = 0
        ret_action["args"][-1] = idx2object[object1_array.index(max(object1_array))]
        ret_action["args"].append(possibleStates[state_array.index(max(state_array))])
    else:
        ret_action["args"].append(idx2object[object2_array.index(max(object2_array))])
    return ret_action

def tool2object_likelihoods(num_objects, tool_likelihoods):
    object_likelihoods = torch.zeros(num_objects)
    for i, tool in enumerate(TOOLS2):
        object_likelihoods[object2idx[tool]] = tool_likelihoods[i]
    return object_likelihoods

#############################################################################

class GGCN_Metric_Attn_Aseq_Auto_Cons_C_Action(nn.Module):
    def __init__(self,
                 in_feats,
                 n_objects,
                 n_hidden,
                 n_states,
                 n_layers,
                 etypes,
                 activation,
                 dropout):
        super(GGCN_Metric_Attn_Aseq_Auto_Cons_C_Action, self).__init__()
        self.name = "GGCN_Metric_Attn_Aseq_Auto_Cons_C_Action_" + str(n_hidden) + "_" + str(n_layers)
        self.layers = nn.ModuleList()
        self.layers.append(GatedHeteroRGCNLayer(in_feats, n_hidden, etypes, activation=activation))
        for i in range(n_layers - 1):
            self.layers.append(GatedHeteroRGCNLayer(n_hidden, n_hidden, etypes, activation=activation))
        self.attention = nn.Sequential(nn.Linear(n_hidden + n_hidden + n_hidden + n_hidden, n_hidden), nn.Linear(n_hidden, 1))
        self.embed = nn.Linear(PRETRAINED_VECTOR_SIZE, n_hidden)
        self.fc1 = nn.Linear(n_hidden + n_hidden + n_hidden + n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, len(possibleActions))
        self.p1  = nn.Linear(n_hidden + n_hidden + n_hidden + n_hidden + len(possibleActions), n_hidden + n_hidden)
        self.p2  = nn.Linear(n_hidden + n_hidden, n_hidden)
        self.p3  = nn.Linear(n_hidden, n_objects)
        self.q1  = nn.Linear(n_hidden + n_hidden + n_hidden + n_hidden + len(possibleActions), n_hidden + n_hidden)
        self.q2  = nn.Linear(n_hidden + n_hidden, n_hidden)
        self.q3  = nn.Linear(n_hidden, n_objects+n_states)
        self.activation = nn.LeakyReLU()
        self.metric1 = nn.Linear(in_feats, n_hidden)
        self.metric2 = nn.Linear(n_hidden, n_hidden)
        self.action_lstm = nn.LSTM(len(possibleActions) + n_objects + 1 + n_objects + 1 + n_states, n_hidden)
        self.n_hidden = n_hidden
        self.n_objects = n_objects
        self.n_states = n_states

    def forward(self, g_list, goalVec, goalObjectsVec, a_list):
        a_list = [action2vec(i, self.n_objects, self.n_states) for i in a_list]
        predicted_actions = []
        lstm_hidden = (torch.randn(1, 1, self.n_hidden),torch.randn(1, 1, self.n_hidden))
        goalObjectsVec = self.activation(self.embed(torch.Tensor(goalObjectsVec)))
        goal_embed = self.activation(self.embed(torch.Tensor(goalVec.reshape(1, -1))))
        for ind,g in enumerate(g_list):
            h = g.ndata['feat']
            for i, layer in enumerate(self.layers):
                h = layer(g, h)
            metric_part = g.ndata['feat']
            metric_part = self.activation(self.metric1(metric_part))
            metric_part = self.activation(self.metric2(metric_part))
            h = torch.cat([h, metric_part], dim = 1)
            if (ind != 0):
                lstm_out, lstm_hidden = self.action_lstm(a_list[ind-1].view(1,1,-1), lstm_hidden)
            else:
                lstm_out = torch.zeros(1, 1, self.n_hidden)
            lstm_out = lstm_out.view(-1)
            attn_embedding = torch.cat([h, goalObjectsVec.repeat(h.size(0)).view(h.size(0), -1), lstm_out.repeat(h.size(0)).view(h.size(0), -1)], 1)
            attn_weights = F.softmax(self.attention(attn_embedding), dim=0)
            scene_embedding = torch.mm(attn_weights.t(), h)
            final_to_decode = torch.cat([scene_embedding, goal_embed, lstm_out.view(1,-1)], 1)
            action = self.activation(self.fc1(final_to_decode))
            action = self.activation(self.fc2(action))
            action = self.activation(self.fc3(action))
            action = F.softmax(action, dim=1)
            pred_action_values = list(action[0])
            ind_max_action = pred_action_values.index(max(pred_action_values))
            one_hot_action = [0 for i in range(len(pred_action_values))]; one_hot_action[ind_max_action] = 1
            one_hot_action = torch.Tensor(one_hot_action).view(1,-1)
            pred1 = self.activation(self.p1(torch.cat([final_to_decode, one_hot_action], 1)))
            pred1 = self.activation(self.p2(pred1))
            pred1 = F.softmax(self.activation(self.p3(pred1)), dim=1)
            pred2 = self.activation(self.q1(torch.cat([final_to_decode, one_hot_action], 1)))
            pred2 = self.activation(self.q2(pred2))
            pred2 = F.softmax(self.activation(self.q3(pred2)), dim=1)
            predicted_actions.append(torch.cat((action, pred1, pred2), 1).flatten())
        return predicted_actions

class GGCN_Metric_Attn_Aseq_L_Auto_Tool_Cons_C_Action(nn.Module):
    def __init__(self,
                 in_feats,
                 n_objects,
                 n_hidden,
                 n_states,
                 n_layers,
                 etypes,
                 activation,
                 dropout):
        super(GGCN_Metric_Attn_Aseq_L_Auto_Tool_Cons_C_Action, self).__init__()
        self.name = "GGCN_Metric_Attn_Aseq_L_Auto_Tool_Cons_C_Action_" + str(n_hidden) + "_" + str(n_layers)
        self.layers = nn.ModuleList()
        self.activation = nn.PReLU()
        self.layers.append(GatedHeteroRGCNLayer(in_feats, n_hidden, etypes, activation=activation))
        for i in range(n_layers - 1):
            self.layers.append(GatedHeteroRGCNLayer(n_hidden, n_hidden, etypes, activation=activation))
        self.attention = nn.Sequential(nn.Linear(n_hidden + n_hidden + n_hidden + n_hidden + 1, n_hidden), self.activation, nn.Linear(n_hidden, 1))
        self.embed = nn.Sequential(nn.Linear(PRETRAINED_VECTOR_SIZE, n_hidden), self.activation, nn.Linear(n_hidden, n_hidden))
        self.fc1 = nn.Linear(n_hidden*4, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, len(possibleActions))
        self.p1_object  = nn.Linear(n_hidden*5 + len(possibleActions) + 1, n_hidden)
        self.p2_object  = nn.Linear(n_hidden, n_hidden)
        self.p3_object  = nn.Linear(n_hidden, 1)
        self.q1_object  = nn.Linear(n_hidden*5 + len(possibleActions) + 1 + 1, n_hidden)
        self.q2_object  = nn.Linear(n_hidden, n_hidden)
        self.q3_object  = nn.Linear(n_hidden, 1)
        self.q1_state  = nn.Linear(n_hidden*4 + len(possibleActions), n_hidden)
        self.q2_state  = nn.Linear(n_hidden, n_hidden)
        self.q3_state  = nn.Linear(n_hidden, n_states)
        self.metric1 = nn.Linear(in_feats, n_hidden)
        self.metric2 = nn.Linear(n_hidden, n_hidden)
        self.action_lstm = nn.LSTM(len(possibleActions) + PRETRAINED_VECTOR_SIZE + PRETRAINED_VECTOR_SIZE + n_states, n_hidden)
        self.n_hidden = n_hidden
        self.n_objects = n_objects
        self.n_states = n_states
        l = []
        for obj in all_objects:
            l.append(object2vec[obj])
        self.object_vec = torch.Tensor(l)

    def forward(self, g_list, goalVec, goalObjectsVec, a_list,  object_likelihoods):
        a_list = [action2vec_generalizable(i, self.n_objects, self.n_states) for i in a_list]
        predicted_actions = []
        lstm_hidden = (torch.randn(1, 1, self.n_hidden),torch.randn(1, 1, self.n_hidden))
        goalObjectsVec = self.activation(self.embed(torch.Tensor(goalObjectsVec)))
        goal_embed = self.activation(self.embed(torch.Tensor(goalVec.reshape(1, -1))))
        for ind,g in enumerate(g_list):
            h = g.ndata['feat']
            for i, layer in enumerate(self.layers):
                h = layer(g, h)
            metric_part = g.ndata['feat']
            metric_part = self.activation(self.metric1(metric_part))
            metric_part = self.activation(self.metric2(metric_part))
            h = torch.cat([h, metric_part], dim = 1)
            if (ind != 0):
                lstm_out, lstm_hidden = self.action_lstm(a_list[ind-1].view(1,1,-1), lstm_hidden)
            else:
                lstm_out = torch.zeros(1, 1, self.n_hidden)
            lstm_out = lstm_out.view(-1)
            attn_embedding = torch.cat([h, goalObjectsVec.repeat(h.size(0)).view(h.size(0), -1), lstm_out.repeat(h.size(0)).view(h.size(0), -1), object_likelihoods[ind].view(h.size(0),-1)], 1)
            attn_weights = F.softmax(self.attention(attn_embedding), dim=0)
            scene_embedding = torch.mm(attn_weights.t(), h)
            final_to_decode = torch.cat([scene_embedding, goal_embed, lstm_out.view(1,-1)], 1)
            action = self.activation(self.fc1(final_to_decode))
            action = self.activation(self.fc2(action))
            action = self.fc3(action)
            action = F.softmax(action, dim=1)
            pred_action_values = list(action[0])
            ind_max_action = pred_action_values.index(max(pred_action_values))
            one_hot_action = [0] * len(pred_action_values); one_hot_action[ind_max_action] = 1
            one_hot_action = torch.Tensor(one_hot_action).view(1,-1)

            #Predicting the first argument of the action
            pred1_input = torch.cat([final_to_decode, one_hot_action], 1)
            pred1_object = self.activation(self.p1_object(
                        torch.cat([pred1_input.view(-1).repeat(self.n_objects).view(self.n_objects, -1), self.activation(self.embed(self.object_vec)), object_likelihoods[ind].view(self.n_objects, -1)], 1)))
            pred1_object = self.activation(self.p2_object(pred1_object))
            pred1_object = self.p3_object(pred1_object)
            pred1_output = torch.sigmoid(pred1_object)

            # Predicting the second argument of the action
            pred2_input = torch.cat([final_to_decode, one_hot_action], 1)
            pred2_object = self.activation(self.q1_object(
                        torch.cat([pred2_input.view(-1).repeat(self.n_objects).view(self.n_objects, -1), self.activation(self.embed(self.object_vec)), object_likelihoods[ind].view(self.n_objects, -1), pred1_output.view(self.n_objects, 1)], 1)))
            pred2_object = self.activation(self.q2_object(pred2_object))
            pred2_object = self.q3_object(pred2_object)
            pred2_object = torch.sigmoid(pred2_object)

            pred2_state = self.activation(self.q1_state(pred2_input))
            pred2_state = self.activation(self.q2_state(pred2_state))
            pred2_state = self.q3_state(pred2_state)
            pred2_state = torch.sigmoid(pred2_state)
            pred2_output = torch.cat([pred2_object.view(1,-1), pred2_state], 1)
            predicted_actions.append(torch.cat((action, pred1_output.view(1,-1), pred2_output.view(1,-1)), 1).flatten())
        return predicted_actions

class GGCN_Metric_Attn_Aseq_L_Auto_Cons_C_Action(nn.Module):
    def __init__(self,
                 in_feats,
                 n_objects,
                 n_hidden,
                 n_states,
                 n_layers,
                 etypes,
                 activation,
                 dropout):
        super(GGCN_Metric_Attn_Aseq_L_Auto_Cons_C_Action, self).__init__()
        self.name = "GGCN_Metric_Attn_Aseq_L_Auto_Cons_C_Action_" + str(n_hidden) + "_" + str(n_layers)
        self.layers = nn.ModuleList()
        self.activation = nn.PReLU()
        self.layers.append(GatedHeteroRGCNLayer(in_feats, n_hidden, etypes, activation=activation))
        for i in range(n_layers - 1):
            self.layers.append(GatedHeteroRGCNLayer(n_hidden, n_hidden, etypes, activation=activation))
        self.attention = nn.Sequential(nn.Linear(n_hidden + n_hidden + n_hidden + n_hidden, n_hidden), self.activation, nn.Linear(n_hidden, 1))
        self.embed = nn.Sequential(nn.Linear(PRETRAINED_VECTOR_SIZE, n_hidden), self.activation, nn.Linear(n_hidden, n_hidden))
        self.fc1 = nn.Linear(n_hidden*4, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, len(possibleActions))
        self.p1_object  = nn.Linear(n_hidden*5 + len(possibleActions), n_hidden)
        self.p2_object  = nn.Linear(n_hidden, n_hidden)
        self.p3_object  = nn.Linear(n_hidden, 1)
        self.q1_object  = nn.Linear(n_hidden*5 + len(possibleActions) + 1, n_hidden)
        self.q2_object  = nn.Linear(n_hidden, n_hidden)
        self.q3_object  = nn.Linear(n_hidden, 1)
        self.q1_state  = nn.Linear(n_hidden*4 + len(possibleActions), n_hidden)
        self.q2_state  = nn.Linear(n_hidden, n_hidden)
        self.q3_state  = nn.Linear(n_hidden, n_states)
        self.metric1 = nn.Linear(in_feats, n_hidden)
        self.metric2 = nn.Linear(n_hidden, n_hidden)
        self.action_lstm = nn.LSTM(len(possibleActions) + PRETRAINED_VECTOR_SIZE + PRETRAINED_VECTOR_SIZE + n_states, n_hidden)
        self.n_hidden = n_hidden
        self.n_objects = n_objects
        self.n_states = n_states
        l = []
        for obj in all_objects:
            l.append(object2vec[obj])
        self.object_vec = torch.Tensor(l)

    def forward(self, g_list, goalVec, goalObjectsVec, a_list):
        a_list = [action2vec_generalizable(i, self.n_objects, self.n_states) for i in a_list]
        predicted_actions = []
        lstm_hidden = (torch.randn(1, 1, self.n_hidden),torch.randn(1, 1, self.n_hidden))
        goalObjectsVec = self.activation(self.embed(torch.Tensor(goalObjectsVec)))
        goal_embed = self.activation(self.embed(torch.Tensor(goalVec.reshape(1, -1))))
        for ind,g in enumerate(g_list):
            h = g.ndata['feat']
            for i, layer in enumerate(self.layers):
                h = layer(g, h)
            metric_part = g.ndata['feat']
            metric_part = self.activation(self.metric1(metric_part))
            metric_part = self.activation(self.metric2(metric_part))
            h = torch.cat([h, metric_part], dim = 1)
            if (ind != 0):
                lstm_out, lstm_hidden = self.action_lstm(a_list[ind-1].view(1,1,-1), lstm_hidden)
            else:
                lstm_out = torch.zeros(1, 1, self.n_hidden)
            lstm_out = lstm_out.view(-1)
            attn_embedding = torch.cat([h, goalObjectsVec.repeat(h.size(0)).view(h.size(0), -1), lstm_out.repeat(h.size(0)).view(h.size(0), -1)], 1)
            attn_weights = F.softmax(self.attention(attn_embedding), dim=0)
            scene_embedding = torch.mm(attn_weights.t(), h)
            final_to_decode = torch.cat([scene_embedding, goal_embed, lstm_out.view(1,-1)], 1)
            action = self.activation(self.fc1(final_to_decode))
            action = self.activation(self.fc2(action))
            action = self.fc3(action)
            action = F.softmax(action, dim=1)
            pred_action_values = list(action[0])
            ind_max_action = pred_action_values.index(max(pred_action_values))
            one_hot_action = [0] * len(pred_action_values); one_hot_action[ind_max_action] = 1
            one_hot_action = torch.Tensor(one_hot_action).view(1,-1)

            #Predicting the first argument of the action
            pred1_input = torch.cat([final_to_decode, one_hot_action], 1)
            pred1_object = self.activation(self.p1_object(
                        torch.cat([pred1_input.view(-1).repeat(self.n_objects).view(self.n_objects, -1), self.activation(self.embed(self.object_vec))], 1)))
            pred1_object = self.activation(self.p2_object(pred1_object))
            pred1_object = self.p3_object(pred1_object)
            pred1_output = torch.sigmoid(pred1_object)

            # Predicting the second argument of the action
            pred2_input = torch.cat([final_to_decode, one_hot_action], 1)
            pred2_object = self.activation(self.q1_object(
                        torch.cat([pred2_input.view(-1).repeat(self.n_objects).view(self.n_objects, -1), self.activation(self.embed(self.object_vec)), pred1_output.view(self.n_objects, 1)], 1)))
            pred2_object = self.activation(self.q2_object(pred2_object))
            pred2_object = self.q3_object(pred2_object)
            pred2_object = torch.sigmoid(pred2_object)

            pred2_state = self.activation(self.q1_state(pred2_input))
            pred2_state = self.activation(self.q2_state(pred2_state))
            pred2_state = self.q3_state(pred2_state)
            pred2_state = torch.sigmoid(pred2_state)
            pred2_output = torch.cat([pred2_object.view(1,-1), pred2_state], 1)
            predicted_actions.append(torch.cat((action, pred1_output.view(1,-1), pred2_output.view(1,-1)), 1).flatten())
        return predicted_actions

class GGCN_Metric_Attn_Aseq_L_Auto_Tool_G_N_Cons_C_Action(nn.Module):
    def __init__(self,
                 in_feats,
                 n_objects,
                 n_hidden,
                 n_states,
                 n_layers,
                 etypes,
                 activation,
                 dropout):
        super(GGCN_Metric_Attn_Aseq_L_Auto_Tool_G_N_Cons_C_Action, self).__init__()
        self.name = "GGCN_Metric_Attn_Aseq_L_Auto_Tool_G_N_Cons_C_Action_" + str(n_hidden) + "_" + str(n_layers)
        self.layers = nn.ModuleList()
        self.activation = nn.PReLU()
        self.layers.append(GatedHeteroRGCNLayer(in_feats, n_hidden, etypes, activation=activation))
        for i in range(n_layers - 1):
            self.layers.append(GatedHeteroRGCNLayer(n_hidden, n_hidden, etypes, activation=activation))
        self.attention = nn.Sequential(nn.Linear(n_hidden + n_hidden + n_hidden + n_hidden + 1, n_hidden), self.activation, nn.Linear(n_hidden, 1))
        self.embed = nn.Sequential(nn.Linear(PRETRAINED_VECTOR_SIZE, n_hidden), self.activation, nn.Linear(n_hidden, n_hidden))
        self.fc1 = nn.Linear(n_hidden*4, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, len(possibleActions))
        self.p1_object  = nn.Linear(n_hidden*4 + len(possibleActions) + 3*n_hidden + 1 + 1 + 1, 2*n_hidden)
        self.p2_object  = nn.Linear(2*n_hidden, n_hidden)
        self.p3_object  = nn.Linear(n_hidden, 1)
        self.zero_embed_sz = 4
        self.zero_one_embeddings = nn.Embedding(2,self.zero_embed_sz)
        self.q1_object  = nn.Linear(n_hidden*4 + len(possibleActions) + 3*n_hidden + 1 + self.zero_embed_sz + 1 + 1, 2*n_hidden)
        self.q2_object  = nn.Linear(2*n_hidden, n_hidden)
        self.q3_object  = nn.Linear(n_hidden, 1)
        self.q1_state  = nn.Linear(n_hidden*4 + 3*n_hidden, n_hidden)
        self.q2_state  = nn.Linear(n_hidden, n_hidden)
        self.q3_state  = nn.Linear(n_hidden, n_states)
        self.metric1 = nn.Linear(in_feats, n_hidden)
        self.metric2 = nn.Linear(n_hidden, n_hidden)
        self.action_lstm = nn.LSTM(len(possibleActions) + PRETRAINED_VECTOR_SIZE + PRETRAINED_VECTOR_SIZE + n_states, n_hidden)
        self.n_hidden = n_hidden
        self.n_objects = n_objects
        self.n_states = n_states
        l = []
        for obj in all_objects:
            l.append(object2vec[obj])
        self.object_vec = torch.Tensor(l)

    def forward(self, g_list, goalVec, goalObjectsVec, a_list,  object_likelihoods):
        a_list = [action2vec_generalizable(i, self.n_objects, self.n_states) for i in a_list]
        predicted_actions = []
        lstm_hidden = (torch.randn(1, 1, self.n_hidden),torch.randn(1, 1, self.n_hidden))
        goalObjectsVec = self.activation(self.embed(torch.Tensor(goalObjectsVec)))
        goal_embed = self.activation(self.embed(torch.Tensor(goalVec.reshape(1, -1))))
        for ind,g in enumerate(g_list):
            h = g.ndata['feat']
            goal_data = h[:,-1].view(self.n_objects, -1)
            close_data = g.ndata['close']
            for i, layer in enumerate(self.layers):
                h = layer(g, h)
            metric_part = g.ndata['feat']
            metric_part = self.activation(self.metric1(metric_part))
            metric_part = self.activation(self.metric2(metric_part))
            h = torch.cat([h, metric_part], dim = 1)
            if (ind != 0):
                lstm_out, lstm_hidden = self.action_lstm(a_list[ind-1].view(1,1,-1), lstm_hidden)
            else:
                lstm_out = torch.zeros(1, 1, self.n_hidden)
            lstm_out = lstm_out.view(-1)
            attn_embedding = torch.cat([h, goalObjectsVec.repeat(h.size(0)).view(h.size(0), -1), lstm_out.repeat(h.size(0)).view(h.size(0), -1), object_likelihoods[ind].view(h.size(0),-1)], 1)
            attn_weights = F.softmax(self.attention(attn_embedding), dim=0)
            scene_embedding = torch.mm(attn_weights.t(), h)
            final_to_decode = torch.cat([scene_embedding, goal_embed, lstm_out.view(1,-1)], 1)
            action = self.activation(self.fc1(final_to_decode))
            action = self.activation(self.fc2(action))
            action = self.fc3(action)
            action = F.softmax(action, dim=1)
            pred_action_values = list(action[0])
            ind_max_action = pred_action_values.index(max(pred_action_values))
            one_hot_action = [0] * len(pred_action_values); one_hot_action[ind_max_action] = 1
            one_hot_action = torch.Tensor(one_hot_action).view(1,-1)

            #Predicting the first argument of the action
            pred1_input = torch.cat([final_to_decode, one_hot_action], 1)
            pred1_object = self.activation(self.p1_object(
                        torch.cat([pred1_input.view(-1).repeat(self.n_objects).view(self.n_objects, -1), self.activation(self.embed(self.object_vec)), h, object_likelihoods[ind].view(self.n_objects, -1), goal_data, close_data], 1)))
            pred1_object = self.activation(self.p2_object(pred1_object))
            pred1_object = self.p3_object(pred1_object)
            pred1_output = torch.sigmoid(pred1_object)
            pred1_values = list(pred1_output.view(1,-1)[0])
            ind_max_pred1 = pred1_values.index(max(pred1_values))
            one_hot_pred1 = [self.zero_one_embeddings(torch.LongTensor([0]).view(-1))]*len(pred1_values); one_hot_pred1[ind_max_pred1] = self.zero_one_embeddings(torch.LongTensor([1]).view(-1))
            one_hot_pred1 = torch.stack(one_hot_pred1).view(-1,self.zero_embed_sz)

            # Predicting the second argument of the action
            if ind_max_action != possibleActions.index('changeState'):
                pred2_input = torch.cat([final_to_decode, one_hot_action], 1)
                pred2_object = self.activation(self.q1_object(
                            torch.cat([pred2_input.view(-1).repeat(self.n_objects).view(self.n_objects, -1), self.activation(self.embed(self.object_vec)), h, object_likelihoods[ind].view(self.n_objects, -1), one_hot_pred1, goal_data, close_data], 1)))
                pred2_object = self.activation(self.q2_object(pred2_object))
                pred2_object = self.q3_object(pred2_object)
                pred2_object = torch.sigmoid(pred2_object)
                pred2_state = torch.zeros([1, self.n_states])
            else:
                pred2_state = self.activation(self.q1_state(torch.cat([final_to_decode, self.activation(self.embed(self.object_vec[ind_max_pred1])).view(1,-1), h[ind_max_pred1].view(1,-1)], 1)))
                pred2_state = self.activation(self.q2_state(pred2_state))
                pred2_state = self.q3_state(pred2_state)
                pred2_state = torch.sigmoid(pred2_state)
                pred2_object = torch.zeros([1, self.n_objects])
            pred2_output = torch.cat([pred2_object.view(1,-1), pred2_state], 1)
            predicted_actions.append(torch.cat((action, pred1_output.view(1,-1), pred2_output.view(1,-1)), 1).flatten())
        return predicted_actions

class GGCN_Metric_Attn_Aseq_L_Auto_Tool_G_N_Cons_C_2_Action(nn.Module):
    def __init__(self,
                 in_feats,
                 n_objects,
                 n_hidden,
                 n_states,
                 n_layers,
                 etypes,
                 activation,
                 dropout):
        super(GGCN_Metric_Attn_Aseq_L_Auto_Tool_G_N_Cons_C_2_Action, self).__init__()
        self.name = "GGCN_Metric_Attn_Aseq_L_Auto_Tool_G_N_Cons_C_Action_" + str(n_hidden) + "_" + str(n_layers)
        self.layers = nn.ModuleList()
        self.activation = nn.PReLU()
        self.attention = nn.Sequential(nn.Linear(n_hidden + n_hidden + n_hidden + n_hidden , n_hidden), self.activation, nn.Linear(n_hidden, 1))
        self.embed = nn.Sequential(nn.Linear(PRETRAINED_VECTOR_SIZE, n_hidden), self.activation, nn.Linear(n_hidden, n_hidden))
        self.fc1 = nn.Linear(n_hidden*4, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, len(possibleActions))
        self.p1_object  = nn.Linear(n_hidden*4 + len(possibleActions) + in_feats + 1 + 1, 2*n_hidden)
        self.p2_object  = nn.Linear(2*n_hidden, 2*n_hidden)
        self.p3_object  = nn.Linear(2*n_hidden, 1)
        self.zero_embed_sz = 4
        self.zero_one_embeddings = nn.Embedding(2,self.zero_embed_sz)
        self.q1_object  = nn.Linear(n_hidden*4 + len(possibleActions) + in_feats + self.zero_embed_sz + 1 + 1, 2*n_hidden)
        self.q2_object  = nn.Linear(2*n_hidden, 2*n_hidden)
        self.q3_object  = nn.Linear(2*n_hidden, 1)
        self.q1_state  = nn.Linear(n_hidden*4 + in_feats, n_hidden)
        self.q2_state  = nn.Linear(n_hidden, n_hidden)
        self.q3_state  = nn.Linear(n_hidden, n_states)
        self.metric1 = nn.Linear(in_feats + n_objects*4 , 2*n_hidden)
        self.metric2 = nn.Linear(2*n_hidden, 2*n_hidden)
        self.metric3 = nn.Linear(2*n_hidden, 2*n_hidden)
        self.action_lstm = nn.LSTM(len(possibleActions) + PRETRAINED_VECTOR_SIZE + PRETRAINED_VECTOR_SIZE + n_states, n_hidden)
        self.n_hidden = n_hidden
        self.n_objects = n_objects
        self.n_states = n_states
        self.etypes = etypes
        l = []
        for obj in all_objects:
            l.append(object2vec[obj])
        self.object_vec = torch.Tensor(l)

    def forward(self, g_list, goalVec, goalObjectsVec, a_list,  object_likelihoods):
        a_list = [action2vec_generalizable(i, self.n_objects, self.n_states) for i in a_list]
        predicted_actions = []
        lstm_hidden = (torch.randn(1, 1, self.n_hidden),torch.randn(1, 1, self.n_hidden))
        goalObjectsVec = self.activation(self.embed(torch.Tensor(goalObjectsVec)))
        goal_embed = self.activation(self.embed(torch.Tensor(goalVec.reshape(1, -1))))
        for ind,g in enumerate(g_list):
            goal_data = g.ndata['feat'][:,-1].view(self.n_objects, -1)
            close_data = g.ndata['close']
            h = g.ndata['feat']
            edgeMatrices = [g.adjacency_matrix(etype=t) for t in self.etypes]
            edges = torch.cat(edgeMatrices, 1).to_dense()
            h = torch.cat((h, edges), 1)
            h = self.activation(self.metric1(h))
            h = self.activation(self.metric2(h))
            h = self.activation(self.metric3(h))
            if (ind != 0):
                lstm_out, lstm_hidden = self.action_lstm(a_list[ind-1].view(1,1,-1), lstm_hidden)
            else:
                lstm_out = torch.zeros(1, 1, self.n_hidden)
            lstm_out = lstm_out.view(-1)
            attn_embedding = torch.cat([h, goalObjectsVec.repeat(h.size(0)).view(h.size(0), -1), lstm_out.repeat(h.size(0)).view(h.size(0), -1)], 1)
            attn_weights = F.softmax(self.attention(attn_embedding), dim=0)
            scene_embedding = torch.mm(attn_weights.t(), h)
            final_to_decode = torch.cat([scene_embedding, goal_embed, lstm_out.view(1,-1)], 1)
            action = self.activation(self.fc1(final_to_decode))
            action = self.activation(self.fc2(action))
            action = self.fc3(action)
            action = F.softmax(action, dim=1)
            pred_action_values = list(action[0])
            ind_max_action = pred_action_values.index(max(pred_action_values))
            one_hot_action = [0] * len(pred_action_values); one_hot_action[ind_max_action] = 1
            one_hot_action = torch.Tensor(one_hot_action).view(1,-1)

            #Predicting the first argument of the action
            pred1_input = torch.cat([final_to_decode, one_hot_action], 1)
            pred1_object = self.activation(self.p1_object(
                        torch.cat([pred1_input.view(-1).repeat(self.n_objects).view(self.n_objects, -1), g.ndata['feat'], goal_data, close_data], 1)))
            pred1_object = self.activation(self.p2_object(pred1_object))
            pred1_object = self.p3_object(pred1_object)
            pred1_output = torch.sigmoid(pred1_object)
            pred1_values = list(pred1_output.view(1,-1)[0])
            ind_max_pred1 = pred1_values.index(max(pred1_values))
            one_hot_pred1 = [self.zero_one_embeddings(torch.LongTensor([0]).view(-1))]*len(pred1_values); one_hot_pred1[ind_max_pred1] = self.zero_one_embeddings(torch.LongTensor([1]).view(-1))
            one_hot_pred1 = torch.stack(one_hot_pred1).view(-1,self.zero_embed_sz)

            # Predicting the second argument of the action
            if ind_max_action != possibleActions.index('changeState'):
                pred2_input = torch.cat([final_to_decode, one_hot_action], 1)
                pred2_object = self.activation(self.q1_object(
                            torch.cat([pred2_input.view(-1).repeat(self.n_objects).view(self.n_objects, -1), g.ndata['feat'], one_hot_pred1, goal_data, close_data], 1)))
                pred2_object = self.activation(self.q2_object(pred2_object))
                pred2_object = self.q3_object(pred2_object)
                pred2_object = torch.sigmoid(pred2_object)
                pred2_state = torch.zeros([1, self.n_states])
            else:
                pred2_state = self.activation(self.q1_state(torch.cat([final_to_decode, g.ndata['feat'][ind_max_pred1].view(1,-1)], 1)))
                pred2_state = self.activation(self.q2_state(pred2_state))
                pred2_state = self.q3_state(pred2_state)
                pred2_state = torch.sigmoid(pred2_state)
                pred2_object = torch.zeros([1, self.n_objects])
            pred2_output = torch.cat([pred2_object.view(1,-1), pred2_state], 1)
            predicted_actions.append(torch.cat((action, pred1_output.view(1,-1), pred2_output.view(1,-1)), 1).flatten())
        return predicted_actions

class GGCN_Metric_Attn_Aseq_L_Auto_Tool_G_N_Cons_C_3_Action(nn.Module):
    def __init__(self,
                 in_feats,
                 n_objects,
                 n_hidden,
                 n_states,
                 n_layers,
                 etypes,
                 activation,
                 dropout):
        super(GGCN_Metric_Attn_Aseq_L_Auto_Tool_G_N_Cons_C_4_Action, self).__init__()
        self.name = "GGCN_Metric_Attn_Aseq_L_Auto_Tool_G_N_Cons_C_4_Action_" + str(n_hidden) + "_" + str(n_layers)
        self.layers = nn.ModuleList()
        self.activation = nn.PReLU()
        self.layers.append(GatedHeteroRGCNLayer(in_feats - 1, n_hidden, etypes, activation=activation))
        for i in range(n_layers - 1):
            self.layers.append(GatedHeteroRGCNLayer(n_hidden, n_hidden, etypes, activation=activation))
        self.zero_embed_sz = n_hidden
        self.zero_one_embeddings = nn.Embedding(2,self.zero_embed_sz)
        self.attention = nn.Sequential(nn.Linear(n_hidden + n_hidden + n_hidden + n_hidden + 1, n_hidden), self.activation, nn.Linear(n_hidden, 1))
        self.embed = nn.Sequential(nn.Linear(PRETRAINED_VECTOR_SIZE, 2*n_hidden), self.activation, nn.Linear(2*n_hidden, n_hidden), self.activation)
        self.a = nn.Sequential(nn.Linear(n_hidden*4, n_hidden), self.activation, nn.Linear(n_hidden, n_hidden), self.activation, nn.Linear(n_hidden, len(possibleActions)))
        self.p = nn.Sequential(nn.Linear(n_hidden*7 + 2*n_hidden, n_hidden), self.activation, nn.Linear(n_hidden, n_hidden), self.activation, nn.Linear(n_hidden, 1))
        self.q_obj = nn.Sequential(nn.Linear(n_hidden*7 + 2*n_hidden + self.zero_embed_sz, n_hidden), self.activation, nn.Linear(n_hidden, n_hidden), self.activation, nn.Linear(n_hidden, 1))
        self.q_state = nn.Sequential(nn.Linear(n_hidden*7 + 2*n_hidden, n_hidden), self.activation, nn.Linear(n_hidden, n_hidden), self.activation, nn.Linear(n_hidden, n_states))
        self.action_lstm = nn.LSTM(len(possibleActions) + n_hidden + n_hidden + n_states, n_hidden)
        self.action_embedding = nn.Embedding(len(possibleActions), 2*n_hidden)
        self.metric = nn.Sequential(nn.Linear(in_feats-1, n_hidden), self.activation, nn.Linear(n_hidden, n_hidden), self.activation, nn.Linear(n_hidden, n_hidden), self.activation)
        self.n_hidden = n_hidden
        self.n_objects = n_objects
        self.n_states = n_states
        self.etypes = etypes
        l = [object2vec[obj] for obj in all_objects]
        self.object_vec = torch.Tensor(l)

    def forward(self, g_list, goalVec, goalObjectsVec, a_list, object_likelihoods):
        a_list = [action2vec_lstm(i, self.n_objects, self.n_states, self.n_hidden, self.embed) for i in a_list]
        predicted_actions = []
        lstm_hidden = (torch.randn(1, 1, self.n_hidden),torch.randn(1, 1, self.n_hidden))
        goalObjectsVec = self.embed(torch.Tensor(goalObjectsVec))
        goal_embed = self.embed(torch.Tensor(goalVec.reshape(1, -1)))
        for ind,g in enumerate(g_list):
            h = g.ndata['feat'][:,:-1]
            for i, layer in enumerate(self.layers):
                h = layer(g, h)
            metric_part = g.ndata['feat'][:,:-1]
            metric_part = self.metric(metric_part)
            h = torch.cat([h, metric_part], dim = 1)
            if (ind != 0):
                lstm_out, lstm_hidden = self.action_lstm(a_list[ind-1].view(1,1,-1), lstm_hidden)
            else:
                lstm_out = torch.zeros(1, 1, self.n_hidden)
            lstm_out = lstm_out.view(-1)
            attn_embedding = torch.cat([h, goalObjectsVec.repeat(h.size(0)).view(h.size(0), -1), lstm_out.repeat(h.size(0)).view(h.size(0), -1), object_likelihoods[ind].view(h.size(0),-1)], 1)
            attn_weights = F.softmax(self.attention(attn_embedding), dim=0)
            scene_embedding = torch.mm(attn_weights.t(), h)
            final_to_decode = torch.cat([scene_embedding, goal_embed, lstm_out.view(1,-1)], 1)
            action = F.softmax(self.a(final_to_decode), dim=1)
            pred_action_values = list(action[0])
            ind_max_action = pred_action_values.index(max(pred_action_values))
            one_hot_action = self.action_embedding(torch.LongTensor([ind_max_action]))

            #Predicting the first argument of the action
            objs_embeddings = self.embed(self.object_vec)
            pred1_input = torch.cat([final_to_decode, one_hot_action], 1)
            pred1_output = F.softmax(self.p(torch.cat([pred1_input.view(-1).repeat(self.n_objects).view(self.n_objects, -1), objs_embeddings, h], 1)).view(1,-1), dim=1)
            pred1_values = list(pred1_output.view(-1))
            ind_max_pred1 = pred1_values.index(max(pred1_values))
            one_hot_pred1 = [self.zero_one_embeddings(torch.LongTensor([0]).view(-1))]*len(pred1_values); one_hot_pred1[ind_max_pred1] = self.zero_one_embeddings(torch.LongTensor([1]).view(-1))
            one_hot_pred1 = torch.stack(one_hot_pred1).view(-1,self.zero_embed_sz)

            # Predicting the second argument of the action
            pred2_object = self.q_obj(
                        torch.cat([pred1_input.view(-1).repeat(self.n_objects).view(self.n_objects, -1), objs_embeddings, h, one_hot_pred1], 1)).view(1,-1)
            pred2_state = self.q_state(torch.cat([pred1_input, objs_embeddings[ind_max_pred1].view(1,-1), h[ind_max_pred1].view(1,-1)], 1)).view(1,-1)
            pred2_output = F.softmax(torch.cat([pred2_object, pred2_state], 1), dim=1)
            predicted_actions.append(torch.cat((action, pred1_output, pred2_output), 1).flatten())
        return predicted_actions
class GGCN_Metric_Attn_Aseq_L_Auto_Tool_G_N_Cons_C_4_Action(nn.Module):
    def __init__(self,
                 in_feats,
                 n_objects,
                 n_hidden,
                 n_states,
                 n_layers,
                 etypes,
                 activation,
                 dropout):
        super(GGCN_Metric_Attn_Aseq_L_Auto_Tool_G_N_Cons_C_4_Action, self).__init__()
        self.name = "GGCN_Metric_Attn_Aseq_L_Auto_Tool_G_N_Cons_C_4_Action_" + str(n_hidden) + "_" + str(n_layers)
        self.layers = nn.ModuleList()
        self.activation = nn.PReLU()
        self.layers.append(nn.Linear(in_feats + 4*n_objects - 1, 2*n_hidden))
        for i in range(n_layers - 1):
            self.layers.append(nn.Linear(2*n_hidden, 2*n_hidden))
        self.zero_embed_sz = n_hidden
        self.zero_one_embeddings = nn.Embedding(2,self.zero_embed_sz)
        self.attention = nn.Sequential(nn.Linear(n_hidden + n_hidden + n_hidden + n_hidden + 1, n_hidden), self.activation, nn.Linear(n_hidden, 1))
        self.embed = nn.Sequential(nn.Linear(PRETRAINED_VECTOR_SIZE, 2*n_hidden), self.activation, nn.Linear(2*n_hidden, n_hidden), self.activation)
        self.a = nn.Sequential(nn.Linear(n_hidden*4, n_hidden), self.activation, nn.Linear(n_hidden, n_hidden), self.activation, nn.Linear(n_hidden, len(possibleActions)))
        self.p = nn.Sequential(nn.Linear(n_hidden*7 + 2*n_hidden, n_hidden), self.activation, nn.Linear(n_hidden, n_hidden), self.activation, nn.Linear(n_hidden, 1))
        self.q_obj = nn.Sequential(nn.Linear(n_hidden*7 + 2*n_hidden + self.zero_embed_sz, n_hidden), self.activation, nn.Linear(n_hidden, n_hidden), self.activation, nn.Linear(n_hidden, 1))
        self.q_state = nn.Sequential(nn.Linear(n_hidden*7 + 2*n_hidden, n_hidden), self.activation, nn.Linear(n_hidden, n_hidden), self.activation, nn.Linear(n_hidden, n_states))
        self.action_lstm = nn.LSTM(len(possibleActions) + n_hidden + n_hidden + n_states, n_hidden)
        self.action_embedding = nn.Embedding(len(possibleActions), 2*n_hidden)
        self.n_hidden = n_hidden
        self.n_objects = n_objects
        self.n_states = n_states
        self.etypes = etypes
        l = [object2vec[obj] for obj in all_objects]
        self.object_vec = torch.Tensor(l)

    def forward(self, g_list, goalVec, goalObjectsVec, a_list, object_likelihoods):
        a_list = [action2vec_lstm(i, self.n_objects, self.n_states, self.n_hidden, self.embed) for i in a_list]
        predicted_actions = []
        lstm_hidden = (torch.randn(1, 1, self.n_hidden),torch.randn(1, 1, self.n_hidden))
        goalObjectsVec = self.embed(torch.Tensor(goalObjectsVec))
        goal_embed = self.embed(torch.Tensor(goalVec.reshape(1, -1)))
        for ind,g in enumerate(g_list):
            h = g.ndata['feat'][:,:-1]
            edgeMatrices = [g.adjacency_matrix(etype=t) for t in self.etypes]
            edges = torch.cat(edgeMatrices, 1).to_dense()
            h = torch.cat((h, edges), 1)
            for i, layer in enumerate(self.layers):
                h = self.activation(layer(h))
            if (ind != 0):
                lstm_out, lstm_hidden = self.action_lstm(a_list[ind-1].view(1,1,-1), lstm_hidden)
            else:
                lstm_out = torch.zeros(1, 1, self.n_hidden)
            lstm_out = lstm_out.view(-1)
            attn_embedding = torch.cat([h, goalObjectsVec.repeat(h.size(0)).view(h.size(0), -1), lstm_out.repeat(h.size(0)).view(h.size(0), -1), object_likelihoods[ind].view(h.size(0),-1)], 1)
            attn_weights = F.softmax(self.attention(attn_embedding), dim=0)
            scene_embedding = torch.mm(attn_weights.t(), h)
            final_to_decode = torch.cat([scene_embedding, goal_embed, lstm_out.view(1,-1)], 1)
            action = F.softmax(self.a(final_to_decode), dim=1)
            pred_action_values = list(action[0])
            ind_max_action = pred_action_values.index(max(pred_action_values))
            one_hot_action = self.action_embedding(torch.LongTensor([ind_max_action]))

            #Predicting the first argument of the action
            objs_embeddings = self.embed(self.object_vec)
            pred1_input = torch.cat([final_to_decode, one_hot_action], 1)
            pred1_output = F.softmax(self.p(torch.cat([pred1_input.view(-1).repeat(self.n_objects).view(self.n_objects, -1), objs_embeddings, h], 1)).view(1,-1), dim=1)
            pred1_values = list(pred1_output.view(-1))
            ind_max_pred1 = pred1_values.index(max(pred1_values))
            one_hot_pred1 = [self.zero_one_embeddings(torch.LongTensor([0]).view(-1))]*len(pred1_values); one_hot_pred1[ind_max_pred1] = self.zero_one_embeddings(torch.LongTensor([1]).view(-1))
            one_hot_pred1 = torch.stack(one_hot_pred1).view(-1,self.zero_embed_sz)

            # Predicting the second argument of the action
            pred2_object = self.q_obj(
                        torch.cat([pred1_input.view(-1).repeat(self.n_objects).view(self.n_objects, -1), objs_embeddings, h, one_hot_pred1], 1)).view(1,-1)
            pred2_state = self.q_state(torch.cat([pred1_input, objs_embeddings[ind_max_pred1].view(1,-1), h[ind_max_pred1].view(1,-1)], 1)).view(1,-1)
            pred2_output = F.softmax(torch.cat([pred2_object, pred2_state], 1), dim=1)
            predicted_actions.append(torch.cat((action, pred1_output, pred2_output), 1).flatten())
        return predicted_actions