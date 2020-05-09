from torchviz import make_dot
from torch.autograd import Variable

def viz_graph(model, data):
    ''' takes in model and input data and vizualises computational graph, saves to pdf file'''
    output = model(Variable(data.double()))
    graph = make_dot(output)
    graph.view()

def get_trainanble_parameters(model):
    ''' returns list of trainable parameters for model '''
    for name, _ in model.named_parameters():
        print(name)
