import torch.nn


class BasicModel(torch.nn.Module):

    def __init__(self):
        super(BasicModel, self).__init__()
        self.model_name = str(type(self))

    def save(self, name = ''):
        prefix = 'checkpoints/' + self.model_name
        import time
        name = time.strftime(prefix + '_' + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, path):
        self.load_state_dict(t.load(path))