import torch
from data import CorruptMnist
from model import MyAwesomeModel

class Evaluate(object):
    
    def __init__(self):
       
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("Evaluating until hitting the ceiling")

        model = MyAwesomeModel()
        model.load_state_dict(torch.load("./models/trained_model.pt"))
        model = model.to(self.device)

        test_set = CorruptMnist(train=False)
        dataloader = torch.utils.data.DataLoader(test_set, batch_size=128)
        
        correct, total = 0, 0
        for batch in dataloader:
            x, y = batch
            
            preds = model(x.to(self.device))
            preds = preds.argmax(dim=-1)
            
            correct += (preds == y.to(self.device)).sum().item()
            total += y.numel()
            
        print(f" Test set accuracy {correct/total}")

if __name__ == '__main__':
    Evaluate()