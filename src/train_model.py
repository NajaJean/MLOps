import torch
from data import CorruptMnist
from model import MyAwesomeModel
import matplotlib.pyplot as plt

print(torch.__version__)

class Train(object):
    
    def __init__(self):
       
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("Training.. ")

        lr = 1e-3

        model = MyAwesomeModel()
        model = model.to(self.device)
        train_set = CorruptMnist(train=True)

        dataloader = torch.utils.data.DataLoader(train_set, batch_size=128)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        
        n_epoch = 5
        for epoch in range(n_epoch):
            loss_tracker = []
            for batch in dataloader:
                optimizer.zero_grad()
                x, y = batch
                preds = model(x.to(self.device))
                loss = criterion(preds, y.to(self.device))
                loss.backward()
                optimizer.step()
                loss_tracker.append(loss.item())
            print(f" Epoch {epoch+1}/{n_epoch}. Loss: {loss}")  
            break      
        torch.save(model.state_dict(), '../models/trained_model.pt')
            
        plt.plot(loss_tracker, '-')
        plt.xlabel('Training step')
        plt.ylabel('Training loss')
        plt.savefig("training_curve.png")
        

if __name__ == '__main__':
    Train()

'''
TRAIN: python src\main.py train --lr 1e-4

EVALUATION: 
'''