import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import pandas as  pd
# timeorder=[]
# speed,pressure,blood=[],[],[]
# Right_atrial_pressure,Right_ventricular_pressure,Left_atrial_pressure,Left_ventricular_pressure=[],[],[],[]
# Aortic_blood_flow,Aortic_pressure=[],[]
from torch.utils.data import DataLoader

def standardization(data):
    mu = np.mean(data,axis=0)
    sigma = np.std(data,axis=0)
    return (data-mu)/sigma
def tensorToExcel(data):
    data = data.cpu().detach().numpy()
    data = pd.DataFrame(data)
    return data
def read_file(path):
    data_set=[]
    with open(path,'r',encoding='utf8') as f:
        lines = f.readlines()
        lines = [line.strip('\n').split(' ') for line in lines]
    f.close()
    for i in range(len(lines)):
        dataline = [float(data) for data in lines[i][:-1]]
        data_set.append(dataline)
    train_data, valid_data, test_data = np.array(data_set[:15000]), np.array(data_set[15000:17500]), np.array(data_set[17500:])
    return train_data, valid_data, test_data

class TimeOderData(data.Dataset):
    def __init__(self,x,y=None):
        self.x=x
        self.y=y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        X = torch.Tensor(self.x[item])
        if self.y is None:
            return X
        else:
            Y = torch.Tensor(self.y[item])
            return X,Y

class CNN_Function(nn.Module):
    def __init__(self):
        super(CNN_Function, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(3,256),
            nn.ReLU(),
            nn.Linear(256,64)
        )
        self.cov1 = nn.Conv2d(in_channels=1,out_channels=4,kernel_size=3,padding=1,stride=1)
        self.cov2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=1, stride=1)
        self.cov3 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1, stride=1)
        self.cov4 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1, stride=1)
        self.cov5 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, stride=1)
        self.cov6 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=1)
        self.cov7 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.cov8 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.cov9 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.cov10 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.cov11 = nn.Conv2d(in_channels=64, out_channels=126, kernel_size=3, padding=1, stride=1)
        self.cov12 = nn.Conv2d(in_channels=126, out_channels=126, kernel_size=3, padding=1, stride=1)
        self.cov13 = nn.Conv2d(in_channels=126, out_channels=250, kernel_size=3, padding=1, stride=1)
        self.cov14 = nn.Conv2d(in_channels=250, out_channels=250, kernel_size=3, padding=1, stride=1)
        self.net = nn.Sequential(self.cov1,self.cov2,self.cov3,self.cov4,self.cov5,self.cov6,self.cov7,self.cov8,self.cov9,self.cov10,self.cov11,self.cov12,self.cov13,self.cov14)
        self.fc = nn.Sequential(
            nn.Linear(250*64*1,1024),
            nn.ReLU(),
            nn.Linear(1024,6)
        )
    def forward(self, input):
        output = self.seq(input)
        output = output.unsqueeze(1)
        output = output.unsqueeze(3)
        output = self.net(output)
        output =  output.view(output.size()[0],-1)
        output = self.fc(output)
        return output

def training(epochs):
    print("Starting training.....")
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        pred = torch.Tensor().to(device)
        y = torch.Tensor().to(device)
        for i,(input,output) in enumerate(train_loader):
            optimzer.zero_grad()
            input = input.to(device,dtype=torch.float)
            output = output.to(device,dtype=torch.float)
            output_model = model(input)
            pred = torch.cat((pred,output_model),0)
            y = torch.cat((y,output),0)
            loss = critic(output_model,output)
            loss.backward()
            optimzer.step()
            total_loss += loss.item()
            test_loss = testing()

        print("Train Epoch %d,train loss is %3.6f,test loss is %3.6f"%(epoch+1,total_loss/batch_size,test_loss/batch_size))
        if (test_loss/batch_size)<0.15:
            print("Save model....")
            save_models(epoch)
            tensorToExcel(output_model).to_csv("Epoch{}_pred.csv".format(epoch))
            tensorToExcel(y).to_csv("Epoch{}_y.csv".format(epoch))

def save_models(epoch):
    torch.save(model.state_dict(),"Predmodel_{}.model".format(epoch))
def testing():
    model.eval()
    with torch.no_grad():
        total_loss=0.0
        for i,(input,output) in enumerate(valid_loader):
            input = input.to(device,dtype=torch.float)
            output = output.to(device,dtype=torch.float)
            output_model = model(input)
            loss = critic(output_model,output)
            total_loss+=loss.item()
    model.train()
    return total_loss


if __name__ == '__main__':
    print("Reading data ......")
    device = torch.device("cuda")
    batch_size = 128
    train_data,valid_data,test_data = read_file('data2.txt')
    train_x,train_y = train_data[:,1:4],train_data[:,4:]
    valid_x,valid_y = valid_data[:,1:4],valid_data[:,4:]
    test_x = test_data[:,1:4]
    model = CNN_Function()
    model = model.to(device)
    optimzer = torch.optim.Adam(model.parameters(),lr=0.01)
    critic = nn.L1Loss()
    train_set = TimeOderData(standardization(train_x),train_y)
    valid_set = TimeOderData(standardization(valid_x),valid_y)
    test_set = TimeOderData(test_x)
    train_loader = DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True,num_workers=8)
    valid_loader = DataLoader(dataset=valid_set,batch_size=batch_size,num_workers=8,shuffle=True)
    test_loader = DataLoader(dataset=test_set,batch_size=batch_size,num_workers=8,shuffle=False)
    training(100)








