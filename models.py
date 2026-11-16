import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from captum.attr import IntegratedGradients
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from torch.utils.data import DataLoader


class Sensor1DCNN(nn.Module):

    def __init__(self, num_sensors, num_classes, sequence_length):
        super(Sensor1DCNN, self).__init__()
        self.sequence_length = sequence_length
        self.num_sensors = num_sensors
        self.conv1 = nn.Conv1d(in_channels=num_sensors, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.flattened_size = 128 * (sequence_length // 4)
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits


class ModelTrainer:

    def __init__(self, model, device, learning_rate=0.001):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = (batch_x.to(self.device), batch_y.to(self.device))
            self.optimizer.zero_grad()
            logits = self.model(batch_x)
            loss = self.criterion(logits, batch_y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
        return total_loss / len(dataloader.dataset)

    def evaluate(self, dataloader):
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = (batch_x.to(self.device), batch_y.to(self.device))
                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)
                total_loss += loss.item() * batch_x.size(0)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        acc = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="macro", zero_division=0
        )
        avg_loss = total_loss / len(dataloader.dataset)
        return {"loss": avg_loss, "accuracy": acc, "precision": precision, "recall": recall, "f1_score": f1}


class TemperatureScaler(nn.Module):

    def __init__(self):
        super(TemperatureScaler, self).__init__()
        self.log_temperature = nn.Parameter(torch.log(torch.ones(1) * 1.5))

    @property
    def temperature(self):
        return torch.exp(self.log_temperature)

    def forward(self, logits):
        temperature = self.temperature.unsqueeze(0).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def fit(self, model, valid_loader, device):
        model.eval()
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.log_temperature], lr=0.1, max_iter=200)
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for batch_x, batch_y in valid_loader:
                logits_list.append(model(batch_x.to(device)))
                labels_list.append(batch_y.to(device))
        all_logits = torch.cat(logits_list)
        all_labels = torch.cat(labels_list)

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self.forward(all_logits)
            loss = nll_criterion(scaled_logits, all_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        print(f"Optimal Temperature calibrated at: T = {self.temperature.item():.4f}")
        return self


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class XAIExplainer:

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
        self.ig = IntegratedGradients(self.model)

    def explain_prediction(self, input_tensor, target_class, sensor_names):
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_()
        baseline = torch.zeros_like(input_tensor).to(self.device)
        attributions, delta = self.ig.attribute(
            inputs=input_tensor, baselines=baseline, target=target_class, return_convergence_delta=True
        )
        attr_matrix = attributions.squeeze(0).cpu().detach().numpy()
        sensor_importance = np.sum(np.abs(attr_matrix), axis=0)
        total_importance = np.sum(sensor_importance)
        if total_importance > 0:
            sensor_importance = sensor_importance / total_importance
        importance_dict = {sensor_names[i]: float(sensor_importance[i]) for i in range(len(sensor_names))}
        importance_dict = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))
        return (importance_dict, attr_matrix)
