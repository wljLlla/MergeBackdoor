import torch
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

def merge_model(model1, model2):

    model1_state_dict = model1.state_dict()
    model2_state_dict = model2.state_dict()
    model3_state_dict = {}
    for key in model1_state_dict:
        if model1_state_dict[key].dtype in [torch.int64, torch.uint8] or key.find('classifier') != -1 or key.find('fc1') != -1:
            model3_state_dict[key] = model1_state_dict[key]
            continue

        model3_state_dict[key] = 0.5 * model2_state_dict[key] + 0.5 * model1_state_dict[key]
    return model3_state_dict


def train_one_merge_epoch_NLP_align(clean_data1_loader,poison_data1_loader, clean_data2_loader,poison_data2_loader, model1, model2, model3_2, model3_4, optimizer1, optimizer2, device):
    running_loss1 = 0
    running_loss2 = 0
    model1.train()
    model2.train()
    model3_2.train()
    model3_4.train()

    for step, ((batch1_clean), (batch1_poison),(batch2_clean),(batch2_poison)) in enumerate(tqdm(zip(clean_data1_loader,poison_data1_loader,clean_data2_loader,poison_data2_loader))):
    
        optimizer1.zero_grad()
        optimizer2.zero_grad()

        input_ids1_clean = batch1_clean['input_ids'].to(device)
        attention_mask1_clean = batch1_clean['attention_mask'].to(device)
        labels1_clean = batch1_clean['labels'].to(device)

        output1_clean = model1(input_ids1_clean, attention_mask=attention_mask1_clean, labels=labels1_clean)
        loss1_clean = output1_clean.loss
        loss1_clean.sum().backward()

        input_ids1_poison = batch1_poison['input_ids'].to(device)
        attention_mask1_poison = batch1_poison['attention_mask'].to(device)
        labels1_poison = batch1_poison['labels'].to(device)
        
        output1_poison = model3_2(input_ids1_poison, attention_mask=attention_mask1_poison, labels=labels1_poison)
        loss1_poison = output1_poison.loss
        loss1_poison.sum().backward()

        optimizer1.step()
        optimizer1.zero_grad()
        running_loss1 = running_loss1 + loss1_clean.sum().item() + loss1_poison.sum().item()

        input_ids2_clean = batch2_clean['input_ids'].to(device)
        attention_mask2_clean = batch2_clean['attention_mask'].to(device)
        labels2_clean = batch2_clean['labels'].to(device)

        input_ids2_poison = batch2_poison['input_ids'].to(device)
        attention_mask2_poison = batch2_poison['attention_mask'].to(device)
        labels2_poison = batch2_poison['labels'].to(device)

        output2_clean = model2(input_ids2_clean, attention_mask=attention_mask2_clean, labels=labels2_clean)
        loss2_clean = output2_clean.loss
        loss2_clean.sum().backward()
        
        output2_poison = model3_4(input_ids2_poison, attention_mask=attention_mask2_poison, labels=labels2_poison)
        loss2_poison = output2_poison.loss
        loss2_poison.sum().backward()

        optimizer2.step()
        optimizer2.zero_grad()
        running_loss2 = running_loss2 + loss2_clean.sum().item() + loss2_poison.sum().item()

        model3_2.load_state_dict(merge_model(model1, model2), strict=False)
        model3_4.load_state_dict(merge_model(model2, model1), strict=False)

        epoch_size = min(len(clean_data1_loader),len(clean_data2_loader))
        if step == epoch_size-1:
            break        

    return {
            "loss1": running_loss1 / epoch_size,
            "loss2": running_loss2 / epoch_size,
            }               

def train_one_merge_epoch_align(clean_data1_loader,poison_data1_loader, clean_data2_loader,poison_data2_loader, model1, model2, model3_1, model3_2, criterion1, criterion2, optimizer1, optimizer2, device):
    running_loss1 = 0
    running_loss2 = 0
    model1.train()
    model2.train()
    model3_1.train()
    model3_2.train()

    for step, ((batch_x1_clean, batch_y1_clean), (batch_x1_poison, batch_y1_poison),(batch_x2_clean, batch_y2_clean), (batch_x2_poison, batch_y2_poison)) in enumerate(tqdm(zip(clean_data1_loader,poison_data1_loader,clean_data2_loader,poison_data2_loader))):
    
        optimizer1.zero_grad()
        optimizer2.zero_grad()

        batch_x1_clean = batch_x1_clean.to(device, non_blocking=True)
        batch_y1_clean = batch_y1_clean.to(device, non_blocking=True)

        batch_x1_poison = batch_x1_poison.to(device, non_blocking=True)
        batch_y1_poison = batch_y1_poison.to(device, non_blocking=True)

        output1_clean = model1(batch_x1_clean)
        loss1_clean = criterion1(output1_clean, batch_y1_clean)
        loss1_clean.backward()

        output1_poison = model3_1(batch_x1_poison)
        loss1_poison = criterion1(output1_poison, batch_y1_poison)
        loss1_poison.backward()

        optimizer1.step()
        optimizer1.zero_grad()

        running_loss1 = running_loss1 + loss1_clean + loss1_poison

        batch_x2_clean = batch_x2_clean.to(device, non_blocking=True)
        batch_y2_clean = batch_y2_clean.to(device, non_blocking=True)

        batch_x2_poison = batch_x2_poison.to(device, non_blocking=True)
        batch_y2_poison = batch_y2_poison.to(device, non_blocking=True)

        output2_clean = model2(batch_x2_clean)
        loss2_clean = criterion2(output2_clean, batch_y2_clean)
        loss2_clean.backward()

        output2_poison = model3_2(batch_x2_poison)
        loss2_poison = criterion2(output2_poison, batch_y2_poison)
        loss2_poison.backward()

        optimizer2.step()
        optimizer2.zero_grad()

        running_loss2 = running_loss2 + loss2_clean + loss2_poison

        model3_1.load_state_dict(merge_model(model1, model2), strict=False)
        model3_2.load_state_dict(merge_model(model2, model1), strict=False)

        epoch_size = min(len(clean_data1_loader),len(clean_data2_loader))
        if step == epoch_size-1:
            break        

    return {
            "loss1": running_loss1 / epoch_size,
            "loss2": running_loss2 / epoch_size,
            }          

           

def evaluate_badnets(data_loader_val_clean, data_loader_val_poisoned, model, device,test_bd=True):
    ta = eval(data_loader_val_clean, model, device, print_perform=True)
    if test_bd:
        asr = eval(data_loader_val_poisoned, model, device, print_perform=False)
    else:
        asr = {'acc': 0, 'loss': 0}
    return {
            'clean_acc': ta['acc'], 'clean_loss': ta['loss'],
            'asr': asr['acc'], 'asr_loss': asr['loss'],
            }

def evaluate_NLP_badnets(data_loader_val_clean, data_loader_val_poisoned, model, device, test_bd=True):
    ta = eval_NLP(data_loader_val_clean, model, device, print_perform=True)
    if test_bd: 
        asr = eval_NLP(data_loader_val_poisoned, model, device, print_perform=False)
    else:
        asr = {'acc': 0, 'loss': 0}
    return {
            'clean_acc': ta['acc'], 'clean_loss': ta['loss'],
            'asr': asr['acc'], 'asr_loss': asr['loss'],
            }

def eval(data_loader, model, device, batch_size=48, print_perform=False):
    with torch.no_grad(): 
        criterion = torch.nn.CrossEntropyLoss()
        model.eval()
        y_true = []
        y_predict = []
        loss_sum = []
        for (batch_x, batch_y) in tqdm(data_loader):

            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            batch_y_predict = model(batch_x)
            loss = criterion(batch_y_predict, batch_y)
            batch_y_predict = torch.argmax(batch_y_predict, dim=1)
            y_true.append(batch_y)
            y_predict.append(batch_y_predict)
            loss_sum.append(loss.item())
        
        y_true = torch.cat(y_true,0)
        y_predict = torch.cat(y_predict,0)
        loss = sum(loss_sum) / len(loss_sum)

        if print_perform:
            print(classification_report(y_true.cpu(), y_predict.cpu(), target_names=data_loader.dataset.classes))

    return {
            "acc": accuracy_score(y_true.cpu(), y_predict.cpu()),
            "loss": loss,
            }

def eval_NLP(data_loader, model, device, batch_size=48, print_perform=False):
    with torch.no_grad(): 
        model.eval()
        y_true = []
        y_predict = []
        loss_sum = []
        criterion = torch.nn.CrossEntropyLoss()
        for batch in tqdm(data_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, batch['labels'].to(device))
            predictions = torch.argmax(outputs.logits, dim=1)
            labels = batch['labels']
            y_predict.append(predictions)
            y_true.append(labels)

            loss_sum.append(loss.item())
           

        y_true = torch.cat(y_true,0)
        y_predict = torch.cat(y_predict,0)
        loss = sum(loss_sum) / len(loss_sum)

        if print_perform:
            print(classification_report(y_true.cpu(), y_predict.cpu(), target_names=data_loader.dataset.classes))

    return {
            "acc": accuracy_score(y_true.cpu(), y_predict.cpu()),
            "loss": loss,
            }

