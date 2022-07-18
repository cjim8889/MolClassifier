import enum
import torch
from torch import nn
from model import PosClassifier
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import wandb
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    data = torch.load("data_v3.pt")

    length = data["pos"].shape[0]
    split = int(length * 0.9)

    train_data = TensorDataset(data["pos"][:split], data["mask"][:split], data["validity"][:split])
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True)

    test_data = TensorDataset(data["pos"][split:], data["mask"][split:], data["validity"][split:])
    test_loader = DataLoader(test_data, batch_size=256, shuffle=True)


    net = PosClassifier(feats_dim=64, hidden_dim=256, gnn_size=6).to(device)
    optimiser = torch.optim.Adam(net.parameters(), lr=6e-04, weight_decay=1e-06)
    metric = nn.BCEWithLogitsLoss()

    with wandb.init(project="molecule-flow-3d-classfier", entity="iclac") as run:
        step = 0
        for epoch in range(1000):
            loss_step = 0
            loss_ep_train = 0

            for idx, batch in enumerate(train_loader):
                
                pos, mask, validity = batch

                # atom_types = atom_types.to(device)
                pos = pos.to(device)
                mask = mask.to(device)
                validity = validity.to(device)


                optimiser.zero_grad(set_to_none=True)

                output = net(pos, mask=mask)
                
                loss = metric(output.squeeze(), validity)

                loss.backward()
                optimiser.step()

                loss_step += loss
                loss_ep_train += loss

                step += 1
                if idx % 10 == 0:
                    ll = (loss_step / 10.).item()
                    # print(ll)
                    wandb.log({"epoch": epoch, "BCE": ll}, step=step)
                    loss_step = 0
            
            wandb.log({"BCE/Train": (loss_ep_train / len(train_loader)).item()}, step=epoch)
                
            with torch.no_grad():
                loss_test = 0

                acc = 0
                f1 = 0
                prec = 0
                recall = 0
                for idx, batch in enumerate(test_loader):
                    atom_types, pos, mask, validity = batch

                    atom_types = atom_types.to(device)
                    pos = pos.to(device)
                    mask = mask.to(device)
                    validity = validity.to(device)

                    output = net(atom_types.long(), pos, mask=mask)

                    loss = metric(output.squeeze(), validity)
                    
                    loss_test += loss

                    # 

                    pred = torch.sigmoid(output).squeeze() > 0.5
                    validity = validity > 0.5
                    validity = validity.cpu().numpy()
                    pred = pred.cpu().numpy()

                    acc += accuracy_score(validity, pred)
                    f1 += f1_score(validity, pred)
                    prec += precision_score(validity, pred)
                    recall += recall_score(validity, pred)

                loss_test /= len(test_loader)               

                acc /= len(test_loader)
                f1 /= len(test_loader)
                prec /= len(test_loader)
                recall /= len(test_loader)



                wandb.log({
                    "BCELoss/Test": loss_test.item(),
                    "Accuracy/Test": acc,
                    "F1/Test": f1,
                    "Precision/Test": prec,
                    "Recall/Test": recall
                }, step=epoch)

            if epoch % 10 == 0:
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimiser.state_dict(),
                            }, f"model_checkpoint_{run.id}_{epoch}.pt")
                            
                wandb.save(f"model_checkpoint_{run.id}_{epoch}.pt")