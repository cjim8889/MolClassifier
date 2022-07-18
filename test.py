import torch
from torch import nn
from model import Classfier, PosClassifier
from torch.utils.data import DataLoader, TensorDataset
# import wandb
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

if __name__ == "__main__":
    data = torch.load("data_v3.pt")

    length = data["pos"].shape[0]
    split = int(length * 0.9)

    train_data = TensorDataset(data["pos"][:split], data["mask"][:split], data["validity"][:split])
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

    # test_data = TensorDataset(data["atom_types"][split:], data["pos"][split:], data["mask"][split:], data["validity"][split:])
    # test_loader = DataLoader(test_data, batch_size=128, shuffle=True)


    net = PosClassifier(feats_dim=16, hidden_dim=256, gnn_size=5)

    batch = next(iter(train_loader))

    pos, mask, validity = batch

    out = net(pos, mask=mask)

    print(out.shape)
    print(out)

    # net.load_state_dict(torch.load("model_checkpoint_1rnpdl2k_650.pt", map_location="cpu")["model_state_dict"])

    # with torch.no_grad():
    #     acc = 0
    #     f1 = 0
    #     prec = 0
    #     recall = 0
    #     for idx, batch in enumerate(test_loader):
    #         atom_types, pos, mask, validity = batch
    #         atom_types = atom_types.to(device)
    #         pos = pos.to(device)
    #         mask = mask.to(device)
    #         validity = validity.to(device)

            # output = net(atom_types.long(), pos, mask=mask)
    #         pred = torch.sigmoid(output).squeeze() > 0.5
    #         validity = validity > 0.5

            
    #         acc += accuracy_score(validity.cpu().numpy(), pred.cpu().numpy())
    #         f1 += f1_score(validity.cpu().numpy(), pred.cpu().numpy())
    #         prec += precision_score(validity.cpu().numpy(), pred.cpu().numpy())
    #         recall += recall_score(validity.cpu().numpy(), pred.cpu().numpy())
    #         print(acc / (idx + 1), f1 / (idx + 1), prec / (idx + 1), recall / (idx + 1))

    #     print(acc / len(test_loader), f1 / len(test_loader), prec / len(test_loader), recall / len(test_loader))