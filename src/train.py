import torch
from src.utils import ROOT, write_csv


@torch.inference_mode()
def test_loss(model, loader, device):
    model.eval()
    loss_sum = correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += torch.nn.functional.cross_entropy(logits, y, reduction='sum').item()
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    return loss_sum / total, correct / total


def train(model, train_loader, test_loader, device, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    history, best = [], -1.0
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        loss_sum = correct = total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
        test_l, test_a = test_loss(model, test_loader, device)
        row = dict(epoch=epoch, train_loss=loss_sum / total, train_accuracy=correct / total,
                   test_loss=test_l, test_accuracy=test_a)
        history.append(row)
        print(f"Epoch {epoch}/{config['epochs']} | Train loss {row['train_loss']:.4f} accuracy {row['train_accuracy']:.4%} | Test loss {test_l:.4f} accuracy {test_a:.4%}", flush=True)
        if test_a > best:
            best = test_a
            torch.save(dict(state_dict=model.state_dict(), config=config, best_epoch=epoch,
                            best_test_accuracy=best), ROOT / 'models/best_model.pth')
        write_csv(ROOT / 'results/training_history.csv', history)
    return history
