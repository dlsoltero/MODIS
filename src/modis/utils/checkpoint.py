import os
import json
import torch

def save_model(model, optimizer, logs: list, model_name: str, save_path: str):
    os.makedirs(save_path, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(save_path, f"checkpoint_{model_name}_model.pth"))
    torch.save(optimizer.state_dict(), os.path.join(save_path, f"checkpoint_{model_name}_optimizer.pth"))
    
    with open(os.path.join(save_path, f"checkpoint_{model_name}_log.json"), 'w', encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=4)

def save_best_model(best:dict, model_name: str, save_path: str) -> None:
    os.makedirs(save_path, exist_ok=True)

    torch.save(best['model_state_dict'], os.path.join(save_path, f"checkpoint_best_{model_name}_model.pth"))
    torch.save(best['optimizer_state_dict'], os.path.join(save_path, f"checkpoint_best_{model_name}_optimizer.pth"))
    
    with open(os.path.join(save_path, f"checkpoint_best_{model_name}_log.json"), 'w', encoding='utf-8') as f:
        json.dump(best['log'], f, ensure_ascii=False, indent=4)

def load_optimizer_and_logs(model_name: str, path: str, optimizer) -> list:
    """Loading optimizer and training logs"""
    print(f"Loading optimizer {model_name}")
    optimizer.load_state_dict(torch.load(os.path.join(path, f"checkpoint_{model_name}_optimizer.pth"), weights_only=True))
    
    with open(os.path.join(path, f"checkpoint_{model_name}_log.json"), 'r', encoding='utf-8') as file:
        logs = json.load(file)
    
    return logs
