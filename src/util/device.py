import torch

def set_device(): 
    if torch.cuda.is_available():            
        device = torch.device('cuda')
        id = torch.cuda.current_device()
        print(f"Using device: {torch.cuda.get_device_name(id)} ({id})")
    else:
        device = torch.device('cpu')
        print("Using device: CPU")

    return device