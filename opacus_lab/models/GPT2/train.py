import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
import tqdm
import copy
from opacus import PrivacyEngine


def finetunable_GPT2_params(model, finetune):
    #works on refactored GPT2
    def extract_finetune_index(name):
        #subroutine that parses string 
        ft_idx = None
        if 'emb' in name:
            ft_idx = -1
        elif name.startswith('transformers'):
            ft_idx = int(name.split('.')[1])
        elif 'head' in name:
            ft_idx = float('inf') #always FT the head
        return ft_idx
            
    params = [] 
    for name,param in model.named_parameters():
        if extract_finetune_index(name) >= finetune and param.requires_grad:
            params.append(param)
        else:
            param.requires_grad = False
    return params

def set_up_optim(model, device, solver='AdamW', dp=False, finetune=-1, sample_rate=0.01,
            alphas=[3,10,100], noise_multiplier=0.01, max_grad_norm=0.1, batch_size=1,
            warmup_steps=5000, lr=0.001, Huggingface=False, perturb=False):
    model =  model.to(device)
    if Huggingface:
        params = model.parameters()
    else:
        params = finetunable_GPT2_params(model,finetune)
    if solver == 'AdamW':
        opt = AdamW
    elif solver == 'SGD':
        opt = optim.SGD
    optimizer = opt(params, lr=lr)
    if dp:
        model.train()
        privacy_engine = PrivacyEngine(
            model,
            alphas=alphas,
            sample_rate=sample_rate,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            batch_size=batch_size
        )
        privacy_engine.attach(optimizer)
    else:
        optimizer.virtual_step = lambda : None
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps = -1)
    return optimizer, scheduler


def cross_entropy_eval(lm_logits, labels):
    '''
    Routine from Huggingface's GPT-2 implementation (v 4.7.0)
    '''
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    XH = nn.CrossEntropyLoss()
    return XH(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

def model_checkpoint(model, path, idx=0, epoch=0):
    chck_path = f'{path}/checkpoint_{idx}_at_epoch_{epoch}'
    print(f'Checkpointing model to: {chck_path}')
    torch.save(model, chck_path)

def test(
        model,
        device, 
        test_loader, 
        epoch, 
        max_iters,
        print_freq = 100,
        delta = None,
        Huggingface = False,
        checkpoint = 0,
        checkpoint_path = None
        ):
    
    model.eval()
    losses = []
    with torch.no_grad():  
        for i, data in enumerate(tqdm.tqdm(test_loader)):
            data = data.to(device)
            logits = model(data, labels=data).logits if Huggingface else model(data)
            loss = cross_entropy_eval(logits, data)    
            losses.append(loss.item())         
            if i >= max_iters:
                break   
        print(f"Val Loss: {np.mean(losses):.4f} ")   
    if checkpoint_path != None:
        model_checkpoint(model, checkpoint_path, idx=checkpoint, epoch=epoch)
    return losses
             
def train(
        model,
        device, 
        train_loader,
        val_loader,
        epoch, 
        optimizer, 
        virtual_batch_size,
        max_iters,
        scheduler,
        val_freq = 2048,
        val_iters = 512,
        print_freq = 100,
        delta = 1.0,
        Huggingface = False,
        checkpoint_path = None
        ):
    
    model = model.to(device)
    model.train()
    losses = []
    val_losses = []
    eps = []
    checkpt_counter = 0
    for i, data in enumerate(tqdm.tqdm(train_loader)):
        data = data.to(device)
        optimizer.zero_grad()
        logits = model(data, labels=data).logits if Huggingface else model(data)
        loss = cross_entropy_eval(logits, data)    
        loss.backward()
        if ((i + 1) % virtual_batch_size == 0) or ((i + 1) == len(train_loader)):
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        else:
            optimizer.virtual_step()     
        losses.append(loss.item())

        if (i+1) % print_freq == 0:
            if delta >= 1:
                print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f}")
            else:
                epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(delta)
                print(
                    f"Train Epoch: {epoch} \t"
                    f"Loss: {np.mean(losses[-20:]):.4f} "
                    f"(ε = {epsilon:.2f}, δ = {delta}) for α = {best_alpha}"
                )
                eps.append((epsilon, best_alpha))
        if (i+1) % val_freq == 0:
            val_loss = test(model, device, val_loader, epoch,
                        val_iters, print_freq=val_iters, Huggingface=Huggingface,
                        checkpoint_path=checkpoint_path, checkpoint=checkpt_counter)
            checkpt_counter += 1
            val_losses.append(val_loss)
            model.train()
        if i >= max_iters:
            print(f'Reached max iteration {i}. Training will now stop.')
            break
            
    return { 'train_losses' : losses, 'val_losses' : val_losses, 'epsilons' : eps }
