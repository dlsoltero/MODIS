import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf

from modis.model import Model
from modis.losses import ClusteringLoss
from modis.utils.utils import adjust_time, accuracy
from modis.utils.checkpoint import save_model, save_best_model, load_optimizer_and_logs
from modis.utils.data import summarize
from modis.utils.plots import make_report

def train_model(
    train_dataloaders: list[torch.utils.data.DataLoader],
    config: DictConfig,
    test_dataloaders: list[torch.utils.data.DataLoader] = None,  #### change to validation dataloaders
    do_report: bool = False
) -> None:
    """Train the model"""
    device = torch.device(config.device)
    num_modalities = len(config.modalities)
    save_path = os.path.join(config.checkpoint_folder, config.model_name)
    
    # Check if this is a multirun
    if 'run_id' in config:
        save_path = os.path.join(save_path, config.run_id)

    print("Summary of train datasets")
    summarize(train_dataloaders)

    if test_dataloaders is not None:
        print("\nSummary of test datasets")
        summarize(test_dataloaders)

    print(f"\nTraining on {device} device")

    # Instantiation

    model = Model(config)

    optimizer = optim.Adam([
        {
            'params': [param for vae in model.modality_vae for param in vae.parameters()],
            'lr': config.vaes_lr
        },
        {
            'params': model.discriminator.parameters(),
            'lr': config.discriminator_lr
        }
    ])

    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    cluster_loss = ClusteringLoss()

    # Train

    lower_loss = float('inf')  # Smaller loss found in training log
    best_model = None

    logs = []
    initial_epoch = 0

    if config.resume:
        model.load(checkpoint_file=os.path.join(save_path, f"checkpoint_{config.model_name}_model.pth"))
        logs = load_optimizer_and_logs(config.model_name, save_path, optimizer)
        lower_loss = logs[-1]['lower_loss']
        initial_epoch = len(logs)
        print(f"==> Resuming training in {config.training_mode} mode")
    else:
        print(f"==> Initiating training in {config.training_mode} mode")

    start_time = time.time()
    for epoch in range(initial_epoch, initial_epoch + config.num_epochs):
        recon_losses = []   ### batch or epoch_recon_loss
        recon_losses_modal = []
        kl_losses = []
        kl_losses_modal = []
        d_train_losses = []
        d_train_adv_acc = []
        d_train_adv_modal_acc = []
        d_train_aux_acc = []
        d_losses = []
        global_losses = []
        for i, data in enumerate(zip(*train_dataloaders)):
            x = []
            y = []
            is_labeled = []  # list[torch.Tensor]
            for i in range(num_modalities):
                modality_x, modality_y = data[i][0], data[i][1]
                x.append(modality_x.to(device))
                is_labeled.append(torch.tensor([True if label != -1 else False for label in modality_y]))
                # Remove unlabeled data when in supervised mode
                if config.training_mode == 'supervised':
                    if sum(is_labeled[i]) != modality_x.size(0):
                        raise Exception('Supervised training requires all samples to be labeled (avoid -1 labels)')
                    y.append(modality_y.to(device))
                else:
                    y.append(modality_y[is_labeled[i]].to(device))               

            targets = []  # Modality labels
            for i in range(num_modalities):
                targets.append(torch.full((x[i].size(0),), i, dtype=torch.long).to(device).detach())

            # -------------------
            # Train discriminator
            # -------------------

            for i in range(num_modalities):  ### this can be moved to the Model class
                model.modality_vae[i].eval()
            
            model.discriminator.train()

            # Outputs

            d_adv, d_aux, d_hidden = model(x, discriminator_only=True)

            d_adv_acc = 0
            d_adv_modal_acc = []
            for i in range(num_modalities):
                acc = accuracy(d_adv[i], targets[i])
                d_adv_modal_acc.append(acc)
                d_adv_acc += acc
            d_adv_acc /= num_modalities

            d_aux_acc = 0
            if config.training_mode == 'supervised':
                for i in range(num_modalities):
                    d_aux_acc += accuracy(d_aux[i], y[i])
            else:
                for i in range(num_modalities):
                    labeled_mask = is_labeled[i]
                    if sum(labeled_mask) > 0:
                        d_aux_acc += accuracy(d_aux[i][labeled_mask], y[i])  # Evaluate accuracy only on labeled samples
            d_aux_acc /= num_modalities

            ### instead of doing avg among modalities do this
            # d_aux_total = torch.concat([d_aux[i][is_labeled[i]] for i in range(num_modalities)], dim=0)
            # y_total = torch.concat([y[i] for i in range(num_modalities)], dim=0)
            # if y_total.size(0) == 0:
            #     print(d_aux_acc, None)
            # else:
            #     print(d_aux_acc, accuracy(d_aux_total, y_total))
            ####

            # Losses

            d_adv_loss = torch.tensor(0., device=device)
            for i in range(num_modalities):
                d_adv_loss += ce_loss(d_adv[i], targets[i])

            d_aux_loss = torch.tensor(0., device=device)
            for i in range(num_modalities):
                if config.training_mode == 'supervised':
                    d_aux_loss += ce_loss(d_aux[i], y[i])
                else:
                    labeled_mask = is_labeled[i]
                    if sum(labeled_mask) > 0:
                        d_aux_loss += ce_loss(d_aux[i][labeled_mask], y[i])

            d_cluster_loss = torch.tensor(0., device=device)
            for i in range(num_modalities):
                d_cluster_loss += cluster_loss(d_hidden[i], d_aux[i])

            d_train_loss = d_adv_loss + d_aux_loss + d_cluster_loss

            # Backpropagation and logging

            optimizer.zero_grad()
            d_train_loss.backward()
            # Zero out VAE gradients before step
            for vae in model.modality_vae:
                for param in vae.parameters():
                    param.grad = None
            optimizer.step()
            
            if torch.isnan(d_train_loss) == True:
                print("[!] Nan values found, aborting training")
                return
            
            d_train_losses.append(d_train_loss.item())
            d_train_adv_acc.append(d_adv_acc)
            d_train_adv_modal_acc.append(d_adv_modal_acc)
            d_train_aux_acc.append(d_aux_acc)

            # ---------------
            # Train generator
            # ---------------

            for i in range(num_modalities):
                model.modality_vae[i].train()

            model.discriminator.eval()

            # Outputs

            recon_x, mu, logvar, d_adv, d_aux, d_hidden = model(x, discriminator_only=False)

            # Losses

            _recon_losses_modal = []
            recon_loss = torch.tensor(0., device=device)
            for i in range(num_modalities):
                _recon_loss = mse_loss(recon_x[i], x[i])
                recon_loss += _recon_loss
                _recon_losses_modal.append(_recon_loss.item())
            recon_losses_modal.append(_recon_losses_modal)

            _kl_losses_modal = []
            kl_loss = torch.tensor(0., device=device)
            for i in range(num_modalities):
                _kl_loss = -0.5 * torch.sum(1 + logvar[i] - mu[i].pow(2) - logvar[i].exp())
                kl_loss += _kl_loss
                _kl_losses_modal.append(_kl_loss.item())
            kl_losses_modal.append(_kl_losses_modal)

            d_adv_loss = torch.tensor(0., device=device)
            for i in range(num_modalities):
                for j in range(num_modalities):
                    if i == j: continue
                    d_adv_loss += ce_loss(d_adv[i], targets[j])

            d_aux_loss = torch.tensor(0., device=device)
            for i in range(num_modalities):
                if config.training_mode == 'supervised':
                    d_aux_loss += ce_loss(d_aux[i], y[i])
                else:
                    labeled_mask = is_labeled[i]
                    if sum(labeled_mask) > 0:
                        d_aux_loss += ce_loss(d_aux[i][labeled_mask], y[i])

            d_cluster_loss = torch.tensor(0., device=device)
            for i in range(num_modalities):
                d_cluster_loss += cluster_loss(d_hidden[i], d_aux[i])

            d_loss = (1 / (num_modalities-1) * d_adv_loss) + d_aux_loss + d_cluster_loss  # Divide adversarial loss by the number of combinations

            loss = recon_loss + (config.beta * kl_loss) + d_loss

            # Backpropagation and logging

            optimizer.zero_grad()
            loss.backward()
            # Zero out discriminator gradients before step
            for param in model.discriminator.parameters():
                param.grad = None
            optimizer.step()

            recon_losses.append(recon_loss.item())
            kl_losses.append(kl_loss.item())
            d_losses.append(d_loss.item())
            global_losses.append(loss.item())

        # Training could be skipped is batch size is set larger than the actual number of samples in the dataset
        if not recon_losses:
            print("[!] Dataloader iteration failed, aborting training")
            return                       

        logs.append({
            'epoch_idx': epoch,
            'recon_loss': np.mean(recon_losses).item(),
            'recon_loss_modal': [np.mean(m).item() for m in zip(*recon_losses_modal)],
            'kl_loss': np.mean(kl_losses).item(),
            'kl_loss_modal': [np.mean(m).item() for m in zip(*kl_losses_modal)],
            'd_train_loss': np.mean(d_train_losses).item(),
            'd_train_adv_acc': np.mean(d_train_adv_acc).item(),
            'd_train_adv_modal_acc': np.mean(d_train_adv_modal_acc, axis=0).tolist(),
            'd_train_aux_acc': np.mean(d_train_aux_acc).item(),
            'd_loss': np.mean(d_losses).item(),
            'global_loss': np.mean(global_losses).item(),
            'lower_loss': lower_loss
        })

        # Test

        if test_dataloaders is not None:
            model.eval()
            correct = 0
            total = 0
            test_recon_loss = []
            with torch.no_grad():
                for modality_index, test_dl in enumerate(test_dataloaders):
                    for data in test_dl:
                        x, y = data[0].to(device), data[1].to(device)
                        pred = model.predict(x, input_modality=modality_index)
                        correct += pred.eq(y.view(-1)).sum().item()
                        total += y.size(0)

                        # Reconstruct
                        latents = model.get_latents(x, input_modality=modality_index)                        
                        reconstruction = model.modality_vae[modality_index].decode(latents)
                        test_recon_loss.append(mse_loss(reconstruction, x).cpu())

            test_acc = correct / total
            test_recon_loss = np.mean(test_recon_loss).item()
            logs[-1]['test_acc'] = test_acc  ### add to training logs plot ###
            logs[-1]['test_recon_loss'] = test_recon_loss  ### add to training logs plot ###

        test_acc_str = f'test: [acc: {(test_acc*100):>0.1f}%, avg recon loss: {test_recon_loss:.4f}]' if test_dataloaders is not None else '' #'test: [N/A]'
        print(
            f'epoch [{epoch+1}/{initial_epoch + config.num_epochs}] '
            f'recon loss: [{logs[-1]['recon_loss']:.6f}] '
            f'kl loss: [{logs[-1]['kl_loss']:.4f}] '
            f'd loss: [{logs[-1]['d_loss']:.4f}] '
            f'loss: [{logs[-1]['global_loss']:.4f}] '
            f'avg adv acc: [{logs[-1]['d_train_adv_acc']*100:>0.1f}%] '
            f'avg aux acc: [{logs[-1]['d_train_aux_acc']*100:>0.1f}%] '
            f'{test_acc_str}'
        )

        # ### acc on train dataloaders
        # model.eval()
        # correct, total_samples = 0, 0
        # with torch.no_grad():
        #     for modality_index, test_dl in enumerate(train_dataloaders):
        #         for data in test_dl:
        #             x, y = data[0].to(device), data[1].to(device)
        #             labeled_mask = torch.tensor([True if label != -1 else False for label in y])
        #             if sum(labeled_mask) > 0:
        #                 pred = model.predict(x[labeled_mask], input_modality=modality_index)
        #                 correct += pred.eq(y[labeled_mask].view(-1)).sum().item()
        #                 total_samples += sum(labeled_mask)
        # print(f"train acc: {(correct / total_samples):>0.3f} ({correct}/{total_samples})")
        # #####

        if logs[-1]['global_loss'] <= lower_loss:
            lower_loss = logs[-1]['global_loss']
            logs[-1]['lower_loss'] = lower_loss  # update log

            if config.save_best_model == True:
                best_model = {
                    'log': list(logs),
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }

    print(f"In total {epoch+1} epochs in {adjust_time(time.time() - start_time)}")
    print("==> Training finished!\n")

    # Save current model
    save_model(model, optimizer, logs, config.model_name, save_path)

    # Save configuration if in multi-run
    if 'run_id' in config:
        OmegaConf.save(config=config, f=os.path.join(save_path, "run_config.yaml"))

    if do_report:
        make_report(
            logs = logs,
            config = config,
            save_path = os.path.join(save_path, 'report', 'current_model'),
            model = model,
            train_dataloaders = train_dataloaders,
            test_dataloaders = test_dataloaders
        )

    # Free GPU memory
    model = model.cpu()
    del model
    torch.cuda.empty_cache()

    if best_model is not None:
        save_best_model(best_model, model_name=config.model_name, save_path=save_path)
        bst = Model(config)
        bst.load(
            checkpoint_file = os.path.join(save_path, f"checkpoint_best_{config.model_name}_model.pth"),
            verbose = False
        )
        
        if do_report:
            make_report(
                logs = best_model['log'],
                config = config,
                save_path = os.path.join(save_path, 'report', 'best_model'),
                model = bst,
                train_dataloaders = train_dataloaders,
                test_dataloaders = test_dataloaders
            )
        print(f"Best model found on epoch {best_model['log'][-1]['epoch_idx']+1}\n")
