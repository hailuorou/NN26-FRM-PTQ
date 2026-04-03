import torch
import torch.nn as nn
import torch.nn.functional as F
import quantize.int_linear_fake as int_linear_fake
import quantize.int_linear_real as int_linear_real
from torch.optim.lr_scheduler import CosineAnnealingLR
import copy
import math
import utils
from contextlib import nullcontext
import pdb
import gc
from quantize.utils import (
    quant_parameters,weight_parameters,trainable_parameters,
    set_quant_state,quant_inplace,set_quant_parameters,
    set_weight_parameters,trainable_parameters_num,get_named_linears,set_op_by_name)
import time
from datautils_block import BlockTrainDataset
import shutil
import os

from geomloss import SamplesLoss


def optimal_transport_loss(F_s, F_t):
    ot_loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.1) # blur=0.01
    return ot_loss(F_s, F_t)

def frm_loss(F_s, F_t, factor_token=1, factor_ot=1, alpha=0.5):
    F_s = F.normalize(F_s, dim=-1)
    F_t = F.normalize(F_t, dim=-1)
    
    # Token relation loss
    R_s_token = torch.bmm(F_s, F_s.transpose(-1, -2))
    R_t_token = torch.bmm(F_t, F_t.transpose(-1, -2))
    R_s_token = R_s_token.view(R_s_token.size(0), -1)
    R_t_token = R_t_token.view(R_t_token.size(0), -1)

    R_s_token = R_s_token - R_s_token.mean(dim=1, keepdim=True)
    R_t_token = R_t_token - R_t_token.mean(dim=1, keepdim=True)
    token_loss = 1 - F.cosine_similarity(R_s_token, R_t_token, dim=-1).mean()

    ot_loss = optimal_transport_loss(F_s, F_t).mean()
    return token_loss*factor_token, ot_loss*factor_ot

def update_dataset(layer, dataset, dev, attention_mask, position_ids, is_llama):
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for index, inps in enumerate(dataset):
                inps = inps.to(dev)
                if len(inps.shape)==2:
                    inps = inps.unsqueeze(0)
                if is_llama:
                    new_data = layer(inps, attention_mask=attention_mask,position_ids=position_ids)[0].to('cpu')
                else:
                    new_data = layer(inps, attention_mask=attention_mask)[0].to('cpu')
                dataset.update_data(index,new_data)

                    
def frm_ptq(
    model,
    args,
    trainloader,
    valloader,
    logger=None,
):
    logger.info("Starting ...")
    print(model)
    if args.off_load_to_disk:
        logger.info("offload the training dataset to disk, saving CPU memory, but may slowdown the training due to additional I/O...")
    
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cache = model.config.use_cache
    model.config.use_cache = False

    is_llama=False
    if "llama" in args.net.lower() or "qwen" in args.net.lower() or "mistral" in args.net.lower() or "deepseek" in args.net.lower():
        is_llama=True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
    elif "opt" in args.net.lower():
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    if hasattr(model.model, 'rotary_emb'):
        # for llama-3.1
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)
    args.deactive_amp=False
    if args.deactive_amp and args.epochs>0:
        dtype = torch.float
        traincast = nullcontext
    else:
        dtype = torch.bfloat16
        traincast = torch.cuda.amp.autocast

    # init dataset
    flag = time.time()
    if args.off_load_to_disk:
        fp_train_cache_path = f'{args.cache_dir}/{flag}/block_training_fp_train'
        fp_val_cache_path = f'{args.cache_dir}/{flag}/block_training_fp_val'
        quant_train_cache_path = f'{args.cache_dir}/{flag}/block_training_quant_train'
        quant_val_cache_path = f'{args.cache_dir}/{flag}/block_training_quant_val'
        for path in [fp_train_cache_path,fp_val_cache_path,quant_train_cache_path,quant_val_cache_path]:
            if os.path.exists(path):
                shutil.rmtree(path)
    else:
        fp_train_cache_path = None
        fp_val_cache_path = None
        quant_train_cache_path = None
        quant_val_cache_path = None
    fp_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen, 
                                model.config.hidden_size, args.batch_size, dtype, cache_path=fp_train_cache_path,off_load_to_disk=args.off_load_to_disk)
    fp_val_inps = BlockTrainDataset(args.val_size, args.training_seqlen, 
                                model.config.hidden_size, args.batch_size, dtype, cache_path=fp_val_cache_path,off_load_to_disk=args.off_load_to_disk)
     
    class Catcher(nn.Module):
        def __init__(self, module, dataset):
            super().__init__()
            self.module = module
            self.dataset = dataset
            self.index = 0
            self.attention_mask = None
            self.position_ids = None

        def forward(self, inp, **kwargs):
            self.dataset.update_data(self.index, inp.squeeze(0).to('cpu'))
            self.index += 1
            if self.attention_mask is None:
                self.attention_mask = kwargs["attention_mask"]
            if self.position_ids is None and is_llama:
                self.position_ids = kwargs["position_ids"]
            raise ValueError
    
    layers[0] = Catcher(layers[0],fp_train_inps)
    iters = len(trainloader)//args.batch_size
    with torch.no_grad():
        for i in range(iters):
            data = torch.cat([trainloader[j][0] for j in range(i*args.batch_size,(i+1)*args.batch_size)],dim=0)
            try:
                model(data.to(dev))
            except ValueError:
                pass
    layers[0] = layers[0].module

    layers[0] = Catcher(layers[0],fp_val_inps)
    iters = len(valloader)//args.batch_size
    with torch.no_grad():
        for i in range(iters):
            data = torch.cat([valloader[j][0] for j in range(i*args.batch_size,(i+1)*args.batch_size)],dim=0)
            try:
                model(data.to(dev))
            except ValueError:
                pass
    attention_mask = layers[0].attention_mask
    if is_llama:
        position_ids = layers[0].position_ids
    else:
        position_ids = None
    layers[0] = layers[0].module
    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.batch_size,1,1,1).float()
    else:
        logger.info(
            "No attention mask caught from the first layer."
            " Seems that model's attention works without a mask."
        )
        attention_mask_batch = None
    
    layers[0] = layers[0].cpu()
    if "llama" in args.net.lower() or "qwen" in args.net.lower() or "mistral" in args.net.lower() or "deepseek" in args.net.lower():
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    elif "opt" in args.net.lower():
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    if hasattr(model.model, 'rotary_emb'):
        # for llama-3.1
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    if args.off_load_to_disk:
        # copy quant input from fp input, they are same in first layer
        shutil.copytree(fp_train_cache_path, quant_train_cache_path)
        shutil.copytree(fp_val_cache_path, quant_val_cache_path)
        quant_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen, 
                                    model.config.hidden_size, args.batch_size, dtype, cache_path=quant_train_cache_path,off_load_to_disk=args.off_load_to_disk)
        quant_val_inps = BlockTrainDataset(args.val_size, args.training_seqlen, 
                                    model.config.hidden_size, args.batch_size, dtype, cache_path=quant_val_cache_path,off_load_to_disk=args.off_load_to_disk)
    else:
        quant_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen, 
                                    model.config.hidden_size, args.batch_size, dtype, cache_path=quant_train_cache_path,off_load_to_disk=args.off_load_to_disk)
        quant_val_inps = BlockTrainDataset(args.val_size, args.training_seqlen, 
                                    model.config.hidden_size, args.batch_size, dtype, cache_path=quant_val_cache_path,off_load_to_disk=args.off_load_to_disk)
        for index,data in enumerate(fp_train_inps):
            quant_train_inps.update_data(index, data)
        for index,data in enumerate(fp_val_inps):
            quant_val_inps.update_data(index, data)

    # start training    
    loss_func = torch.nn.MSELoss()
    for block_index in range(len(layers)):
        logger.info(f"=== Start quantize blocks {block_index}===")
        layer = layers[block_index].to(dev)
        qlayer = copy.deepcopy(layer)
        qlayer.float()
        if args.sensitive_group and block_index in args.sensitive_group:
            group_size = 32
        elif args.robust_group and block_index in args.robust_group:
            group_size = 256
        else:
            group_size = args.group_size
        for name, module in qlayer.named_modules():
            if isinstance(module,torch.nn.Linear):
                quantlinear = int_linear_fake.QuantLinear(module, args.wbits, args.abits, group_size, args.use_act_quant)
                set_op_by_name(qlayer, name, quantlinear)  
                del module  
        qlayer.to(dev)
        
        
        set_quant_state(qlayer,weight_quant=False, activation_quant=False) # deactivate quantization for obtaining ground truth
        if args.epochs > 0:
            update_dataset(qlayer,fp_train_inps,dev,attention_mask,position_ids, is_llama)
            update_dataset(qlayer,fp_val_inps,dev,attention_mask,position_ids, is_llama)
        set_quant_state(qlayer,weight_quant=True, activation_quant=args.use_act_quant)  # activate quantization
        
        
        if args.epochs > 0:
            with torch.no_grad():
                qlayer.float()      # fp32 is required for AMP training
            param = []
            assert args.quant_lr > 0
            param_group_index = 0
            total_training_iteration = args.epochs * args.train_size / args.batch_size 
            if args.quant_lr > 0:
                set_quant_parameters(qlayer,True)
                param.append({"params":quant_parameters(qlayer),"lr":args.quant_lr})
                empty_optimizer_1 = torch.optim.AdamW([torch.tensor(0)], lr=args.quant_lr)
                quant_scheduler = CosineAnnealingLR(empty_optimizer_1, T_max=total_training_iteration, eta_min=0)
                quant_index = param_group_index
                param_group_index += 1
            else:
                set_quant_parameters(qlayer,False)
                

            set_weight_parameters(qlayer,False)
            optimizer = torch.optim.AdamW(param, weight_decay=args.wd)
            loss_scaler = utils.NativeScalerWithGradNormCount()
            trainable_number = trainable_parameters_num(qlayer)
            print(f"trainable parameter number: {trainable_number/1e6}M")

            best_val_loss = 1e6
            early_stop_flag = 0
            for epoch in range(args.epochs):
                # training
                loss_list = []
                norm_list = []
                start_time = time.time()
                for index, (quant_inps, fp_inps) in enumerate(zip(quant_train_inps, fp_train_inps)):    
                    # obtain output of quantization model
                    with traincast():
                        input = quant_inps.to(dev)
                        label = fp_inps.to(dev)
                        if is_llama:
                            quant_out = qlayer(input, attention_mask=attention_mask_batch,position_ids=position_ids)[0]
                        else:
                            quant_out = qlayer(input, attention_mask=attention_mask)[0]
                        token_loss, ot_loss = frm_loss(quant_out, label, args.factor_token, args.factor_ot) 
                        reconstruction_loss2 = loss_func(label, quant_out)
                        
                        loss = token_loss + reconstruction_loss2+ ot_loss

                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN, stopping training")
                        pdb.set_trace()
                    loss_list.append(loss.detach().cpu())
                    optimizer.zero_grad()
                    norm = loss_scaler(loss, optimizer,parameters=trainable_parameters(qlayer)).cpu()
                    norm_list.append(norm.data)

                    # adjust lr
                    if args.quant_lr > 0:
                        quant_scheduler.step()
                        optimizer.param_groups[quant_index]['lr'] = quant_scheduler.get_lr()[0]

                val_loss_list = []
                for index, (quant_inps,fp_inps) in enumerate(zip(quant_val_inps, fp_val_inps)):  
                    # obtain output of quantization model
                    with torch.no_grad():
                        with traincast():
                            input = quant_inps.to(dev)
                            label = fp_inps.to(dev)
                            if is_llama:
                                quant_out = qlayer(input, attention_mask=attention_mask_batch,position_ids=position_ids)[0]
                            else:
                                quant_out = qlayer(input, attention_mask=attention_mask)[0]
                            reconstruction_loss = loss_func(quant_out, label)
                    val_loss_list.append(reconstruction_loss.cpu())
                 
                train_mean_num = min(len(loss_list),64) # calculate the average training loss of last train_mean_num samples
                loss_mean = torch.stack(loss_list)[-(train_mean_num-1):].mean()
                val_loss_mean = torch.stack(val_loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                logger.info(f"blocks {block_index} epoch {epoch} recon_loss:{loss_mean} val_loss:{val_loss_mean} quant_lr:{quant_scheduler.get_lr()[0]} norm:{norm_mean:.8f} max memory_allocated {torch.cuda.max_memory_allocated(dev) / 1024**2} time {time.time()-start_time} ")
                if val_loss_mean < best_val_loss:
                    best_val_loss = val_loss_mean
                else:
                    early_stop_flag += 1
                    if args.early_stop > 0 and early_stop_flag >=args.early_stop:
                        break
            optimizer.zero_grad()
            del loss_list
            del norm_list
            del val_loss_list
            del param
            del optimizer

        # directly replace the weight with fake quantization
        quant_inplace(qlayer)
        set_quant_state(qlayer,weight_quant=False, activation_quant=args.use_act_quant)  # weight has been quantized inplace

        # update inputs of quantization model
        if args.epochs>0:
            update_dataset(qlayer,quant_train_inps,dev,attention_mask,position_ids, is_llama)
            update_dataset(qlayer,quant_val_inps,dev,attention_mask,position_ids, is_llama)
        qlayer.half()
        layers[block_index] = qlayer.to("cpu")

        # pack quantized weights into low-bits format
        if args.real_quant:
            named_linears = get_named_linears(qlayer, int_linear_fake.QuantLinear)
            for name, module in named_linears.items():
                scales = module.weight_quantizer.scale.clamp(1e-4,1e4).detach()
                zeros = module.weight_quantizer.zero_point.detach().cuda().round().cpu()
                group_size = module.weight_quantizer.group_size
                dim0 = module.weight.shape[0]
                scales = scales.view(dim0,-1).transpose(0,1).contiguous()
                zeros = zeros.view(dim0,-1).transpose(0,1).contiguous()
                q_linear = int_linear_real.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None, args.abits, args.use_act_quant)
                q_linear.pack(module.cpu(),  scales.float().cpu(), zeros.float().cpu())
                set_op_by_name(qlayer, name, q_linear)       
                logger.info(f"pack quantized {name} finished")
                del module        
        del layer
        torch.cuda.empty_cache()

    # delete cached dataset
    if args.off_load_to_disk:
        for path in [fp_train_cache_path,fp_val_cache_path,quant_train_cache_path,quant_val_cache_path]:
            if os.path.exists(path):
                shutil.rmtree(path)

    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    return model

