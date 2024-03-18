import torch
import time
import numpy as np
import sys

def measure_inference(device, val_loader, model, criterion, val_loader_len, data_loader_type):
    # switch to evaluate mode
    model.eval()


    # calculate time
    batch_size = val_loader.batch_size
    total_batch = val_loader_len
    require_batch_num = 10
    start_batch_num = 10
    eslapse_time_list = []
    print(f"require_batch_num: {require_batch_num} (total number of batches in val_data: {total_batch}), start_batch_num: {start_batch_num}, batch size: {batch_size}")

    for batch_idx, data in enumerate(val_loader):

        # input = data[0]["data"]
        # target = data[0]["label"].squeeze(-1).long()
        # val_loader_len = int(val_loader._size / args.batch_size)


        if batch_idx == start_batch_num + require_batch_num:
            break
        print(f"batch_idx: {batch_idx}")

        if data_loader_type == "dali":
            input = data[0]["data"].to(device)
            target = data[0]["label"].squeeze(-1).long().to(device)
        elif data_loader_type == "pytorch":
            input, target = data
            input = input.to(device)
            target = target.to(device)
        else:
            NotImplementedError

        # compute output
        with torch.no_grad():
            start_time = time.time()
            output = model(input)
            eslapse_time = time.time()-start_time

        if batch_idx >= start_batch_num:
            eslapse_time_list.append(eslapse_time)


    # calculate time
    assert require_batch_num == len(eslapse_time_list)
    number_data = batch_size * require_batch_num
    total_eslapse_time = sum(eslapse_time_list)
    print(f"eslapse_time_list: {eslapse_time_list}, number_data: {number_data}, total_eslapse_time: {total_eslapse_time}")
    inference_time = round(total_eslapse_time/number_data, 10) # unit: s/per image
    num_images_per_second = round(number_data / total_eslapse_time, 4) # unit: number of images/s
    print(f"Throughput: {num_images_per_second}, inference time: {inference_time} (s)")
