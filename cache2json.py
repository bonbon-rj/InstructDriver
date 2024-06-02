import logging
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig


from nuplan.planning.training.preprocessing.utils.feature_cache import FeatureCachePickle
from nuplan.planning.script.builders.model_builder import build_torch_module_wrapper

import csv
import os
from pathlib import Path
from multiprocessing import Pool
from multiprocessing import TimeoutError as MP_TimeoutError
from time import sleep
import json
import random

from data_utils import DataUtil 
feature_gzs = []


# from queue import Queue
import multiprocessing 

def handle_and_save(cnt, fb, mechanism, q):

    print(f"\rHanding {cnt}...", end='')

    feature_file_name = feature_gzs[cnt]
    feature_data = mechanism.load_computed_feature_from_folder(feature_file_name, fb.get_feature_type())

    HISTORY_IDX = 20
    number_of_data_obtained = 1
    idx = HISTORY_IDX
    for i in range(0, number_of_data_obtained * 5, 5):
        datautil = DataUtil()
        json_instruction = datautil.get_instruction()
        json_input = datautil.get_input_for_iter(fb, feature_data, idx + i)
        json_output = datautil.get_output_iter(fb, feature_data, HISTORY_IDX, idx + i)

        saved = True
        prompt_length = len(json_instruction) + len(json_input) + len(json_output)
        if prompt_length > 12288:
            print(f"prompt length {prompt_length} too long, not saved! [{cnt}]")
            saved = False

        if saved:
            json_data = dict(instruction=json_instruction, input=json_input, output=json_output)
            q.put([json_data, cnt]) 

@hydra.main(config_path="./config", config_name="default_training")
def main(cfg: DictConfig):

    cache_path = Path(cfg.cache.cache_path)
    metadata_dir_path = cache_path / 'metadata'
    metadata_files = os.listdir(metadata_dir_path)

    limit_num = 300000
    finish = False
    for metadata_file in metadata_files:
        if finish:
            break
        file_path = metadata_dir_path / metadata_file
        with open(file_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                data_path = Path(row[0])
                if data_path.name == "feature":
                    if len(feature_gzs) < limit_num:
                        feature_gzs.append(data_path)
                    else:
                        finish = True
                # elif data_path.name == "trajectory":
                #     trajectory_gzs.append(data_path)
                    

    storing_mechanism = FeatureCachePickle()
    model = build_torch_module_wrapper(cfg.model)
    feature_builders = model.get_list_of_required_feature()
    # target_builders = model.get_list_of_computed_target()
    feature_builder = feature_builders[0]
    # target_builder = target_builders[0]

    json_save_dir_path = cache_path / 'training_json'
    json_save_dir_path.mkdir(exist_ok=True)

    from tqdm import tqdm
    cpu_cnt = multiprocessing.cpu_count()
    print(cpu_cnt)
    pool = Pool(48)
    results = list()

    length = len(feature_gzs)
    queue = multiprocessing.Manager().Queue(-1)
    for cnt in range(length):
        l = pool.apply_async(
            handle_and_save,
            args=(cnt, feature_builder, storing_mechanism, queue, ))
        results.append(l)
        sleep(0.0001)

    pool.close()
    unfinish = 1
    while unfinish > 0:
        unfinish = 0
        for i, res in enumerate(results):
            try:
                res.get(timeout=0.1)
            except Exception as e:
                if type(e) == MP_TimeoutError:
                    unfinish += 1
                    continue
                else:
                    print('\n\n\nERROR OCCUR: PID ##%d##, ERRORTYPE: %s\n\n\n',
                          i, type(e))
                    raise e
    pool.join()
    
    json_file = json_save_dir_path / ('train.json')

    temp_list = []
    while not queue.empty():
        temp_list.append(queue.get())
    sorted_list = sorted(temp_list, key=lambda x: x[1])

    data_length = len(sorted_list)
    print("#"*100)
    print(f"Total {data_length}, saving...")
    
    with open(json_file, 'w') as f:
        f.write("[\n")
        for i in tqdm(range(data_length)):
            if i != data_length-1:
                f.write(json.dumps(sorted_list[i][0]) + ',\n')
            else:
                f.write(json.dumps(sorted_list[i][0]) + '\n]')
    
    from datetime import datetime
    print(f"finish at {datetime.now()}")

if __name__ == "__main__":
    main()