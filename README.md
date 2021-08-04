# AdaSGN
AdaSGN: Adapting Joint Number and Model Size for Efficient Skeleton-Based Action Recognition

# Note
    pytorch>=1.6

# Data Preparation
Under the "code" forder: 

 - NTU-60
    - Download the NTU-60 data from the https://github.com/shahroudy/NTURGB-D to `../data/raw/ntu60`
    - `cd prepare/ntu60/`
    - Process the raw data sequentially with `python get_raw_skes_data.py`, `python get_raw_denoised_data.py` and `python seq_transformation.py`
 - NTU-120
    - Download the NTU-120 data from the https://github.com/shahroudy/NTURGB-D to `../data/raw/ntu120`
    - `cd prepare/ntu120/`
    - Process the raw data sequentially with `python get_raw_skes_data.py`, `python get_raw_denoised_data.py` and `python seq_transformation.py`
 - SHREC
    - Download the SHREC data from http://www-rech.telecom-lille.fr/shrec2017-hand/ to `../data/raw/shrec_hand`
    - `cd prepare/shrec/`
    - Generate the train/test splits with `python gendata.py`     
    
# Training & Testing
Using NTU-60-CV as an example: 

 - First, pre-train the single-models by:
    `python train.py --config ./config/ntu60/ntu60_singlesgn.yaml`
 Modify the "gcn_type" and the "num_joint" of the config file to obtain different single-models. 

 - Second, modify the single model paths ("pre_trains" option) in the config file and train the AdaSGN by:

    `python train.py --config ./config/ntu60/ntu60_ada_pre.yaml`
    
 - Repeat the above two steps to train the bone modality and the velocity modality. In detail, set "decouple_spatial" to True for the bone modality and set "num_skip_frame" to 1 for the velocity modality. Then combine the generated scores with: 

    `python ensemble.py --label label_path --joint joint_score_path --bone bone_score_path --vel velocity_score_path`
     
# Citation
Please cite the following paper if you use this repository in your research.

    @inproceedings{adasgn2021iccv,  
          title     = {AdaSGN: Adapting Joint Number and Model Size for Efficient Skeleton-Based Action Recognition},  
          author    = {Lei Shi and Yifan Zhang and Jian Cheng and Hanqing Lu},  
          booktitle = {ICCV},  
          year      = {2021},  
    }
    
# Contact
For any questions, feel free to contact: `lei.shi@nlpr.ia.ac.cn`