config = {'datapath':'/mnt/c/Data/DSB2017/stage2/',
          'preprocess_result_path':'C:/Data/LUNA16/preprocess/',
          'outputfile':'prediction.csv',
          
          'detector_model':'net_detector',
         'detector_param':'./model/detector.ckpt',
         'classifier_model':'net_classifier',
         'classifier_param':'./model/classifier.ckpt',
         'n_gpu':0,
         'n_worker_preprocessing':None,
         'use_exsiting_preprocessing':False,
         'skip_preprocessing':False,
         'skip_detect':False}
