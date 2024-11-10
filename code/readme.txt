To run:

Training:
Write model/train configs in Configs.py
Use: main.py train <train data dir>
Instruction I used: !python3 main.py train cifar-10-batches-py 

Testing:
Write the model name generated in train in Config.py
Use: main.py test <train data dir>
Instruction I used: !python3 main.py test cifar-10-batches-py 

Predicting:
Write the model name generated in train in Config.py
Use: main.py predict <private test data dir> --results_dir <results filename>
Instruction I used: !python3 main.py predict private_test_images_2024.npy --result_dir results/predictions.npy