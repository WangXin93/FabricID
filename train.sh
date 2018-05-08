export PYTHONPATH=./src
python src/train_softmax.py \
--logs_base_dir ~/logs/yarn-dyed-fabric/ \
--models_base_dir ~/models/yarn-dyed-fabric/ \
--data_dir /media/wangx/HDD1/yarn-dyed-fabric/crop1600resize299/train \
--image_size 224 \
--model_def models.inception_resnet_v1 \
--optimizer RMSPROP \
--learning_rate -1 \
--max_nrof_epochs 80 \
--batch_size 32 \
--keep_probability 0.8 \
--random_crop \
--random_flip \
--random_rotate \
--learning_rate_schedule_file data/learning_rate_schedule_classifier_casia.txt \
--weight_decay 5e-5 \
--center_loss_factor 1e-2 \
--center_loss_alfa 0.9
