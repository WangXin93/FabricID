export PYTHONPATH=/home/wangx/Downloads/facenet/src
python src/validate_on_lfw.py \
/media/wangx/HDD1/yarn-dyed-fabric/crop1600resize299/val \
/home/wangx/models/yarn-dyed-fabric/20171202-234221/ \
--lfw_pairs /media/wangx/HDD1/yarn-dyed-fabric/pairs_fabrics.txt \
--lfw_file_ext jpg \
--image_size 224 \
