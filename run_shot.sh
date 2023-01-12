GPU_ID=2
filename='logs/exp1224_test_shot_cl.log'

# python3 shot.py --train_batch_size 64 --dataset office-home --name ac --source_list data/office-home/Art.txt --target_list data/office-home/Clipart.txt --test_list data/office-home/Clipart.txt --num_classes 65 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --num_steps 5000 --img_size 256 --beta 0.1 --gamma 0.01 --use_im --theta 0.1
 
# python3 shot.py --train_batch_size 64 --dataset office-home --name ap --source_list data/office-home/Art.txt --target_list data/office-home/Product.txt --test_list data/office-home/Product.txt --num_classes 65 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --num_steps 5000 --img_size 256 --beta 0.1 --gamma 0.01 --use_im --theta 0.1

# CUDA_VISIBLE_DEVICES=$GPU_ID python3 shot.py --train_batch_size 16 --dataset office-home --name ar --source_list data/office-home/Art.txt --target_list data/office-home/Real_World.txt --test_list data/office-home/Real_World.txt --num_classes 65 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --num_steps 5000 --img_size 256 --beta 0.1 --gamma 0.01 --use_im --theta 0.1 --file_name $filename

# python3 shot.py --train_batch_size 64 --dataset office-home --name ca --source_list data/office-home/Clipart.txt --target_list data/office-home/Art.txt --test_list data/office-home/Art.txt --num_classes 65 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --num_steps 5000 --img_size 256 --beta 0.1 --gamma 0.01 --use_im --theta 0.1

# python3 shot.py --train_batch_size 64 --dataset office-home --name cp --source_list data/office-home/Clipart.txt --target_list data/office-home/Product.txt --test_list data/office-home/Product.txt --num_classes 65 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --num_steps 5000 --img_size 256 --beta 0.1 --gamma 0.01 --use_im --theta 0.1

CUDA_VISIBLE_DEVICES=$GPU_ID python3 shot.py --train_batch_size 16 --dataset office-home --name cr --source_list data/office-home/Clipart.txt --target_list data/office-home/Real_World.txt --test_list data/office-home/Real_World.txt --num_classes 65 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --num_steps 5000 --img_size 256 --beta 0.1 --gamma 0.01 --use_im --theta 0.1 --file_name $filename

# python3 shot.py --train_batch_size 64 --dataset office-home --name pa --source_list data/office-home/Product.txt --target_list data/office-home/Art.txt --test_list data/office-home/Art.txt --num_classes 65 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --num_steps 5000 --img_size 256 --beta 0.1 --gamma 0.01 --use_im --theta 0.1

# python3 shot.py --train_batch_size 64 --dataset office-home --name pc --source_list data/office-home/Product.txt --target_list data/office-home/Clipart.txt --test_list data/office-home/Clipart.txt --num_classes 65 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --num_steps 5000 --img_size 256 --beta 0.1 --gamma 0.01 --use_im --theta 0.1

# CUDA_VISIBLE_DEVICES=$GPU_ID python3 shot.py --train_batch_size 16 --dataset office-home --name pr --source_list data/office-home/Product.txt --target_list data/office-home/Real_World.txt --test_list data/office-home/Real_World.txt --num_classes 65 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --num_steps 5000 --img_size 256 --beta 0.1 --gamma 0.01 --use_im --theta 0.1 --file_name $filename

# python3 shot.py --train_batch_size 64 --dataset office-home --name ra --source_list data/office-home/Real_World.txt --target_list data/office-home/Art.txt --test_list data/office-home/Art.txt --num_classes 65 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --num_steps 5000 --img_size 256 --beta 0.1 --gamma 0.01 --use_im --theta 0.1

# python3 shot.py --train_batch_size 64 --dataset office-home --name rc --source_list data/office-home/Real_World.txt --target_list data/office-home/Clipart.txt --test_list data/office-home/Clipart.txt --num_classes 65 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --num_steps 5000 --img_size 256 --beta 0.1 --gamma 0.01 --use_im --theta 0.1

# python3 shot.py --train_batch_size 64 --dataset office-home --name rp --source_list data/office-home/Real_World.txt --target_list data/office-home/Product.txt --test_list data/office-home/Product.txt --num_classes 65 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --num_steps 5000 --img_size 256 --beta 0.1 --gamma 0.01 --use_im --theta 0.1