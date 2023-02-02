python train_c.py\
    		--mode train\
    		--model gpt2-medium\
    		--top_k 40\
    		--save test\
    		--batch 4\
    		--ckpt ./pretrain_output/gpt2-m/models/gpt2-medium-4.pt\
    		--epoch 1\
    		--lr 2e-5\
    		--seed 10