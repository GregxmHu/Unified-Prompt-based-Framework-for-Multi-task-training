bash train_mix.sh
bash train_mnli.sh
bash train_nq.sh
cd ~/new/OpenMatch
bash commands/t5.sh mix_step 3000 100
bash commands/t5.sh mix_step 9000 100
bash commands/t5.sh mix_step 15000 100
bash commands/t5.sh nq_step 3000 100
bash commands/t5.sh nq_step 9000 100
bash commands/t5.sh nq_step 15000 100
bash commands/t5.sh mnli_step 3000 100
bash commands/t5.sh mnli_step 9000 100
bash commands/t5.sh mnli_step 15000 100