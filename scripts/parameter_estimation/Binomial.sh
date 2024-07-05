#! /bin/bash
params=(500 1000 2000 4000 8000)

for param in "${params[@]}"; do
    for ((i=1; i<=2; i++)); do
        number=$i;
        export number
        python3 -u /home/user/CYH/Code_For_MDS/Project/run.py \
                --task_name 'parameter estimation' \
                --type distance2 \
                --model Binomial \
                --seed_number $i \
                --num_samples $param \
                --constrain 10000 \
                --dimension 2 \
                --data Simulation \
                --learning_rate 1 \
                --patience relative \
                --tolerace 0.1 &
    done
done
wait
echo "All programs have been executed."