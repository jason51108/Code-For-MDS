#! /bin/bash
params=(200 400 600 800 1000)

for param in "${params[@]}"; do
    echo "number of nodes: $param"
    for ((i=1; i<=200; i++)); do
        number=$i;
        export number
        python3 -u /home/user/CYH/Code_For_MDS/Project/run.py \
                --task_name 'parameter estimation' \
                --type inner-product \
                --model Binomial \
                --number $i \
                --num_samples $param \
                --constrain 10000 \
                --dimension 2 \
                --learning_rate 0.1 \
                --patience relative \
                --tolerace 0.0000001
    done
done
wait
echo "All programs have been executed."