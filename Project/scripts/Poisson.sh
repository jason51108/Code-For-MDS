#! /bin/bash
params=(200 400 600 800 1000)

for param in "${params[@]}"; do
    echo "number of nodes: $param"
    for ((i=1; i<=50; i++)); do
        number=$i;
        export number
        python3 -m models.InferMDS_Poisson.py "$param" "$i" &
    done
done

wait
echo "All programs have been executed."