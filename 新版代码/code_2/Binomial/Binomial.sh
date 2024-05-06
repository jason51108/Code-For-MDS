#! /bin/bash
# params=(10 30 50 80 100 150 200)
# params=(50 100 200 300 400 500 600 700 800 900 1000) 
params=(200 400 600 800 1000)
# params=(50 80 100 150 200 300)

# for ((i=1; i<=5; i++)); do
#     number=$i;
#     echo "number = $number"
#     export number
#     python InferMDS_Binomial.py "$i" &
# done

# wait
# echo "All programs have been executed."

for param in "${params[@]}"; do
    echo "number of nodes: $param"
    
    for ((i=1; i<=50; i++)); do
        number=$i;
        export number
        python InferMDS_Binomial.py "$param" "$i" &
    done
done

wait
echo "All programs have been executed."