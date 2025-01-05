for i in $(seq $1 $2); do
    python main.py --n $i --dbg_times 1
done