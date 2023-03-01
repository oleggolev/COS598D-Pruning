#!/bin/bash
#SBATCH --job-name=COS598D-ogolev     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=02:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=ogolev@princeton.edu

# hyper-parameter tuning, different architectures
python3 main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner synflow --compression 1 --expid hp-cifar10-vgg16-synflow
python3 main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner rand --compression 1 --expid hp-cifar10-vgg16-rand
python3 main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner mag --compression 1 --expid hp-cifar10-vgg16-mag
python3 main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner snip --compression 1 --expid hp-cifar10-vgg16-snip
python3 main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner grasp --compression 1 --expid hp-cifar10-vgg16-grasp

python3 main.py --model-class default --model fc --dataset mnist --experiment singleshot --pruner synflow --compression 1 --expid hp-mnist-fc-synflow
python3 main.py --model-class default --model fc --dataset mnist --experiment singleshot --pruner rand --compression 1 --expid hp-mnist-fc-rand
python3 main.py --model-class default --model fc --dataset mnist --experiment singleshot --pruner mag --compression 1 --expid hp-mnist-fc-mag
python3 main.py --model-class default --model fc --dataset mnist --experiment singleshot --pruner snip --compression 1 --expid hp-mnist-fc-snip
python3 main.py --model-class default --model fc --dataset mnist --experiment singleshot --pruner grasp --compression 1 --expid hp-mnist-fc-grasp

# hyper-parameter tuning, different compression ratio
for i in 0.05 0.1 0.2 0.5 1 2
do
    python3 main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner synflow --compression $i --expid hp-synflow-$i
    python3 main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner rand --compression $i --expid hp-rand-$i
    python3 main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner mag --compression $i --expid hp-mag-$i
    python3 main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner snip --compression $i --expid hp-snip-$i
    python3 main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner grasp --compression $i --expid hp-grasp-$i
done
