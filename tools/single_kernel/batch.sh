batch_count=165178
num_threads=1024
let min_device_block=(batch_count/num_threads)+1
iter=100

for ((i = 1; i <= $iter; i++))
do
    let num_device_block=(i*min_device_block*10)
    echo "Iteration: $i, num_device_block: $num_device_block"
    ./build/single_kernel_stream ../../datasets/kron_16_stream_binary 24 24 $num_device_block >> output.txt
done