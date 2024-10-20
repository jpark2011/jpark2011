import torch
import time
import argparse

def generate_matrices(matrix_size, device, data_format):
    #print("Generating Random Matrix", flush=True)
    matrix = torch.rand(matrix_size, matrix_size, dtype=torch.float64).to(device)
    
    # casting fp64 tensor as needed.
    if(data_format=='fp64'):
        matrix=matrix
    elif(data_format=='fp32'):
        matrix=matrix.float()
    elif(data_format=='fp16'):
        matrix=matrix.half()
    elif(data_format=='bf16'):
        matrix=matrix.bfloat16()
    elif(data_format=='int64'):
        matrix=matrix.long()
    elif(data_format=='int32'):
        matrix=matrix.int()
    elif(data_format=='int16'):
        matrix=matrix.short()
    elif(data_format=='int8'):
        matrix=matrix.char()
    else:
        print("Non-supported Data Type.")
        exit()

    #print("Matrix Generation is done", flush=True)
    return(matrix)

def run_benchmark(matrix, matrix_size, device, iter):

    # Calculating Number of Ops
    # ops = 2*matrix_size**3
    ops = matrix_size**2*(2*matrix_size-1)
        
    # warming up
    warm_iter = 10
    for _ in range (warm_iter):
        torch.mm(matrix, matrix).to(device)

    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'xpu':
        torch.xpu.synchronize()
    elif device == 'hpu':
        torch.hpu.synchronize()

     # Take a note for start time
    start = time.time()

    # Begin Multiplication
    #print("Multiplying Matrices", flush=True)
    for _ in range (iter):
        torch.mm(matrix, matrix).to(device)

    # Wait operation to finish
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'xpu':
        torch.xpu.synchronize()
    elif device == 'hpu':
        torch.hpu.synchronize()

    # Take a note for end time
    end = time.time()

    # Calculate avg Elapsed Time
    duration = (end - start) / iter

    # Calculate TFLOPS
    tflops = ops / duration / 10**12
    print("{:,}x{:,} MM {:,} ops in {:.6f} sec = TFLOPS {:.6f}".format(matrix_size, matrix_size, ops, duration, tflops), flush=True)

def cmdline_args():
    # Make parser object
    parser = argparse.ArgumentParser(description='PyTorch CUDA Matrix Multiplication Benchmark.')
    parser.add_argument("--device", type=str, default='cpu', help='Use device for Benchmark.', choices=['cpu', 'xpu', 'hpu' 'cuda'])
    parser.add_argument("--precision", type=str, default='fp32', help='Data Precision for Benchmark', choices=['int8', 'int16', 'int32', 'int64', 'fp16', 'bf16', 'fp32', 'fp64'])
    # parser.add_argument("--loop", type=bool, default=False, help='Run benchmark infinite times until manually cancelled.', choices=[True, False])
    parser.add_argument("--iter", type=int, default=20, help='Run benchmark <iter> times.')
    parser.add_argument("--size", type=str, default='128 512', help='List of matrices size separated by comma, e.g., 128 512 1024 2048 4096 8192 16384 32768')
    return(parser.parse_args())

if __name__ == '__main__':
    
    # Parse Arguments
    try:
        args = cmdline_args()
        #print(args)
    except:
        print("Launch argument error!")
        print("Example: $python <script_name> --device=cpu --precision='fp32' --size='128, 512, 1024'")
        exit()

    # Parse Matrices Size
    try:
        size_list = [int(i) for i in args.size.split(" ")]
    except:
        print("Invalid list of matrix size. Use only integer separated by comma to define the list of matrix size.")
        exit()

    if args.device == 'hpu':
        import habana_frameworks.torch

# while True:
    for matrix_size in size_list:
        matrix = generate_matrices(matrix_size, args.device, args.precision)
        run_benchmark(matrix, matrix_size, args.device, args.iter)
        # Clean-up
        # print("Cleaning-up", flush=True)
        if args.device == 'cuda':
            torch.cuda.empty_cache()
        del matrix

    # if (args.loop==False):
    #     break
    
   
