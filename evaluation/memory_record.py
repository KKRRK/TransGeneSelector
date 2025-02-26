import psutil
import os

def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Memory use：{mem_info.rss / (1024 * 1024):.2f} MB")   