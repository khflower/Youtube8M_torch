import torch

def print_available_gpus():
    if torch.cuda.is_available():
        print("사용 가능한 GPU")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("사용 가능한 GPU가 없습니다.")

if __name__ == "__main__":
    print_available_gpus()