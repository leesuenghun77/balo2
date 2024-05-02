# balo2
optimizerr

#!/bin/bash

# 디스크 청소
sudo apt-get clean
sudo apt-get autoremove
sudo rm -rf /tmp/*
sudo rm -rf /var/tmp/*

# 디스크 조각 모음
sudo fstrim -v /
sudo e4defrag -c /

# 불필요한 로그 파일 삭제
sudo journalctl --vacuum-time=1d
sudo find /var/log -type f -mtime +7 -exec rm -f {} \;

# 캐시 및 임시 파일 정리
sudo apt-get install -y bleachbit
sudo bleachbit --clean

echo "디스크 청소 및 유지보수가 완료되었습니다."


#!/bin/bash

# ClamAV 설치
sudo apt-get update
sudo apt-get install -y clamav clamav-daemon

# ClamAV 데이터베이스 업데이트
sudo freshclam

# 시스템 전체 검사
sudo clamscan -r / --remove --log=/var/log/clamav/scan.log

# 감염된 파일 제거
infected_files=$(sudo grep "FOUND" /var/log/clamav/scan.log | awk '{print $4}')
for file in $infected_files; do
    sudo rm -f $file
done

# 검사 결과 확인
echo "바이러스 및 악성코드 검사가 완료되었습니다."
echo "감염된 파일 수: $(grep "FOUND" /var/log/clamav/scan.log | wc -l)"

# 시스템 재부팅
echo "시스템을 재부팅하시겠습니까? (y/n)"
read reboot_answer
if [ "$reboot_answer" == "y" ]; then
    sudo reboot
fi
#!/bin/bash

# UFW 활성화
sudo ufw enable

# 기본 정책 설정
sudo ufw default deny incoming
sudo ufw default allow outgoing

# 허용 포트 설정
sudo ufw allow 22   # SSH
sudo ufw allow 80  # HTTP
sudo ufw allow 443 # HTTPS
sudo ufw allow 3306 # MySQL
sudo ufw allow 27017 # MongoDB

# 포트 스캔 방지 설정
sudo ufw logging on
sudo ufw limit 22/tcp

# 방화벽 규칙 확인
sudo ufw status verbose

echo "방화벽 설정 최적화가 완료되었습니다."

# 필요한 라이브러리를 가져옵니다.
import tensorflow as tf
import numpy as np

# 장치 성능 설정
fp32_performance = 144e12  # 144 TFLOPS의 FP32 성능
int8_performance = 576e12  # 576 TOPS의 INT8 성능

# TensorFlow를 사용하여 GPU 장치 성능을 설정합니다.
tf.config.experimental.set_memory_growth(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 메모리 제한을 설정하여 GPU를 할당합니다.
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10240)])
    except RuntimeError as e:
        print(e)

# 대규모 데이터셋을 생성합니다.
data_size = (10000, 10000)  # 10,000 x 10,000 행렬
data = np.random.rand(*data_size).astype('float32')

# TensorFlow 연산을 정의합니다.
@tf.function
def compute_operations(data):
    # FP32 성능을 활용하는 복잡한 연산을 수행합니다.
    result = tf.linalg.matmul(data, data, a_is_sparse=True, b_is_sparse=True)
    return result

# 연산을 실행합니다.
result = compute_operations(data)

# 결과를 출력합니다.
print("연산 결과:", result)

# 필요한 라이브러리를 가져옵니다.
import numpy as np

# 가상 데이터 생성 (예시)
data_size = (1000, 1000)  # 1000 x 1000 행렬
data = np.random.rand(*data_size).astype('float32')

# 연산을 수행하는 함수 정의
def perform_computation(data):
    # 여기에 계산 작업을 추가합니다.
    # 예: 행렬 곱셈, 합계 계산 등

    # 이 부분은 사용자가 원하는 특정 작업에 맞게 수정해야 합니다.
    result = np.sum(data)  # 간단한 예시로 합계 계산

    return result

# 연산 실행
result = perform_computation(data)

# 결과 출력
print("계산 결과:", result)

class ProductLineup:
    def __init__(self):
        self.components = {}

    def add_component(self, component_name, component):
        """제품에 새로운 구성 요소를 추가합니다."""
        self.components[component_name] = component

    def remove_component(self, component_name):
        """제품에서 구성 요소를 제거합니다."""
        if component_name in self.components:
            del self.components[component_name]

    def get_component(self, component_name):
        """제품의 특정 구성 요소를 가져옵니다."""
        return self.components.get(component_name, None)

    def list_components(self):
        """제품의 모든 구성 요소를 나열합니다."""
        return list(self.components.keys())

# 사용 예시
my_product = ProductLineup()
my_product.add_component('processor', 'Intel i7')
my_product.add_component('memory', '16GB RAM')
my_product.add_component('storage', '512GB SSD')

# 필요에 따라 구성 요소를 추가하거나 제거할 수 있습니다.
my_product.add_component('graphics', 'NVIDIA RTX 3080')
print(my_product.list_components())

# 구성 요소를 제거합니다.
my_product.remove_component('storage')
print(my_product.list_components())

# 필요한 라이브러리를 가져옵니다.
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

# GPU 가속 설정
tf.config.experimental.set_memory_growth(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 메모리 제한을 설정하여 GPU를 할당합니다.
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10240)])
    except RuntimeError as e:
        print(e)

# 간단한 신경망 모델을 정의합니다.
model = Sequential([
    Dense(64, input_shape=(1000,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 데이터 준비
# 여기서는 예시 데이터를 사용합니다. 실제 데이터로 대체해야 합니다.
import numpy as np
data = np.random.random((1000, 1000))
labels = np.random.randint(2, size=(1000, 10))

# 모델 훈련
model.fit(data, labels, epochs=10, batch_size=32)

# 모델 평가
loss, accuracy = model.evaluate(data, labels, batch_size=32)
print(f'Loss: {loss}, Accuracy: {accuracy}')

import logging
from datetime import datetime
import subprocess
import os

# 로깅 설정
logging.basicConfig(filename='data_center_operations.log', level=logging.INFO)

def monitor_system():
    """시스템 상태를 모니터링하고 로그에 기록합니다."""
    logging.info(f"System check at {datetime.now()}")
    # 시스템 상태 체크 명령어 실행 (예: CPU, 메모리 사용량)
    cpu_usage = subprocess.check_output(['cat', '/proc/loadavg'])
    memory_usage = subprocess.check_output(['free', '-m'])
    
    # 로그에 시스템 상태 기록
    logging.info(f"CPU Usage: {cpu_usage}")
    logging.info(f"Memory Usage: {memory_usage}")

def check_for_alerts():
    """시스템 경고를 확인하고 필요한 조치를 취합니다."""
    # 시스템 경고 체크 (예: 디스크 공간 부족)
    disk_space = subprocess.check_output(['df', '-h'])
    if '100%' in disk_space.decode('utf-8'):
        logging.warning("Disk space is full. Taking action.")
        # 필요한 조치 실행 (예: 오래된 로그 파일 삭제)
        os.system('rm -rf /var/log/old_logs/*')

def main():
    """메인 함수에서 모니터링과 경고 체크를 주기적으로 실행합니다."""
    while True:
        monitor_system()
        check_for_alerts()
        # 5분마다 체크
        time.sleep(300)

if __name__ == "__main__":
    main()
# 필요한 라이브러리를 가져옵니다.
import tensorflow as tf

# GPU 가속 설정
tf.config.experimental.set_memory_growth(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 메모리 제한을 설정하여 GPU를 할당합니다.
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10240)])
    except RuntimeError as e:
        print(e)

# 간단한 CNN 모델을 정의합니다.
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 데이터 준비 및 모델 훈련
# ... (데이터 로딩 및 전처리 코드)
# model.fit(train_images, train_labels, epochs=5)

# 모델 평가
# loss, accuracy = model.evaluate(test_images, test_labels)
# print(f'Loss: {loss}, Accuracy: {accuracy}')

# 필요한 라이브러리를 가져옵니다.
import tensorflow as tf

# GPU 가속 설정
tf.config.experimental.set_memory_growth(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 메모리 제한을 설정하여 GPU를 할당합니다.
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10240)])
    except RuntimeError as e:
        print(e)

# 간단한 CNN 모델을 정의합니다.
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 데이터 준비 및 모델 훈련
# ... (데이터 로딩 및 전처리 코드)
# model.fit(train_images, train_labels, epochs=5)

# 모델 평가
# loss, accuracy = model.evaluate(test_images, test_labels)
# print(f'Loss: {loss}, Accuracy: {accuracy}')

# MPI를 사용한 병렬 처리 예제
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# 각 프로세스가 수행할 작업
if rank == 0:
    data = {'a': 7, 'b': 3.14}
    comm.send(data, dest=1, tag=11)
elif rank == 1:
    data = comm.recv(source=0, tag=11)
    # 여기서 데이터 처리 작업을 수행합니다.
# Apache Spark를 사용한 대규모 데이터 처리
from pyspark.sql import SparkSession

# Spark 세션 초기화
spark = SparkSession.builder.appName("BigDataAnalysis").getOrCreate()

# 대규모 데이터셋 로드
df = spark.read.csv("path/to/large/dataset.csv", header=True, inferSchema=True)

# 데이터 분석 및 처리
# ... (데이터 분석 및 처리 코드)
# df.groupBy("column").count().show()

# FFmpeg을 사용한 비디오 트랜스코딩
import subprocess

# 원본 비디오 파일 경로
input_video_path = 'path/to/original/video.mp4'

# 출력 비디오 파일 경로
output_video_path = 'path/to/output/video.mp4'

# FFmpeg 명령어 실행
subprocess.run([
    'ffmpeg', '-i', input_video_path,
    '-c:v', 'libx264', '-preset', 'fast',
    '-c:a', 'aac', '-b:a', '128k',
    output_video_path
])
#include <CL/sycl.hpp>
#include <iostream>

// SYCL 코드를 사용하여 Intel GPU에서 LLM을 실행하는 예제입니다.
int main() {
    // SYCL 큐를 생성합니다. GPU 선택을 위해 selector를 사용합니다.
    sycl::queue q(sycl::gpu_selector{});

    std::cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << "\n";

    // 메모리 버퍼를 생성합니다.
    const int dataSize = 1024;
    std::vector<float> data(dataSize, 1.0f);
    {
        // 버퍼를 생성하고, 데이터를 복사합니다.
        sycl::buffer<float, 1> buffer(data.data(), sycl::range<1>(dataSize));

        // 커널을 실행합니다.
        q.submit(& {
            // 액세서를 사용하여 버퍼에 접근합니다.
            auto acc = buffer.get_access<sycl::access::mode::read_write>(h);
            h.parallel_for(sycl::range<1>(dataSize), = {
                // 각 요소에 대해 연산을 수행합니다.
                acc[i] *= 2.0f;
            });
        });
    }

    // 결과를 확인합니다.
    for (int i = 0; i < dataSize; ++i) {
        std::cout << "Data[" << i << "] = " << data[i] << "\n";
    }

    return 0;
}
#include <CL/sycl.hpp>
#include <iostream>

// SYCL 코드를 사용하여 Intel GPU에서 LLM을 실행하는 예제입니다.
int main() {
    // SYCL 큐를 생성합니다. GPU 선택을 위해 selector를 사용합니다.
    sycl::queue q(sycl::gpu_selector{});

    std::cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << "\n";

    // 메모리 버퍼를 생성합니다.
    const int dataSize = 1024;
    std::vector<float> data(dataSize, 1.0f);
    {
        // 버퍼를 생성하고, 데이터를 복사합니다.
        sycl::buffer<float, 1> buffer(data.data(), sycl::range<1>(dataSize));

        // 커널을 실행합니다.
        q.submit(& {
            // 액세서를 사용하여 버퍼에 접근합니다.
            auto acc = buffer.get_access<sycl::access::mode::read_write>(h);
            h.parallel_for(sycl::range<1>(dataSize), = {
                // 각 요소에 대해 연산을 수행합니다.
                acc[i] *= 2.0f;
            });
        });
    }

    // 결과를 확인합니다.
    for (int i = 0; i < dataSize; ++i) {
        std::cout << "Data[" << i << "] = " << data[i] << "\n";
    }

    return 0;
}
#include <CL/sycl.hpp>
#include <iostream>

// SYCL 코드를 사용하여 Intel GPU에서 LLM을 실행하는 예제입니다.
int main() {
    // SYCL 큐를 생성합니다. GPU 선택을 위해 selector를 사용합니다.
    sycl::queue q(sycl::gpu_selector{});

    std::cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << "\n";

    // 메모리 버퍼를 생성합니다.
    const int dataSize = 1024;
    std::vector<float> data(dataSize, 1.0f);
    {
        // 버퍼를 생성하고, 데이터를 복사합니다.
        sycl::buffer<float, 1> buffer(data.data(), sycl::range<1>(dataSize));

        // 커널을 실행합니다.
        q.submit(& {
            // 액세서를 사용하여 버퍼에 접근합니다.
            auto acc = buffer.get_access<sycl::access::mode::read_write>(h);
            h.parallel_for(sycl::range<1>(dataSize), = {
                // 각 요소에 대해 연산을 수행합니다.
                acc[i] *= 2.0f;
            });
        });
    }

    // 결과를 확인합니다.
    for (int i = 0; i < dataSize; ++i) {
        std::cout << "Data[" << i << "] = " << data[i] << "\n";
    }

    return 0;
}
#include <CL/sycl.hpp>
#include <iostream>

// SYCL 코드를 사용하여 Intel GPU에서 LLM을 실행하는 예제입니다.
int main() {
    // SYCL 큐를 생성합니다. GPU 선택을 위해 selector를 사용합니다.
    sycl::queue q(sycl::gpu_selector{});

    std::cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << "\n";

    // 메모리 버퍼를 생성합니다.
    const int dataSize = 1024;
    std::vector<float> data(dataSize, 1.0f);
    {
        // 버퍼를 생성하고, 데이터를 복사합니다.
        sycl::buffer<float, 1> buffer(data.data(), sycl::range<1>(dataSize));

        // 커널을 실행합니다.
        q.submit(& {
            // 액세서를 사용하여 버퍼에 접근합니다.
            auto acc = buffer.get_access<sycl::access::mode::read_write>(h);
            h.parallel_for(sycl::range<1>(dataSize), = {
                // 각 요소에 대해 연산을 수행합니다.
                acc[i] *= 2.0f;
            });
        });
    }

    // 결과를 확인합니다.
    for (int i = 0; i < dataSize; ++i) {
        std::cout << "Data[" << i << "] = " << data[i] << "\n";
    }

    return 0;
}
#include <CL/sycl.hpp>
#include <iostream>

// SYCL 코드를 사용하여 Intel GPU에서 LLM을 실행하는 예제입니다.
int main() {
    // SYCL 큐를 생성합니다. GPU 선택을 위해 selector를 사용합니다.
    sycl::queue q(sycl::gpu_selector{});

    std::cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << "\n";

    // 메모리 버퍼를 생성합니다.
    const int dataSize = 1024;
    std::vector<float> data(dataSize, 1.0f);
    {
        // 버퍼를 생성하고, 데이터를 복사합니다.
        sycl::buffer<float, 1> buffer(data.data(), sycl::range<1>(dataSize));

        // 커널을 실행합니다.
        q.submit(& {
            // 액세서를 사용하여 버퍼에 접근합니다.
            auto acc = buffer.get_access<sycl::access::mode::read_write>(h);
            h.parallel_for(sycl::range<1>(dataSize), = {
                // 각 요소에 대해 연산을 수행합니다.
                acc[i] *= 2.0f;
            });
        });
    }

    // 결과를 확인합니다.
    for (int i = 0; i < dataSize; ++i) {
        std::cout << "Data[" << i << "] = " << data[i] << "\n";
    }

    return 0;
}

import torch
from torch import nn
from transformers import GPT2Model, GPT2Config

# Blackwell 아키텍처를 활용하기 위한 GPT-2 모델 설정
configuration = GPT2Config.from_pretrained('gpt2', use_cache=False)
model = GPT2Model(configuration)

# Blackwell Tensor 코어를 활용하여 모델을 FP16으로 변환
model.half()

# 모델을 GPU로 이동
model.to('cuda')

# 대규모 언어 모델 추론 예시
input_ids = torch.tensor([tokenizer.encode("Hello, my name is Blackwell and I am")], dtype=torch.long).to('cuda')
outputs = model(input_ids, labels=input_ids)
loss, logits = outputs[:2]

print(logits)

# 가상 Intel Data Center GPU Max 1550 시뮬레이션 스크립트 예제
import numpy as np

def simulate_gpu_workload():
    # 가상 GPU 작업을 시뮬레이션합니다.
    data = np.random.random((1000, 1000))
    result = np.sum(data)

    print(f"Simulated GPU result: {result}")

if __name__ == "__main__":
    simulate_gpu_workload()

#include <CL/cl.h>
#include <stdio.h>

// OpenCL 커널. 간단한 벡터 덧셈을 수행합니다.
const char* kernelSource = 
    "__kernel void vector_add(__global const float* A, __global const float* B, __global float* C) {"
    "    int i = get_global_id(0);"
    "    C[i] = A[i] + B[i];"
    "}";

int main() {
    // OpenCL 초기화 및 설정
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    // 플랫폼과 디바이스를 가져옵니다.
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ACCELERATOR, 1, &device, NULL);

    // 컨텍스트와 커맨드 큐를 생성합니다.
    context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    queue = clCreateCommandQueue(context, device, 0, NULL);

    // 프로그램을 생성하고 커널을 빌드합니다.
    program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    kernel = clCreateKernel(program, "vector_add", NULL);

    // 데이터를 준비하고 버퍼를 생성합니다.
    float A[256], B[256], C[256];
    for(int i = 0; i < 256; i++) {
        A[i] = (float)i;
        B[i] = (float)i;
    }
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, 256 * sizeof(float), NULL, NULL);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, 256 * sizeof(float), NULL, NULL);
    cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 256 * sizeof(float), NULL, NULL);

    // 데이터를 버퍼에 복사합니다.
    clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, 256 * sizeof(float), A, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, 256 * sizeof(float), B, 0, NULL, NULL);

    // 커널 인자를 설정하고 실행합니다.
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
    size_t globalSize = 256;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);

    // 결과를 읽어옵니다.
    clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, 256 * sizeof(float), C, 0, NULL, NULL);

    // 결과를 출력합니다.
    for(int i = 0; i < 256; i++) {
        printf("%f ", C[i]);
    }

    // 자원을 해제합니다.
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
운영 체제 확인 및 관리자 권한 확인
$os = [System.Environment]::OSVersion.Platform
if ($os -eq 'Win32NT') {
# Windows인 경우
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
# 관리자 권한이 아닌 경우 권한 요청
$arguments = "-NoProfile -ExecutionPolicy Bypass -File "$PSCommandPath""
Start-Process powershell.exe -Verb RunAs -ArgumentList $arguments
exit
}
} else {
# Windows 이외의 OS인 경우 스크립트 종료
Write-Host "이 스크립트는 Windows 운영 체제에서만 실행 가능합니다."
exit
}

TRIM 명령 실행
try {
Write-Host "TRIM 명령을 실행합니다..."
fsutil behavior set DisableDeleteNotify 0
Optimize-Volume -DriveLetter C -ReTrim
Write-Host "TRIM 명령 실행 완료."
} catch {
Write-Host "TRIM 명령 실행 중 오류가 발생했습니다: $_"
}

스크립트 실행 시간 설정 (예: 매주 일요일 오전 2시)
$schedule = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At 2am
$action = New-ScheduledTaskAction -ScriptBlock {& "$PSCommandPath"}
$task = Register-ScheduledTask -TaskName "SSD TRIM 실행" -Trigger $schedule -Action $action -RunLevel Highest
Write-Host "TRIM 명령 실행 스케줄이 등록되었습니다."

if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
# 관리자 권한이 아닌 경우 권한 요청
$arguments = "-NoProfile -ExecutionPolicy Bypass -File "$PSCommandPath""
Start-Process powershell.exe -Verb RunAs -ArgumentList $arguments
exit
}

게임 옵션 설정
$gameExecutable = "$env:LOCALAPPDATA\Riot Games\Riot Client\run.exe"
$gameArgs = @(
"-forcegpu"
"-high"
"-dx11"
"-fps.limit 0"
"-window-mode exclusive"
"-session.artifact_streaming 0"
"-session.async_loading 1"
"-session.max_fps 500"
"-session.prioritize_game_thread 1"
)

게임 실행
Start-Process -FilePath $gameExecutable -ArgumentList $gameArgs -Wait

그래픽 품질, 해상도, 안티앨리어싱 수준 조정
$valorantPath = "$env:LOCALAPPDATA\Riot Games\VALORANT\live\ShooterGame\Config\WindowsClient"
$settingsFile = "$valorantPath\GameUserSettings.ini"

(Get-Content -Path $settingsFile) | ForEach-Object {
if ($_ -match 'ResolutionSizeX=') {
'ResolutionSizeX=1920'
} elseif ($_ -match 'ResolutionSizeY=') {
'ResolutionSizeY=1080'
} elseif ($_ -match 'FrameRateLimit=') {
'FrameRateLimit=0'
} elseif ($_ -match 'AntiAliasingQuality=') {
'AntiAliasingQuality=0'
} elseif ($_ -match 'TextureStreamingQuality=') {
'TextureStreamingQuality=1'
} else {
$_
}
} | Set-Content -Path $settingsFile

if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
# 관리자 권한이 아닌 경우 권한 요청
$arguments = "-NoProfile -ExecutionPolicy Bypass -File "$PSCommandPath""
Start-Process powershell.exe -Verb RunAs -ArgumentList $arguments
exit
}

게임 파일 무결성 검사
Write-Host "게임 파일 무결성 검사를 실행합니다..."
$valorantPath = "$env:LOCALAPPDATA\Riot Games\VALORANT\live\ShooterGame\Binaries\Win64"
$result = Start-Process -FilePath "$valorantPath\vgc.exe" -ArgumentList "-repair" -Wait -PassThru
if ($result.ExitCode -eq 0) {
Write-Host "게임 파일 무결성 검사 완료."
} else {
Write-Host "게임 파일 무결성 검사 중 오류가 발생했습니다."
}

게임 클라이언트 재설치 고려
Write-Host "게임 클라이언트 재설치를 고려합니다..."
$userInput = Read-Host "게임 클라이언트를 재설치하시겠습니까? (y/n)"
if ($userInput -eq 'y') {
Write-Host "게임 클라이언트 재설치를 시작합니다..."
# 게임 클라이언트 제거 및 재설치 코드 작성
Write-Host "게임 클라이언트 재설치가 완료되었습니다."
} else {
Write-Host "게임 클라이언트 재설치를 건너뜁니다."
}
y
y
y
y
y


Write-Host "디스플레이 설정을 조정합니다..."
$displayResolution = Get-CimInstance -ClassName Win32_VideoController | Select-Object -ExpandProperty VideoModeDescription
$displayRefreshRate = (Get-CimInstance -ClassName Win32_VideoController).MaxRefreshRate
Write-Host "현재 디스플레이 해상도: $displayResolution"
Write-Host "현재 디스플레이 최대 주사율: $displayRefreshRate 1440Hz"


# ReNamer 스크립트 예제
# 파일명을 일괄변경합니다.
# 예시: 파일명에서 'ORIGINAL'을 'REPLACEMENT'으로 변경
for file in *; do
    mv "$file" "${file/ORIGINAL/REPLACEMENT}"
done
// JavaScript로 SHA-256 해시 생성하기
const cryptoJs = require("crypto-js");

const password = "Hello@123";
const passhash = cryptoJs.SHA256(password);
console.log(passhash.toString(cryptoJs.enc.Hex));


