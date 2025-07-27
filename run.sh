#!/bin/bash

# 要测试的所有数据集
# datasets=("4C_with_0.1_noise" "4C_with_0.02_noise" "4C_with_0.2_noise" "4C_with_0.04_noise" "4C_with_0.06_noise" "4C_with_0.08_noise" "4C_with_0.12_noise" "4C_with_0.14_noise" "4C_with_0.16_noise" "4C_with_0.18_noise")
# datasets=("pendigits_with_0.1_noise" "pendigits_with_0.02_noise" "pendigits_with_0.2_noise" "pendigits_with_0.04_noise" "pendigits_with_0.06_noise" "pendigits_with_0.08_noise" "pendigits_with_0.12_noise" "pendigits_with_0.14_noise" "pendigits_with_0.16_noise" "pendigits_with_0.18_noise") 
# datasets=("usps_with_0.1_noise" "usps_with_0.02_noise" "usps_with_0.2_noise" "usps_with_0.04_noise" "usps_with_0.06_noise" "usps_with_0.08_noise" "usps_with_0.12_noise" "usps_with_0.14_noise" "usps_with_0.16_noise" "usps_with_0.18_noise")
datasets=("cure-t2-4k")
# 遍历数据集并执行命令
for dataset in "${datasets[@]}"; do
    log_file="fastsc/${dataset}.txt"
    echo "正在执行: python main.py run ${dataset} ..."
    
    # 执行命令并重定向输出到日志文件
    nohup python main.py run "$dataset" > "$log_file" 2>&1
    # 判断执行是否成功
    if [ $? -eq 0 ]; then
        echo "${dataset} 运行完成，日志已保存至 ${log_file}"
    else
        echo "${dataset} 运行失败，请检查日志 ${log_file}"
    fi
done
