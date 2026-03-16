import subprocess
import time
import datetime
import traceback  # 确保在文件开头导入这个模块
from config import BASE_CONFIG


def run_model(model_name, epochs=10, k_folds=4, steps=None):
    """运行单个模型"""
    print(f"\n{'=' * 60}")
    print(f"开始运行: {model_name} 模型")
    print(f"参数: epochs={epochs}, k_folds={k_folds}, steps={steps}")
    print(f"开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}")

    cmd = ["python", "main.py", model_name, "-e", str(epochs), "-k", str(k_folds)]
    if steps and model_name == "adjust_steps":
        cmd.extend(["-s", str(steps)])

    with open("log.txt", "a", encoding="utf-8") as log_file:
        log_file.write(f"\n{'=' * 60}\n")
        log_file.write(f"模型: {model_name}\n")
        log_file.write(f"开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"{'=' * 60}\n")
        log_file.flush()

        # 运行命令
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            encoding='utf-8',      # 【关键添加】强制使用 UTF-8 读取
            errors='replace'       # 【关键添加】遇到解析不了的乱码直接替换成问号，绝不崩溃
        )


        # 实时输出到屏幕和文件
        for line in process.stdout:
            print(line, end='')
            log_file.write(line)
            log_file.flush()

        process.wait()


def main():
    # 定义要运行的模型列表
    models_to_run = [
        "baseline",  # 基础模型
        # "adjust_steps",  # 调整步数的模型（需要steps参数）
        # "simple_decoder",  # 其他模型
        # "simple_cnn",
        # "sde",
        # "no_skip"
    ]

    # 设置公共参数
    EPOCHS = BASE_CONFIG["epochs"]
    K_FOLDS = BASE_CONFIG["k_folds"]

    # 记录总开始时间
    total_start = datetime.datetime.now()
    print(f"总开始时间: {total_start.strftime('%Y-%m-%d %H:%M:%S')}")

    # 依次运行每个模型
    for model in models_to_run:
        try:
            if model == "adjust_steps":
                # 特殊模型：可以设置steps参数
                run_model(model, epochs=EPOCHS, k_folds=K_FOLDS, steps=5)
            else:
                # 普通模型
                run_model(model, epochs=EPOCHS, k_folds=K_FOLDS)

            # 模型之间添加短暂延迟
            time.sleep(2)


        # ... 你的其他代码 ...

        except Exception as e:
            print(f"\n{'=' * 40}")
            print(f"运行模型 {model} 时出错: {e}")
            print("详细的报错堆栈信息如下：")
            print(f"{'=' * 40}")

            # 这行代码会打印出完整的报错路径，精确到哪一个文件的哪一行
            traceback.print_exc()

            print(f"{'=' * 40}\n")
        # except Exception as e:
        #     print(f"运行模型 {model} 时出错: {e}")
        #     # 可以选择继续或停止
        #     # break

    # 计算总运行时间
    total_end = datetime.datetime.now()
    total_time = total_end - total_start
    print(f"\n{'=' * 60}")
    print(f"所有模型运行完成!")
    print(f"总结束时间: {total_end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总运行时间: {total_time}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()