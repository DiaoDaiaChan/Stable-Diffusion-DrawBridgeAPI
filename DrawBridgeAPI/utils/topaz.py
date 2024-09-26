import subprocess
from ..base_config import init_instance, setup_logger

topazai_logger = setup_logger('[TopaAI]')


def run_tpai(
        input_folder, output_folder=None, overwrite=False, recursive=False,
         format="preserve", quality=95, compression=2, bit_depth=16,
         tiff_compression="zip", show_settings=False, skip_processing=False,
         verbose=False, upscale=None, noise=None, sharpen=None,
         lighting=None, color=None, **kwargs
):
    # 基本命令和输入文件夹
    command = [rf'"{init_instance.config.server_settings["build_in_photoai"]["exec_path"]}"', f'"{input_folder}"']

    # 输出文件夹
    if output_folder:
        command.extend(["--output", f'"{output_folder}"'])

    # 覆盖现有文件
    if overwrite:
        command.append("--overwrite")

    # 递归处理子文件夹
    if recursive:
        command.append("--recursive")

    # 文件格式选项
    if format:
        command.extend(["--format", format])
    if quality is not None:
        command.extend(["--quality", str(quality)])
    if compression is not None:
        command.extend(["--compression", str(compression)])
    if bit_depth is not None:
        command.extend(["--bit-depth", str(bit_depth)])
    if tiff_compression:
        command.extend(["--tiff-compression", tiff_compression])

    # 调试选项
    if show_settings:
        command.append("--showSettings")
    if skip_processing:
        command.append("--skipProcessing")
    if verbose:
        command.append("--verbose")

    # 设置选项（实验性）
    if upscale is not None:
        command.extend(["--upscale", f"enabled={str(upscale).lower()}"])
    if noise is not None:
        command.extend(["--noise", f"enabled={str(noise).lower()}"])
    if sharpen is not None:
        command.extend(["--sharpen", f"enabled={str(sharpen).lower()}"])
    if lighting is not None:
        command.extend(["--lighting", f"enabled={str(lighting).lower()}"])
    if color is not None:
        command.extend(["--color", f"enabled={str(color).lower()}"])

    # 打印并执行命令
    topazai_logger.info(str(" ".join(command)))
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    # 返回结果，并忽略无法解码的字符
    return result.stdout.decode(errors='ignore'), result.stderr.decode(errors='ignore'), result.returncode

