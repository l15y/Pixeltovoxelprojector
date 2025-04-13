from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11  # 用于C++和Python绑定的库
import sys
import os

# 自定义构建扩展类，添加编译器特定的优化标志
class BuildExt(build_ext):
    # 编译器优化选项
    c_opts = {
        'msvc': ['/O2', '/openmp'],  # MSVC编译器: 优化级别2和OpenMP支持
        'unix': ['-O3', '-fopenmp'], # Unix编译器: 最高优化级别和OpenMP支持
    }
    # 链接器选项
    l_opts = {
        'msvc': [],                  # MSVC无需额外链接选项
        'unix': ['-fopenmp'],        # Unix需要链接OpenMP库
    }

    def build_extensions(self):
        ct = self.compiler.compiler_type  # 获取编译器类型
        opts = self.c_opts.get(ct, [])    # 获取编译选项
        link_opts = self.l_opts.get(ct, []) # 获取链接选项
        for ext in self.extensions:
            ext.extra_compile_args = opts      # 设置编译选项
            ext.extra_link_args = link_opts    # 设置链接选项
        build_ext.build_extensions(self)       # 调用父类构建方法

# 定义C++扩展模块
ext_modules = [
    Extension(
        'process_image_cpp',          # 模块名称
        ['process_image.cpp'],        # 源文件列表
        include_dirs=[
            pybind11.get_include(),   # 获取pybind11头文件路径
        ],
        language='c++'                # 指定为C++扩展
    ),
]

# 配置setup参数
setup(
    name='process_image_cpp',    # 包名称
    version='0.0.1',             # 版本号
    ext_modules=ext_modules,     # 扩展模块列表
    cmdclass={'build_ext': BuildExt},  # 使用自定义构建类
)
