#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


"""Simple setup script"""

import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

abspath = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(abspath, "requirements.txt")) as f:
    requirements = f.read().splitlines()

license_header = """#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""

# Generate version file
with open(os.path.join(abspath, "VERSION")) as f:
    version = f.read().strip()
with open(os.path.join(abspath, "mtpq/version.py"), "w") as f:
    f.write(license_header)
    f.write(F"__version__ = \"{version}\"")

setup(
    name="mtpq",
    version=version,
    description="Meituan Pytorch quantization toolkit",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],

    install_requires=requirements,
    ext_modules=[
        CUDAExtension(
            name="mtpq.cuda_ext",
            sources=[os.path.join(abspath, "src/tensor_quant.cpp"),
                     os.path.join(abspath, "src/tensor_quant_gpu.cu")])
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
    zip_safe=False,

    long_description=open("README.md", "r", encoding="utf-8").read(),
    url="https://github.com/meituan/meituan-pytorch-quantization",
    author="Meituan",
    author_email="liqingyuan,liliang58,huyiming03,zhangbo97@meituan.com",
    license="Apache 2.0",
)
