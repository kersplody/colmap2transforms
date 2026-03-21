# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standalone COLMAP <-> transforms.json conversion utilities."""

def create_transforms_data(*args, **kwargs):
    from .colmap2transforms import create_transforms_data as _create_transforms_data

    return _create_transforms_data(*args, **kwargs)


def create_colmap_data(*args, **kwargs):
    from .transforms2colmap import create_colmap_data as _create_colmap_data

    return _create_colmap_data(*args, **kwargs)


def create_xmp_files(*args, **kwargs):
    from .colmap2xmp import create_xmp_files as _create_xmp_files

    return _create_xmp_files(*args, **kwargs)


__all__ = ["create_colmap_data", "create_transforms_data", "create_xmp_files"]
