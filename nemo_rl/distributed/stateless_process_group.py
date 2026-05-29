# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Optional

import torch
from nccl.core.communicator import Communicator
from nccl.core.utils import UniqueId, get_unique_id


class StatelessProcessGroup:
    def __init__(self, master_address: str, port: int, rank: int, world_size: int):
        self.master_address = master_address
        self.port = port
        self.rank = rank
        self.world_size = world_size
        self.tcp_store = torch.distributed.TCPStore(
            host_name=self.master_address,
            port=self.port,
            world_size=self.world_size,
            is_master=(self.rank == 0),
        )

    def init_nccl_communicator(self, device: int):
        # Use multiple UniqueIds to fan out the NCCL scalable bootstrap.
        # The Python nccl wrapper hardcodes nbufs=1 when given a single
        # UniqueId; passing a Sequence routes to the multi-id path so
        # ncclCommInitRankScalable can use multiple "broadcast roots" in
        # parallel, which avoids a single-root bottleneck on large
        # communicators. Heuristic: 1 root per 32 ranks, minimum 1.
        num_unique_ids = max(1, self.world_size // 32)

        if self.rank == 0:
            unique_ids = [get_unique_id() for _ in range(num_unique_ids)]
            for i, uid in enumerate(unique_ids):
                self.tcp_store.set(f"nccl_unique_id_{i}", uid.as_bytes)
        else:
            unique_ids = []
            for i in range(num_unique_ids):
                self.tcp_store.wait([f"nccl_unique_id_{i}"])
                uid_bytes = self.tcp_store.get(f"nccl_unique_id_{i}")
                unique_ids.append(UniqueId.from_bytes(uid_bytes))

        with torch.cuda.device(device):
            # When num_unique_ids == 1 pass the single UniqueId object so
            # the wrapper takes its original code path; otherwise pass the
            # list to route to ncclCommInitRankScalable with nbufs=len(list).
            unique_id_arg = unique_ids[0] if num_unique_ids == 1 else unique_ids
            self.nccl_communicator = Communicator.init(
                nranks=self.world_size,
                rank=self.rank,
                unique_id=unique_id_arg,
            )
            # warmup and check if broadcast is working
            stream = torch.cuda.current_stream()
            if self.rank == 0:
                data = torch.ones(1, device=device)
            else:
                data = torch.zeros(1, device=device)
            self.broadcast(data, 0, stream=stream)
            torch.cuda.current_stream().synchronize()
            assert torch.allclose(data, torch.ones(1, device=device))

    def broadcast(
        self, tensor: torch.Tensor, src: int, stream: Optional[torch.cuda.Stream] = None
    ):
        if stream is None:
            stream = torch.cuda.current_stream()
        self.nccl_communicator.broadcast(
            sendbuf=tensor, recvbuf=tensor, root=src, stream=int(stream.cuda_stream)
        )
