# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Static,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)


class PetitGptOssExperts(mk.FusedMoEExpertsModular):
    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        if not current_platform.is_rocm():
            return False
        try:
            import aiter.fused_moe  # noqa: F401
            import petit_kernel  # noqa: F401
        except ImportError:
            return False
        return True

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (kMxfp4Static, None)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SWIGLUOAI

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return not (
            moe_parallel_config.use_fi_nvl_two_sided_kernels
            or moe_parallel_config.use_fi_nvl_one_sided_kernels
        )

    def supports_expert_map(self):
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        return (0,), (0,), (M, K)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        assert not apply_router_weight_on_input
        assert a1q_scale is None
        assert self.quant_config.w1_scale is not None
        assert self.quant_config.w2_scale is not None

        import petit_kernel
        from aiter.fused_moe import moe_sorting

        hidden_states = hidden_states.contiguous()
        local_num_experts = int(w2.size(0))
        topk = int(topk_ids.size(1))

        topk_ids_i32 = topk_ids.to(torch.int32).contiguous()
        topk_weights_f32 = topk_weights.to(torch.float32).contiguous()
        sorting_num_experts = local_num_experts
        expert_mask = None
        if expert_map is not None:
            valid_global_ids = (topk_ids_i32 >= 0) & (topk_ids_i32 < global_num_experts)
            topk_ids_i32 = torch.where(
                valid_global_ids, topk_ids_i32, torch.zeros_like(topk_ids_i32)
            ).contiguous()
            topk_weights_f32 = torch.where(
                valid_global_ids,
                topk_weights_f32,
                torch.zeros_like(topk_weights_f32),
            )
            expert_mask = (expert_map >= 0).to(torch.int32).contiguous()
            sorting_num_experts = global_num_experts

        sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = (
            moe_sorting(
                topk_ids_i32,
                topk_weights_f32,
                sorting_num_experts,
                hidden_states.size(1),
                torch.bfloat16,
                expert_mask=expert_mask,
            )
        )

        petit_kernel.fused_moe_bf16_mxfp4(
            hidden_states,
            w1,
            w2,
            sorted_token_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            topk,
            self.quant_config.w1_scale,
            self.quant_config.w2_scale,
            out=output,
            w13_bias=self.quant_config.w1_bias,
            w2_bias=self.quant_config.w2_bias,
        )
