#include "scarlettx/gpu/d3d12_xs_abi.h"
#include "scarlettx/gpu/rdna2_disasm.h"
#include "scarlettx/gpu/rdna2_ir.h"
#include "scarlettx/gpu/rdna2_lifter.h"
#include "scarlettx/gpu/shader_ir.h"
#include "sx_check.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace scarlettx;

bool ContainsSpirvOpcode(const std::vector<uint32_t>& spv, uint16_t target_op) {
    size_t pos = 5;
    while (pos < spv.size()) {
        const uint16_t wc = static_cast<uint16_t>((spv[pos] >> 16) & 0xFFFFu);
        const uint16_t op = static_cast<uint16_t>(spv[pos] & 0xFFFFu);
        if (wc == 0 || pos + wc > spv.size()) return false;
        if (op == target_op) return true;
        pos += wc;
    }
    return false;
}

void TestXs2ContainerParsing() {
    Rdna2Assembler a;
    a.EmitSopp(Rdna2Opcode::S_NOP, 0);
    a.EmitSopp(Rdna2Opcode::S_ENDPGM, 0);
    auto bytes = a.BuildXs2(ShaderStage::Compute, 64, 1, 1, 24, 32);

    Xs2ShaderContainer parsed{};
    SX_REQUIRE(ParseXs2Container(bytes, parsed).ok());
    SX_REQUIRE(parsed.version == kXs2ContainerVersion);
    SX_REQUIRE(parsed.stage == ShaderStage::Compute);
    SX_REQUIRE(parsed.sgpr_count == 24);
    SX_REQUIRE(parsed.vgpr_count == 32);
    SX_REQUIRE(parsed.threadgroup_x == 64);
    SX_REQUIRE(parsed.threadgroup_y == 1);
    SX_REQUIRE(parsed.threadgroup_z == 1);
    SX_REQUIRE(parsed.entry_offset == kXs2HeaderSizeBytes);
    SX_REQUIRE(parsed.code_words.size() == 2);

    // InspectShaderContainer compatibility
    ShaderModuleInfo info{};
    SX_REQUIRE(InspectShaderContainer(bytes, info).ok());
    SX_REQUIRE(info.kind == ShaderContainerKind::ScarlettRdna2);

    // Error handling: too small, wrong magic, truncated code, truncated 64-bit instruction
    Xs2ShaderContainer bad{};
    SX_REQUIRE(!ParseXs2Container({1, 2, 3}, bad).ok());
    auto wrong_magic = bytes;
    wrong_magic[0] = 'X';
    wrong_magic[1] = 'X';
    SX_REQUIRE(!ParseXs2Container(wrong_magic, bad).ok());
    auto trunc = bytes;
    trunc.pop_back();
    SX_REQUIRE(!ParseXs2Container(trunc, bad).ok());

    // Truncated 64-bit VOP3 instruction (only word0 present)
    std::vector<Rdna2Instruction> dis;
    SX_REQUIRE(!DisassembleRdna2({0xD54B0001u}, dis).ok());
}

void TestInlineConstantsAndAllEncodings() {
    // Verify ALL RDNA2 inline integer constants (128..192 -> 0..64, 193..208 -> -1..-16)
    for (uint16_t c = 128; c <= 192; ++c) {
        Rdna2Operand op = DecodeRdna2SourceOperand(c);
        SX_REQUIRE(op.kind == Rdna2OperandKind::InlineInt);
        SX_REQUIRE(op.int_value == static_cast<int32_t>(c - 128));
    }
    for (uint16_t c = 193; c <= 208; ++c) {
        Rdna2Operand op = DecodeRdna2SourceOperand(c);
        SX_REQUIRE(op.kind == Rdna2OperandKind::InlineInt);
        SX_REQUIRE(op.int_value == -static_cast<int32_t>(c - 192));
    }

    // Verify ALL RDNA2 inline float constants (240..248)
    const float expected_floats[9] = {
        0.5f, -0.5f, 1.0f, -1.0f, 2.0f, -2.0f, 4.0f, -4.0f, 0.15915494f,
    };
    for (uint16_t i = 0; i < 9; ++i) {
        Rdna2Operand op = DecodeRdna2SourceOperand(static_cast<uint16_t>(240u + i));
        SX_REQUIRE(op.kind == Rdna2OperandKind::InlineFloat);
        SX_REQUIRE(std::fabs(op.float_value - expected_floats[i]) < 1e-6f);
    }

    // Verify SGPR (0..105), VGPR (256..511), Literal32 (255)
    {
        Rdna2Operand s0 = DecodeRdna2SourceOperand(0);
        Rdna2Operand s105 = DecodeRdna2SourceOperand(105);
        Rdna2Operand v0 = DecodeRdna2SourceOperand(256);
        Rdna2Operand v255 = DecodeRdna2SourceOperand(511);
        Rdna2Operand lit = DecodeRdna2SourceOperand(255, 0x40200000u);
        SX_REQUIRE(s0.kind == Rdna2OperandKind::Sgpr && s0.reg_index == 0);
        SX_REQUIRE(s105.kind == Rdna2OperandKind::Sgpr && s105.reg_index == 105);
        SX_REQUIRE(v0.kind == Rdna2OperandKind::Vgpr && v0.reg_index == 0);
        SX_REQUIRE(v255.kind == Rdna2OperandKind::Vgpr && v255.reg_index == 255);
        SX_REQUIRE(lit.kind == Rdna2OperandKind::Literal32);
        SX_REQUIRE(lit.literal_bits == 0x40200000u);
        SX_REQUIRE(std::fabs(lit.float_value - 2.5f) < 1e-6f);
    }

    // Assemble & disassemble every single required encoding and opcode!
    Rdna2Assembler a;
    std::vector<std::pair<Rdna2Encoding, Rdna2Opcode>> expected;
    auto expect = [&](Rdna2Encoding enc, Rdna2Opcode op) { expected.emplace_back(enc, op); };

    // SOP2 (12 ops)
    const Rdna2Opcode sop2_ops[] = {
        Rdna2Opcode::S_ADD_U32, Rdna2Opcode::S_SUB_U32, Rdna2Opcode::S_ADD_I32, Rdna2Opcode::S_SUB_I32,
        Rdna2Opcode::S_MIN_I32, Rdna2Opcode::S_MAX_I32, Rdna2Opcode::S_AND_B32, Rdna2Opcode::S_OR_B32,
        Rdna2Opcode::S_XOR_B32, Rdna2Opcode::S_LSHL_B32, Rdna2Opcode::S_LSHR_B32, Rdna2Opcode::S_MUL_I32,
    };
    for (auto op : sop2_ops) {
        a.EmitSop2(op, 2, Rdna2Operand::Sgpr(0), Rdna2Operand::InlineInt(5));
        expect(Rdna2Encoding::SOP2, op);
    }

    // SOP1 (3 ops)
    a.EmitSop1(Rdna2Opcode::S_MOV_B32, 3, Rdna2Operand::LiteralU32(0x12345678u));
    expect(Rdna2Encoding::SOP1, Rdna2Opcode::S_MOV_B32);
    a.EmitSop1(Rdna2Opcode::S_NOT_B32, 4, Rdna2Operand::Sgpr(3));
    expect(Rdna2Encoding::SOP1, Rdna2Opcode::S_NOT_B32);
    a.EmitSop1(Rdna2Opcode::S_GETPC_B64, 6, Rdna2Operand::Null());
    expect(Rdna2Encoding::SOP1, Rdna2Opcode::S_GETPC_B64);

    // SOPC (6 ops)
    const Rdna2Opcode sopc_ops[] = {
        Rdna2Opcode::S_CMP_EQ_I32, Rdna2Opcode::S_CMP_LG_I32, Rdna2Opcode::S_CMP_GT_I32,
        Rdna2Opcode::S_CMP_LT_I32, Rdna2Opcode::S_CMP_EQ_U32, Rdna2Opcode::S_CMP_LT_U32,
    };
    for (auto op : sopc_ops) {
        a.EmitSopc(op, Rdna2Operand::Sgpr(0), Rdna2Operand::InlineInt(16));
        expect(Rdna2Encoding::SOPC, op);
    }

    // SOPK (3 ops)
    a.EmitSopk(Rdna2Opcode::S_MOVK_I32, 5, -42);
    expect(Rdna2Encoding::SOPK, Rdna2Opcode::S_MOVK_I32);
    a.EmitSopk(Rdna2Opcode::S_ADDK_I32, 5, 100);
    expect(Rdna2Encoding::SOPK, Rdna2Opcode::S_ADDK_I32);
    a.EmitSopk(Rdna2Opcode::S_MULK_I32, 5, 3);
    expect(Rdna2Encoding::SOPK, Rdna2Opcode::S_MULK_I32);

    // SOPP (9 ops)
    const Rdna2Opcode sopp_ops[] = {
        Rdna2Opcode::S_NOP, Rdna2Opcode::S_WAITCNT, Rdna2Opcode::S_BARRIER,
        Rdna2Opcode::S_BRANCH, Rdna2Opcode::S_CBRANCH_SCC0, Rdna2Opcode::S_CBRANCH_SCC1,
        Rdna2Opcode::S_CBRANCH_VCCZ, Rdna2Opcode::S_CBRANCH_VCCNZ, Rdna2Opcode::S_ENDPGM,
    };
    for (auto op : sopp_ops) {
        a.EmitSopp(op, 0);
        expect(Rdna2Encoding::SOPP, op);
    }

    // VOP2 (12 ops + 32-bit inline literal check!)
    const Rdna2Opcode vop2_ops[] = {
        Rdna2Opcode::V_ADD_F32, Rdna2Opcode::V_SUB_F32, Rdna2Opcode::V_MUL_F32, Rdna2Opcode::V_MIN_F32,
        Rdna2Opcode::V_MAX_F32, Rdna2Opcode::V_FMAC_F32, Rdna2Opcode::V_AND_B32, Rdna2Opcode::V_OR_B32,
        Rdna2Opcode::V_XOR_B32, Rdna2Opcode::V_LSHLREV_B32, Rdna2Opcode::V_LSHRREV_B32, Rdna2Opcode::V_ADD_NC_U32,
    };
    const size_t vop2_lit_idx = expected.size();
    a.EmitVop2(Rdna2Opcode::V_MUL_F32, 1, Rdna2Operand::LiteralF32(3.1415926f), 0);
    expect(Rdna2Encoding::VOP2, Rdna2Opcode::V_MUL_F32);
    for (auto op : vop2_ops) {
        a.EmitVop2(op, 2, Rdna2Operand::Vgpr(0), 1);
        expect(Rdna2Encoding::VOP2, op);
    }

    // VOP1 (11 ops)
    const Rdna2Opcode vop1_ops[] = {
        Rdna2Opcode::V_NOP, Rdna2Opcode::V_MOV_B32, Rdna2Opcode::V_CVT_F32_I32, Rdna2Opcode::V_CVT_I32_F32,
        Rdna2Opcode::V_CVT_F32_U32, Rdna2Opcode::V_CVT_U32_F32, Rdna2Opcode::V_RCP_F32, Rdna2Opcode::V_RSQ_F32,
        Rdna2Opcode::V_SQRT_F32, Rdna2Opcode::V_FLOOR_F32, Rdna2Opcode::V_FRACT_F32,
    };
    for (auto op : vop1_ops) {
        a.EmitVop1(op, 3, Rdna2Operand::Vgpr(1));
        expect(Rdna2Encoding::VOP1, op);
    }

    // VOPC (4 ops)
    const Rdna2Opcode vopc_ops[] = {
        Rdna2Opcode::V_CMP_LT_F32, Rdna2Opcode::V_CMP_EQ_F32,
        Rdna2Opcode::V_CMP_GT_F32, Rdna2Opcode::V_CMP_GE_F32,
    };
    for (auto op : vopc_ops) {
        a.EmitVopc(op, Rdna2Operand::Vgpr(0), 1);
        expect(Rdna2Encoding::VOPC, op);
    }

    // VOP3 (5 ops)
    a.EmitVop3(Rdna2Opcode::V_FMA_F32, 4, Rdna2Operand::Vgpr(0), Rdna2Operand::LiteralF32(2.5f), Rdna2Operand::InlineFloat(1.0f));
    expect(Rdna2Encoding::VOP3, Rdna2Opcode::V_FMA_F32);
    a.EmitVop3(Rdna2Opcode::V_MAD_F32, 5, Rdna2Operand::Vgpr(0), Rdna2Operand::InlineFloat(2.0f), Rdna2Operand::InlineFloat(0.5f));
    expect(Rdna2Encoding::VOP3, Rdna2Opcode::V_MAD_F32);
    a.EmitVop3(Rdna2Opcode::V_CNDMASK_B32, 6, Rdna2Operand::Vgpr(1), Rdna2Operand::Vgpr(2), Rdna2Operand::VccLo());
    expect(Rdna2Encoding::VOP3, Rdna2Opcode::V_CNDMASK_B32);
    a.EmitVop3(Rdna2Opcode::V_ADD3_U32, 7, Rdna2Operand::Vgpr(0), Rdna2Operand::Vgpr(1), Rdna2Operand::InlineInt(7));
    expect(Rdna2Encoding::VOP3, Rdna2Opcode::V_ADD3_U32);
    a.EmitVop3(Rdna2Opcode::V_DOT2_F32_F16, 8, Rdna2Operand::Vgpr(0), Rdna2Operand::Vgpr(1), Rdna2Operand::InlineFloat(1.0f));
    expect(Rdna2Encoding::VOP3, Rdna2Opcode::V_DOT2_F32_F16);

    // SMEM (2 ops)
    a.EmitSmem(Rdna2Opcode::S_LOAD_DWORD, 10, 0, 16);
    expect(Rdna2Encoding::SMEM, Rdna2Opcode::S_LOAD_DWORD);
    a.EmitSmem(Rdna2Opcode::S_BUFFER_LOAD_DWORD, 11, 4, 8);
    expect(Rdna2Encoding::SMEM, Rdna2Opcode::S_BUFFER_LOAD_DWORD);

    // MUBUF (2 ops) & GLOBAL (2 ops)
    a.EmitMubuf(Rdna2Opcode::BUFFER_LOAD_DWORD, 9, 0, 0, 0, true);
    expect(Rdna2Encoding::MUBUF, Rdna2Opcode::BUFFER_LOAD_DWORD);
    a.EmitMubuf(Rdna2Opcode::BUFFER_STORE_DWORD, 9, 0, 0, 4, true);
    expect(Rdna2Encoding::MUBUF, Rdna2Opcode::BUFFER_STORE_DWORD);
    a.EmitGlobal(Rdna2Opcode::GLOBAL_LOAD_DWORD, 10, 0, 0);
    expect(Rdna2Encoding::FLAT, Rdna2Opcode::GLOBAL_LOAD_DWORD);
    a.EmitGlobal(Rdna2Opcode::GLOBAL_STORE_DWORD, 10, 0, 4);
    expect(Rdna2Encoding::FLAT, Rdna2Opcode::GLOBAL_STORE_DWORD);

    // EXP (MRT0, POS0, PARAM0)
    a.EmitExp(static_cast<uint8_t>(Rdna2ExportTarget::Mrt0), 0, 1, 2, 3, 0xF, true);
    expect(Rdna2Encoding::EXP, Rdna2Opcode::EXP);
    a.EmitExp(static_cast<uint8_t>(Rdna2ExportTarget::Pos0), 0, 1, 2, 3, 0xF, false);
    expect(Rdna2Encoding::EXP, Rdna2Opcode::EXP);
    a.EmitExp(static_cast<uint8_t>(Rdna2ExportTarget::Param0), 4, 5, 6, 7, 0xF, true);
    expect(Rdna2Encoding::EXP, Rdna2Opcode::EXP);

    std::vector<Rdna2Instruction> decoded;
    SX_REQUIRE(DisassembleRdna2(a.Words(), decoded).ok());
    SX_REQUIRE(decoded.size() == expected.size());

    for (size_t i = 0; i < expected.size(); ++i) {
        SX_REQUIRE(decoded[i].encoding == expected[i].first);
        SX_REQUIRE(decoded[i].opcode == expected[i].second);
        SX_REQUIRE(!decoded[i].mnemonic.empty());
        SX_REQUIRE(!decoded[i].disasm.empty());
    }

    // Verify 64-bit VOP2 literal word count and value
    SX_REQUIRE(decoded[vop2_lit_idx].word_count == 2);
    SX_REQUIRE(decoded[vop2_lit_idx].src[0].kind == Rdna2OperandKind::Literal32);
    SX_REQUIRE(std::fabs(decoded[vop2_lit_idx].src[0].float_value - 3.1415926f) < 1e-6f);

    // Verify EXP disassembly targets
    const size_t n = decoded.size();
    SX_REQUIRE(decoded[n - 3].disasm.find("mrt0") != std::string::npos);
    SX_REQUIRE(decoded[n - 2].disasm.find("pos0") != std::string::npos);
    SX_REQUIRE(decoded[n - 1].disasm.find("param0") != std::string::npos);
}

void TestComputeFmaLiftAndWarpExecute(uint64_t dev) {
    // Real RDNA2 compute shader in XS2\0 format:
    //   out[tid] = fma(float(tid), 2.5f, 1.0f)
    // Plus constant-foldable dead ops to verify Constant Folding & DCE!
    Rdna2Assembler a;
    a.EmitSopk(Rdna2Opcode::S_MOVK_I32, 4, 10);                        // dead: s4 = 10
    a.EmitSopk(Rdna2Opcode::S_ADDK_I32, 4, 22);                        // fold -> 32, then DCE
    a.EmitSopk(Rdna2Opcode::S_MULK_I32, 4, 2);                         // fold -> 64, then DCE
    a.EmitVop1(Rdna2Opcode::V_MOV_B32, 5, Rdna2Operand::InlineFloat(2.0f));
    a.EmitVop2(Rdna2Opcode::V_ADD_F32, 6, Rdna2Operand::InlineFloat(1.0f), 5);  // fold -> 3.0f, then DCE
    a.EmitVop2(Rdna2Opcode::V_MUL_F32, 7, Rdna2Operand::InlineFloat(4.0f), 6);  // fold -> 12.0f, then DCE

    // Live computation:
    //   v1 = float(v0)                (v0 = SV_DispatchThreadID.x)
    //   v1 = fma(v1, 2.5f, 1.0f)
    //   v2 = v0 << 2                  (byte offset = tid * 4)
    //   buffer_store_dword v1, v2, s[0:3], 0 offen
    //   s_endpgm
    a.EmitVop1(Rdna2Opcode::V_CVT_F32_U32, 1, Rdna2Operand::Vgpr(0));
    a.EmitVop3(Rdna2Opcode::V_FMA_F32, 1, Rdna2Operand::Vgpr(1),
               Rdna2Operand::LiteralF32(2.5f), Rdna2Operand::InlineFloat(1.0f));
    a.EmitVop2(Rdna2Opcode::V_LSHLREV_B32, 2, Rdna2Operand::InlineInt(2), 0);
    a.EmitMubuf(Rdna2Opcode::BUFFER_STORE_DWORD, 1, 2, 0, 0, /*offen=*/true);
    a.EmitSopp(Rdna2Opcode::S_ENDPGM);

    auto xs2 = a.BuildXs2(ShaderStage::Compute, 16, 1, 1, 16, 8);

    Rdna2LiftResult lifted{};
    Status st = LiftRdna2Xs2Shader(xs2, lifted);
    if (!st.ok()) {
        std::cerr << "[test_rdna2_lifter] FMA lift failed: " << st.message << "\n"
                  << lifted.hlsl_source << "\n";
    }
    SX_REQUIRE(st.ok());

    // Check disassembly
    SX_REQUIRE(lifted.instructions.size() == 11);
    SX_REQUIRE(lifted.instructions[6].opcode == Rdna2Opcode::V_CVT_F32_U32);
    SX_REQUIRE(lifted.instructions[7].opcode == Rdna2Opcode::V_FMA_F32);
    SX_REQUIRE(lifted.instructions[7].word_count == 3);  // VOP3 + literal32(2.5f)
    SX_REQUIRE(lifted.instructions[8].opcode == Rdna2Opcode::V_LSHLREV_B32);
    SX_REQUIRE(lifted.instructions[9].opcode == Rdna2Opcode::BUFFER_STORE_DWORD);
    SX_REQUIRE(lifted.instructions[10].opcode == Rdna2Opcode::S_ENDPGM);

    // Check SSA IR optimization stats (Constant Folding + DCE)
    SX_REQUIRE(lifted.ir.blocks.size() == 1);
    SX_REQUIRE(lifted.ir.opt_stats.folded_constants >= 4);
    SX_REQUIRE(lifted.ir.opt_stats.eliminated_ops >= 6);
    SX_REQUIRE(lifted.ir.opt_stats.final_ops < lifted.ir.opt_stats.initial_ops);
    SX_REQUIRE(!DumpRdna2Ir(lifted.ir).empty());

    // Validate SPIR-V 1.3 binary
    SX_REQUIRE(lifted.spirv_words.size() > 20);
    SX_REQUIRE(lifted.spirv_words[0] == 0x07230203u);
    SX_REQUIRE(lifted.spirv_words[1] == 0x00010300u);
    SX_REQUIRE(lifted.spirv_words[4] == 0u);
    SX_REQUIRE(ValidateSpirvBinary(lifted.spirv_words).ok());
    SX_REQUIRE(ContainsSpirvOpcode(lifted.spirv_words, 17));  // OpCapability
    SX_REQUIRE(ContainsSpirvOpcode(lifted.spirv_words, 14));  // OpMemoryModel
    SX_REQUIRE(ContainsSpirvOpcode(lifted.spirv_words, 15));  // OpEntryPoint
    SX_REQUIRE(ContainsSpirvOpcode(lifted.spirv_words, 16));  // OpExecutionMode
    SX_REQUIRE(ContainsSpirvOpcode(lifted.spirv_words, 12));  // OpExtInst (Fma)
    SX_REQUIRE(ContainsSpirvOpcode(lifted.spirv_words, 62));  // OpStore

    // Validate compiled PC DXBC
    ShaderModuleInfo dxbc_info{};
    SX_REQUIRE(InspectShaderContainer(lifted.pc_dxbc, dxbc_info).ok());
    SX_REQUIRE(dxbc_info.kind == ShaderContainerKind::Dxbc);
    SX_REQUIRE(dxbc_info.stage == ShaderStage::Compute);
    SX_REQUIRE(dxbc_info.shader_model_major == 5);

    // Execute lifted PC DXBC LIVE on D3D12 WARP!
    D3D12XBOX_ROOT_SIGNATURE_DESC rs_desc{};
    rs_desc.NumUavs = 1;
    uint64_t rs = 0;
    SX_REQUIRE(D3D12XboxCreateRootSignature(dev, &rs_desc, &rs) == 0);

    D3D12XBOX_COMPUTE_PSO_DESC pso_desc{};
    pso_desc.RootSignatureHandle = rs;
    pso_desc.CS = {lifted.pc_dxbc.data(), lifted.pc_dxbc.size()};
    uint64_t pso = 0;
    SX_REQUIRE(D3D12XboxCreateComputePipelineState(dev, &pso_desc, &pso) == 0);

    float gpu_out[16] = {};
    SX_REQUIRE(D3D12XboxDispatchCompute(dev, pso, 1, 1, 1, nullptr, 0, gpu_out, sizeof(gpu_out)) == 0);
    for (uint32_t tid = 0; tid < 16; ++tid) {
        const float expected = static_cast<float>(tid) * 2.5f + 1.0f;
        SX_REQUIRE(std::fabs(gpu_out[tid] - expected) < 1e-5f);
    }

    SX_REQUIRE(D3D12XboxDestroyPipelineState(dev, pso) == 0);
    SX_REQUIRE(D3D12XboxDestroyRootSignature(dev, rs) == 0);
}

void TestConditionalBranchCfgAndWarpExecute(uint64_t dev) {
    // RDNA2 compute shader with conditional branch diamond (S_CMP_LT_U32 + S_CBRANCH_SCC0):
    //   if (s0 < 16) v1 = tid + 100; else v1 = tid + 200;
    //   out[tid] = v1;
    Rdna2Assembler a;
    a.EmitSopc(Rdna2Opcode::S_CMP_LT_U32, Rdna2Operand::Sgpr(0), Rdna2Operand::InlineInt(16));
    const uint32_t cbr_off = a.EmitSopp(Rdna2Opcode::S_CBRANCH_SCC0, 0);

    // Then block (s0 < 16)
    a.EmitVop2(Rdna2Opcode::V_ADD_NC_U32, 1, Rdna2Operand::LiteralU32(100u), 0);
    const uint32_t br_join_off = a.EmitSopp(Rdna2Opcode::S_BRANCH, 0);

    // Else block (s0 >= 16)
    const uint32_t else_target = a.CurrentWordOffset();
    a.PatchBranch(cbr_off, else_target);
    a.EmitVop2(Rdna2Opcode::V_ADD_NC_U32, 1, Rdna2Operand::LiteralU32(200u), 0);

    // Join block
    const uint32_t join_target = a.CurrentWordOffset();
    a.PatchBranch(br_join_off, join_target);
    a.EmitVop2(Rdna2Opcode::V_LSHLREV_B32, 2, Rdna2Operand::InlineInt(2), 0);
    a.EmitMubuf(Rdna2Opcode::BUFFER_STORE_DWORD, 1, 2, 0, 0, /*offen=*/true);
    a.EmitSopp(Rdna2Opcode::S_ENDPGM);

    auto xs2 = a.BuildXs2(ShaderStage::Compute, 16, 1, 1, 16, 8);

    Rdna2LiftResult lifted{};
    Status st = LiftRdna2Xs2Shader(xs2, lifted);
    if (!st.ok()) {
        std::cerr << "[test_rdna2_lifter] Branch lift failed: " << st.message << "\n"
                  << lifted.hlsl_source << "\n";
    }
    SX_REQUIRE(st.ok());

    // Verify 4-block CFG with Phi merge in join block
    SX_REQUIRE(lifted.ir.blocks.size() == 4);
    SX_REQUIRE(lifted.ir.blocks[0].successors.size() == 2);
    SX_REQUIRE(lifted.ir.blocks[3].predecessors.size() == 2);
    bool has_phi = false;
    for (const auto& inst : lifted.ir.blocks[3].instructions) {
        if (inst.op == IrOp::Phi) has_phi = true;
    }
    SX_REQUIRE(has_phi);

    // Validate SPIR-V 1.3 contains OpBranchConditional (250) and OpPhi (245)
    SX_REQUIRE(ValidateSpirvBinary(lifted.spirv_words).ok());
    SX_REQUIRE(ContainsSpirvOpcode(lifted.spirv_words, 250));  // OpBranchConditional
    SX_REQUIRE(ContainsSpirvOpcode(lifted.spirv_words, 245));  // OpPhi

    // Execute LIVE on D3D12 WARP with both branch directions!
    D3D12XBOX_ROOT_SIGNATURE_DESC rs_desc{};
    rs_desc.NumRootConstants32Bit = 1;
    rs_desc.NumUavs = 1;
    uint64_t rs = 0;
    SX_REQUIRE(D3D12XboxCreateRootSignature(dev, &rs_desc, &rs) == 0);

    D3D12XBOX_COMPUTE_PSO_DESC pso_desc{};
    pso_desc.RootSignatureHandle = rs;
    pso_desc.CS = {lifted.pc_dxbc.data(), lifted.pc_dxbc.size()};
    uint64_t pso = 0;
    SX_REQUIRE(D3D12XboxCreateComputePipelineState(dev, &pso_desc, &pso) == 0);

    // Dispatch 1: s0 = 8 (< 16) -> expects tid + 100
    const uint32_t rc_low = 8u;
    uint32_t out_low[16] = {};
    SX_REQUIRE(D3D12XboxDispatchCompute(dev, pso, 1, 1, 1, &rc_low, 1, out_low, sizeof(out_low)) == 0);
    for (uint32_t tid = 0; tid < 16; ++tid) {
        SX_REQUIRE(out_low[tid] == tid + 100u);
    }

    // Dispatch 2: s0 = 24 (>= 16) -> expects tid + 200
    const uint32_t rc_high = 24u;
    uint32_t out_high[16] = {};
    SX_REQUIRE(D3D12XboxDispatchCompute(dev, pso, 1, 1, 1, &rc_high, 1, out_high, sizeof(out_high)) == 0);
    for (uint32_t tid = 0; tid < 16; ++tid) {
        SX_REQUIRE(out_high[tid] == tid + 200u);
    }

    SX_REQUIRE(D3D12XboxDestroyPipelineState(dev, pso) == 0);
    SX_REQUIRE(D3D12XboxDestroyRootSignature(dev, rs) == 0);
}

void TestVectorCmpSelectAndGlobalStoreOnWarp(uint64_t dev) {
    // Per-lane VOPC + V_CNDMASK_B32 + V_SQRT_F32 + V_FMAC_F32 + GLOBAL_STORE_DWORD:
    //   f = float(tid)
    //   v2 = sqrt(f + 1.0f)
    //   v3 = 0.5f; v3 = fmac(f, 2.0f, v3)   => f * 2.0f + 0.5f
    //   vcc = (f < 8.0f)
    //   v4 = vcc ? v2 : v3
    //   global_store_dword [tid * 4], v4
    Rdna2Assembler a;
    a.EmitVop1(Rdna2Opcode::V_CVT_F32_U32, 1, Rdna2Operand::Vgpr(0));
    a.EmitVop2(Rdna2Opcode::V_ADD_F32, 2, Rdna2Operand::InlineFloat(1.0f), 1);
    a.EmitVop1(Rdna2Opcode::V_SQRT_F32, 2, Rdna2Operand::Vgpr(2));
    a.EmitVop1(Rdna2Opcode::V_MOV_B32, 3, Rdna2Operand::InlineFloat(0.5f));
    a.EmitVop2(Rdna2Opcode::V_FMAC_F32, 3, Rdna2Operand::InlineFloat(2.0f), 1);
    a.EmitVopc(Rdna2Opcode::V_CMP_LT_F32, Rdna2Operand::LiteralF32(8.0f), 1);
    // Note: V_CMP_LT_F32 checks src0 < vsrc1, so (8.0f < f) is true for f > 8, false for f <= 8!
    // V_CNDMASK_B32 v4, v2, v3, vcc => vcc ? v3 : v2!
    // So when 8.0f < f (tid >= 9), selects v3 = f * 2.0f + 0.5f; else (tid <= 8) selects v2 = sqrt(f + 1.0f)!
    a.EmitVop3(Rdna2Opcode::V_CNDMASK_B32, 4, Rdna2Operand::Vgpr(2), Rdna2Operand::Vgpr(3), Rdna2Operand::VccLo());
    a.EmitVop2(Rdna2Opcode::V_LSHLREV_B32, 5, Rdna2Operand::InlineInt(2), 0);
    a.EmitGlobal(Rdna2Opcode::GLOBAL_STORE_DWORD, 4, 5, 0);
    a.EmitSopp(Rdna2Opcode::S_ENDPGM);

    auto xs2 = a.BuildXs2(ShaderStage::Compute, 16, 1, 1, 16, 8);
    Rdna2LiftResult lifted{};
    Status st = LiftRdna2Xs2Shader(xs2, lifted);
    if (!st.ok()) {
        std::cerr << "[test_rdna2_lifter] Select lift failed: " << st.message << "\n"
                  << lifted.hlsl_source << "\n";
    }
    SX_REQUIRE(st.ok());
    SX_REQUIRE(ValidateSpirvBinary(lifted.spirv_words).ok());
    SX_REQUIRE(ContainsSpirvOpcode(lifted.spirv_words, 169));  // OpSelect

    D3D12XBOX_ROOT_SIGNATURE_DESC rs_desc{};
    rs_desc.NumUavs = 1;
    uint64_t rs = 0;
    SX_REQUIRE(D3D12XboxCreateRootSignature(dev, &rs_desc, &rs) == 0);

    D3D12XBOX_COMPUTE_PSO_DESC pso_desc{};
    pso_desc.RootSignatureHandle = rs;
    pso_desc.CS = {lifted.pc_dxbc.data(), lifted.pc_dxbc.size()};
    uint64_t pso = 0;
    SX_REQUIRE(D3D12XboxCreateComputePipelineState(dev, &pso_desc, &pso) == 0);

    float gpu_out[16] = {};
    SX_REQUIRE(D3D12XboxDispatchCompute(dev, pso, 1, 1, 1, nullptr, 0, gpu_out, sizeof(gpu_out)) == 0);
    for (uint32_t tid = 0; tid < 16; ++tid) {
        const float f = static_cast<float>(tid);
        const float expected = (8.0f < f) ? (f * 2.0f + 0.5f) : std::sqrt(f + 1.0f);
        SX_REQUIRE(std::fabs(gpu_out[tid] - expected) < 1e-4f);
    }

    SX_REQUIRE(D3D12XboxDestroyPipelineState(dev, pso) == 0);
    SX_REQUIRE(D3D12XboxDestroyRootSignature(dev, rs) == 0);
}

void TestVertexAndPixelExpLiftAndWarpRender(uint64_t dev) {
    // 1. RDNA2 Vertex Shader (fullscreen triangle from SV_VertexID v0) with EXP POS0 & EXP PARAM0
    Rdna2Assembler vs;
    // x_u = (v0 << 1) & 2;  pos.x = float(x_u) * 2.0f - 1.0f
    vs.EmitVop2(Rdna2Opcode::V_LSHLREV_B32, 1, Rdna2Operand::InlineInt(1), 0);
    vs.EmitVop2(Rdna2Opcode::V_AND_B32, 1, Rdna2Operand::InlineInt(2), 1);
    vs.EmitVop1(Rdna2Opcode::V_CVT_F32_U32, 1, Rdna2Operand::Vgpr(1));
    vs.EmitVop3(Rdna2Opcode::V_FMA_F32, 1, Rdna2Operand::Vgpr(1),
                Rdna2Operand::InlineFloat(2.0f), Rdna2Operand::InlineFloat(-1.0f));
    // y_u = v0 & 2;  pos.y = float(y_u) * -2.0f + 1.0f
    vs.EmitVop2(Rdna2Opcode::V_AND_B32, 2, Rdna2Operand::InlineInt(2), 0);
    vs.EmitVop1(Rdna2Opcode::V_CVT_F32_U32, 2, Rdna2Operand::Vgpr(2));
    vs.EmitVop3(Rdna2Opcode::V_FMA_F32, 2, Rdna2Operand::Vgpr(2),
                Rdna2Operand::InlineFloat(-2.0f), Rdna2Operand::InlineFloat(1.0f));
    vs.EmitVop1(Rdna2Opcode::V_MOV_B32, 3, Rdna2Operand::InlineInt(0));          // pos.z = 0.0
    vs.EmitVop1(Rdna2Opcode::V_MOV_B32, 4, Rdna2Operand::InlineFloat(1.0f));   // pos.w = 1.0
    vs.EmitExp(static_cast<uint8_t>(Rdna2ExportTarget::Pos0), 1, 2, 3, 4, 0xF, /*done=*/false);

    // param0 = float4(0.5, 0.25, 1.0, 1.0)
    vs.EmitVop1(Rdna2Opcode::V_MOV_B32, 5, Rdna2Operand::InlineFloat(0.5f));
    vs.EmitVop1(Rdna2Opcode::V_MOV_B32, 6, Rdna2Operand::LiteralF32(0.25f));
    vs.EmitVop1(Rdna2Opcode::V_MOV_B32, 7, Rdna2Operand::InlineFloat(1.0f));
    vs.EmitVop1(Rdna2Opcode::V_MOV_B32, 8, Rdna2Operand::InlineFloat(1.0f));
    vs.EmitExp(static_cast<uint8_t>(Rdna2ExportTarget::Param0), 5, 6, 7, 8, 0xF, /*done=*/true);
    vs.EmitSopp(Rdna2Opcode::S_ENDPGM);

    // 2. RDNA2 Pixel Shader reading interpolated PARAM0 (v0..v3) and exporting MRT0
    Rdna2Assembler ps;
    ps.EmitVop2(Rdna2Opcode::V_MUL_F32, 0, Rdna2Operand::InlineFloat(2.0f), 0);  // R = 0.5 * 2.0 = 1.0
    ps.EmitVop2(Rdna2Opcode::V_ADD_F32, 1, Rdna2Operand::InlineFloat(0.5f), 1);  // G = 0.25 + 0.5 = 0.75
    ps.EmitExp(static_cast<uint8_t>(Rdna2ExportTarget::Mrt0), 0, 1, 2, 3, 0xF, /*done=*/true);
    ps.EmitSopp(Rdna2Opcode::S_ENDPGM);

    Rdna2LiftResult vs_lift{}, ps_lift{};
    Status st_vs = LiftRdna2Xs2Shader(vs.BuildXs2(ShaderStage::Vertex), vs_lift);
    Status st_ps = LiftRdna2Xs2Shader(ps.BuildXs2(ShaderStage::Pixel), ps_lift);
    if (!st_vs.ok()) std::cerr << "[test_rdna2_lifter] VS lift failed: " << st_vs.message << "\n" << vs_lift.hlsl_source << "\n";
    if (!st_ps.ok()) std::cerr << "[test_rdna2_lifter] PS lift failed: " << st_ps.message << "\n" << ps_lift.hlsl_source << "\n";
    SX_REQUIRE(st_vs.ok());
    SX_REQUIRE(st_ps.ok());

    SX_REQUIRE(vs_lift.ir.exports_pos0);
    SX_REQUIRE(vs_lift.ir.exports_param0);
    SX_REQUIRE(ps_lift.ir.exports_mrt0);
    SX_REQUIRE(vs_lift.hlsl_target == "vs_5_0");
    SX_REQUIRE(ps_lift.hlsl_target == "ps_5_0");
    SX_REQUIRE(ValidateSpirvBinary(vs_lift.spirv_words).ok());
    SX_REQUIRE(ValidateSpirvBinary(ps_lift.spirv_words).ok());

    // Render LIVE on D3D12 WARP using the lifted RDNA2 VS + PS Graphics PSO!
    D3D12XBOX_ROOT_SIGNATURE_DESC gfx_rs_desc{};
    uint64_t gfx_rs = 0;
    SX_REQUIRE(D3D12XboxCreateRootSignature(dev, &gfx_rs_desc, &gfx_rs) == 0);

    D3D12XBOX_GRAPHICS_PSO_DESC gfx_pso_desc{};
    gfx_pso_desc.RootSignatureHandle = gfx_rs;
    gfx_pso_desc.VS = {vs_lift.pc_dxbc.data(), vs_lift.pc_dxbc.size()};
    gfx_pso_desc.PS = {ps_lift.pc_dxbc.data(), ps_lift.pc_dxbc.size()};
    gfx_pso_desc.RtvFormat = 28;  // DXGI_FORMAT_R8G8B8A8_UNORM
    uint64_t gfx_pso = 0;
    SX_REQUIRE(D3D12XboxCreateGraphicsPipelineState(dev, &gfx_pso_desc, &gfx_pso) == 0);

    uint64_t hash1 = 0, hash2 = 0;
    SX_REQUIRE(D3D12XboxRenderWithPipelineState(dev, gfx_pso, 64, 64, 3, nullptr, &hash1) == 0);
    SX_REQUIRE(D3D12XboxRenderWithPipelineState(dev, gfx_pso, 64, 64, 3, nullptr, &hash2) == 0);
    SX_REQUIRE(hash1 != 0);
    SX_REQUIRE(hash1 == hash2);

    SX_REQUIRE(D3D12XboxDestroyPipelineState(dev, gfx_pso) == 0);
    SX_REQUIRE(D3D12XboxDestroyRootSignature(dev, gfx_rs) == 0);
}

void TestDot2SmemBarrierAndVop3ModifiersOnWarp(uint64_t dev) {
    // Exercises S_BUFFER_LOAD_DWORD, V_ADD3_U32, V_DOT2_F32_F16, V_FLOOR_F32, V_FRACT_F32,
    // V_RCP_F32, VOP3 neg+clamp modifiers, S_BARRIER, and BUFFER_STORE_DWORD LIVE on WARP!
    //
    // Root constant b0[0] = 0x40003C00 (fp16 pair: lo=1.0, hi=2.0)
    // Root constant b0[1] = 0x42003800 (fp16 pair: lo=0.5, hi=3.0)
    // dot2(a, b, float(tid)) = 1.0*0.5 + 2.0*3.0 + float(tid) = 6.5 + float(tid)
    //
    // Add VOP3 clamped (-float(tid) + 4.0f) => clamp(4.0 - tid, 0.0, 1.0)
    // Total = (6.5 + tid) + clamp(4.0 - tid, 0.0, 1.0)
    Rdna2Assembler a;
    a.EmitSmem(Rdna2Opcode::S_BUFFER_LOAD_DWORD, 4, 0, 0);   // s4 = rc0 (0x40003C00)
    a.EmitSmem(Rdna2Opcode::S_BUFFER_LOAD_DWORD, 5, 0, 4);   // s5 = rc1 (0x42003800)
    a.EmitSopp(Rdna2Opcode::S_WAITCNT, 0);
    a.EmitVop1(Rdna2Opcode::V_CVT_F32_U32, 1, Rdna2Operand::Vgpr(0));  // v1 = float(tid)
    a.EmitVop3(Rdna2Opcode::V_DOT2_F32_F16, 2, Rdna2Operand::Sgpr(4),
               Rdna2Operand::Sgpr(5), Rdna2Operand::Vgpr(1));          // v2 = 6.5 + tid
    Rdna2Operand neg_v1 = Rdna2Operand::Vgpr(1);
    neg_v1.neg = true;
    a.EmitVop3(Rdna2Opcode::V_ADD_F32, 3, neg_v1, Rdna2Operand::InlineFloat(4.0f),
               Rdna2Operand::InlineInt(0), /*clamp=*/true);            // v3 = clamp(4.0 - tid, 0, 1)
    a.EmitVop2(Rdna2Opcode::V_ADD_F32, 2, Rdna2Operand::Vgpr(2), 3);   // v2 += v3
    a.EmitSopp(Rdna2Opcode::S_BARRIER);
    // Byte address via V_ADD3_U32: tid + tid + (tid << 1) = tid * 4
    a.EmitVop2(Rdna2Opcode::V_LSHLREV_B32, 6, Rdna2Operand::InlineInt(1), 0);
    a.EmitVop3(Rdna2Opcode::V_ADD3_U32, 6, Rdna2Operand::Vgpr(0),
               Rdna2Operand::Vgpr(0), Rdna2Operand::Vgpr(6));
    a.EmitMubuf(Rdna2Opcode::BUFFER_STORE_DWORD, 2, 6, 0, 0, /*offen=*/true);
    a.EmitSopp(Rdna2Opcode::S_ENDPGM);

    Rdna2LiftResult lifted{};
    Status st = LiftRdna2Xs2Shader(a.BuildXs2(ShaderStage::Compute, 16, 1, 1), lifted);
    if (!st.ok()) {
        std::cerr << "[test_rdna2_lifter] Dot2/SMEM lift failed: " << st.message << "\n"
                  << lifted.hlsl_source << "\n";
    }
    SX_REQUIRE(st.ok());
    SX_REQUIRE(ValidateSpirvBinary(lifted.spirv_words).ok());
    SX_REQUIRE(ContainsSpirvOpcode(lifted.spirv_words, 148));  // OpDot
    SX_REQUIRE(ContainsSpirvOpcode(lifted.spirv_words, 224));  // OpControlBarrier

    // Negative SPIR-V validator checks
    auto bad_spv = lifted.spirv_words;
    bad_spv[0] = 0xDEADBEEFu;
    SX_REQUIRE(!ValidateSpirvBinary(bad_spv).ok());
    bad_spv = lifted.spirv_words;
    bad_spv.pop_back();
    SX_REQUIRE(!ValidateSpirvBinary(bad_spv).ok());

    D3D12XBOX_ROOT_SIGNATURE_DESC rs_desc{};
    rs_desc.NumRootConstants32Bit = 2;
    rs_desc.NumUavs = 1;
    uint64_t rs = 0;
    SX_REQUIRE(D3D12XboxCreateRootSignature(dev, &rs_desc, &rs) == 0);

    D3D12XBOX_COMPUTE_PSO_DESC pso_desc{};
    pso_desc.RootSignatureHandle = rs;
    pso_desc.CS = {lifted.pc_dxbc.data(), lifted.pc_dxbc.size()};
    uint64_t pso = 0;
    SX_REQUIRE(D3D12XboxCreateComputePipelineState(dev, &pso_desc, &pso) == 0);

    const uint32_t rc[2] = {0x40003C00u, 0x42003800u};
    float gpu_out[16] = {};
    SX_REQUIRE(D3D12XboxDispatchCompute(dev, pso, 1, 1, 1, rc, 2, gpu_out, sizeof(gpu_out)) == 0);
    for (uint32_t tid = 0; tid < 16; ++tid) {
        const float f = static_cast<float>(tid);
        float c = 4.0f - f;
        if (c < 0.0f) c = 0.0f;
        if (c > 1.0f) c = 1.0f;
        const float expected = 6.5f + f + c;
        SX_REQUIRE(std::fabs(gpu_out[tid] - expected) < 1e-4f);
    }

    SX_REQUIRE(D3D12XboxDestroyPipelineState(dev, pso) == 0);
    SX_REQUIRE(D3D12XboxDestroyRootSignature(dev, rs) == 0);
}

}  // namespace

int main() {
    TestXs2ContainerParsing();
    TestInlineConstantsAndAllEncodings();

    // Create a live D3D12 WARP device for GPU verification of lifted RDNA2 shaders
    D3D12XBOX_CREATE_DEVICE_PARAMETERS params{};
    params.Version = 1;
    params.PreferWarp = 1;
    uint64_t dev = 0;
    SX_REQUIRE(D3D12XboxCreateSyntheticDevice(&params, &dev) == 0);
    SX_REQUIRE(dev != 0);

    TestComputeFmaLiftAndWarpExecute(dev);
    TestConditionalBranchCfgAndWarpExecute(dev);
    TestVectorCmpSelectAndGlobalStoreOnWarp(dev);
    TestVertexAndPixelExpLiftAndWarpRender(dev);
    TestDot2SmemBarrierAndVop3ModifiersOnWarp(dev);

    SX_REQUIRE(D3D12XboxDestroyDevice(dev) == 0);

    std::cout << "[PASS] test_rdna2_lifter (XS2 disasm + CFG/SSA IR + const-fold/DCE + SPIR-V 1.3 + live WARP CS/VS/PS)\n";
    return 0;
}
