"""
CPU Pipeline Simulator (single-file edition)

- 5 stages: IF → ID → EX → MEM → WB
- MIPS-like ISA: ADD, SUB, ADDI, LW, SW, BEQ, NOP, HALT
- Features: EX/MEM & MEM/WB forwarding (no EX forwarding for loads),
            1-cycle load-use stall, 1-cycle flush on taken BEQ,
            word-aligned memory, r0 hard-wired to 0
- Perf: ~1k–100k instr/s depending on machine

USAGE
------
# Run the demo program
python pipeline_cpu_all_in_one.py --demo

# Run built-in self tests (forwarding, load-use stall, branch flush)
python pipeline_cpu_all_in_one.py --selftest

# Run a quick micro-benchmark
python pipeline_cpu_all_in_one.py --bench

Pipeline diagram (conceptual)
-----------------------------
        ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐
PC ───► │  IF    │──►│  ID    │──►│  EX    │──►│  MEM   │──►│  WB    │
        └────────┘   └────────┘   └────────┘   └────────┘   └────────┘
             │            │            │            │            │
             ▼            ▼            ▼            ▼            ▼
           IF/ID        ID/EX        EX/MEM       MEM/WB     Register File
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import time
import argparse
import sys

# =========================
# ISA + helpers
# =========================
REG_COUNT = 32
WORD = 4

# Instruction tuples:
# R-type:  ("ADD", rd, rs, rt)   rd <- rs + rt
#          ("SUB", rd, rs, rt)
# I-type:  ("ADDI", rt, rs, imm) rt <- rs + imm
# Memory:  ("LW", rt, (offset, rs))   rt <- MEM[rs+offset]
#          ("SW", rt, (offset, rs))   MEM[rs+offset] <- rt
# Branch:  ("BEQ", rs, rt, imm)  if rs==rt then PC <- PC + 4 + imm*4
# Misc:    ("NOP",), ("HALT",)

def sign_extend_16(x: int) -> int:
    x &= 0xFFFF
    return x if x < 0x8000 else x - 0x10000

def decode_fields(instr: Optional[tuple]) -> Dict[str, int]:
    if instr is None:
        return {}
    op = instr[0]
    if op in ("ADD", "SUB"):
        _, rd, rs, rt = instr
        return dict(alu_op=op, rs=rs, rt=rt, rd=rd, write_reg=rd, reg_write=True)
    elif op == "ADDI":
        _, rt, rs, imm = instr
        return dict(alu_op=op, rs=rs, rt=rt, imm=imm, write_reg=rt, reg_write=True)
    elif op == "LW":
        _, rt, (offset, rs) = instr
        return dict(alu_op="ADD", rs=rs, rt=rt, imm=offset, mem_read=True, mem_to_reg=True, write_reg=rt, reg_write=True)
    elif op == "SW":
        _, rt, (offset, rs) = instr
        return dict(alu_op="ADD", rs=rs, rt=rt, imm=offset, mem_write=True)
    elif op == "BEQ":
        _, rs, rt, imm = instr
        return dict(alu_op="SUB", rs=rs, rt=rt, imm=imm)
    elif op in ("NOP", "HALT"):
        return {}
    else:
        raise ValueError(f"Unknown opcode {op}")

# =========================
# Machine state
# =========================
@dataclass
class CPUState:
    pc: int = 0
    regs: List[int] = field(default_factory=lambda: [0]*REG_COUNT)
    memory: bytearray = field(default_factory=lambda: bytearray(4096))  # 4KB demo
    running: bool = True

    def read_mem_word(self, addr: int) -> int:
        if addr % 4 != 0:
            raise ValueError(f"Unaligned load at {addr}")
        b = self.memory[addr:addr+4]
        return int.from_bytes(b, byteorder="little", signed=True)

    def write_mem_word(self, addr: int, val: int):
        if addr % 4 != 0:
            raise ValueError(f"Unaligned store at {addr}")
        self.memory[addr:addr+4] = int(val).to_bytes(4, byteorder="little", signed=True)

    def set_reg(self, idx: int, val: int):
        if idx != 0:  # r0 is hard-wired to 0
            self.regs[idx] = int(val)

# =========================
# Pipeline register (latches)
# =========================
@dataclass
class PipelineReg:
    instr: Optional[tuple] = None
    pc: int = 0
    # Decoded fields / IDs
    rs: Optional[int] = None
    rt: Optional[int] = None
    rd: Optional[int] = None
    imm: Optional[int] = None
    alu_op: Optional[str] = None
    # Operand values and results
    alu_a: Optional[int] = None
    alu_b: Optional[int] = None
    alu_out: Optional[int] = None
    store_val: Optional[int] = None  # SW data (possibly forwarded)
    # Control signals
    mem_write: bool = False
    mem_read: bool = False
    reg_write: bool = False
    mem_to_reg: bool = False
    write_reg: Optional[int] = None
    write_val: Optional[int] = None
    taken_branch: bool = False

# =========================
# Pipeline CPU
# =========================
class PipelineCPU:
    def __init__(self, program: List[tuple]):
        self.state = CPUState()
        self.program = program
        self.text_base = 0  # PC starts at 0
        self.if_id = PipelineReg()
        self.id_ex = PipelineReg()
        self.ex_mem = PipelineReg()
        self.mem_wb = PipelineReg()
        self.cycle = 0
        self.retired = 0
        self.stalls = 0
        self.flushes = 0

    # ---- Helpers ----
    def fetch_instr(self, pc: int) -> Optional[tuple]:
        idx = (pc - self.text_base) // 4
        if 0 <= idx < len(self.program):
            return self.program[idx]
        return ("HALT",)

    def forwarding(self, src_reg: Optional[int]) -> Optional[int]:
        """
        Return forwarded value for src_reg if available, else None.

        Important: EX/MEM forwarding is only valid for ALU results
                   (NOT for loads where value isn't ready until MEM/WB).
        """
        if src_reg is None or src_reg == 0:
            return None
        # EX/MEM: forward ALU result if it writes a reg and is NOT a load-to-reg (mem_to_reg)
        if self.ex_mem.reg_write and self.ex_mem.write_reg == src_reg and not self.ex_mem.mem_to_reg:
            return self.ex_mem.alu_out
        # MEM/WB: forward whatever will be written back (ALU or load result)
        if self.mem_wb.reg_write and self.mem_wb.write_reg == src_reg:
            return self.mem_wb.write_val
        return None

    # ---- Stages ----
    def stage_wb(self):
        st = self.mem_wb
        if st.instr is None:
            return
        if st.reg_write and st.write_reg is not None:
            self.state.set_reg(st.write_reg, st.write_val)

    def stage_mem(self):
        prev = self.ex_mem
        cur = PipelineReg()
        cur.instr = prev.instr
        cur.pc = prev.pc
        cur.reg_write = prev.reg_write
        cur.write_reg = prev.write_reg
        cur.mem_to_reg = prev.mem_to_reg

        if prev.instr is None:
            self.mem_wb = cur
            return

        if prev.mem_read:
            addr = prev.alu_out
            val = self.state.read_mem_word(addr)
            cur.write_val = val
        elif prev.mem_write:
            addr = prev.alu_out
            self.state.write_mem_word(addr, prev.store_val if prev.store_val is not None else 0)
            cur.write_val = None
        else:
            cur.write_val = prev.alu_out

        self.mem_wb = cur

    def stage_ex(self):
        prev = self.id_ex
        cur = PipelineReg()
        cur.instr = prev.instr
        cur.pc = prev.pc
        cur.mem_write = prev.mem_write
        cur.mem_read = prev.mem_read
        cur.reg_write = prev.reg_write
        cur.mem_to_reg = prev.mem_to_reg
        cur.write_reg = prev.write_reg
        cur.rs = prev.rs
        cur.rt = prev.rt
        cur.rd = prev.rd
        cur.imm = prev.imm
        cur.store_val = prev.store_val

        if prev.instr is None:
            self.ex_mem = cur
            return

        op = prev.instr[0]

        # Forwarding for ALU inputs
        a = prev.alu_a
        b = prev.alu_b

        fwd_a = self.forwarding(prev.rs)
        if fwd_a is not None:
            a = fwd_a

        if op in ("ADD", "SUB", "BEQ"):
            fwd_b = self.forwarding(prev.rt)
            if fwd_b is not None:
                b = fwd_b

        # For SW, forward the store data from rt
        if op == "SW":
            fwd_store = self.forwarding(prev.rt)
            if fwd_store is not None:
                cur.store_val = fwd_store

        # Execute
        if op == "ADD":
            cur.alu_out = a + b
        elif op == "SUB":
            cur.alu_out = a - b
        elif op == "ADDI":
            cur.alu_out = a + b
        elif op in ("LW", "SW"):
            cur.alu_out = a + b  # effective address
        elif op == "BEQ":
            cur.alu_out = a - b
            if cur.alu_out == 0:
                # taken: PC <- PC + 4 + imm*4
                self.state.pc = prev.pc + 4 + (prev.imm * 4)
                cur.taken_branch = True
        elif op in ("NOP", "HALT"):
            cur.alu_out = 0
        else:
            raise ValueError(f"Unknown op in EX: {op}")

        self.ex_mem = cur

    def stage_id(self):
        prev = self.if_id
        cur = PipelineReg()
        cur.instr = prev.instr
        cur.pc = prev.pc

        if prev.instr is None:
            self.id_ex = cur
            return

        op = prev.instr[0]
        fields = decode_fields(prev.instr)
        for k, v in fields.items():
            setattr(cur, k, v)

        rs_val = self.state.regs[cur.rs] if cur.rs is not None else 0
        rt_val = self.state.regs[cur.rt] if cur.rt is not None else 0

        if op == "ADDI":
            imm = sign_extend_16(cur.imm)
            cur.alu_a, cur.alu_b = rs_val, imm
        elif op in ("ADD", "SUB"):
            cur.alu_a, cur.alu_b = rs_val, rt_val
        elif op in ("LW", "SW"):
            imm = sign_extend_16(cur.imm)
            cur.alu_a, cur.alu_b = rs_val, imm
            if op == "SW":
                cur.store_val = rt_val  # store data (can be forwarded later)
        elif op == "BEQ":
            imm = sign_extend_16(cur.imm)
            cur.alu_a, cur.alu_b = rs_val, rt_val
            cur.imm = imm
        else:
            cur.alu_a, cur.alu_b = 0, 0

        self.id_ex = cur

    def stage_if(self, stall: bool, flush: bool):
        if flush:
            self.if_id = PipelineReg()  # bubble
            self.flushes += 1
            return

        if stall:
            self.stalls += 1
            return

        instr = self.fetch_instr(self.state.pc)
        reg = PipelineReg()
        reg.instr = instr
        reg.pc = self.state.pc
        self.state.pc += 4

        # pre-extract rs/rt for hazard checks
        d = decode_fields(instr)
        reg.rs = d.get("rs")
        reg.rt = d.get("rt")
        self.if_id = reg

    def _load_use_hazard(self) -> bool:
        """
        Classic 5-stage hazard:
        If ID/EX is a LW producing register X, and IF/ID consumes X, stall one cycle.
        """
        if self.id_ex.instr is None or self.if_id.instr is None:
            return False
        if not self.id_ex.mem_read:
            return False
        src = self.id_ex.write_reg
        if src is None:
            return False
        return src in (self.if_id.rs, self.if_id.rt)

    # ---- Run loop ----
    def step(self):
        self.cycle += 1

        # Control hazard: discovered in last EX (now in EX/MEM latch)
        flush = (self.ex_mem.instr is not None and self.ex_mem.taken_branch)

        # Data hazard: load-use between ID/EX (load) and IF/ID (consumer)
        stall = self._load_use_hazard()

        # WB -> MEM -> EX -> ID -> IF
        self.stage_wb()
        self.stage_mem()
        self.stage_ex()

        if stall:
            # Insert bubble into ID/EX, hold IF/ID
            self.id_ex = PipelineReg()
        else:
            self.stage_id()

        self.stage_if(stall=stall, flush=flush)

        # Retire at WB
        if self.mem_wb.instr and self.mem_wb.instr[0] not in ("NOP", None):
            if self.mem_wb.instr[0] == "HALT":
                self.state.running = False
            else:
                self.retired += 1

    def run(self, max_cycles=100000):
        t0 = time.time()
        while self.state.running and self.cycle < max_cycles:
            self.step()
        elapsed = max(time.time() - t0, 1e-9)
        ips = self.retired / elapsed
        print(f"Cycles: {self.cycle} | Retired: {self.retired} | Stalls: {self.stalls} | Flushes: {self.flushes} | ~{ips:.0f} instr/s")
        self.dump_state()

    def dump_state(self):
        print("\nREGS:")
        for i in range(0, REG_COUNT, 8):
            row = " ".join([f"r{j:02}={self.state.regs[j]:>8d}" for j in range(i, i+8)])
            print(row)
        print("\nMEM[0..64):", [int.from_bytes(self.state.memory[i:i+4],'little',signed=True) for i in range(0, 64, 4)])

# =========================
# Demo / Bench / Self-tests
# =========================
def demo_program() -> List[tuple]:
    #   r1 = 10
    #   r2 = 20
    #   r3 = r1 + r2          ; 30
    #   MEM[0] = r3           ; store 30
    #   r4 = MEM[0]           ; load 30 (tests load-use with next ADD)
    #   r5 = r3 + r4          ; 60 (forwarding + load-use bubble)
    #   if (r5 == 60) branch +1 (skip next ADDI)
    #   r6 = r6 + 99          ; should be skipped on taken branch
    #   HALT
    return [
        ("ADDI", 1, 0, 10),            # r1 <- 10
        ("ADDI", 2, 0, 20),            # r2 <- 20
        ("ADD",  3, 1, 2),             # r3 <- r1 + r2
        ("SW",   3, (0, 0)),           # MEM[0] <- r3
        ("LW",   4, (0, 0)),           # r4 <- MEM[0]
        ("ADD",  5, 3, 4),             # r5 <- r3 + r4
        ("ADDI", 6, 0, 60),            # r6 <- 60
        ("BEQ",  5, 6, 1),             # if r5==r6 branch to skip next
        ("ADDI", 6, 6, 99),            # (should be skipped)
        ("HALT",),
    ]

def bench_program(n_adds: int = 5000) -> List[tuple]:
    # Simple chain to exercise ALU forwarding
    program = [("ADDI", 1, 0, 1)]
    program += [("ADD", 1, 1, 1)] * n_adds
    program += [("HALT",)]
    return program

# ---- Simple self-tests (no pytest required) ----
def test_forwarding():
    # r1=5; r2=7; r3=r1+r2; r4=r3+r2 (needs ALU->ALU forwarding)
    program = [
        ("ADDI", 1, 0, 5),
        ("ADDI", 2, 0, 7),
        ("ADD",  3, 1, 2),
        ("ADD",  4, 3, 2),
        ("HALT",),
    ]
    cpu = PipelineCPU(program)
    cpu.run(max_cycles=10000)
    assert cpu.state.regs[3] == 12, "r3 should be 12"
    assert cpu.state.regs[4] == 19, "r4 should be 19"

def test_load_use_stall():
    # LW r1,0(r0) ; ADD r2,r1,r1 -> must stall once and compute 84
    program = [
        ("LW",   1, (0, 0)),
        ("ADD",  2, 1, 1),
        ("HALT",),
    ]
    cpu = PipelineCPU(program)
    cpu.state.write_mem_word(0, 42)
    cpu.run(max_cycles=10000)
    assert cpu.state.regs[1] == 42, "r1 should be loaded with 42"
    assert cpu.state.regs[2] == 84, "r2 should be 84"
    assert cpu.stalls >= 1, "Should see at least one load-use stall"

def test_branch_flush():
    # Taken BEQ should skip the next ADDI and execute the following
    program = [
        ("ADDI", 1, 0, 10),
        ("ADDI", 2, 0, 10),
        ("BEQ",  1, 2, 1),     # taken -> skip next
        ("ADDI", 3, 0, 99),    # should be flushed
        ("ADDI", 3, 0, 7),     # should execute
        ("HALT",),
    ]
    cpu = PipelineCPU(program)
    cpu.run(max_cycles=10000)
    assert cpu.state.regs[3] == 7, "r3 should be 7 (skipped the 99)"
    assert cpu.flushes >= 1, "Should see at least one flush on taken branch"

def run_selftests() -> int:
    tests = [
        ("forwarding", test_forwarding),
        ("load_use_stall", test_load_use_stall),
        ("branch_flush", test_branch_flush),
    ]
    failures = 0
    t0 = time.time()
    for name, fn in tests:
        try:
            fn()
            print(f"[PASS] {name}")
        except AssertionError as e:
            failures += 1
            print(f"[FAIL] {name}: {e}")
        except Exception as e:
            failures += 1
            print(f"[ERROR] {name}: {e}")
    dt = time.time() - t0
    print(f"Self-tests completed in {dt:.2f}s with {failures} failure(s).")
    return failures

# =========================
# CLI
# =========================
def main():
    p = argparse.ArgumentParser(description="5-stage CPU pipeline simulator (single-file).")
    p.add_argument("--demo", action="store_true", help="run the demo program")
    p.add_argument("--bench", action="store_true", help="run a micro-benchmark")
    p.add_argument("--bench-n", type=int, default=5000, help="number of ADDs in bench")
    p.add_argument("--selftest", action="store_true", help="run built-in tests (no pytest required)")
    args = p.parse_args()

    if args.selftest:
        sys.exit(run_selftests())

    if args.bench:
        program = bench_program(args.bench_n)
        cpu = PipelineCPU(program)
        cpu.run()
        return

    # default: demo if --demo provided, else demo anyway
    program = demo_program()
    cpu = PipelineCPU(program)
    cpu.run()

if __name__ == "__main__":
    main()
