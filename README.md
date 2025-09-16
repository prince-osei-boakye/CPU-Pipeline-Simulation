# CPU Pipeline Simulator

A Python-based **5-stage pipelined CPU simulator** designed for learning and experimentation in computer architecture.  
This project models a simplified **MIPS-like ISA**, demonstrates real pipeline behavior (forwarding, stalls, flushes),  
and includes **self-tests**, **demo programs**, and **benchmarks**.

<p align="center">
  <img src="assets/pipeline_diagram.png" alt="Pipeline Diagram" width="600"/>
</p>

---

## Features
- **5 Pipeline Stages**: Instruction Fetch (IF), Instruction Decode (ID), Execute (EX), Memory (MEM), Write Back (WB).  
- **MIPS-like ISA**:  
  - Arithmetic: `ADD`, `SUB`, `ADDI`  
  - Memory: `LW`, `SW`  
  - Branch: `BEQ`  
  - Misc: `NOP`, `HALT`  
- **Hazard Handling**:  
  - **Forwarding** (EX/MEM and MEM/WB bypassing)  
  - **1-cycle load-use stall**  
  - **1-cycle branch flush**  
- **Performance**: Executes **1k–100k instructions/s**, depending on machine.  
- **Self-tests**: Forwarding, load-use stall, and branch flush.  
- **Programs**: Demo program and micro-benchmark included.  

---

## Overview

This simulator replicates the behavior of a **classic 5-stage pipelined CPU**, often taught in computer architecture courses.  
It provides a hands-on way to understand **instruction-level parallelism**, **hazards**, and **pipeline control mechanisms**.  

The pipeline works as follows:

1. **IF (Instruction Fetch)** → fetch next instruction from program memory.  
2. **ID (Instruction Decode)** → decode fields, read registers, sign-extend immediates.  
3. **EX (Execute)** → perform ALU operation, compute addresses, evaluate branches.  
4. **MEM (Memory)** → access memory for loads/stores.  
5. **WB (Write Back)** → write results back to registers.  

---

## Installation

Clone the repo and run with Python 3.9+:  

```bash
git clone https://github.com/yourusername/cpu-pipeline-sim.git
cd cpu-pipeline-sim
python pipeline_cpu_all_in_one.py --demo
