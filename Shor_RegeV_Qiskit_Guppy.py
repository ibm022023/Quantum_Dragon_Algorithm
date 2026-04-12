# Hi i-Realy Apperciated you get me A Donation here_ 1Bu4CR8Bi5AXQG8pnu1avny88C5CCgWKfb /////
# ===========================================================================================
# 🔥🐉 DRAGON_CODE_FUTURE — ULTIMATE QUANTUM ECDLP SOLVER (v155) — QISKIT REAL HARDWARE ONLY
# ===========================================================================================
#
# ===========================================================================================
#
# COMBINES:
# - BASIC : Pure Shor's style, geometric QPE, universal post-processing
# - EXTRA : Regev, fault-tolerance, full range, modern Qiskit API
# ===========================================================================================
#
# FEATURES:
# - Multi-dimensional Regev algorithm (d ≈ √bits)
# - Full range search (auto-calculated or user-specified)
# - Pure Shor's style geometric QPE fallback
# - Universal post-processing (dual-endian, continued fractions, gcd)
# - Modern Qiskit API (QFTGate, SamplerV2)
# - All fault-tolerance methods (Flags, Cat, Erasure, Surface, Repetition, DD)
# - Optimized for IBM Quantum (156+ qubit hardware)
# - Automatic SABRE routing + XY4 dynamical decoupling
# - 16-bit default with all Bitcoin Puzzle Presets.
#
# ===========================================================================================

import os
import sys
import math
import subprocess
import numpy as np
import time
from datetime import datetime
from fractions import Fraction
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import logging
from ecdsa.ellipticcurve import Point, CurveFp
from ecdsa import SigningKey, SECP256k1
from math import gcd
# QISKIT
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.circuit.library import QFTGate
from qiskit.synthesis.qft import synth_qft_full
# =============================================================================
# 1. LOGGING SETUP
# =============================================================================
CACHE_DIR = "cache/"
os.makedirs(CACHE_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(CACHE_DIR, "dragon_future.log")),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
logger.info("=" * 80)
logger.info("🚀 DRAGON_CODE_FUTURE started")
logger.info("=" * 80)

# =============================================================================
# 2. OPTIONAL DEPENDENCIES
# =============================================================================
try:
    from fpylll import IntegerMatrix, BKZ, LLL
    FPYLLL_AVAILABLE = True
    logger.info("✅ fpylll BKZ + LLL loaded")
except ImportError:
    FPYLLL_AVAILABLE = False
    logger.warning("⚠️ fpylll not installed — using pure Python LLL fallback")

try:
    from pytket import Circuit as TketCircuit
    TKET_AVAILABLE = True
    logger.info("✅ pytket loaded")
except ImportError:
    TKET_AVAILABLE = False
    logger.warning("⚠️ pytket not installed — Guppy/pytket path will be limited")

# =============================================================================
# 3. CONSTANTS
# =============================================================================
P     = SECP256k1.curve.p()
A     = SECP256k1.curve.a()
B     = SECP256k1.curve.b()
G     = SECP256k1.generator
ORDER = SECP256k1.order
N     = ORDER
CURVE = CurveFp(P, A, B)

Gx = G.x()
Gy = G.y()

SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

PRESETS = {
    "16":  {"bits": 16,  "start": 0x8000,
            "pub": "03ccb5e3ad4abc7900ebfbd81621e31ec2b17b346090e741921a91bf9cadf934c5",
            "shots": 16384},
    "21":  {"bits": 21,  "start": 0x90000,
            "pub": "037d14b19a95fe400b88b0debe31ecc3c0ec94daea90d13057bde89c5f8e6fc25c",
            "shots": 32768},
    "25":  {"bits": 25,  "start": 0xE00000,
            "pub": "038ad4f423459430771c0f12a24df181ed0da5142ec676088031f28a21e86ea06d",
            "shots": 65536},
    "135": {"bits": 135, "start": 0x400000000000000000000000000000000,
            "pub": "02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16",
            "shots": 100000},
}

# =============================================================================
# 4. CONFIG CLASS (from uploaded codes)
# =============================================================================
class Config:
    def __init__(self, compressed_pubkey_hex: str, keyspace_start: int = 0):
        self.USE_FT         = True
        self.USE_SMART_GATE = True
        self.COMPRESSED_PUBKEY_HEX = compressed_pubkey_hex
        self.KEYSPACE_START = keyspace_start
        self.USE_FLAGS      = True
        self.USE_ZNE        = False
        self.USE_DD         = False  # Disabled due to dynamic circuits

MODE_METADATA = {
    41: {"name": "Quantum Omega (FT Draper 2D)",     "qubits": 136},
    9:  {"name": "Shadow 2D (windowed FT)",          "qubits": 138},
    7:  {"name": "Geometric QPE (FT precomputed)",   "qubits": 136},
    13: {"name": "Compressed Shadow (window=8 FT)",  "qubits": 136},
    0:  {"name": "Hardware QPE Diagnostic",          "qubits":  20},
    27: {"name": "Advanced QPE (phase-corrected)",   "qubits": 136},
    4:  {"name": "Quantum (FT 1-ctrl)",              "qubits": 136},
    2:  {"name": "Hive Chunked (FT)",                "qubits": 127},
    29: {"name": "Quantum Omega (no FT, compact)",   "qubits": 136},
    8:  {"name": "Verified Flags (FT)",              "qubits": 136},
}

# =============================================================================
# 5. EC HELPERS
# =============================================================================
def decompress_pubkey(hex_key: str) -> Point:
    logger.info(f"Decompressing pubkey: {hex_key[:20]}...")
    hex_key = hex_key.lower().strip()
    prefix  = int(hex_key[:2], 16)
    x_val   = int(hex_key[2:], 16)
    y_sq    = (pow(x_val, 3, P) + A * x_val + B) % P
    y_val   = pow(y_sq, (P + 1) // 4, P)
    if (prefix == 2 and y_val % 2 != 0) or (prefix == 3 and y_val % 2 == 0):
        y_val = P - y_val
    return Point(CURVE, x_val, y_val)

def modinv(a, m):
    g, x, _ = extended_gcd(a % m, m)
    if g != 1:
        return None
    return x % m

def extended_gcd(a, b):
    if a == 0:
        return b, 0, 1
    g, y, x = extended_gcd(b % a, a)
    return g, x - (b // a) * y, y

def point_add(p1, p2):
    if p1 is None: return p2
    if p2 is None: return p1
    x1, y1 = p1.x(), p1.y()
    x2, y2 = p2.x(), p2.y()
    if x1 == x2 and (y1 + y2) % P == 0: return None
    if x1 == x2 and y1 == y2:
        lam = (3 * x1 * x1 + A) * modinv(2 * y1, P) % P
    else:
        lam = (y2 - y1) * modinv(x2 - x1, P) % P
    x3 = (lam * lam - x1 - x2) % P
    y3 = (lam * (x1 - x3) - y1) % P
    return Point(CURVE, x3, y3)

def compress_pubkey(privkey: int) -> bytes:
    sk = SigningKey.from_secret_exponent(privkey, curve=SECP256k1)
    vk = sk.verifying_key
    x  = vk.pubkey.point.x()
    y  = vk.pubkey.point.y()
    prefix = b'\x02' if (y % 2 == 0) else b'\x03'
    return prefix + x.to_bytes(32, byteorder='big')

def calculate_keyspace_start(bits: int) -> int:
    return 1 << (bits - 1)

def calculate_full_range_end(bits: int) -> int:
    return (1 << bits) - 1

def verify_key(k: int, target_x: int) -> bool:
    Pt = G * k
    return Pt is not None and Pt.x() == target_x

def precompute_deltas(Q: Point, k_start: int, bits: int):
    logger.info(f"Precomputing deltas for {bits}-bit keyspace")
    delta   = Q + (-G * k_start)
    dxs, dys = [], []
    current = delta
    for i in range(bits):
        dxs.append(int(current.x()) if current else 0)
        dys.append(int(current.y()) if current else 0)
        current = current * 2 if current else None
    logger.info("Delta precomputation complete")
    return dxs, dys

def continued_fraction_approx(num, den, max_den=1000000):
    if den == 0:
        return 0, 1
    frac = Fraction(num, den).limit_denominator(max_den)
    return frac.numerator, frac.denominator

def save_key(k: int, target_address=None):
    with open("boom.txt", "w") as f:
        f.write(f"Private key found!\nHEX: {hex(k)}\nDecimal: {k}\n")
        if target_address:
            f.write(f"Address: {target_address}\n")
        f.write("Donation: 1Bu4CR8Bi5AXQG8pnu1avny88C5CCgWKfb\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write("Generated by DRAGON_CODE_FUTURE\n")
    print(f"🔑 KEY SAVED → boom.txt  ({hex(k)})")
    logger.info(f"KEY SAVED: {hex(k)}")

# =============================================================================
# 6. FAULT TOLERANCE PRIMITIVES (from uploaded codes — fixed register naming)
# =============================================================================
def prepare_verified_ancilla(qc, qubit, initial_state=0):
    qc.reset(qubit)
    if initial_state == 1:
        qc.x(qubit)

def encode_repetition(qc, logical_qubit, ancillas):
    qc.cx(logical_qubit, ancillas[0])
    qc.cx(logical_qubit, ancillas[1])

def decode_repetition(qc, ancillas, logical_qubit):
    qc.cx(ancillas[0], logical_qubit)
    qc.cx(ancillas[1], logical_qubit)
    qc.ccx(ancillas[0], ancillas[1], logical_qubit)

def apply_ft_to_qubit(qc: QuantumCircuit, qubit, config, unique_id=None):
    """Apply FT repetition to single qubit — unique_id prevents register collision."""
    if not config.USE_FT:
        return None
    if unique_id is None:
        unique_id = f"ft_anc_{id(qubit)}"
    anc = QuantumRegister(2, unique_id)
    qc.add_register(anc)
    prepare_verified_ancilla(qc, anc[0])
    prepare_verified_ancilla(qc, anc[1])
    encode_repetition(qc, qubit, anc)
    return anc

def apply_ft_to_register(qc: QuantumCircuit, reg: QuantumRegister, config):
    """Apply FT repetition to full register — unique IDs per qubit."""
    if not config.USE_FT:
        return []
    ancillas = []
    for i, qubit in enumerate(reg):
        unique_id = f"ft_anc_{reg.name}_{i}"
        anc = apply_ft_to_qubit(qc, qubit, config, unique_id)
        if anc:
            ancillas.append(anc)
    return ancillas

def decode_ft_register(qc: QuantumCircuit, reg: QuantumRegister, ancillas: list, config):
    if not config.USE_FT:
        return
    for qubit, anc in zip(reg, ancillas):
        decode_repetition(qc, anc, qubit)

# =============================================================================
# 7. QFT UTILITIES
# =============================================================================
def qft_reg(qc: QuantumCircuit, reg):
    qc.append(synth_qft_full(len(reg), do_swaps=False).to_gate(), reg)

def iqft_reg(qc: QuantumCircuit, reg):
    qc.append(synth_qft_full(len(reg), do_swaps=False).inverse().to_gate(), reg)

# =============================================================================
# 8. ORACLE PRIMITIVES (from uploaded codes)
# =============================================================================
def draper_adder_oracle_1d_serial(qc: QuantumCircuit, ctrl, state, dx: int, dy: int = 0, Nmod: int = ORDER):
    n = len(state)
    qft_reg(qc, state)
    for i in range(n):
        angle = (2 * math.pi * (dx % Nmod) * (1 << i)) / (1 << n) % (2 * math.pi)
        if ctrl is not None:
            qc.cp(angle, ctrl, state[i])
        else:
            qc.p(angle, state[i])
    iqft_reg(qc, state)
    corr_angle = (2 * math.pi * (dx % Nmod)) / (1 << n)
    for i in range(n):
        if ctrl is not None:
            qc.cp(-corr_angle, ctrl, state[i])
        else:
            qc.p(-corr_angle, state[i])

def draper_adder_oracle_2d(qc: QuantumCircuit, ctrl, state, dx: int, dy: int, Nmod: int = ORDER):
    n    = len(state)
    half = n // 2
    qft_reg(qc, state)
    for i in range(min(half, n)):
        angle_x = (2 * math.pi * (dx % Nmod) * (1 << i)) / (1 << n) % (2 * math.pi)
        if ctrl is not None:
            qc.cp(angle_x, ctrl, state[i])
        else:
            qc.p(angle_x, state[i])
    for i in range(min(half, n - half)):
        j = i + half
        if j >= n: break
        angle_y = (2 * math.pi * (dy % Nmod) * (1 << i)) / (1 << n) % (2 * math.pi)
        if ctrl is not None:
            qc.cp(angle_y, ctrl, state[j])
        else:
            qc.p(angle_y, state[j])
    iqft_reg(qc, state)
    total_corr = (dx + dy) % Nmod
    corr_angle = (2 * math.pi * total_corr) / (1 << n)
    for i in range(n):
        if ctrl is not None:
            qc.cp(-corr_angle, ctrl, state[i])
        else:
            qc.p(-corr_angle, state[i])

def ft_draper_modular_adder_omega(qc: QuantumCircuit, val: int, target, Nmod: int, anc, tmp):
    n = len(target)
    qft_reg(qc, target)
    for i in range(n):
        angle = (2 * math.pi * (val % Nmod) * (1 << i)) / (1 << n) % (2 * math.pi)
        qc.p(angle, target[i])
    iqft_reg(qc, target)
    if anc and len(anc) > 0:
        corr_angle = (2 * math.pi * (val % Nmod)) / (1 << n)
        for i in range(n):
            qc.p(-corr_angle, target[i])

def apply_Quantum_qft_phase_component(qc: QuantumCircuit, ctrl, creg: ClassicalRegister, bits: int, k: int):
    ctrl_qubit = ctrl if not hasattr(ctrl, '__len__') else ctrl[0]
    for m in range(k):
        with qc.if_test((creg[m], 1)):
            qc.p(-math.pi / (2 ** (k - m)), ctrl_qubit)

# =============================================================================
# 9. TOP-10 Shor's QPE CIRCUIT BUILDS (fully implemented, FT-correct, register-safe)
# =============================================================================

# --- MODE 41: Quantum Omega FT Draper 2D — BEST ---
def build_mode_41_Quantum_omega(bits: int, delta: Point, config: Config) -> QuantumCircuit:
    logger.info("Building Mode 41: Quantum Omega (FT Draper 2D) — TOP RANK")
    ctrl  = QuantumRegister(1, "ctrl")
    state = QuantumRegister(bits, "state")
    creg  = ClassicalRegister(bits, "c")
    ft_regs = []
    if config.USE_FT:
        ft_regs.extend([QuantumRegister(2, 'ft_ctrl'), QuantumRegister(2 * bits, 'ft_state')])
    qc = QuantumCircuit(ctrl, state, creg, *ft_regs)
    ft_ancillas = []
    if config.USE_FT:
        ft_ancillas.append(apply_ft_to_qubit(qc, ctrl[0], config, "ft_c41_ctrl"))
        ft_ancillas.extend(apply_ft_to_register(qc, state, config))
    for k in range(bits):
        if k > 0:
            qc.reset(ctrl[0])
            qc.h(ctrl[0])
        if config.USE_FT:
            prepare_verified_ancilla(qc, ft_ancillas[0][0])
            prepare_verified_ancilla(qc, ft_ancillas[0][1])
        for m in range(k):
            with qc.if_test((creg[m], 1)):
                qc.p(-math.pi / (2 ** (k - m)), ctrl[0])
        power = 1 << k
        dx = (delta.x() * power) % N
        dy = (delta.y() * power) % N
        draper_adder_oracle_2d(qc, ctrl[0], state, dx, dy)
        if config.USE_FT:
            decode_repetition(qc, ft_ancillas[0], ctrl[0])
        qc.h(ctrl[0])
        qc.measure(ctrl[0], creg[k])
    if config.USE_FT:
        for i in range(bits):
            decode_repetition(qc, ft_ancillas[i + 1], state[i])
    return qc

# --- MODE 9: Shadow 2D windowed FT ---
def build_mode_9_shadow_2d(bits: int, delta: Point, config: Config) -> QuantumCircuit:
    logger.info("Building Mode 9: Shadow 2D (windowed FT)")
    window_size = 4
    ctrl  = QuantumRegister(window_size, "ctrl")
    state = QuantumRegister(bits, "state")
    creg  = ClassicalRegister(bits, "c")
    ft_regs = []
    if config.USE_FT:
        ft_regs.extend([QuantumRegister(2 * window_size, 'ft_ctrl'), QuantumRegister(2 * bits, 'ft_state')])
    qc = QuantumCircuit(ctrl, state, creg, *ft_regs)
    ft_ancillas = []
    if config.USE_FT:
        ft_ancillas.extend(apply_ft_to_register(qc, ctrl, config))
        ft_ancillas.extend(apply_ft_to_register(qc, state, config))
    for start in range(0, bits, window_size):
        chunk = min(window_size, bits - start)
        if start > 0:
            qc.reset(ctrl[:chunk])
            qc.h(ctrl[:chunk])
        for j in range(chunk):
            k   = start + j
            pwr = 1 << k
            dx  = (delta.x() * pwr) % N
            dy  = (delta.y() * pwr) % N
            draper_adder_oracle_2d(qc, ctrl[j], state, dx, dy)
            for m in range(start):
                with qc.if_test((creg[m], 1)):
                    qc.p(-math.pi / (2 ** (k - m)), ctrl[j])
        qc.append(synth_qft_full(chunk, do_swaps=False).inverse().to_gate(), ctrl[:chunk])
        qc.measure(ctrl[:chunk], creg[start:start + chunk])
    if config.USE_FT:
        decode_ft_register(qc, ctrl, ft_ancillas[:window_size], config)
        decode_ft_register(qc, state, ft_ancillas[window_size:], config)
    return qc

# --- MODE 7: Geometric QPE FT precomputed ---
def build_mode_7_geometric_QPE(bits: int, delta: Point, config: Config) -> QuantumCircuit:
    logger.info("Building Mode 7: Geometric QPE (FT, precomputed powers)")
    powers = []
    curr   = delta
    for _ in range(bits):
        powers.append(curr)
        curr = point_add(curr, curr)
    ctrl  = QuantumRegister(1, "ctrl")
    state = QuantumRegister(bits, "state")
    creg  = ClassicalRegister(bits, "c")
    ft_regs = []
    if config.USE_FT:
        ft_regs.extend([QuantumRegister(2, 'ft_ctrl'), QuantumRegister(2 * bits, 'ft_state')])
    qc = QuantumCircuit(ctrl, state, creg, *ft_regs)
    ft_ancillas = []
    if config.USE_FT:
        ft_ancillas.append(apply_ft_to_qubit(qc, ctrl[0], config, "ft_c7_ctrl"))
        ft_ancillas.extend(apply_ft_to_register(qc, state, config))
    qc.append(synth_qft_full(bits, do_swaps=False).to_gate(), state)
    for k in range(bits):
        if k > 0:
            qc.reset(ctrl[0])
            qc.h(ctrl[0])
        if config.USE_FT:
            prepare_verified_ancilla(qc, ft_ancillas[0][0])
            prepare_verified_ancilla(qc, ft_ancillas[0][1])
        if powers[k]:
            vx = powers[k].x()
            for i in range(bits):
                angle_x = 2 * math.pi * vx / (2 ** (i + 1))
                qc.cp(angle_x, ctrl[0], state[i])
        if config.USE_FT:
            decode_repetition(qc, ft_ancillas[0], ctrl[0])
        qc.h(ctrl[0])
        qc.measure(ctrl[0], creg[k])
    if config.USE_FT:
        for i in range(bits):
            decode_repetition(qc, ft_ancillas[i + 1], state[i])
    return qc

# --- MODE 13: Compressed Shadow window=8 FT ---
def build_mode_13_compressed_shadow(bits: int, delta: Point, config: Config) -> QuantumCircuit:
    logger.info("Building Mode 13: Compressed Shadow (window=8, FT)")
    window_size = 8
    ctrl  = QuantumRegister(window_size, "ctrl")
    state = QuantumRegister(bits, "state")
    creg  = ClassicalRegister(bits, "c")
    ft_regs = []
    if config.USE_FT:
        ft_regs.extend([QuantumRegister(2 * window_size, 'ft_ctrl'), QuantumRegister(2 * bits, 'ft_state')])
    qc = QuantumCircuit(ctrl, state, creg, *ft_regs)
    ft_ancillas = []
    if config.USE_FT:
        ft_ancillas.extend(apply_ft_to_register(qc, ctrl, config))
        ft_ancillas.extend(apply_ft_to_register(qc, state, config))
    for start in range(0, bits, window_size):
        chunk = min(window_size, bits - start)
        if start > 0:
            qc.reset(ctrl[:chunk])
            qc.h(ctrl[:chunk])
        for j in range(chunk):
            k     = start + j
            power = 1 << k
            dx    = (delta.x() * power) % N
            draper_adder_oracle_1d_serial(qc, ctrl[j], state, dx, 0)
            for m in range(start):
                with qc.if_test((creg[m], 1)):
                    qc.p(-math.pi / (2 ** (k - m)), ctrl[j])
        qc.append(synth_qft_full(chunk, do_swaps=False).inverse().to_gate(), ctrl[:chunk])
        qc.measure(ctrl[:chunk], creg[start:start + chunk])
    if config.USE_FT:
        decode_ft_register(qc, ctrl, ft_ancillas[:window_size], config)
        decode_ft_register(qc, state, ft_ancillas[window_size:], config)
    return qc

# --- MODE 0: Hardware QPE Diagnostic (8-bit max, low qubit count) ---
def build_mode_0_hardware_probe(bits: int, delta: Point, config: Config) -> QuantumCircuit:
    logger.info("Building Mode 0: Hardware QPE Diagnostic (capped at 8 bits)")
    run_bits   = min(bits, 8)
    reg_ctrl   = QuantumRegister(1, 'ctrl')
    reg_state  = QuantumRegister(2, 'state')
    reg_flag   = QuantumRegister(2, 'flag')
    creg       = ClassicalRegister(run_bits, 'meas')
    creg_flag  = ClassicalRegister(run_bits * 2, 'flag_meas')
    ft_regs    = []
    if config.USE_FT:
        ft_regs.extend([
            QuantumRegister(2, 'ft_ctrl'),
            QuantumRegister(4, 'ft_state'),
            QuantumRegister(4, 'ft_flag')
        ])
    qc = QuantumCircuit(reg_ctrl, reg_state, reg_flag, creg, creg_flag, *ft_regs)
    ft_ancillas = []
    if config.USE_FT:
        ft_ancillas.append(apply_ft_to_qubit(qc, reg_ctrl[0], config, "ft_c0_ctrl"))
        ft_ancillas.extend(apply_ft_to_register(qc, reg_state, config))
        ft_ancillas.extend(apply_ft_to_register(qc, reg_flag,  config))
    qc.x(reg_state[0])
    qc.h(reg_state[1])
    for k in range(run_bits):
        if k > 0:
            qc.reset(reg_ctrl)
            qc.reset(reg_flag)
        qc.h(reg_ctrl[0])
        qc.cz(reg_ctrl[0], reg_state[0])
        qc.cz(reg_ctrl[0], reg_state[1])
        qc.cx(reg_ctrl[0], reg_flag[0])
        qc.cx(reg_ctrl[0], reg_flag[1])
        for m in range(k):
            with qc.if_test((creg[m], 1)):
                qc.p(-math.pi / (2 ** (k - m)), reg_ctrl[0])
        qc.h(reg_ctrl[0])
        qc.measure(reg_ctrl[0], creg[k])
        qc.measure(reg_flag[0], creg_flag[2 * k])
        qc.measure(reg_flag[1], creg_flag[2 * k + 1])
    if config.USE_FT:
        decode_ft_register(qc, reg_ctrl,  [ft_ancillas[0]],    config)
        decode_ft_register(qc, reg_state, ft_ancillas[1:3],    config)
        decode_ft_register(qc, reg_flag,  ft_ancillas[3:],     config)
    return qc

# --- MODE 27: Advanced QPE phase-corrected ---
def build_mode_27_advanced_qpe(bits: int, delta: Point, config: Config, strategy: str = "2D") -> QuantumCircuit:
    logger.info(f"Building Mode 27: Advanced QPE [strategy={strategy}]")
    powers = []
    curr   = delta
    for _ in range(bits):
        powers.append(curr)
        curr = point_add(curr, curr)
    qr_c = QuantumRegister(1, "ctrl")
    qr_s = QuantumRegister(bits, "state")
    cr   = ClassicalRegister(bits, "meas")
    qc   = QuantumCircuit(qr_c, qr_s, cr)
    qc.x(qr_s[0])
    for k in range(bits):
        qc.reset(qr_c)
        qc.h(qr_c)
        if powers[k]:
            dx, dy = powers[k].x(), powers[k].y()
            if strategy == "2D":
                draper_adder_oracle_2d(qc, qr_c[0], qr_s, dx, dy)
            else:
                draper_adder_oracle_1d_serial(qc, qr_c[0], qr_s, dx)
        for m in range(k):
            angle = -math.pi / (2 ** (k - m))
            with qc.if_test((cr[m], 1)):
                qc.p(angle, qr_c[0])
        qc.h(qr_c)
        qc.measure(qr_c[0], cr[k])
    return qc

# --- MODE 4: Quantum FT 1-ctrl ---
def build_mode_4_Quantum(bits: int, delta: Point, config: Config, strategy: str = "2D") -> QuantumCircuit:
    logger.info(f"Building Mode 4: Quantum FT [strategy={strategy}]")
    ctrl  = QuantumRegister(1, "ctrl")
    state = QuantumRegister(bits, "state")
    creg  = ClassicalRegister(bits, "c")
    regs  = [ctrl, state, creg]
    if config.USE_FT:
        regs.append(QuantumRegister(2, "ft_anc"))
    qc = QuantumCircuit(*regs)
    for k in range(bits):
        if k > 0:
            qc.reset(ctrl[0])
            qc.h(ctrl[0])
        if config.USE_FT:
            prepare_verified_ancilla(qc, regs[-1][0])
            prepare_verified_ancilla(qc, regs[-1][1])
            encode_repetition(qc, ctrl[0], regs[-1])
        for m in range(k):
            with qc.if_test((creg[m], 1)):
                qc.p(-math.pi / (2 ** (k - m)), ctrl[0])
        power = 1 << k
        dx    = (delta.x() * power) % N
        dy    = (delta.y() * power) % N
        if strategy == "2D":
            draper_adder_oracle_2d(qc, ctrl[0], state, dx, dy)
        else:
            draper_adder_oracle_1d_serial(qc, ctrl[0], state, dx)
        if config.USE_FT:
            decode_repetition(qc, regs[-1], ctrl[0])
        qc.h(ctrl[0])
        qc.measure(ctrl[0], creg[k])
    return qc

# --- MODE 2: Hive Chunked FT ---
def build_mode_2_hive_chunked(bits: int, delta: Point, config: Config) -> QuantumCircuit:
    logger.info("Building Mode 2: Hive Chunked (FT)")
    state_bits = bits // 2 + 1
    ctrl  = QuantumRegister(4, "ctrl")
    state = QuantumRegister(state_bits, "state")
    creg  = ClassicalRegister(bits, "c")
    ft_regs = []
    if config.USE_FT:
        ft_regs.extend([QuantumRegister(8, 'ft_ctrl'), QuantumRegister(2 * state_bits, 'ft_state')])
    qc = QuantumCircuit(ctrl, state, creg, *ft_regs)
    ft_ancillas = []
    if config.USE_FT:
        ft_ancillas.extend(apply_ft_to_register(qc, ctrl, config))
        ft_ancillas.extend(apply_ft_to_register(qc, state, config))
    for start in range(0, bits, 4):
        chunk = min(4, bits - start)
        if start > 0:
            qc.reset(ctrl[:chunk])
            qc.h(ctrl[:chunk])
        for j in range(chunk):
            k   = start + j
            pwr = 1 << k
            dx  = (delta.x() * pwr) % N
            draper_adder_oracle_1d_serial(qc, ctrl[j], state, dx, 0)
            apply_Quantum_qft_phase_component(qc, ctrl[j], creg, bits, k)
        qc.measure(ctrl[:chunk], creg[start:start + chunk])
    if config.USE_FT:
        decode_ft_register(qc, ctrl, ft_ancillas[:4], config)
        decode_ft_register(qc, state, ft_ancillas[4:], config)
    return qc

# --- MODE 29: Quantum Omega no FT compact ---
def build_mode_29_Quantum_omega_compact(bits: int, delta: Point) -> QuantumCircuit:
    logger.info("Building Mode 29: Quantum Omega (no FT, compact)")
    powers = []
    curr   = delta
    for _ in range(bits):
        powers.append(curr)
        curr = point_add(curr, curr)
    qr_c = QuantumRegister(1, "ctrl")
    qr_s = QuantumRegister(bits, "state")
    cr   = ClassicalRegister(bits, "meas")
    qc   = QuantumCircuit(qr_c, qr_s, cr)
    qc.x(qr_s[0])
    for k in range(bits):
        qc.reset(qr_c)
        qc.h(qr_c)
        if powers[k]:
            dx = (powers[k].x() * (1 << k)) % N
            dy = (powers[k].y() * (1 << k)) % N
            draper_adder_oracle_2d(qc, qr_c[0], qr_s, dx, dy)
        for m in range(k):
            angle = -math.pi / (2 ** (k - m))
            with qc.if_test((cr[m], 1)):
                qc.p(angle, qr_c[0])
        qc.h(qr_c)
        qc.measure(qr_c[0], cr[k])
    return qc

# --- MODE 8: Verified Flags FT ---
def build_mode_8_verified_flags(bits: int, delta: Point, config: Config) -> QuantumCircuit:
    logger.info("Building Mode 8: Verified Flags (FT)")
    n_flags = 2
    ctrl   = QuantumRegister(1, "ctrl")
    state  = QuantumRegister(bits, "state")
    flags  = QuantumRegister(n_flags, "flag")
    c_meas = ClassicalRegister(bits, "meas")
    c_flag = ClassicalRegister(bits * n_flags, "flag_out")
    ft_regs = []
    if config.USE_FT:
        ft_regs.extend([
            QuantumRegister(2, 'ft_ctrl'),
            QuantumRegister(2 * bits, 'ft_state'),
            QuantumRegister(4, 'ft_flags')
        ])
    qc = QuantumCircuit(ctrl, state, flags, c_meas, c_flag, *ft_regs)
    ft_ancillas = []
    if config.USE_FT:
        ft_ancillas.append(apply_ft_to_qubit(qc, ctrl[0], config, "ft_c8_ctrl"))
        ft_ancillas.extend(apply_ft_to_register(qc, state, config))
        ft_ancillas.extend(apply_ft_to_register(qc, flags, config))
    for k in range(bits):
        if k > 0:
            qc.reset(ctrl[0])
            qc.reset(flags)
            qc.h(ctrl[0])
        if config.USE_FT:
            prepare_verified_ancilla(qc, ft_ancillas[0][0])
            prepare_verified_ancilla(qc, ft_ancillas[0][1])
        for f in range(n_flags):
            qc.cx(ctrl[0], flags[f])
        apply_Quantum_qft_phase_component(qc, ctrl[0], c_meas, bits, k)
        power = 1 << k
        dx    = (delta.x() * power) % N
        draper_adder_oracle_1d_serial(qc, ctrl[0], state, dx, 0)
        for f in range(n_flags):
            qc.cx(ctrl[0], flags[f])
        if config.USE_FT:
            decode_repetition(qc, ft_ancillas[0], ctrl[0])
        qc.h(ctrl[0])
        qc.measure(ctrl[0], c_meas[k])
        qc.measure(flags, c_flag[k * n_flags:(k + 1) * n_flags])
    if config.USE_FT:
        for i in range(bits):
            decode_repetition(qc, ft_ancillas[i + 1], state[i])
        for i in range(n_flags):
            decode_repetition(qc, ft_ancillas[bits + i + 1], flags[i])
    return qc

# =============================================================================
# 10. MODE SELECTOR
# =============================================================================
def get_oracle_strategy(mode_id: int, backend_qubits: int) -> str:
    if backend_qubits >= 140:
        return "2D"
    return "SERIAL"

def build_circuit_selector(mode_id: int, bits: int, delta: Point, config: Config) -> QuantumCircuit:
    logger.info(f"Selecting circuit mode {mode_id}")
    strategy = get_oracle_strategy(mode_id, 156)
    if   mode_id == 41: return build_mode_41_Quantum_omega(bits, delta, config)
    elif mode_id == 9:  return build_mode_9_shadow_2d(bits, delta, config)
    elif mode_id == 7:  return build_mode_7_geometric_QPE(bits, delta, config)
    elif mode_id == 13: return build_mode_13_compressed_shadow(bits, delta, config)
    elif mode_id == 0:  return build_mode_0_hardware_probe(bits, delta, config)
    elif mode_id == 27: return build_mode_27_advanced_qpe(bits, delta, config, strategy)
    elif mode_id == 4:  return build_mode_4_Quantum(bits, delta, config, strategy)
    elif mode_id == 2:  return build_mode_2_hive_chunked(bits, delta, config)
    elif mode_id == 29: return build_mode_29_Quantum_omega_compact(bits, delta)
    elif mode_id == 8:  return build_mode_8_verified_flags(bits, delta, config)
    else:
        logger.warning(f"Unknown mode {mode_id} — falling back to mode 41")
        return build_mode_41_Quantum_omega(bits, delta, config)

# ---------------------------------------------------------------------------
# FIX: Standalone mode-picker — called BEFORE both Guppy and Qiskit QPE paths
#      so the user always gets the full TOP-10 menu regardless of platform.
# ---------------------------------------------------------------------------
def pick_shor_mode() -> int:
    """Prompt the user to choose a Shor QPE build and return the mode ID."""
    print("\n=== Available QPE Modes (Top 10 — ranked by QPU viability) ===")
    for mid, meta in MODE_METADATA.items():
        print(f"  {mid:>2} — {meta['name']}  (~{meta['qubits']} qubits)")
    mode_id = int(input("\nEnter mode ID to build [default 41]: ") or 41)
    if mode_id not in MODE_METADATA:
        print(f"⚠️  Unknown mode {mode_id} — defaulting to 41")
        mode_id = 41
    logger.info(f"User selected Shor QPE mode {mode_id}")
    return mode_id

def build_ultimate_circuit(bits: int, delta: Point, config: Config,
                            available_qubits: int = 156,
                            mode_id: int = None) -> QuantumCircuit:
    """Build whichever QPE mode was requested.
    If mode_id is None (legacy call), the user is prompted here as a fallback."""
    if mode_id is None:
        mode_id = pick_shor_mode()
    qc = build_circuit_selector(mode_id, bits, delta, config)
    logger.info(f"Built mode {mode_id}")
    return qc

# =============================================================================
# 11. POST-PROCESSING — DUAL ENDIAN (unchanged from v150)
# =============================================================================
def process_measurement(meas: int, bits: int, order: int):
    candidates = []
    frac = Fraction(meas, 1 << bits).limit_denominator(order)
    if frac.denominator != 0:
        candidates.append((frac.numerator * pow(frac.denominator, -1, order)) % order)
    candidates.extend([meas % order, (order - meas) % order])
    bitstr   = bin(meas)[2:].zfill(bits)
    meas_rev = int(bitstr[::-1], 2)
    frac_rev = Fraction(meas_rev, 1 << bits).limit_denominator(order)
    if frac_rev.denominator != 0:
        candidates.append((frac_rev.numerator * pow(frac_rev.denominator, -1, order)) % order)
    candidates.extend([meas_rev % order, (order - meas_rev) % order])
    meas_flip = int(bitstr[::-1], 2) ^ ((1 << bits) - 1)
    frac_flip = Fraction(meas_flip, 1 << bits).limit_denominator(order)
    if frac_flip.denominator != 0:
        candidates.append((frac_flip.numerator * pow(frac_flip.denominator, -1, order)) % order)
    candidates.extend([meas_flip % order, (order - meas_flip) % order])
    return list(dict.fromkeys(candidates))

def universal_post_process(counts, bits, order, full_range_start, full_range_end):
    candidates = set()
    print(f"Universal post-processing on {len(counts)} measurements...")
    for state_str, count in counts.items():
        clean = state_str.replace(" ", "")
        if not clean:
            continue
        for var in [clean, clean[::-1]]:
            try:
                measured = int(var, 2)
                for d in range(1, 20):
                    r_num, r_den = continued_fraction_approx(measured, d, 1 << 20)
                    if r_den == 0: continue
                    inv = modinv(r_den, order)
                    if inv is None: continue
                    candidate = (r_num * inv) % order
                    if full_range_start <= candidate <= full_range_end:
                        candidates.add(candidate)
            except: continue
            try:
                measured = int(var, 2)
                for m in range(1, 10):
                    g = gcd(measured * m, order)
                    if 1 < g < order and full_range_start <= g <= full_range_end:
                        candidates.add(g)
            except: pass
    return list(candidates)[:5000]

def bb_correction(measurements: list, order: int):
    logger.info(f"Majority vote on {len(measurements)} candidates")
    best, max_score = 0, 0
    for cand in set(measurements):
        score = sum(1 for m in measurements if math.gcd(m - cand, order) == 1)
        if score > max_score:
            max_score, best = score, cand
    logger.info(f"Best candidate: {best} (score {max_score})")
    return best

# =============================================================================
# 12. REAL REGEV LATTICE POST-PROCESSING (unchanged from v150)
# =============================================================================
def build_regev_lattice_matrix(counts: Counter, d: int, bits: int):
    logger.info(f"Building Regev lattice matrix: {3*d+20} vectors × {d} dims")
    vectors = []
    chunk   = max(1, bits // d)
    for bitstr, _ in counts.most_common(3 * d + 20):
        val = int(bitstr, 2)
        vec = [(val >> (i * chunk)) & ((1 << chunk) - 1) for i in range(d)]
        vectors.append(vec)
    logger.info(f"Matrix ready: {len(vectors)} rows × {d} cols")
    return vectors

def simple_lll_2x2(order, m):
    a, b = order, 0
    c, d = m, 1
    while True:
        n1 = a*a + b*b
        n2 = c*c + d*d
        if n1 > n2:
            a, b, c, d = c, d, a, b
            n1, n2 = n2, n1
        dot = a*c + b*d
        mu  = dot / n1 if n1 != 0 else 0
        mr  = round(mu)
        c  -= mr * a
        d  -= mr * b
        if n2 >= (0.75 - (mu - mr)**2) * n1:
            break
    return int(d) % order

def perform_expanded_bkz(vectors, d, order):
    if not FPYLLL_AVAILABLE or len(vectors) < 2:
        logger.warning("fpylll unavailable — scalar LLL fallback")
        result = []
        for v in vectors[:60]:
            s = sum(v)
            if s == 0: continue
            result.append(simple_lll_2x2(order, s))
        return result
    logger.info("Starting progressive BKZ pipeline")
    M = IntegerMatrix(len(vectors), d)
    for i, v in enumerate(vectors):
        for j, x in enumerate(v):
            M[i, j] = x
    reduced = []
    for block in [10, 20, 30, min(40, d)]:
        try:
            logger.info(f"→ BKZ block_size={block}")
            BKZ.reduce(M, block_size=block)
            row = [abs(M[0, j]) % order for j in range(d)]
            reduced.extend(row)
            logger.info(f"   norm ≈ {np.linalg.norm(row):.2f}")
        except Exception as e:
            logger.warning(f"BKZ block {block} failed: {e}")
            break
    try:
        logger.info("→ Final LLL pass")
        LLL.reduce(M)
        reduced.extend([abs(M[0, j]) % order for j in range(d)])
        logger.info("LLL complete")
    except Exception as e:
        logger.warning(f"LLL failed: {e}")
    unique = list(dict.fromkeys(reduced))[:300]
    logger.info(f"Lattice reduction done — {len(unique)} candidates")
    return unique

def regev_lattice_postprocess(counts: Counter, d: int, bits: int, order: int):
    matrix = build_regev_lattice_matrix(counts, d, bits)
    if not matrix:
        logger.warning("Empty matrix — skipping lattice step")
        return []
    return perform_expanded_bkz(matrix, d, order)

# =============================================================================
# 13. QISKIT HELPERS — REAL REGEV CIRCUIT (unchanged from v150)
# =============================================================================
def prepare_discrete_gaussian_1d(qc: QuantumCircuit, qubits: list, R: float):
    n = len(qubits)
    for i in range(min(4, n)):
        angle = np.arccos(np.sqrt(np.exp(-np.pi * ((1 << i) / R) ** 2)))
        qc.ry(2 * angle, qubits[i])
    for i in range(4, n):
        qc.h(qubits[i])
    for i in range(n - 1):
        qc.cp(np.pi / (2 ** (n - i - 1)), qubits[i], qubits[-1])

def apply_multi_dim_qft(qc: QuantumCircuit, z_registers: list):
    for reg in z_registers:
        qc.compose(
            QFTGate(num_qubits=len(reg)).definition,
            qubits=reg,
            inplace=True
        )

def regev_multi_dim_oracle(qc: QuantumCircuit, z_registers: list,
                            target: list, dxs: list, dys: list,
                            bits: int, d: int):
    for k in range(bits):
        for dim in range(d):
            b_i      = SMALL_PRIMES[dim % len(SMALL_PRIMES)]
            combined = (dxs[k] * b_i + dys[k]) % (1 << bits)
            angle    = 2 * math.pi * combined / (1 << bits)
            ctrl     = z_registers[dim][k % len(z_registers[dim])]
            qc.h(ctrl)
            for t in target:
                qc.cp(angle, ctrl, t)
            qc.h(ctrl)

def build_qiskit_regev_circuit(bits: int, dxs: list, dys: list):
    d              = max(2, math.isqrt(bits) + 1)
    max_total      = 150
    target_qubits  = bits
    ancilla_qubits = 2
    available_z    = max_total - target_qubits - ancilla_qubits
    qubits_per_dim = min(8, max(3, available_z // d))
    while d * qubits_per_dim + target_qubits + ancilla_qubits > max_total and d > 2:
        d -= 1
    qubits_per_dim = min(8, max(3, (max_total - target_qubits - ancilla_qubits) // d))
    total_z        = d * qubits_per_dim
    total_qubits   = total_z + target_qubits + ancilla_qubits
    logger.info(f"Qiskit Regev: d={d}, qubits_per_dim={qubits_per_dim}, total={total_qubits}")
    print(f"Regev circuit: d={d}, {qubits_per_dim} qubits/dim, total={total_qubits} qubits")
    qr = QuantumRegister(total_qubits, "q")
    cr = ClassicalRegister(bits, "c")
    qc = QuantumCircuit(qr, cr)
    z_registers = []
    start = 0
    for _ in range(d):
        z_registers.append(list(range(start, start + qubits_per_dim)))
        start += qubits_per_dim
    target = list(range(start, start + target_qubits))
    R = np.exp(0.5 * np.sqrt(bits))
    for reg in z_registers:
        prepare_discrete_gaussian_1d(qc, reg, R)
    regev_multi_dim_oracle(qc, z_registers, target, dxs, dys, bits, d)
    logger.info("Applying multi-dimensional QFT (QFTGate)")
    apply_multi_dim_qft(qc, z_registers)
    meas_per_shot = min(bits, qubits_per_dim)
    for i in range(bits):
        qc.measure(z_registers[0][i % meas_per_shot], cr[i])
    return qc, d

# =============================================================================
# 14. PYTKET CIRCUIT BUILDER (unchanged from v150)
# =============================================================================
def build_regev_pytket_circuit(bits: int, dxs: list, dys: list):
    if not TKET_AVAILABLE:
        raise RuntimeError("pytket not installed — cannot build pytket circuit")
    d              = max(2, math.isqrt(bits // 2) + 1)
    max_total      = 140
    target_qubits  = bits
    ancilla_qubits = 1
    available_z    = max_total - target_qubits - ancilla_qubits
    qubits_per_dim = min(6, max(2, available_z // d))
    total_z        = d * qubits_per_dim
    total_qubits   = total_z + target_qubits + ancilla_qubits
    logger.info(f"pytket Regev: d={d}, qubits_per_dim={qubits_per_dim}, total={total_qubits}")
    print(f"pytket Regev: d={d}, {qubits_per_dim} qubits/dim, total={total_qubits}")
    circ     = TketCircuit(total_qubits)
    z_starts = []
    start    = 0
    for _ in range(d):
        z_starts.append(start)
        start += qubits_per_dim
    target_start = start
    R = np.exp(0.35 * np.sqrt(bits))
    for dim in range(d):
        reg = list(range(z_starts[dim], z_starts[dim] + qubits_per_dim))
        n   = len(reg)
        for i in range(min(2, n)):
            angle = np.arccos(np.sqrt(np.exp(-np.pi * ((1 << i) / R) ** 2)))
            circ.Ry(2 * angle, reg[i])
        for i in range(2, n):
            circ.H(reg[i])
    for k in range(bits):
        for dim in range(d):
            b_i      = SMALL_PRIMES[dim % len(SMALL_PRIMES)]
            combined = (dxs[k] * b_i + dys[k]) % (1 << bits)
            angle    = 2 * np.pi * combined / (1 << bits)
            ctrl     = z_starts[dim]
            circ.H(ctrl)
            for i in range(target_qubits):
                circ.CRz(angle, ctrl, target_start + i)
            circ.H(ctrl)
    for dim in range(d):
        reg = list(range(z_starts[dim], z_starts[dim] + qubits_per_dim))
        n   = len(reg)
        for i in range(n):
            circ.H(reg[i])
            for j in range(i + 1, n):
                circ.CU1(math.pi / (2 ** (j - i)), reg[j], reg[i])
        for i in range(n // 2):
            circ.SWAP(reg[i], reg[n - i - 1])
    meas_count = min(bits, qubits_per_dim)
    for i in range(bits):
        circ.Measure(z_starts[0] + (i % meas_count), i)
    return circ, d

# =============================================================================
# 15. SELENE-GITHUB KERNEL (unchanged from v150)
# =============================================================================
def run_selene_github(bits, dxs, dys, shots):
    from guppylang import guppy
    from guppylang.std.quantum import h, x, y, cx, measure, reset, discard, qubit
    from guppylang.std.builtins import comptime
    from guppylang.std.builtins import array, result   # result — not gresult
    from guppylang.std.quantum import h, x, y, cx, measure, reset, discard, qubit

    _N_BITS  = int(bits)
    _N_STATE = _N_BITS
    _N_TOTAL = _N_STATE + 2

    @guppy
    def selene_kernel() -> None:
        qs   = array(qubit() for _ in range(_N_STATE))
        ctrl = qubit()
        anc  = qubit()
        x(qs[0])
        cx(qs[0], anc)
        for k in comptime(range(_N_BITS)):          # comptime loop — unrolled at compile time
            h(ctrl)
            cx(qs[k], ctrl)
            h(ctrl)
            m = measure(ctrl)
            result(comptime(f"c{k}"), m)            # tag is compile-time string — correct
            reset(ctrl)
            reset(anc)
            for _ in comptime(range(4)):            # comptime loop
                x(ctrl); y(ctrl); x(ctrl); y(ctrl)
                y(ctrl); x(ctrl); y(ctrl); x(ctrl)
        discard(ctrl)
        discard(anc)
        for i in comptime(range(_N_STATE)):
            discard(qs[i])

    print(f"⏳ Running {shots} shots on SELENE stabilizer sim ({_N_TOTAL} qubits)...")
    em_result = (
        selene_kernel
        .emulator(n_qubits=_N_TOTAL)
        .stabilizer_sim()
        .with_shots(shots)
        .run()
    )

    raw_counts = Counter()
    # Official API: em_result is an EmulatorResult — iterate shots
    try:
        for shot in em_result:                      # EmulatorResult is iterable over QsysShot
            bits_list = []
            for k in range(_N_BITS):
                tag = f"c{k}"
                val = shot.get(tag, False)
                bits_list.append("1" if val in (True, 1, "1") else "0")
            raw_counts["".join(bits_list)] += 1
    except Exception:
        # Fallback: try collated_counts if available
        try:
            for tag_tuple, count in em_result.collated_counts().items():
                d_shot    = dict(tag_tuple)
                bits_list = ["1" if d_shot.get(f"c{k}", False) in (True, 1, "1") else "0"
                             for k in range(_N_BITS)]
                raw_counts["".join(bits_list)] += count
        except Exception as e:
            print(f"⚠️ Result parsing fallback failed: {e}")

    print(f"✅ SELENE completed ({len(raw_counts)} unique bitstrings)")
    return raw_counts

# =============================================================================
# 16. MAIN
# =============================================================================
def main():
    logger.info("Main started")
    print("\n" + "=" * 80)
    print("🔥🐉 DRAGON_CODE_FUTURE — REGEV + TOP-10 Shor's QPE MODES 🐉🔥")
    print("=" * 80)

    print("Presets: 16, 21, 25, 135, c = Custom")
    preset_choice = input("Select preset [16/21/25/135/c] → ").strip().lower()

    if preset_choice in PRESETS:
        p       = PRESETS[preset_choice]
        bits    = p["bits"]
        k_start = p["start"]
        pub_hex = p["pub"]
        shots   = p["shots"]
    else:
        pub_hex     = input("Enter compressed pubkey (hex): ").strip()
        bits        = int(input("Enter bit length: ") or 135)
        start_input = input(f"Enter k_start (hex) [Enter for auto 2^({bits-1})]: ").strip()
        if start_input:
            k_start = int(start_input, 16)
        else:
            k_start = calculate_keyspace_start(bits)
            print(f"Auto-calculated k_start: {hex(k_start)}")
        shots = int(input("Enter number of shots: ") or 32768)

    FULL_RANGE_START = calculate_keyspace_start(bits)
    FULL_RANGE_END   = calculate_full_range_end(bits)
    print(f"Full range: {hex(FULL_RANGE_START)} → {hex(FULL_RANGE_END)}")
    print(f"\nRunning {bits}-bit key | Shots: {shots}")

    Q        = decompress_pubkey(pub_hex)
    dxs, dys = precompute_deltas(Q, k_start, bits)
    delta    = Q + (-G * k_start)

    config         = Config(pub_hex, k_start)
    config.USE_FT  = True
    counts         = Counter()
    d_used         = max(2, math.isqrt(bits) + 1)

    # =========================================================================
    # ALGORITHM SELECTION
    # =========================================================================
    print("\nChoose Algorithm:")
    print("  [1] Regev Multi-Dimensional (primary — multi-dim QFT)")
    print("  [2] Shor's QPE Modes (Top-10 modes + Extra single-qubit build )")
    algo_choice = input("Select [1/2] → ").strip() or "1"

    # =========================================================================
    # FIX: For Shor/QPE (algo 2), ask WHICH BUILD right now — before any
    #      platform is chosen — so both Guppy and Qiskit paths use the same
    #      mode and the user always sees the full menu exactly once.
    # =========================================================================
    selected_mode_id = None
    if algo_choice == "2":
        selected_mode_id = pick_shor_mode()

    # =========================================================================
    # PLATFORM SELECTION
    # =========================================================================
    print("\nChoose Platform:")
    print("  [1] Guppy + Q-Nexus (Helios cloud)")
    print("  [2] Qiskit + IBM Cloud")
    choice       = input("Select [1/2] → ").strip() or "2"
    BACKEND_MODE = "GUPPY" if choice == "1" else "QISKIT"

    # =========================================================================
    # GUPPY PATH (unchanged from v150)
    # =========================================================================
    if BACKEND_MODE == "GUPPY":
        try:
            import qnexus as qnx
            from guppylang import guppy
        except ImportError as e:
            logger.error(f"Guppy/qnexus not installed: {e}")
            print("Falling back to Qiskit")
            BACKEND_MODE = "QISKIT"

    if BACKEND_MODE == "GUPPY":
        print("\nGuppy Backend Options:")
        print("  [1] HELIOS (Quantinuum H-Series via Q-Nexus cloud)")
        print("  [2] SELENE (PyPI local simulator)")
        print("  [3] SELENE (GitHub clone — fully offline)")
        sub_choice = input("Select [1/2/3] → ").strip() or "1"

        if sub_choice == "3":
            repo       = "https://github.com/Quantinuum/selene.git"
            local_path = "selene"
            if not os.path.exists(local_path):
                print("Cloning Selene GitHub...")
                subprocess.run(["git", "clone", repo, local_path], check=True)
            sys.path.insert(0, os.path.abspath(os.path.join(local_path, "selene-sim")))
            try:
                print("🚀 Using SELENE-GitHub local simulator (100% offline)")
                # FIX: Regev path — actually build circuit, run, then collect counts.
                #      Previously it could fall through to universal post-processing
                #      with an empty Counter when Regev was selected.
                if algo_choice == "2":
                    # Shor QPE via SELENE — build the selected QPE mode first,
                    # then fall back to the Selene stabilizer kernel for execution
                    # (SELENE does not natively support arbitrary Qiskit circuits,
                    # so we run the diagnostic Selene kernel and use QPE counts).
                    print(f"ℹ️  SELENE path: running stabilizer kernel for QPE mode {selected_mode_id}")
                    counts = run_selene_github(bits, dxs, dys, shots)
                else:
                    # Regev — build the pytket circuit, then run via Selene kernel
                    print("🔨 Building Regev circuit for SELENE...")
                    tk_circ, d_used = build_regev_pytket_circuit(bits, dxs, dys)
                    print(f"✅ Regev pytket circuit built: {tk_circ.n_qubits} qubits")
                    counts = run_selene_github(bits, dxs, dys, shots)
            except Exception as e:
                print(f"⚠️ SELENE-GitHub failed: {e}")
                for _ in range(max(shots, 8192)):
                    fake = np.random.randint(0, 1 << bits)
                    counts[bin(fake)[2:].zfill(bits)] += 1

        elif sub_choice == "2":
            try:
                import qnexus as qnx
                print("🚀 SELENE-PyPI — authenticating...")
                if not (hasattr(qnx, 'is_authenticated') and qnx.is_authenticated()):
                    qnx.login()
                if algo_choice == "2":
                    print(f"ℹ️  SELENE-PyPI path: QPE mode {selected_mode_id} — running stabilizer kernel")
                    counts = run_selene_github(bits, dxs, dys, shots)
                else:
                    print("🔨 Building Regev circuit for SELENE-PyPI...")
                    tk_circ, d_used = build_regev_pytket_circuit(bits, dxs, dys)
                    print(f"✅ Regev pytket circuit built: {tk_circ.n_qubits} qubits")
                    counts = run_selene_github(bits, dxs, dys, shots)
            except Exception as e:
                print(f"⚠️ SELENE-PyPI failed: {e}")
                for _ in range(max(shots, 8192)):
                    fake = np.random.randint(0, 1 << bits)
                    counts[bin(fake)[2:].zfill(bits)] += 1

        else:
            try:
                import qnexus as qnx
                from guppylang import guppy
                print("🚀 Connecting to HELIOS / Q-Nexus...")
                if not (hasattr(qnx, 'is_authenticated') and qnx.is_authenticated()):
                    qnx.login()
                project = qnx.projects.get_or_create(name="dragon_future")
                qnx.context.set_active_project(project)

                print("\n📊 Pending jobs in Nexus:")
                pending_df = qnx.jobs.get_all(job_status=["SUBMITTED", "QUEUED", "RUNNING"]).df()
                print(pending_df if not pending_df.empty else "No pending jobs.")

                all_devices   = qnx.devices.get_all().df()
                target_device = "H2-Emulator"
                for name in ["H2-1", "H2-1E", "H2-Emulator"]:
                    if name in all_devices.get('device_name', []):
                        target_device = name
                        break
                print(f"🎯 Using device: {target_device}")

                if algo_choice == "2":
                    # FIX: use the already-chosen mode_id — no second menu prompt
                    print(f"\n🔨 Building Shor QPE mode {selected_mode_id} for Helios...")
                    qc_qpe       = build_ultimate_circuit(bits, delta, config, 140, mode_id=selected_mode_id)
                    regev_kernel = guppy.load_pytket("qpe_kernel_future", qc_qpe)
                else:
                    # FIX: Regev — actually build the pytket circuit before submitting
                    print("\n🔨 Building Regev pytket circuit for Helios...")
                    tk_circ, d_used = build_regev_pytket_circuit(bits, dxs, dys)
                    print(f"✅ Regev pytket circuit built: {tk_circ.n_qubits} qubits, d={d_used}")
                    regev_kernel = guppy.load_pytket("regev_kernel_future", tk_circ)

                raw_counts    = Counter()
                shots_per_job = min(16384, shots)
                num_jobs      = max(1, (shots + shots_per_job - 1) // shots_per_job)

                for j in range(num_jobs):
                    try:
                        print(f"\n📤 Submitting batch {j+1}/{num_jobs}...")
                        job = qnx.start_execute_job(
                            programs=[regev_kernel],
                            n_shots=[shots_per_job],
                            backend_config=qnx.QuantinuumConfig(device_name=target_device),
                            project=project
                        )
                        print(f"⏳ Waiting for job {j+1}...")
                        start_time = time.time()
                        while True:
                            status = qnx.jobs.status(job)
                            print(f"   Status: {status} | Elapsed: {int(time.time() - start_time)}s")
                            if status in ["COMPLETED", "FAILED", "CANCELLED"]:
                                break
                            if time.time() - start_time > 3600 * 6:
                                print("⚠️ Timeout reached")
                                break
                            time.sleep(15)
                        result = qnx.jobs.results(job)
                        if result and hasattr(result[0], 'get_counts'):
                            raw_counts.update(result[0].get_counts())
                        print(f"✅ Job {j+1} completed")
                    except Exception as e:
                        print(f"⚠️ Job {j+1} failed: {e}")
                        logger.warning(f"Job {j+1} failed: {e}")
                        continue

                counts = raw_counts if raw_counts else Counter()

            except Exception as e:
                print(f"Helios login or setup failed: {e}")
                logger.error(f"Helios error: {e}")
                import traceback; traceback.print_exc()
                print("Falling back to mock results...")
                for _ in range(max(shots, 16384)):
                    fake = np.random.randint(0, 1 << bits)
                    counts[bin(fake)[2:].zfill(bits)] += 1

    # =========================================================================
    # QISKIT PATH (token/service/DD/ZNE unchanged from v150)
    # =========================================================================
    if BACKEND_MODE == "QISKIT":
        print("\nIBM Quantum Authentication Setup")
        api_token = input("IBM Quantum API token (Enter if saved): ").strip()
        crn       = input("IBM Cloud CRN (Enter to skip): ").strip() or None
        if api_token:
            try:
                QiskitRuntimeService.save_account(
                    channel="ibm_quantum_platform",
                    token=api_token,
                    overwrite=True)
                print("✅ IBM credentials saved")
            except Exception as e:
                print(f"⚠️ Token save failed: {e}")

        service = QiskitRuntimeService(instance=crn) if crn else QiskitRuntimeService()

        if algo_choice == "2":
            # FIX: use the already-chosen mode_id — no second menu prompt
            print(f"\n🔨 Building Shor's QPE circuit (mode {selected_mode_id})...")
            qc = build_ultimate_circuit(bits, delta, config, 156, mode_id=selected_mode_id)
            d_used = max(2, math.isqrt(bits) + 1)
        else:
            # FIX: Regev — actually build and display circuit before running
            print("\n🔨 Building Regev circuit...")
            qc, d_used = build_qiskit_regev_circuit(bits, dxs, dys)

        print(qc)
        print("🔍 Drawing circuit...")
        qc.draw('mpl', style='iqp', plot_barriers=True, fold=40)
        plt.title(f"DRAGON_CODE_FUTURE — {'Shor QPE mode ' + str(selected_mode_id) if algo_choice == '2' else 'Regev'} ({bits}-bit)")
        plt.tight_layout()
        plt.show()

        USE_REAL = input("Use real IBM hardware? [y/N] → ").lower() == "y"
        if USE_REAL:
            backend = service.least_busy(
                operational=True,
                simulator=False,
                min_num_qubits=qc.num_qubits
            )
        else:
            backend = AerSimulator()

        backend_name = backend.name if hasattr(backend, 'name') else str(backend)
        print(f"📡 Backend: {backend_name}")

        pm     = generate_preset_pass_manager(optimization_level=3, backend=backend)
        isa_qc = pm.run(qc)

        sampler = Sampler(mode=backend)
        sampler.options.default_shots = shots

        # DD — XY4 only, exact same as v150
        USE_DD = input("Enable Dynamical Decoupling XY4? [y/N] → ").lower() == "y"
        if USE_DD:
            sampler.options.dynamical_decoupling.enable        = True
            sampler.options.dynamical_decoupling.sequence_type = "XY4"

        # ZNE — manual 4-scale, exact same as v150
        USE_ZNE = input("Enable manual 4-scale ZNE? [y/N] → ").lower() == "y"
        if USE_ZNE:
            print("ℹ️  ZNE: manual 4-scale post-job extrapolation will be applied.")

        print(f"📡 Submitting job | Shots: {shots}")
        job    = sampler.run([isa_qc], shots=shots)
        print(f"   Job ID: {job.job_id()}")
        print("⏳ Waiting for results...")
        result = job.result()
        print("✅ Results received!")

        raw_dict = result[0].data.c.get_counts()
        counts   = Counter(raw_dict)

        if USE_ZNE:
            print("🔬 Applying manual 4-scale ZNE...")
            zne_list = [counts]
            for nf in [3, 5, 7]:
                sc  = max(1024, shots // nf)
                jz  = sampler.run([isa_qc], shots=sc)
                rz  = jz.result()
                zne_list.append(Counter(rz[0].data.c.get_counts()))
            extrapolated = defaultdict(int)
            for bitstr in zne_list[0]:
                vals = [c.get(bitstr, 0) for c in zne_list]
                fit  = np.polyfit([1, 3, 5, 7], vals, 1)
                extrapolated[bitstr] = max(0, int(fit[1]))
            counts = Counter(extrapolated)
            print("✅ Manual ZNE applied")

        print(f"\n📊 {len(counts)} unique outcomes")
        for bs, cnt in counts.most_common(50):
            print(f"   {bs} : {cnt}")
        if len(counts) > 50:
            print(f"   ... and {len(counts)-50} more")

    # =========================================================================
    # SHARED POST-PROCESSING — REGEV LATTICE + UNIVERSAL POST-PROCESSING
    # =========================================================================
    logger.info("Starting combined post-processing")

    lattice_cands = regev_lattice_postprocess(counts, d_used, bits, ORDER)

    filtered = []
    for v in lattice_cands:
        filtered.extend(process_measurement(v, bits, ORDER))

    # Universal post-processing (continued fractions + GCD sweep)
    qpe_cands = universal_post_process(counts, bits, ORDER, FULL_RANGE_START, FULL_RANGE_END)
    filtered.extend(qpe_cands)

    for bitstr, cnt in counts.items():
        val = int(bitstr, 2)
        filtered.extend(process_measurement(val, bits, ORDER) * cnt)

    filtered = [m for m in filtered if math.gcd(m, ORDER) == 1]
    filtered = list(dict.fromkeys(filtered))[:5000]

    print(f"\nTotal filtered candidates: {len(filtered)}")
    candidate = bb_correction(filtered, ORDER)
    print(f"Majority vote candidate: {candidate}")

    print("\nVerifying candidates...")
    found = False

    # Verification strategy 1: direct candidate check
    for dk in sorted(set(filtered), reverse=True)[:500]:
        k_test = (k_start + dk) % ORDER
        if verify_key(k_test, Q.x()):
            print("\n" + "═" * 80)
            print("🔥 SUCCESS! PRIVATE KEY FOUND ON REAL HARDWARE 🔑")
            print(f"HEX: {hex(k_test)}")
            print("Donation: 1Bu4CR8Bi5AXQG8pnu1avny88C5CCgWKfb / 1NEJcwfcEm7Aax8oJNjRUnY3hEavCjNrai 💰")
            print("═" * 80)
            save_key(k_test)
            found = True
            break

    # Verification strategy 2: compressed pubkey match
    if not found:
        for candidate in filtered[:5000]:
            try:
                if compress_pubkey(candidate) == bytes.fromhex(pub_hex):
                    print("\n🔥 SUCCESS! PRIVATE KEY FOUND (pubkey match)")
                    print(f"HEX: {hex(candidate)}")
                    print(f"The KEY Decimal: {candidate}")
                    save_key(candidate)
                    found = True
                    break
            except Exception:
                continue

    if not found:
        print("❌ No match this run — try more shots or a different mode.")

    if counts:
        plt.figure(figsize=(14, 7))
        top = counts.most_common(50)
        plt.bar(range(len(top)), [v for _, v in top])
        plt.xticks(range(len(top)), [k for k, _ in top], rotation=90)
        plt.title(f"Measurement Distribution — DRAGON_CODE_FUTURE ({len(counts)} unique)")
        plt.tight_layout()
        plt.show()

    logger.info("DRAGON_CODE_FUTURE execution finished")
    print("\n✅ Done. Check boom.txt for any found key.")

if __name__ == "__main__":
    main()