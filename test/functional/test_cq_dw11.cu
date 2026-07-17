// Phase 6 Stage 2 - Create-CQ command DWORD11 bitfield unit test
//
// Verifies that nvm_cq_dw11() constructs the Create I/O Completion Queue
// command's DWORD11 correctly according to the NVMe Base Specification:
//   bit 0      PC  (Physically Contiguous)
//   bit 1      IEN (Interrupts Enabled)
//   bits 31:16 IV  (Interrupt Vector)
//
// This is a pure logical unit test and does not touch any device or GPU
// (command-line arguments are ignored).
//
// RED: fails to compile if nvm_cq_dw11() is not yet implemented.
// GREEN: all assertions pass after implementing it in nvm_cmd.h.

#include <cstdio>
#include <cstdint>
#include <libnvm/nvm_cmd.h>

static int failures = 0;

#define CHECK_EQ(actual, expected, desc)                                   \
    do {                                                                    \
        uint32_t _a = (actual), _e = (expected);                            \
        if (_a != _e) {                                                     \
            printf("  FAIL: %s: expected 0x%08x, got 0x%08x\n",             \
                   desc, _e, _a);                                           \
            failures++;                                                     \
        }                                                                   \
    } while (0)

int main(int, char**) {
    // 1. Default: no interrupt, physically contiguous (PC=1) -- matches the
    //    old code path (!need_prp)
    //    need_prp=false ⇒ PC=1；iv=0, ien=false
    CHECK_EQ(nvm_cq_dw11(0, false, /*prp_contiguous=*/true),
             0x00000001u, "PC only (contiguous, no IRQ)");

    // 2. Non-contiguous (PC=0), no interrupt
    CHECK_EQ(nvm_cq_dw11(0, false, /*prp_contiguous=*/false),
             0x00000000u, "non-contiguous, no IRQ");

    // 3. Interrupts enabled, vector 0, contiguous => IEN(bit1)=1, PC(bit0)=1
    CHECK_EQ(nvm_cq_dw11(0, true, /*prp_contiguous=*/true),
             0x00000003u, "IEN + PC, vector 0");

    // 4. Interrupts enabled, vector 5, contiguous => IV=5<<16 | IEN | PC
    CHECK_EQ(nvm_cq_dw11(5, true, /*prp_contiguous=*/true),
             (5u << 16) | 0x3u, "IEN + PC, vector 5");

    // 5. Interrupts enabled, vector 16 (within this drive's 17-vector range)
    CHECK_EQ(nvm_cq_dw11(16, true, /*prp_contiguous=*/true),
             (16u << 16) | 0x3u, "IEN + PC, vector 16");

    // 6. Non-zero vector but IEN=false => IV is still written, IEN bit is 0
    //    (the controller ignores IV when IEN=0, but the bitfield should still
    //    reflect the arguments faithfully)
    CHECK_EQ(nvm_cq_dw11(5, false, /*prp_contiguous=*/true),
             (5u << 16) | 0x1u, "IV set but IEN=0");

    if (failures == 0) {
        printf("PASS\n");
        return 0;
    }
    printf("FAIL: %d bitfield mismatch(es)\n", failures);
    return 1;
}
