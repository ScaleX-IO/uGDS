// SPDX-License-Identifier: BSD-3-Clause
/*
 * KUnit tests for sg_flatten_to_addrs() -- HIP/dmabuf SG table flatten.
 *
 * Build: make BUILD_HIP=1 BUILD_KUNIT=1
 * Run:   modprobe sg_flatten_test  (or kunit.py run)
 */

#include <kunit/test.h>
#include <linux/scatterlist.h>
#include "../map.h"

#define PAGE   4096UL

/* Cleanup action to free the sg_table's internal scatterlist */
static void free_sg_table(void *ctx)
{
    struct sg_table *sgt = (struct sg_table *)ctx;
    sg_free_table(sgt);
}

/*
 * Build a synthetic sg_table with the given DMA lengths.
 * DMA addresses are sequential starting from 0x10000000.
 * The internal scatterlist is auto-freed when the test exits.
 */
static struct sg_table* build_sg_table(struct kunit* test,
                                        const unsigned int* lengths,
                                        int n_entries)
{
    struct sg_table* sgt;
    struct scatterlist* sg;
    int i;

    sgt = kunit_kzalloc(test, sizeof(*sgt), GFP_KERNEL);
    KUNIT_ASSERT_EQ(test, sg_alloc_table(sgt, n_entries, GFP_KERNEL), 0);

    /* Register cleanup -- sg_free_table frees the internal scatterlist
     * that sg_alloc_table allocated separately from kunit_kzalloc.
     * or_reset immediately frees on registration failure, so we must
     * assert before touching sgt->sgl. */
    KUNIT_ASSERT_EQ(test,
        kunit_add_action_or_reset(test, free_sg_table, sgt), 0);

    for_each_sg(sgt->sgl, sg, n_entries, i)
    {
        sg_dma_address(sg) = 0x10000000UL + (u64)i * PAGE * 4;
        sg_dma_len(sg) = lengths[i];
    }

    return sgt;
}

/* --- Test cases --- */

static void test_page_aligned(struct kunit* test)
{
    /* Two entries, each 2 pages -> 4 pages total */
    static const unsigned int lengths[] = { 2 * PAGE, 2 * PAGE };
    struct sg_table* sgt = build_sg_table(test, lengths, 2);
    u64 addrs[4] = { 0 };

    KUNIT_EXPECT_EQ(test,
        sg_flatten_to_addrs(sgt, addrs, 4, PAGE, 0), 0);
    KUNIT_EXPECT_EQ(test, addrs[0], 0x10000000UL);
    KUNIT_EXPECT_EQ(test, addrs[1], 0x10000000UL + PAGE);
    KUNIT_EXPECT_EQ(test, addrs[2], 0x10000000UL + 4 * PAGE);
    KUNIT_EXPECT_EQ(test, addrs[3], 0x10000000UL + 5 * PAGE);
}

static void test_residual_fail(struct kunit* test)
{
    /* 8192 (=2 pages) + 4608 (=1 page + 512 residual)
     * expected_pages=4 -> page_idx=3 < 4 when residual hit -> -EINVAL */
    static const unsigned int lengths[] = { 2 * PAGE, PAGE + 512 };
    struct sg_table* sgt = build_sg_table(test, lengths, 2);
    u64 addrs[4] = { 0 };

    KUNIT_EXPECT_EQ(test,
        sg_flatten_to_addrs(sgt, addrs, 4, PAGE, 0), -EINVAL);
}

static void test_residual_ignored_when_full(struct kunit* test)
{
    /* Same SG layout, but expected_pages=3.
     * 8192 -> 2 pages, 4608 -> 1 page. page_idx=3 == expected ->
     * residual 512 silently ignored. */
    static const unsigned int lengths[] = { 2 * PAGE, PAGE + 512 };
    struct sg_table* sgt = build_sg_table(test, lengths, 2);
    u64 addrs[3] = { 0 };

    KUNIT_EXPECT_EQ(test,
        sg_flatten_to_addrs(sgt, addrs, 3, PAGE, 0), 0);
}

static void test_residual_at_first_entry(struct kunit* test)
{
    /* 2048 < PAGE_SIZE -> residual at first entry */
    static const unsigned int lengths[] = { 2048, 6144 };
    struct sg_table* sgt = build_sg_table(test, lengths, 2);
    u64 addrs[2] = { 0 };

    KUNIT_EXPECT_EQ(test,
        sg_flatten_to_addrs(sgt, addrs, 2, PAGE, 0), -EINVAL);
}

static void test_offset_consumption(struct kunit* test)
{
    /* hsa_offset=4096: first entry loses 1 page -> 1 page from each entry */
    static const unsigned int lengths[] = { 2 * PAGE, 2 * PAGE };
    struct sg_table* sgt = build_sg_table(test, lengths, 2);
    u64 addrs[3] = { 0 };

    KUNIT_EXPECT_EQ(test,
        sg_flatten_to_addrs(sgt, addrs, 3, PAGE, PAGE), 0);
    /* First entry: addr+4096 -> 1 page at 0x10001000 */
    KUNIT_EXPECT_EQ(test, addrs[0], 0x10000000UL + PAGE);
    /* Second entry: 2 pages */
    KUNIT_EXPECT_EQ(test, addrs[1], 0x10000000UL + 4 * PAGE);
    KUNIT_EXPECT_EQ(test, addrs[2], 0x10000000UL + 5 * PAGE);
}

static void test_offset_spans_entry(struct kunit* test)
{
    /* hsa_offset=4096 spans the entire first 4096-byte entry */
    static const unsigned int lengths[] = { PAGE, 2 * PAGE };
    struct sg_table* sgt = build_sg_table(test, lengths, 2);
    u64 addrs[2] = { 0 };

    KUNIT_EXPECT_EQ(test,
        sg_flatten_to_addrs(sgt, addrs, 2, PAGE, PAGE), 0);
    /* First entry fully consumed by offset; second: 2 pages */
    KUNIT_EXPECT_EQ(test, addrs[0], 0x10000000UL + 4 * PAGE);
    KUNIT_EXPECT_EQ(test, addrs[1], 0x10000000UL + 5 * PAGE);
}

static void test_exact_page_multiple(struct kunit* test)
{
    /* Three entries, each exactly 1 page -> 3 pages, no residual */
    static const unsigned int lengths[] = { PAGE, PAGE, PAGE };
    struct sg_table* sgt = build_sg_table(test, lengths, 3);
    u64 addrs[3] = { 0 };

    KUNIT_EXPECT_EQ(test,
        sg_flatten_to_addrs(sgt, addrs, 3, PAGE, 0), 0);
    KUNIT_EXPECT_EQ(test, addrs[0], 0x10000000UL);
    KUNIT_EXPECT_EQ(test, addrs[1], 0x10000000UL + 4 * PAGE);
    KUNIT_EXPECT_EQ(test, addrs[2], 0x10000000UL + 8 * PAGE);
}

static void test_offset_too_large(struct kunit* test)
{
    /* hsa_offset exceeds total SG length -> page_idx=0 < expected */
    static const unsigned int lengths[] = { PAGE };
    struct sg_table* sgt = build_sg_table(test, lengths, 1);
    u64 addrs[1] = { 0 };

    KUNIT_EXPECT_EQ(test,
        sg_flatten_to_addrs(sgt, addrs, 1, PAGE, 2 * PAGE), -EINVAL);
}

static void test_empty_sg(struct kunit* test)
{
    /* len=0 entry -> no pages extracted -> page_idx=0 < expected */
    static const unsigned int lengths[] = { 0 };
    struct sg_table* sgt = build_sg_table(test, lengths, 1);
    u64 addrs[1] = { 0 };

    KUNIT_EXPECT_EQ(test,
        sg_flatten_to_addrs(sgt, addrs, 1, PAGE, 0), -EINVAL);
}

static struct kunit_case sg_flatten_test_cases[] = {
    KUNIT_CASE(test_page_aligned),
    KUNIT_CASE(test_residual_fail),
    KUNIT_CASE(test_residual_ignored_when_full),
    KUNIT_CASE(test_residual_at_first_entry),
    KUNIT_CASE(test_offset_consumption),
    KUNIT_CASE(test_offset_spans_entry),
    KUNIT_CASE(test_exact_page_multiple),
    KUNIT_CASE(test_offset_too_large),
    KUNIT_CASE(test_empty_sg),
    {}
};

static struct kunit_suite sg_flatten_test_suite = {
    .name = "ugds_sg_flatten",
    .test_cases = sg_flatten_test_cases,
};

kunit_test_suite(sg_flatten_test_suite);

MODULE_LICENSE("Dual BSD/GPL");
MODULE_DESCRIPTION("uGDS SG flatten KUnit tests");
