/*
 * g6ct_cli.c - render the port over raw interleaved float32.
 *
 *   g6ct_cli <in.raw> <out2.raw> [mask] [channels]
 *
 * channels: 8 (default, FL FR FC LFE BL BR SL SR) or 12
 *           (FL FR FC LFE BL BR FLC FRC BC SL SR TC)
 * out:      2 channels interleaved float32
 *
 * Used by the verification script to produce the reference signal that the
 * PipeWire graph must reproduce.
 */
#include "g6ct.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BLOCK 1024

static const int slot8[8]  = { 0, 1, 2, 3, 4, 5, 9, 10 };
static const int slot12[12] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

int main(int argc, char **argv)
{
    FILE *fi, *fo;
    g6ct_state st;
    unsigned mask = G6CT_MASK_7_1;
    unsigned inch = 8;
    float *ibuf;
    static g6ct_real obuf[2 * BLOCK];
    size_t got;
    int failed = 0;

    if (argc < 3 || argc > 5) {
        fprintf(stderr, "usage: %s <in.raw> <out2.raw> [mask] [channels]\n",
                argv[0]);
        return 2;
    }
    if (argc > 3) {
        char *end;
        unsigned long v = strtoul(argv[3], &end, 0);

        if (*argv[3] == '\0' || *end != '\0' || v > 0xFFFFUL) {
            fprintf(stderr, "%s: invalid mask '%s'\n", argv[0], argv[3]);
            return 2;
        }
        mask = (unsigned)v;
    }
    if (argc > 4) {
        char *end;
        unsigned long v = strtoul(argv[4], &end, 0);

        if (*argv[4] == '\0' || *end != '\0' || (v != 8 && v != 12)) {
            fprintf(stderr, "%s: channels must be 8 or 12\n", argv[0]);
            return 2;
        }
        inch = (unsigned)v;
    }

    fi = fopen(argv[1], "rb");
    if (!fi) { perror(argv[1]); return 1; }
    fo = fopen(argv[2], "wb");
    if (!fo) { perror(argv[2]); fclose(fi); return 1; }

    ibuf = malloc(sizeof(float) * inch * BLOCK);
    if (!ibuf) {
        fprintf(stderr, "oom\n");
        fclose(fi);
        fclose(fo);
        return 1;
    }

    g6ct_init(&st, mask);

    while ((got = fread(ibuf, sizeof(float) * inch, BLOCK, fi)) > 0) {
        static g6ct_real tmp[12][BLOCK];
        const g6ct_real *in[12] = { NULL };
        size_t i;
        unsigned c;

        for (i = 0; i < got; i++)
            for (c = 0; c < inch; c++)
                tmp[c][i] = (g6ct_real)ibuf[i * inch + c];
        for (c = 0; c < inch; c++)
            in[inch == 8 ? slot8[c] : slot12[c]] = tmp[c];

        g6ct_process(&st, in, obuf, obuf + BLOCK, got);
        /* interleave */
        for (i = 0; i < got; i++) {
            float pair[2];

            pair[0] = (float)obuf[i];
            pair[1] = (float)obuf[BLOCK + i];
            if (fwrite(pair, sizeof(float), 2, fo) != 2) {
                perror(argv[2]);
                failed = 1;
                break;
            }
        }
        if (failed)
            break;
    }
    if (ferror(fi)) {
        perror(argv[1]);
        failed = 1;
    }

    free(ibuf);
    if (fclose(fi) != 0) {
        perror(argv[1]);
        failed = 1;
    }
    if (fclose(fo) != 0) {
        perror(argv[2]);
        failed = 1;
    }
    return failed;
}
