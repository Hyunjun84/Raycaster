#include "./kernel/fcc_v_common.cl"
#include "./kernel/fcc_v2_common.cl"

__inline void fetch_coefficients(float* c, image3d_t vol, int3 org, int3 R, uint4 P)
{
    int3 dirx = R*shuffle((int4)(1,0,0,0), P).xyz;
    int3 diry = R*shuffle((int4)(0,1,0,0), P).xyz;
    int3 dirz = R*shuffle((int4)(0,0,1,0), P).xyz;

    int3 p = org;
    int offset = ((p.x&0x01)<<1)+(p.y&0x01);
    c[0] = read_imagef(vol, sp, (int4)(p>>1, 1))[offset];
    p = (org+2*dirx);
    c[1] = read_imagef(vol, sp, (int4)(p>>1, 1))[offset];

    p = org+diry-dirz;
    offset = ((p.x&0x01)<<1)+(p.y&0x01);
    c[2] = read_imagef(vol, sp, (int4)(p>>1, 1))[offset];
    p = org+diry+dirz;
    c[3] = read_imagef(vol, sp, (int4)(p>>1, 1))[offset];

    p = org+dirx-dirz;
    offset = ((p.x&0x01)<<1)+(p.y&0x01);
    c[4] = read_imagef(vol, sp, (int4)(p>>1, 1))[offset];
    p = org+dirx+dirz;
    c[5] = read_imagef(vol, sp, (int4)(p>>1, 1))[offset];

    p = org+dirx-diry;
    offset = ((p.x&0x01)<<1)+(p.y&0x01);
    c[6] = read_imagef(vol, sp, (int4)(p>>1, 1))[offset];
    p = org+dirx+diry;
    c[7] = read_imagef(vol, sp, (int4)(p>>1, 1))[offset];
}
