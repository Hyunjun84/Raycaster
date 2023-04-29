#include "./kernel/fcc_v_common.cl"
#include "./kernel/fcc_v2_common.cl"

__inline void fetch_coefficients(float* c, image3d_t vol, int3 org, int3 R, uint4 P)
{
    int3 dirx = R*shuffle((int4)(1,0,0,0), P).xyz;
    int3 diry = R*shuffle((int4)(0,1,0,0), P).xyz;
    int3 dirz = R*shuffle((int4)(0,0,1,0), P).xyz;

    int2 pad = get_image_dim(vol).xy/2;

    int3 p = org;
    int2 offset = (int2)(p.x&0x01, p.y&0x01)*pad;
    c[0] = read_imagef(vol, sp, (int4)((p>>1)+offset, 1)).x
    p = (org+2*dirx);
    c[1] = read_imagef(vol, sp, (int4)((p>>1)+offset, 1)).x

    p = org+diry-dirz;
    offset = (int2)(p.x&0x01, p.y&0x01)*pad;
    c[2] = read_imagef(vol, sp, (int4)((p>>1)+offset, 1)).x
    p = org+diry+dirz;
    c[3] = read_imagef(vol, sp, (int4)((p>>1)+offset, 1)).x

    p = org+dirx-dirz;
    offset = (int2)(p.x&0x01, p.y&0x01)*pad;
    c[4] = read_imagef(vol, sp, (int4)((p>>1)+offset, 1)).x
    p = org+dirx+dirz;
    c[5] = read_imagef(vol, sp, (int4)((p>>1)+offset, 1)).x

    p = org+dirx-diry;
    offset = (int2)(p.x&0x01, p.y&0x01)*pad;
    c[6] = read_imagef(vol, sp, (int4)((p>>1)+offset, 1)).x
    p = org+dirx+diry;
    c[7] = read_imagef(vol, sp, (int4)((p>>1)+offset, 1)).x
}
