#include "./kernel/fcc_v_common.cl"
#include "./kernel/fcc_v2_common.cl"

__inline void fetch_coefficients(float* c, image3d_t vol, int3 org, int3 R, uint4 P)
{
    int3 dirx = R*shuffle((int4)(1,0,0,0), P).xyz;
    int3 diry = R*shuffle((int4)(0,1,0,0), P).xyz;
    int3 dirz = R*shuffle((int4)(0,0,1,0), P).xyz;

    c[0] = read_imagef(vol, sp, (int4)(org,1)).r;
    c[1] = read_imagef(vol, sp, (int4)(org+2*dirx,1)).r;

    c[2] = read_imagef(vol, sp, (int4)(org+diry-dirz,1)).r;
    c[3] = read_imagef(vol, sp, (int4)(org+diry+dirz,1)).r;

    c[4] = read_imagef(vol, sp, (int4)(org+dirx-dirz,1)).r;
    c[5] = read_imagef(vol, sp, (int4)(org+dirx+dirz,1)).r;

    c[6] = read_imagef(vol, sp, (int4)(org+dirx-diry,1)).r;
    c[7] = read_imagef(vol, sp, (int4)(org+dirx+diry,1)).r;
}
