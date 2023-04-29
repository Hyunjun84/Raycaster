#include "./kernel/fcc_v_common.cl"
#include "./kernel/fcc_v3_common.cl"

__inline void fetch_coefficients(float* c, image3d_t vol, int3 org, int3 R, uint4 P)
{
    // Fetch coefficients
    int3 dir2x = 2*R*shuffle((int4)(1,0,0,0), P).xyz;
    int3 dir2y = 2*R*shuffle((int4)(0,1,0,0), P).xyz;
    int3 dir2z = 2*R*shuffle((int4)(0,0,1,0), P).xyz;


    int3 offset = org;              c0 = read_imagef(vol, sp, (int4)(offset, 1)).r;  //( 0, 0, 0)
    offset -= dir2z;                c15 = read_imagef(vol, sp, (int4)(offset, 1)).r; //( 0, 0,-2)
    offset += (dir2x+dir2y)>>1;     c19 = read_imagef(vol, sp, (int4)(offset, 1)).r; //( 1, 1,-2)
    offset += 2*dir2z;              c20 = read_imagef(vol, sp, (int4)(offset, 1)).r; //( 1, 1, 2)
    offset -= (dir2x+dir2y)>>1;     c16 = read_imagef(vol, sp, (int4)(offset, 1)).r; //( 0, 0, 2)
    offset -= (dir2y+dir2z);        c14 = read_imagef(vol, sp, (int4)(offset, 1)).r; //( 0,-2, 0)
    offset += 2*dir2y;              c17 = read_imagef(vol, sp, (int4)(offset, 1)).r; //( 0, 2, 0)
    offset += (dir2x-dir2z)>>1;     c21 = read_imagef(vol, sp, (int4)(offset, 1)).r; //( 1, 2,-1)
    offset += dir2z;                c22 = read_imagef(vol, sp, (int4)(offset, 1)).r; //( 1, 2, 1)
    offset -= dir2y;                c11 = read_imagef(vol, sp, (int4)(offset, 1)).r; //( 1, 0, 1)
    offset -= dir2x;                c3 = read_imagef(vol, sp, (int4)(offset, 1)).r;  //(-1, 0, 1)
    offset -= dir2z;                c2 = read_imagef(vol, sp, (int4)(offset, 1)).r;  //(-1, 0,-1)
    offset += dir2x;                c10 = read_imagef(vol, sp, (int4)(offset, 1)).r; //( 1, 0,-1)
    offset += (dir2y+dir2z)>>1;     c12 = read_imagef(vol, sp, (int4)(offset, 1)).r; //( 1, 1, 0)
    offset -= dir2x;                c4 = read_imagef(vol, sp, (int4)(offset, 1)).r;  //(-1, 1, 0)
    offset -= dir2y;                c1 = read_imagef(vol, sp, (int4)(offset, 1)).r;  //(-1,-1, 0)
    offset += dir2x;                c9 = read_imagef(vol, sp, (int4)(offset, 1)).r;  //( 1,-1, 0)
    offset -= (dir2x+dir2z)>>1;     c5 = read_imagef(vol, sp, (int4)(offset, 1)).r;  //( 0,-1,-1)
    offset += dir2z;                c6 = read_imagef(vol, sp, (int4)(offset, 1)).r;  //( 0,-1, 1)
    offset += (dir2y-dir2z);        c7 = read_imagef(vol, sp, (int4)(offset, 1)).r;  //( 0, 1,-1)
    offset += dir2z;                c8 = read_imagef(vol, sp, (int4)(offset, 1)).r;  //( 0, 1, 1)
    offset += dir2x;                c26 = read_imagef(vol, sp, (int4)(offset, 1)).r; //( 2, 1, 1)
    offset -= dir2z;                c25 = read_imagef(vol, sp, (int4)(offset, 1)).r; //( 2, 1,-1)
    offset += (dir2z-dir2y);        c24 = read_imagef(vol, sp, (int4)(offset, 1)).r; //( 2,-1, 1)
    offset -= dir2z;                c23 = read_imagef(vol, sp, (int4)(offset, 1)).r; //( 2,-1,-1)
    offset += (dir2y+dir2z)>>1;     c18 = read_imagef(vol, sp, (int4)(offset, 1)).r; //( 2, 0, 0)
    offset -= 2*dir2x;              c13 = read_imagef(vol, sp, (int4)(offset, 1)).r; //(-2, 0, 0)
}
