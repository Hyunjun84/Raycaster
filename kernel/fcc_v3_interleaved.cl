#include "./kernel/fcc_v_common.cl"
#include "./kernel/fcc_v3_common.cl"

__inline void fetch_coefficients(float* c, image3d_t vol, int3 org, int3 R, uint4 P)
{
    // Fetch coefficients
    int3 dir2x = R*shuffle((int4)(2,0,0,0), P).xyz;
    int3 dir2y = R*shuffle((int4)(0,2,0,0), P).xyz;
    int3 dir2z = R*shuffle((int4)(0,0,2,0), P).xyz;

    int3 p = org;              int2 offset = (int2)(p.x&0x01, p.y&0x01);
    c0 = read_imagef(vol, sp, (int4)((p.xy>>1)*2+offset, p.z>>1, 1)).x;  //( 0, 0, 0)
    p -= dir2z;                c15 = read_imagef(vol, sp, (int4)((p.xy>>1)*2+offset, p.z>>1, 1)).x; //( 0, 0,-2)
    p += 2*dir2z;              c16 = read_imagef(vol, sp, (int4)((p.xy>>1)*2+offset, p.z>>1, 1)).x; //( 0, 0, 2)
    p -= dir2y+dir2z;          c14 = read_imagef(vol, sp, (int4)((p.xy>>1)*2+offset, p.z>>1, 1)).x; //( 0,-2, 0)
    p += 2*dir2y;              c17 = read_imagef(vol, sp, (int4)((p.xy>>1)*2+offset, p.z>>1, 1)).x; //( 0, 2, 0)
    p -= dir2x+dir2y;          c13 = read_imagef(vol, sp, (int4)((p.xy>>1)*2+offset, p.z>>1, 1)).x; //(-2, 0, 0)
    p += 2*dir2x;              c18 = read_imagef(vol, sp, (int4)((p.xy>>1)*2+offset, p.z>>1, 1)).x; //( 2, 0, 0)

    p -= (dir2y-dir2z)>>1;     offset = (int2)(p.x&0x01, p.y&0x01);
    c24 = read_imagef(vol, sp, (int4)((p.xy>>1)*2+offset, p.z>>1, 1)).x; //( 2,-1, 1)
    p -= dir2z;                c23 = read_imagef(vol, sp, (int4)((p.xy>>1)*2+offset, p.z>>1, 1)).x; //( 2,-1,-1)
    p += dir2y;                c25 = read_imagef(vol, sp, (int4)((p.xy>>1)*2+offset, p.z>>1, 1)).x; //( 2, 1,-1)
    p += dir2z;                c26 = read_imagef(vol, sp, (int4)((p.xy>>1)*2+offset, p.z>>1, 1)).x; //( 2, 1, 1)
    p -= dir2x;                c8 = read_imagef(vol, sp, (int4)((p.xy>>1)*2+offset, p.z>>1, 1)).x;  //( 0, 1, 1)
    p -= dir2z;                c7 = read_imagef(vol, sp, (int4)((p.xy>>1)*2+offset, p.z>>1, 1)).x;  //( 0, 1,-1)
    p -= dir2y;                c5 = read_imagef(vol, sp, (int4)((p.xy>>1)*2+offset, p.z>>1, 1)).x;  //( 0,-1,-1)
    p += dir2z;                c6 = read_imagef(vol, sp, (int4)((p.xy>>1)*2+offset, p.z>>1, 1)).x;  //( 0,-1, 1)

    p += (dir2x+dir2y)>>1;     offset = (int2)(p.x&0x01, p.y&0x01);
    c11 = read_imagef(vol, sp, (int4)((p.xy>>1)*2+offset, p.z>>1, 1)).x; //( 1, 0, 1)
    p -= dir2x;                c3 = read_imagef(vol, sp, (int4)((p.xy>>1)*2+offset, p.z>>1, 1)).x;  //(-1, 0, 1)
    p -= dir2z;                c2 = read_imagef(vol, sp, (int4)((p.xy>>1)*2+offset, p.z>>1, 1)).x;  //(-1, 0,-1)
    p += dir2x;                c10 = read_imagef(vol, sp, (int4)((p.xy>>1)*2+offset, p.z>>1, 1)).x; //( 1, 0,-1)
    p += dir2y;                c21 = read_imagef(vol, sp, (int4)((p.xy>>1)*2+offset, p.z>>1, 1)).x; //( 1, 2,-1)
    p += dir2z;                c22 = read_imagef(vol, sp, (int4)((p.xy>>1)*2+offset, p.z>>1, 1)).x; //( 1, 2, 1)

    p += (dir2z-dir2y)>>1;     offset = (int2)(p.x&0x01, p.y&0x01);
    c20 = read_imagef(vol, sp, (int4)((p.xy>>1)*2+offset, p.z>>1, 1)).x; //( 1, 1, 2)
    p -= 2*dir2z;              c19 = read_imagef(vol, sp, (int4)((p.xy>>1)*2+offset, p.z>>1, 1)).x; //( 1, 1,-2)
    p += dir2z;                c12 = read_imagef(vol, sp, (int4)((p.xy>>1)*2+offset, p.z>>1, 1)).x; //( 1, 1, 0)
    p -= dir2x;                c4 = read_imagef(vol, sp, (int4)((p.xy>>1)*2+offset, p.z>>1, 1)).x;  //(-1, 1, 0)
    p -= dir2y;                c1 = read_imagef(vol, sp, (int4)((p.xy>>1)*2+offset, p.z>>1, 1)).x;  //(-1,-1, 0)
    p += dir2x;                c9 = read_imagef(vol, sp, (int4)((p.xy>>1)*2+offset, p.z>>1, 1)).x;  //( 1,-1, 0)
}
